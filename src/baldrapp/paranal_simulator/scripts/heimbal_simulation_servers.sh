#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Baldr/Heimdallr simulation startup helper
# ============================================================
# Starts:
#   - shared-memory creator
#   - simulated MDM server
#   - fake CRED1/MDS ZMQ server
#   - Baldr optical simulator with runtime-control ZMQ
#   - lab MDM GUI
#   - shmview GUI
#   - Baldr simulator control GUI
#
# Usage:
#   ./heimbal_simulation_servers.sh start
#   ./heimbal_simulation_servers.sh stop
#   ./heimbal_simulation_servers.sh status
#   ./heimbal_simulation_servers.sh tail
#
# Optional override:
#   SIM_SCRIPT=baldr_sim_updated_runtime_control.py ./heimbal_simulation_servers.sh start
# ============================================================



# ============================================================
# Baldr/Heimdallr simulation startup helper
# ============================================================

# --- Config ---
# 1. Find where this script lives (now in the 'scripts' folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 2. Set the main simulation directory one level up (paranal_simulator)
SIM_DIR="$(dirname "$SCRIPT_DIR")"

# 3. Find the git root and venv
GIT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
VENV_PATH="${GIT_ROOT}/.venv/bin/activate"

# 4. Run everything from the main paranal_simulator directory

# 4. Run everything from the main paranal_simulator directory
cd "$SIM_DIR"

# CHANGED: Moved logs and run files to the project root
RUNDIR="${GIT_ROOT}/.run"
LOGDIR="${GIT_ROOT}/.logs"
PIDFILE="${RUNDIR}/pids"
mkdir -p "$RUNDIR" "$LOGDIR"



# 5. Update paths to point to the new subdirectories
SIM_SCRIPT="scripts/baldr_sim_updated_runtime_control.py"
GUI_SCRIPT="guis/baldr_sim_control_gui.py"

# Common Python launch prefix.
PY_PREFIX="export PYTHONUNBUFFERED=1 && export PYTHONPATH=\"${GIT_ROOT}/src\":\${PYTHONPATH:-} && source \"${VENV_PATH}\""

# --- Process Definitions ---
# 6. Prepend 'scripts/' to the binaries and python files
declare -A PROCS=(
  [shm_creator_sim]="tail -f /dev/null | ./scripts/shm_creator_sim"
  [sim_mdm_server]="tail -f /dev/null | ./scripts/sim_mdm_server"
  [fake_zmq_server]="bash -c '${PY_PREFIX} && python -u scripts/fake_asgard_ZMQ_CRED1_server.py'"
  [baldr_sim]="bash -c '${PY_PREFIX} && python -u ${SIM_SCRIPT}'"
  [lab_mdm_gui]="bash -c 'source \"${VENV_PATH}\" && lab-MDM-control'"
  [shmview_gui]="bash -c 'source \"${VENV_PATH}\" && shmview /dev/shm/cred1.im.shm'"
  [baldr_sim_gui]="bash -c '${PY_PREFIX} && python -u ${GUI_SCRIPT}'"
)


START_ORDER=(
  shm_creator_sim
  sim_mdm_server
  fake_zmq_server
  baldr_sim
  lab_mdm_gui
  shmview_gui
  baldr_sim_gui
)

# --- Helpers ---
have_setsid() { command -v setsid >/dev/null 2>&1; }

is_running_pid() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

pgid_of() {
  local pid="${1:-}"
  [[ -z "$pid" ]] && return 0
  command -v ps >/dev/null || return 0
  ps -o pgid= -p "$pid" 2>/dev/null | awk '{$1=$1};1'
}

check_required_files() {
  local missing=0

  for f in "scripts/shm_creator_sim" "scripts/sim_mdm_server" "scripts/fake_asgard_ZMQ_CRED1_server.py" "$SIM_SCRIPT" "$GUI_SCRIPT"; do
    if [[ ! -e "$f" ]]; then
      echo "[ERR] Missing required file in ${SIM_DIR}: $f" >&2
      missing=1
    fi
  done

  if [[ ! -f "$VENV_PATH" ]]; then
    echo "[ERR] Missing virtualenv activate script: $VENV_PATH" >&2
    missing=1
  fi

  if [[ "$missing" -ne 0 ]]; then
    exit 1
  fi
}

start_one() {
  local cmd="$1" name="$2" log="${LOGDIR}/${name}.log"
  echo "======== $(date -Is) START ${name} ========" >>"$log"
  echo "[START] ${name}: ${cmd}"

  # Soft existence check for simple './binary'.
  if [[ "$cmd" =~ ^\./[^[:space:]\|\;\(\)]+$ ]]; then
    local word0
    word0="${cmd%% *}"
    if [[ ! -x "$word0" ]]; then
      echo "[ERR] ${name}: command not found or not executable: ${word0}" | tee -a "$log" >&2
      return 1
    fi
  fi

  if have_setsid; then
    nohup bash -c "exec setsid ${cmd}" >>"$log" 2>&1 &
  else
    nohup bash -c "exec ${cmd}" >>"$log" 2>&1 &
  fi

  local pid="$!"
  sleep 0.5

  local pgid
  pgid="$(pgid_of "$pid" || true)"

  if is_running_pid "$pid"; then
    echo "[OK]  ${name} started (pid ${pid}, pgid ${pgid:-?}) --> ${log}"
    printf '%s:%s:%s\n' "$name" "$pid" "${pgid:-}" >> "$PIDFILE"
  else
    echo "[ERR] ${name} failed to start. Last 40 log lines:"
    tail -n 40 "$log" || true
    echo "[HINT] Full log: $log"
    return 1
  fi
}

start_all() {
  if [[ -f "$PIDFILE" ]]; then
    echo "[WARN] PID file exists. Run '$0 status' or '$0 stop' first: $PIDFILE"
    exit 1
  fi

  check_required_files

  : > "$PIDFILE"

  echo "[INFO] Using simulator script: ${SIM_SCRIPT}"
  echo "[INFO] Using GUI script:       ${GUI_SCRIPT}"

  for name in "${START_ORDER[@]}"; do
    cmd="${PROCS[$name]}"
    start_one "$cmd" "$name"

    case "$name" in
      shm_creator_sim|sim_mdm_server|fake_zmq_server)
        sleep 1
        ;;
    esac
  done

  echo "[INFO] Wrote PIDs to $PIDFILE"
}

stop_all() {
  if [[ ! -f "$PIDFILE" ]]; then
    echo "[INFO] Nothing to stop (no $PIDFILE)."
    exit 0
  fi

  while IFS=':' read -r name pid pgid; do
    [[ -z "${name}${pid}" ]] && continue
    if is_running_pid "$pid"; then
      if [[ -n "${pgid:-}" ]]; then
        echo "[STOP] SIGTERM group ${name} (pgid ${pgid})"
        kill -TERM -"${pgid}" 2>/dev/null || true
      else
        echo "[STOP] SIGTERM ${name} (pid ${pid})"
        kill -TERM "${pid}" 2>/dev/null || true
      fi
    else
      echo "[INFO] ${name} already stopped."
    fi
  done < "$PIDFILE"

  for _ in {1..16}; do
    sleep 0.5
    any_left=0
    while IFS=':' read -r _ pid _; do
      [[ -z "$pid" ]] && continue
      if is_running_pid "$pid"; then
        any_left=1
        break
      fi
    done < "$PIDFILE"
    [[ $any_left -eq 0 ]] && break
  done

  while IFS=':' read -r name pid pgid; do
    [[ -z "${name}${pid}" ]] && continue
    if is_running_pid "$pid"; then
      if [[ -n "${pgid:-}" ]]; then
        echo "[FORCE] SIGKILL group ${name} (pgid ${pgid})"
        kill -KILL -"${pgid}" 2>/dev/null || true
      else
        echo "[FORCE] SIGKILL ${name} (pid ${pid})"
        kill -KILL "${pid}" 2>/dev/null || true
      fi
    fi
  done < "$PIDFILE"

  rm -f "$PIDFILE"
  echo "[DONE] All processes stopped."
}

status_all() {
  if [[ ! -f "$PIDFILE" ]]; then
    echo "[INFO] No pidfile; nothing running under this script."
    exit 0
  fi

  while IFS=':' read -r name pid pgid; do
    [[ -z "${name}${pid}" ]] && continue
    if is_running_pid "$pid"; then
      current_pgid="$(pgid_of "$pid" || true)"
      badge="[UP]"
      [[ -n "${pgid:-}" && -n "${current_pgid:-}" && "$pgid" != "$current_pgid" ]] && badge="[WARN]"
      echo "${badge} ${name} (pid ${pid}, pgid ${current_pgid:-?})  log: ${LOGDIR}/${name}.log"
    else
      echo "[DOWN] ${name} (pid ${pid})"
    fi
  done < "$PIDFILE"
}

tail_logs() {
  shopt -s nullglob
  files=("${LOGDIR}"/*.log)
  if (( ${#files[@]} == 0 )); then
    echo "[INFO] No logs yet in ${LOGDIR}"
    exit 0
  fi
  echo "[INFO] Tailing: ${files[*]}"
  tail -n 200 -F "${files[@]}"
}

case "${1:-}" in
  start)  start_all ;;
  stop)   stop_all ;;
  status) status_all ;;
  tail)   tail_logs ;;
  *)
    echo "Usage: $0 {start|stop|status|tail}"
    exit 2
    ;;
esac

