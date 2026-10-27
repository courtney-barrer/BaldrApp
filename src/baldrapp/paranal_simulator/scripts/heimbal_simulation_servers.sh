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
#   SIM_SCRIPT=baldr_simulator.py ./heimbal_simulation_servers.sh start
# ============================================================



# ============================================================
# Baldr/Heimdallr simulation startup helper
# ============================================================

# Resolve installed resources; never require Git or a checkout-local virtualenv.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${BALDR_PYTHON:-python3}"
# Enables direct use from a src checkout as well as an installed package.
export PYTHONPATH="$(dirname "$(dirname "$SIM_DIR")")${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
BALDR_CONFIG_PATH="$("$PYTHON" -c 'from baldrapp.paranal_simulator.paths import configuration_path; print(configuration_path())')"
export BALDR_CONFIG_PATH
STATE_DIR="${BALDR_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/baldrapp}"
mkdir -p "$STATE_DIR"
cd "$STATE_DIR"
STATE_DIR="$PWD"
RUNDIR="$STATE_DIR/run"
LOGDIR="$STATE_DIR/logs"
PIDFILE="$RUNDIR/pids"
mkdir -p "$RUNDIR" "$LOGDIR"
SIM_SCRIPT="${SIM_SCRIPT:-$SCRIPT_DIR/baldr_simulator.py}"
GUI_SCRIPT="$SIM_DIR/guis/baldr_sim_control_gui.py"
declare -A PROCS
START_ORDER=(sim_mdm_server fake_zmq_server lab_mdm_gui shmview_gui baldr_sim baldr_sim_gui)

configure_processes() {
  local native_dir split_json native_cmd
  if [[ -n "${BALDR_NATIVE_BIN_DIR:-}" ]]; then
    native_dir="$BALDR_NATIVE_BIN_DIR"
  else
    native_dir="$("$PYTHON" -c 'from baldrapp_simulator_native import binary_directory; print(binary_directory())' 2>/dev/null)" || {
      echo '[ERR] Install simulator support: pip install "BaldrApp[simulator]"' >&2
      return 1
    }
  fi
  for executable in shm_creator_sim sim_mdm_server; do
    [[ -x "$native_dir/$executable" ]] || {
      echo "[ERR] Missing executable: $native_dir/$executable" >&2
      return 1
    }
  done
  split_json="$("$PYTHON" -c '
import json
from baldrapp.paranal_simulator.paths import configuration_path, resource_path
config = configuration_path()
with config.open() as f:
    value = json.load(f).get("simulator_runtime", {}).get("shared_memory", {}).get("split_json", "cred1_split.json")
print(resource_path(value, config))')"
  [[ -f "$split_json" ]] || { echo "[ERR] Missing configuration: $split_json" >&2; return 1; }
  printf -v SHM_CREATE '%q ' "$native_dir/shm_creator_sim" "$split_json"
  printf -v native_cmd '%q ' "$native_dir/sim_mdm_server"
  PROCS[sim_mdm_server]="tail -f /dev/null | $native_cmd"
  printf -v 'PROCS[fake_zmq_server]' '%q ' "$PYTHON" -u "$SCRIPT_DIR/fake_asgard_ZMQ_CRED1_server.py"
  printf -v 'PROCS[baldr_sim]' '%q ' "$PYTHON" -u "$SIM_SCRIPT"
  printf -v 'PROCS[baldr_sim_gui]' '%q ' "$PYTHON" -u "$GUI_SCRIPT"
  PROCS[lab_mdm_gui]="lab-MDM-control"
  PROCS[shmview_gui]="shmview /dev/shm/cred1.im.shm"
}

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
  for f in "$SCRIPT_DIR/fake_asgard_ZMQ_CRED1_server.py" "$SIM_SCRIPT" "$GUI_SCRIPT"; do
    [[ -f "$f" ]] || { echo "[ERR] Missing script: $f" >&2; return 1; }
  done
  for command in lab-MDM-control shmview setsid; do
    command -v "$command" >/dev/null || {
      echo "[ERR] Missing $command; install BaldrApp[simulator] in the active environment." >&2
      return 1
    }
  done
  "$PYTHON" -c 'import xaosim.shmlib, zmq, PyQt5, pyqtgraph'
}

start_one() {
  local cmd="$1" name="$2"
  local log="${LOGDIR}/${name}.log"
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
    nohup setsid bash -c "$cmd" >>"$log" 2>&1 &
  else
    nohup bash -c "$cmd" >>"$log" 2>&1 &
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

  configure_processes
  check_required_files

  # The creator initializes shared memory and exits; it is not a daemon.
  bash -c "$SHM_CREATE" >"$LOGDIR/shm_creator_sim.log" 2>&1
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

