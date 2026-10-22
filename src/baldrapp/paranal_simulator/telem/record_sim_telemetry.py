#!/usr/bin/env python3
"""Record simulator telemetry and existing dmN/baldrN images to FITS chunks.

Example (publisher enabled in simulator_runtime.publish_opd):
    python record_sim_telemetry.py --beams 1 --output /path/to/recordings

OPD image cubes are metres. DM_CMD is the observed 12x12 combined DM stream,
not guaranteed to be the command used for that camera frame. ZWFS holds the
observed int32 subframe. FRAMES stores counters, quality flags and RMS in nm;
FRAME_META preserves all metadata/quality fields as JSON. Missing DM/image
reads use NaN/zero placeholders with false read-ok flags. PUPIL is saved per
frame so internal/on-sky changes remain unambiguous. CONFIG stores run config.

Only completed snapshots are recorded; gaps are counted between recorded
simulator frame IDs. Initial frames before attachment cannot be counted.
Ctrl+C or SIGTERM flushes partial chunks. An abrupt kill may lose the current
in-memory chunk. Completed files are published by an atomic rename.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import time
import uuid

import numpy as np
from astropy.io import fits

try:
    from .sim_telemetry import TelemetryReader, OPD_PLANES
except ImportError:
    from sim_telemetry import TelemetryReader, OPD_PLANES


def write_chunk(frames, output, recording_id, chunk_index):
    meta = frames[0]["metadata"]
    primary = fits.PrimaryHDU()
    primary.header["RUNID"] = meta["run_id"]
    primary.header["BEAM"] = meta["beam"]
    primary.header["SCHEMA"] = meta["schema"]
    primary.header["ALNGUAR"] = (False, "DM/camera alignment is not guaranteed")
    primary.header["COMMENT"] = "Counter matches are evidence only; existing writers are unchanged."
    hdus = [primary]
    for i, name in enumerate(OPD_PLANES):
        hdu = fits.ImageHDU(np.stack([f["opd"][i] for f in frames]), name=name)
        hdu.header["BUNIT"] = "m"
        hdus.append(hdu)
    hdus.append(fits.ImageHDU(np.stack([f["pupil"] for f in frames]), name="PUPIL"))
    dm_shape = tuple(meta["dm_shape"])
    camera_shape = tuple(meta["camera_shape"])
    hdus.append(fits.ImageHDU(np.stack([
        f["dm"] if f["dm"] is not None else np.full(dm_shape, np.nan)
        for f in frames]), name="DM_CMD"))
    hdus.append(fits.ImageHDU(np.stack([
        f["camera"] if f["camera"] is not None else np.zeros(camera_shape, dtype=np.int32)
        for f in frames]), name="ZWFS"))
    fields = {
        "FRAME": ("K", [f["metadata"]["frame"] for f in frames]),
        "TIME_NS": ("K", [f["metadata"]["sample_ns"] for f in frames]),
        "CAM_CNT": ("K", [f["quality"]["camera_counter"] if f["quality"]["camera_counter"] is not None else -1 for f in frames]),
        "DM_CNT": ("K", [f["quality"]["dm_counter"] if f["quality"]["dm_counter"] is not None else -1 for f in frames]),
        "CAM_OK": ("L", [f["quality"]["camera_read_ok"] for f in frames]),
        "DM_OK": ("L", [f["quality"]["dm_read_ok"] for f in frames]),
        "CAM_MATCH": ("L", [f["quality"]["camera_counter_match"] for f in frames]),
        "DM_MATCH": ("L", [f["quality"]["dm_counter_match"] for f in frames]),
        "GAP": ("K", [f["quality"]["missed_frames"] for f in frames]),
    }
    for column, key in (("TIP_NM", "opd_tip_rms_nm"), ("TILT_NM", "opd_tilt_rms_nm"), ("HO_NM", "opd_ho_rms_nm")):
        fields[column] = ("D", [f["metadata"]["diagnostics"].get(key) if f["metadata"]["diagnostics"].get(key) is not None else np.nan for f in frames])
    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name=name, format=fmt, array=values, unit="nm" if name.endswith("_NM") else None)
        for name, (fmt, values) in fields.items()], name="FRAMES"))
    rows = [json.dumps({"metadata": {k: v for k, v in f["metadata"].items() if k != "config"},
                        "quality": f["quality"]}, allow_nan=False) for f in frames]
    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name="JSON", format=f"{max(map(len, rows))}A", array=rows)], name="FRAME_META"))
    config = json.dumps(meta["config"], allow_nan=False)
    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name="JSON", format=f"{len(config)}A", array=[config])], name="CONFIG"))
    name = f"sim_b{meta['beam']}_{meta['run_id']}_{recording_id}_{chunk_index:06d}.fits"
    final = Path(output) / name
    temporary = final.with_suffix(".fits.partial")
    with fits.HDUList(hdus) as hdul:
        hdul.writeto(temporary, checksum=True)
    os.replace(temporary, final)
    print(f"Saved {len(frames)} frames: {final}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--beams", type=int, nargs="+", choices=[1, 2, 3, 4], default=[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shm-dir", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--poll-hz", type=float, default=20)
    parser.add_argument("--chunk-frames", type=int, default=60)
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    args = parser.parse_args()
    if not np.isfinite(args.poll_hz) or args.poll_hz <= 0 or args.chunk_frames < 1:
        parser.error("poll-hz must be positive and finite; chunk-frames must be positive")
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("duration must be positive and finite")
    return args


def record(args):
    args.output.mkdir(parents=True, exist_ok=True)
    readers = {b: TelemetryReader(b, args.shm_dir) for b in args.beams}
    buffers = {b: [] for b in readers}
    last = {}
    chunk_index = 0
    recording_id = uuid.uuid4().hex[:12]
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    def flush(beam):
        nonlocal chunk_index
        if buffers[beam]:
            write_chunk(buffers[beam], args.output, recording_id, chunk_index)
            buffers[beam].clear()
            chunk_index += 1

    previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    start = time.monotonic()
    print(f"Watching beams {list(readers)} in {args.shm_dir}; Ctrl+C saves the final chunk.", flush=True)
    try:
        while not stop and (args.duration is None or time.monotonic() - start < args.duration):
            for beam, reader in readers.items():
                previous = last.get(beam)
                snapshot = reader.read_latest(after=previous[:2] if previous else None)
                if snapshot is None:
                    continue
                meta = snapshot["metadata"]
                identity = (meta["run_id"], meta["sequence"])
                if previous and identity == previous[:2]:
                    continue
                new_run = previous is None or meta["run_id"] != previous[0]
                if new_run:
                    flush(beam)
                    print(f"Beam {beam}: attached to run {meta['run_id']}", flush=True)
                gap = 0 if new_run else max(0, meta["frame"] - previous[2] - 1)
                snapshot["quality"]["missed_frames"] = gap
                if gap:
                    print(f"Beam {beam}: missed {gap} simulator frames", flush=True)
                buffers[beam].append(snapshot)
                last[beam] = (*identity, meta["frame"])
                if len(buffers[beam]) >= args.chunk_frames:
                    flush(beam)
            time.sleep(1.0 / args.poll_hz)
    finally:
        try:
            for beam in readers:
                flush(beam)
        finally:
            for s, handler in previous_handlers.items():
                signal.signal(s, handler)


def main():
    record(parse_args())


if __name__ == "__main__":
    main()




"""
Example to read in telemetry and look on slider 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Extract your data cube (shape: 47, 256, 256)
data_cube = d["ATM_OPD"].data
num_frames = data_cube.shape[0]

#  Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.2)  # Leave space at the bottom for the slider

#  Display the initial frame
initial_frame = 0
im = ax.imshow(data_cube[initial_frame], cmap='viridis', origin='lower')
fig.colorbar(im, ax=ax, label='ATM_OPD')
ax.set_title(f'Frame {initial_frame}')

# Create the slider axis and the Slider object
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]
frame_slider = Slider(
    ax=ax_slider,
    label='Frame',
    valmin=0,
    valmax=num_frames - 1,
    valinit=initial_frame,
    valstep=1  # Ensures the slider only snaps to integer indices
)

#  Define the update function
def update(val):
    frame_idx = int(frame_slider.val)
    # Update the image data without redrawing the whole plot (much faster)
    im.set_data(data_cube[frame_idx])
    
    # Optional: adjust color limits dynamically if your frames vary wildly in values
    # im.set_clim(vmin=data_cube[frame_idx].min(), vmax=data_cube[frame_idx].max())
    
    ax.set_title(f'Frame {frame_idx}')
    fig.canvas.draw_idle()

# Connect the slider to the update function
frame_slider.on_changed(update)

plt.show()
"""