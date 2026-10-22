"""Optional simulator ground truth, independent of camera/DM writers.

Per beam: baldr_sim_telemetry_bN_{opd,pupil,meta}.im.shm. OPD is a
float64 stack in metres (atmosphere after AO, internal, DM, total residual).
Metadata is UTF-8 JSON in a fixed uint8 image. The dedicated metadata file
is also the advisory lock: publishers never wait for readers. An odd
sequence means incomplete publication; an even, nonzero sequence is ready.

Existing dmN/baldrN images are opened O_RDONLY and read with pread, never
through the SHM constructor or semaphore API. Counter checks are evidence
of alignment, not a guarantee of atomic or historical camera/DM snapshots.
"""

import fcntl
import json
import os
from pathlib import Path
import struct
import time
import uuid

import numpy as np
from xaosim.shmlib import shm, hdr_fmt_aln, mtkeys, all_dtypes


SCHEMA_VERSION = 1
OPD_PLANES = ("ATM_OPD", "INTERNAL_OPD", "DM_OPD", "RESIDUAL_OPD")
META_BYTES = 65536
HEADER_BYTES = struct.calcsize(hdr_fmt_aln)


def stream_path(directory, beam, kind):
    return Path(directory) / f"baldr_sim_telemetry_b{beam}_{kind}.im.shm"


def _header(fd):
    raw = os.pread(fd, HEADER_BYTES, 0)
    if len(raw) != HEADER_BYTES:
        raise ValueError("Incomplete SHM header")
    return dict(zip(mtkeys, struct.unpack(hdr_fmt_aln, raw)))


def _read_image(fd):
    """Copy an aligned ImageStreamIO/xaosim image without touching metadata."""
    before = _header(fd)
    if before["write"]:
        raise ValueError("SHM write in progress")
    if not 1 <= before["atype"] <= len(all_dtypes) or not 1 <= before["naxis"] <= 3:
        raise ValueError("Unsupported SHM layout")
    shape = tuple(before[k] for k in ("x", "y", "z")[:before["naxis"]])[::-1]
    dtype = np.dtype(all_dtypes[before["atype"] - 1])
    size = int(np.prod(shape)) * dtype.itemsize
    if size <= 0 or int(np.prod(shape)) != before["nel"] or HEADER_BYTES + size > os.fstat(fd).st_size:
        raise ValueError("Invalid SHM dimensions")
    raw = os.pread(fd, size, HEADER_BYTES)
    after = _header(fd)
    if len(raw) != size or before != after or after["write"]:
        raise ValueError("SHM changed during copy")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy(), int(after["cnt0"])


def read_existing_image(path):
    """Return a best-effort counter-stable snapshot, or None if unavailable."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            image, counter = _read_image(fd)
            a, b = os.fstat(fd), os.stat(path)
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                return None
            return image, counter
        finally:
            os.close(fd)
    except (OSError, ValueError):
        return None


def _encode_metadata(metadata):
    raw = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(raw) >= META_BYTES:
        raise ValueError("Telemetry metadata exceeds reserved space")
    image = np.zeros((1, META_BYTES), dtype=np.uint8)
    image[0, :len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return image


def _decode_metadata(fd):
    image, _ = _read_image(fd)
    return json.loads(image.tobytes().split(b"\0", 1)[0])


class TelemetryPublisher:
    def __init__(self, beam_shapes, config, directory="/dev/shm"):
        self.run_id = uuid.uuid4().hex
        self.config = config
        self.shapes = dict(beam_shapes)
        self.streams = {}
        self.sequences = {}
        self.skipped = {}
        self.closed = False
        try:
            for beam, shape in beam_shapes.items():
                self.streams[beam] = {}
                self.sequences[beam] = 0
                self.skipped[beam] = 0
                # Replace metadata first with a not-ready generation. Readers
                # verify its inode again before accepting any copied snapshot.
                arrays = {
                    "meta": _encode_metadata({"schema": SCHEMA_VERSION, "run_id": self.run_id, "sequence": 0}),
                    "opd": np.zeros((4, *shape), dtype=np.float64),
                    "pupil": np.zeros(shape, dtype=np.float64),
                }
                for kind, data in arrays.items():
                    final = stream_path(directory, beam, kind)
                    temporary = final.with_name(final.name + "." + self.run_id + ".tmp")
                    image = shm(str(temporary), data=data, nosem=True)
                    self.streams[beam][kind] = image
                    os.replace(temporary, final)
        except Exception:
            self.close()
            raise

    def publish_frame(self, beam, atmosphere, internal, dm_opd, pupil, metadata):
        images = self.streams[beam]
        if any(np.shape(a) != self.shapes[beam] for a in (atmosphere, internal, dm_opd, pupil)):
            raise ValueError("Telemetry maps must match the configured pupil shape")
        try:
            fcntl.flock(images["meta"].fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.skipped[beam] += 1
            return False
        try:
            sequence = self.sequences[beam] + 2
            base = dict(metadata, schema=SCHEMA_VERSION, run_id=self.run_id,
                        beam=int(beam), sequence=sequence, published_ns=time.time_ns(),
                        opd_units="m", opd_planes=OPD_PLANES, config=self.config,
                        skipped_publications=self.skipped[beam])
            complete = _encode_metadata(base)
            images["meta"].set_data(_encode_metadata(dict(base, sequence=sequence - 1)))
            planes = np.stack((atmosphere * pupil, internal * pupil, dm_opd * pupil))
            stack = np.concatenate((planes, planes.sum(axis=0, keepdims=True)))
            images["opd"].set_data(np.asarray(stack, dtype=np.float64))
            images["pupil"].set_data(np.asarray(pupil, dtype=np.float64))
            images["meta"].set_data(complete)
            self.sequences[beam] = sequence
            return True
        finally:
            fcntl.flock(images["meta"].fd, fcntl.LOCK_UN)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for images in self.streams.values():
            for image in images.values():
                image.close(erase_file=False)


class TelemetryReader:
    def __init__(self, beam, directory="/dev/shm"):
        self.beam = int(beam)
        self.directory = Path(directory)

    def read_latest(self, after=None):
        """Open fresh handles each poll, automatically following replacements."""
        path = stream_path(self.directory, self.beam, "meta")
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            meta = _decode_metadata(fd)
            sequence = meta.get("sequence", 0)
            if not sequence or sequence % 2 or meta.get("schema") != SCHEMA_VERSION:
                return None
            if after == (meta.get("run_id"), sequence):
                return None
            opd = read_existing_image(stream_path(self.directory, self.beam, "opd"))
            pupil = read_existing_image(stream_path(self.directory, self.beam, "pupil"))
            if opd is None or pupil is None or opd[0].shape != (4, *pupil[0].shape):
                return None
            dm = read_existing_image(self.directory / f"dm{self.beam}.im.shm")
            camera = read_existing_image(self.directory / f"baldr{self.beam}.im.shm")
            if dm is not None and dm[0].shape != tuple(meta["dm_shape"]):
                dm = None
            if camera is not None and camera[0].shape != tuple(meta["camera_shape"]):
                camera = None
            end = _decode_metadata(fd)
            a, b = os.fstat(fd), os.stat(path)
            if meta != end or (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                return None
            quality = {
                "recorded_ns": time.time_ns(),
                "dm_read_ok": dm is not None, "camera_read_ok": camera is not None,
                "dm_counter": dm[1] if dm else None,
                "camera_counter": camera[1] if camera else None,
                "camera_counter_match": camera is not None and camera[1] == meta["camera_counter"],
                "dm_counter_match": dm is not None and dm[1] == meta["dm_counter_after"] == meta["dm_counter_before"],
                "alignment_guaranteed": False,
            }
            return dict(metadata=meta, quality=quality, opd=opd[0], pupil=pupil[0],
                        dm=dm[0] if dm else None, camera=camera[0] if camera else None)
        except (OSError, ValueError, KeyError, TypeError):
            return None
        finally:
            os.close(fd)  # releases the telemetry lock; no semaphore operations
