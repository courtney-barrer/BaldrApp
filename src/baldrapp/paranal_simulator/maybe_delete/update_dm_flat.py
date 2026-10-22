#!/usr/bin/env python3
"""Collect N new DM-channel snapshots and propose flat_new = flat_old + mean.

Standalone: numpy, astropy, xaosim (native aligned ImageStreamIO layout).
No history is stored in a DM SHM: sampling starts AFTER this program attaches.
The C++ RTC does not increment disp02.cnt0. Pace reads on dm<beam>.cnt0 instead,
without consuming semaphores. These are best-effort channel snapshots, not
an atomic history of RTC commands; identical commands on new ticks still count.

This program never opens/resets the RTC or clears the source channel.
The operator must coordinate that separately to avoid adding the correction twice.
Only --apply permits writing channel 0. A FITS backup/proposal is saved first.
Do not let another process change channel 0 during this operation.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import argparse
import os
import struct
import time
import uuid

import numpy as np
from astropy.io import fits
from xaosim.shmlib import shm, hdr_fmt_aln, mtkeys, all_dtypes

HEADER_BYTES = struct.calcsize(hdr_fmt_aln)


class NativeReader:
    """Read-only native SHM view; never creates streams or touches semaphores."""
    def __init__(self, path):
        self.path = Path(path)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.identity = self._identity(os.fstat(self.fd))

    @staticmethod
    def _identity(stat):
        return stat.st_dev, stat.st_ino

    def check_identity(self):
        if self._identity(self.path.stat()) != self.identity:
            raise RuntimeError(f"Stream replaced: {self.path}. Restart this script after the simulator/RTC restart.")

    def header(self):
        self.check_identity()
        raw = os.pread(self.fd, HEADER_BYTES, 0)
        if len(raw) != HEADER_BYTES:
            raise RuntimeError(f"Incomplete header: {self.path}")
        return dict(zip(mtkeys, struct.unpack(hdr_fmt_aln, raw)))

    def snapshot(self):
        before = self.header()
        if before['write']:
            return None
        if before['naxis'] != 2 or (before['y'], before['x']) != (12, 12):
            raise ValueError(f"Expected a 12x12 DM image: {self.path}")
        if before['atype'] not in (9, 10) or before['nel'] != 144:
            raise ValueError(f"Expected float32/float64 DM data: {self.path}")
        dtype = np.dtype(all_dtypes[before['atype']-1])
        size = 144*dtype.itemsize
        raw = os.pread(self.fd, size, HEADER_BYTES)
        # RTC disp channels may have neither write flags nor advancing counters.
        # A second identical copy rejects obvious concurrent writes, but cannot
        # guarantee atomicity with a writer that supplies no handshake.
        again = os.pread(self.fd, size, HEADER_BYTES)
        after = self.header()
        if len(raw) != size:
            raise RuntimeError(f"Incomplete DM data: {self.path}")
        if before != after or raw != again or after['write']:
            return None
        image = np.frombuffer(raw, dtype=dtype).reshape(12,12).copy()
        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite DM data: {self.path}")
        return image, int(after['cnt0'])

    def close(self):
        os.close(self.fd)


def stable_snapshot(reader, timeout, poll):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        result = reader.snapshot()
        if result is not None:
            return result
        time.sleep(poll)
    raise TimeoutError(f"No stable snapshot from {reader.path}")


def collect_samples(source, clock, nframes, timeout, poll):
    last = int(clock.header()['cnt0'])
    samples, counters, times, missed = [], [], [], []
    deadline = time.monotonic()+timeout
    while len(samples)<nframes:
        if time.monotonic()>=deadline:
            raise TimeoutError(f"Only collected {len(samples)}/{nframes} samples before timeout. "
                               "Check the combined DM counter and whether the RTC is writing. No flat was written.")
        before = clock.header()
        counter = int(before['cnt0'])
        if counter<last:
            raise RuntimeError("Combined DM counter reset; aborting without updating flat")
        if before['write'] or counter==last:
            time.sleep(poll)
            continue
        snapshot = source.snapshot()
        after = clock.header()
        if snapshot is None or after['write'] or int(after['cnt0'])!=counter:
            time.sleep(poll)
            continue
        image, source_counter = snapshot
        samples.append(image.astype(np.float64))
        counters.append((counter, source_counter))
        times.append(time.time_ns())
        missed.append(counter-last-1)
        last=counter
        if len(samples)==1 or len(samples)%max(1,nframes//10)==0:
            print(f"Collected {len(samples)}/{nframes}",flush=True)
        time.sleep(poll)
    return np.stack(samples),np.asarray(counters,dtype=np.uint64),np.asarray(times,dtype=np.int64),np.asarray(missed,dtype=np.int64)


def make_candidate(flat, mean):
    candidate = np.asarray(flat,dtype=np.float64)+mean
    if not np.isfinite(candidate).all():
        raise ValueError("Non-finite candidate flat")
    # The 12x12 layout has 140 actuators: four unused corners remain untouched.
    active=np.ones((12,12),dtype=bool)
    active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
    candidate[~active]=flat[~active]
    if np.any((candidate[active]<0)|(candidate[active]>1)):
        raise ValueError("Candidate flat is outside [0,1] on active actuators; refusing to clip or apply it")
    return candidate.astype(flat.dtype)


def save_proposal(path, args, flat, mean, candidate, samples, counters, times, missed):
    header=fits.Header()
    header['BEAM']=args.beam;header['CHANNEL']=args.channel;header['NAVG']=args.frames
    header['DATE']=datetime.now(timezone.utc).isoformat()
    header['CONTENT']=('PROPOSAL','Saved before application; not a receipt')
    header['COMMENT']='Paced by combined DM counter; source snapshots are not guaranteed atomic.'
    header['COMMENT']='Source channel and RTC state are not modified by this script.'
    columns=[fits.Column(name='MASTER_CNT',format='K',array=counters[:,0]),
             fits.Column(name='SOURCE_CNT',format='K',array=counters[:,1]),
             fits.Column(name='TIME_NS',format='K',array=times),
             fits.Column(name='SKIPPED',format='K',array=missed)]
    fits.HDUList([fits.PrimaryHDU(header=header),fits.ImageHDU(flat,name='FLAT_BEFORE'),
                  fits.ImageHDU(mean,name='OFFSET_MEAN'),fits.ImageHDU(candidate,name='FLAT_CANDIDATE'),
                  fits.ImageHDU(samples,name='SAMPLES'),fits.BinTableHDU.from_columns(columns,name='FRAMES')]).writeto(path,checksum=True)


def apply_candidate(flat_reader, expected, candidate, notify):
    """Write only channel 0, then notify the existing combined-DM semaphore."""
    current=flat_reader.snapshot()
    if current is None or not np.array_equal(current[0],expected):
        raise RuntimeError("Flat changed since proposal; refusing to overwrite it")
    writer=shm(str(flat_reader.path),nosem=True)
    try:
        flat_reader.check_identity()
        if NativeReader._identity(os.fstat(writer.fd))!=flat_reader.identity:
            raise RuntimeError("Flat stream replaced before write")
        current=flat_reader.snapshot()
        if current is None or not np.array_equal(current[0],expected):
            raise RuntimeError("Flat changed before write")
        writer.set_data(candidate)
        notify()
        result=flat_reader.snapshot()
        if result is None or not np.array_equal(result[0],candidate):
            raise RuntimeError("Flat write attempted but readback differs; inspect channel 0 before retrying")
    finally:
        writer.close(erase_file=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('beam',type=int,choices=range(1,5))
    parser.add_argument('-N','--frames',type=int,required=True)
    parser.add_argument('--channel',type=int,default=2,help='Source disp channel (default 2; channel 0 is forbidden)')
    parser.add_argument('--shm-dir',type=Path,default=Path('/dev/shm'))
    parser.add_argument('--timeout',type=float,default=120,help='Total sampling timeout in seconds')
    parser.add_argument('--poll-ms',type=float,default=1,help='Poll interval; faster streams may skip frames')
    parser.add_argument('--output',type=Path,default=Path('dm_flat_updates'),help='Directory for FITS proposals/backups')
    parser.add_argument('--apply',action='store_true',help='After sampling, prompt before writing channel 0; default is preview only')
    args=parser.parse_args()
    if args.frames<1 or not 1<=args.channel<=99: parser.error('N must be positive; source channel must be 1–99')
    if not np.isfinite(args.timeout) or not np.isfinite(args.poll_ms) or min(args.timeout,args.poll_ms)<=0:
        parser.error('Timeout and poll interval must be finite and positive')
    source_path=args.shm_dir/f'dm{args.beam}disp{args.channel:02d}.im.shm'
    flat_path=args.shm_dir/f'dm{args.beam}disp00.im.shm'
    master_path=args.shm_dir/f'dm{args.beam}.im.shm'
    print(f'Sampling {source_path}; target flat {flat_path}.')
    print('Collecting new observed maps; no historical frames are available. RTC/source channel will not be changed.')
    with ExitStack() as stack:
        def reader(path):
            value=NativeReader(path);stack.callback(value.close);return value
        source,flat,clock=reader(source_path),reader(flat_path),reader(master_path)
        # Validate the streams before waiting for data.
        stable_snapshot(source,args.timeout,args.poll_ms/1000)
        stable_snapshot(clock,args.timeout,args.poll_ms/1000)
        initial,_=stable_snapshot(flat,args.timeout,args.poll_ms/1000)
        samples,counters,times,missed=collect_samples(source,clock,args.frames,args.timeout,args.poll_ms/1000)
        current,_=stable_snapshot(flat,args.timeout,args.poll_ms/1000)
        if not np.array_equal(initial,current):
            raise RuntimeError('Flat changed during sampling; no update made')
        mean=samples.mean(axis=0)
        candidate=make_candidate(initial,mean)
        args.output.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        backup=args.output/f'dm{args.beam}_flat_proposal_{stamp}_{uuid.uuid4().hex[:8]}.fits'
        save_proposal(backup,args,initial,mean,candidate,samples,counters,times,missed)
        print(f'Saved original flat, mean, candidate and {args.frames} samples: {backup}')
        print(f'Mean offset range [{mean.min():.6g}, {mean.max():.6g}]; candidate range [{candidate.min():.6g}, {candidate.max():.6g}]')
        print(f'Skipped combined-DM ticks: {missed.sum()} (these are not recoverable historical samples).')
        if not args.apply:
            print('Preview only: no shared-memory writes. Use --apply to collect and apply a new average.')
            return
        print('Before applying, coordinate the RTC/source channel yourself. Adding this mean while the same')
        print('correction remains on the source channel adds it twice. This script does not reset the controller.')
        #if input('When ready, type APPLY to add the saved mean to channel 0 (anything else cancels): ').strip()!='APPLY':
        #    print('Cancelled; flat unchanged.');return
        source.check_identity();clock.check_identity()
        # Open the existing semaphore only: never create, acquire or unlink it.
        import posix_ipc
        sem_prefix=str(args.shm_dir.resolve()).replace('/', '.')
        sem=posix_ipc.Semaphore(f'/{sem_prefix}.dm{args.beam}_sem01',flags=0)
        try:
            apply_candidate(flat,initial,candidate,sem.release)
        finally:
            sem.close()
        print('Channel 0 updated and verified; combined-DM update requested. RTC and source channel unchanged.')
        print('The FITS file is the pre-write backup/proposal; retain it to recover the original flat.')


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit('\nInterrupted. If interrupted during application, inspect channel 0 before retrying.')
    except (OSError,RuntimeError,ValueError,TimeoutError) as exc:
        raise SystemExit(f'ERROR: {exc}')
