#!/usr/bin/env python3
"""Analyse chunked simulator FITS without importing or controlling the simulator.

Dependencies: numpy, astropy, matplotlib; ffmpeg for MP4.
Example:
  python analyse_sim_telemetry.py /path/to/run --output /path/to/results --speed 5

RMS is piston-removed, uniformly weighted on PUPIL>0, expressed in nm.
Loop state is a heuristic from DM_OPD changes (the DM used in propagation),
NOT a measurement of RTC servo mode. Pokes/manual changes can resemble closure;
a converged closed loop can resemble an open loop. First-stage ao_enabled is
never interpreted as the RTC loop state. DM_CMD/ZWFS retain their recorded
snapshot alignment flags. Inputs are opened read-only. Existing outputs are
not overwritten. Select one beam and one simulator run at a time.
"""
import argparse
import csv
import json
from pathlib import Path
from contextlib import ExitStack

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.patches import Patch

PLANES = ("ATM_OPD", "INTERNAL_OPD", "DM_OPD", "RESIDUAL_OPD")
LABELS = {0: "Open-like (DM steady)", 1: "Closed-like (DM varying)", -1: "Unknown / gap"}
COLORS = {0: "#e7b65a", 1: "#63b89b", -1: "#b0b6c0"}


def rms_nm(a, mask):
    samples = np.asarray(a)[mask]
    return float(np.std(samples)*1e9) if len(samples) and np.isfinite(samples).all() else np.nan


def load_index(stack, directory, beam, run_id):
    chunks = []
    for path in sorted(directory.glob("*.fits")):
        h = stack.enter_context(fits.open(path, mode="readonly", memmap=True))
        if h[0].header.get("BEAM") != beam:
            continue
        run = h[0].header.get("RUNID")
        if run_id and run != run_id:
            continue
        if not all(name in h for name in (*PLANES, "PUPIL", "FRAMES", "FRAME_META", "DM_CMD", "ZWFS")):
            raise ValueError(f"Incomplete telemetry schema: {path}")
        chunks.append((path, h, run))
    runs = {run for _, _, run in chunks}
    if len(runs) != 1:
        raise ValueError(f"Select exactly one run with --run-id. Found: {sorted(map(str, runs))}")
    rows, seen, duplicates = [], set(), 0
    for path, h, run in chunks:
        for j, frame in enumerate(h["FRAMES"].data):
            identity = (run, int(frame["FRAME"]))
            if identity in seen:
                duplicates += 1
                continue
            seen.add(identity)
            meta = json.loads(h["FRAME_META"].data["JSON"][j])["metadata"]
            rows.append(dict(path=path, h=h, j=j, frame=int(frame["FRAME"]),
                             time_ns=int(frame["TIME_NS"]), meta=meta))
    rows.sort(key=lambda r: (r["time_ns"], r["frame"]))
    if not rows:
        raise ValueError("No telemetry frames")
    return rows, next(iter(runs)), len(chunks), duplicates


def analyse(rows, threshold, window, fraction):
    records, previous = [], None
    for row in rows:
        h, j = row["h"], row["j"]
        mask = np.asarray(h["PUPIL"].data[j]) > 0
        if not mask.any():
            raise ValueError(f"Empty pupil at frame {row['frame']}")
        maps = {name: np.asarray(h[name].data[j]) for name in PLANES}
        table = h["FRAMES"].data[j]
        rec = dict(frame=row["frame"], time_ns=row["time_ns"],
                   elapsed_s=(row["time_ns"]-rows[0]["time_ns"])*1e-9,
                   source=row["meta"].get("source_mode", "unknown"),
                   pupil=row["meta"].get("pupil_model", "unknown"))
        rec.update({name.lower()+"_rms_nm": rms_nm(a, mask) for name,a in maps.items()})
        rec["before_dm_rms_nm"] = rms_nm(maps["ATM_OPD"]+maps["INTERNAL_OPD"],mask)
        for name in ("TIP_NM", "TILT_NM", "HO_NM", "CAM_OK", "DM_OK", "CAM_MATCH", "DM_MATCH"):
            rec[name.lower()] = table[name].item()
        rec["gap_frames"] = max(0,row["frame"]-previous["frame"]-1) if previous else 0
        same_pupil = previous is not None and np.array_equal(mask,previous["mask"])
        rec["dm_opd_step_nm"] = (rms_nm(maps["DM_OPD"]-previous["dm"],mask)
                                   if same_pupil else np.nan)
        rec["dm_cmd_step_rms"] = (float(np.sqrt(np.mean((h["DM_CMD"].data[j]-previous["cmd"])**2)))
                                  if previous and rec["dm_ok"] and previous["cmd_ok"] else np.nan)
        rec["sum_error_nm"] = float(np.max(np.abs(maps["RESIDUAL_OPD"]-
                            maps["ATM_OPD"]-maps["INTERNAL_OPD"]-maps["DM_OPD"]))*1e9)
        rec["pupil_changed"] = previous is not None and not same_pupil
        records.append(rec)
        previous = dict(frame=row["frame"],mask=mask.copy(),dm=maps["DM_OPD"].copy(),
                        cmd=h["DM_CMD"].data[j].copy(),cmd_ok=rec["dm_ok"])
    t = np.array([r["elapsed_s"] for r in records])
    dt = float(np.median(np.diff(t))) if len(t)>1 else 1.0
    boundaries = np.array([i==0 or r["gap_frames"]>0 or r["pupil_changed"] or
                           (i>0 and (t[i]-t[i-1]>3*dt or r["source"]!=records[i-1]["source"]))
                           for i,r in enumerate(records)])
    segments = np.cumsum(boundaries)
    for i,rec in enumerate(records):
        ids=np.arange(max(0,i-window//2), min(len(records),i+window//2+1))
        ids=ids[(segments[ids]==segments[i]) & ~boundaries[ids]]
        vals=np.array([records[k]["dm_opd_step_nm"] for k in ids])
        vals=vals[np.isfinite(vals)]
        activity=float(np.mean(vals>threshold)) if len(vals) else np.nan
        state=-1 if boundaries[i] or len(vals)<min(3,window) else int(activity>=fraction)
        rec.update(activity_fraction=activity, inferred_state=LABELS[state],state_code=state)
    return records, dt


def intervals(records, dt):
    """Sample bins; large unsampled intervals are explicitly unknown."""
    spans=[]
    for i,r in enumerate(records):
        start=r["elapsed_s"]
        end=records[i+1]["elapsed_s"] if i+1<len(records) else start+dt
        gap_after = i+1<len(records) and (records[i+1]["gap_frames"]>0 or end-start>3*dt)
        sampled_end=min(end,start+dt) if gap_after else end
        spans.append((start,sampled_end,r["state_code"]))
        if end>sampled_end:
            spans.append((sampled_end,end,-1))
    merged=[]
    for a,b,state in spans:
        if merged and merged[-1][2]==state and abs(merged[-1][1]-a)<1e-8:
            merged[-1]=(merged[-1][0],b,state)
        else:
            merged.append((a,b,state))
    return merged


def save_products(records, spans, output, summary):
    with (output/"metrics.csv").open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (output/"loop_intervals.csv").open("x",newline="") as f:
        writer=csv.writer(f);writer.writerow(["start_s","end_s","inferred_state"])
        writer.writerows((a,b,LABELS[s]) for a,b,s in spans)
    with (output/"summary.json").open("x") as f:
        json.dump(summary,f,indent=2,allow_nan=False)


def create_figure(rows, records, spans):
    fig=plt.figure(figsize=(13,8.5),layout="constrained")
    grid=fig.add_gridspec(3,3,height_ratios=[1,0.65,1.45])
    ax=fig.add_subplot(grid[0,:]); components=fig.add_subplot(grid[1,:],sharex=ax)
    t=np.array([r["elapsed_s"] for r in records])
    for key,label,color in [("before_dm_rms_nm","Before DM","#777777"),
                            ("residual_opd_rms_nm","Residual total","#164e8a"),
                            ("ho_nm","Residual HO","#b63378")]:
        ax.plot(t,[r[key] for r in records],label=label,color=color,lw=1.2)
    for key,label in [("tip_nm","Tip"),("tilt_nm","Tilt")]:
        components.plot(t,[r[key] for r in records],label=label,lw=1)
    for a,b,state in spans:
        ax.axvspan(a,b,color=COLORS[state],alpha=.18,lw=0)
        components.axvspan(a,b,color=COLORS[state],alpha=.18,lw=0)
    ax.set(ylabel="OPD RMS (nm)",title="RTC state")
    ax.legend(loc="upper right",ncol=3,fontsize=9)
    components.set(xlabel="Seconds since first recorded frame",ylabel="OPD RMS (nm)")
    components.legend(loc="upper right",ncol=2,fontsize=9)
    ax.set_xlim(t[0],spans[-1][1]);ax.grid(alpha=.2);components.grid(alpha=.2)
    cursors=[a.axvline(t[0],color="black",lw=1.5) for a in (ax,components)]
    axs=[fig.add_subplot(grid[2,k]) for k in range(3)]
    # Fixed colour scales across the whole run avoid flickering/autoscale illusions.
    limits=[]
    for name in ("RESIDUAL_OPD","DM_CMD","ZWFS"):
        low,high=[],[]
        for row in rows:
            data=np.asarray(row["h"][name].data[row["j"]])
            if name=="RESIDUAL_OPD":
                data=data[row["h"]["PUPIL"].data[row["j"]]>0]*1e9
                data=data-data.mean()
            elif name=="DM_CMD" and not bool(row["h"]["FRAMES"].data[row["j"]]["DM_OK"]): continue
            elif name=="ZWFS" and not bool(row["h"]["FRAMES"].data[row["j"]]["CAM_OK"]): continue
            data=data[np.isfinite(data)]
            if data.size:
                lo,hi=np.percentile(data,[1,99]);low.append(lo);high.append(hi)
        lo,hi=(min(low),max(high)) if low else (0,1)
        if name=="RESIDUAL_OPD": hi=max(abs(lo),abs(hi),1);lo=-hi
        if hi<=lo: hi=lo+1
        limits.append((lo,hi))
    artists=[]
    for a,(lo,hi),label,cmap in zip(axs,limits,["nm OPD (piston removed)","DM command units","Detector ADU"],["RdBu_r","viridis","inferno"]):
        artist=a.imshow(np.zeros((2,2)),origin="lower",vmin=lo,vmax=hi,cmap=cmap,interpolation="nearest")
        artists.append(artist);fig.colorbar(artist,ax=a,shrink=.8,label=label)
        a.set_xlabel("Column");a.set_ylabel("Row")
    fig.legend(handles=[Patch(facecolor=COLORS[s],alpha=.5,label=LABELS[s]) for s in (0,1,-1)],
               loc="outside lower center",ncol=3,fontsize=9)
    def update(i,display_time=None):
        row,r=rows[i],records[i];h,j=row["h"],row["j"]
        mask=h["PUPIL"].data[j]>0
        residual=h["RESIDUAL_OPD"].data[j]*1e9
        residual=residual-np.mean(residual[mask])
        data=[np.ma.array(residual,mask=~mask),h["DM_CMD"].data[j],h["ZWFS"].data[j]]
        time=r["elapsed_s"] if display_time is None else display_time
        gap_after = i+1<len(records) and (records[i+1]["gap_frames"]>0 or records[i+1]["elapsed_s"]-r["elapsed_s"]>3*summary_dt[0])
        stale=gap_after and time-r["elapsed_s"]>summary_dt[0]
        for k,(artist,a,array) in enumerate(zip(artists,axs,data)):
            missing=stale or (k==1 and not r['dm_ok']) or (k==2 and not r['cam_ok'])
            artist.set_data(np.full(array.shape,np.nan) if missing else array)
            ny,nx=array.shape;artist.set_extent((-.5,nx-.5,-.5,ny-.5))
            a.set_xlim(-.5,nx-.5);a.set_ylim(-.5,ny-.5)
        yy,xx=np.where(mask)
        margin=max(2,int(.08*max(np.ptp(xx),np.ptp(yy))))
        axs[0].set_xlim(xx.min()-margin,xx.max()+margin)
        axs[0].set_ylim(yy.min()-margin,yy.max()+margin)
        axs[0].set_title("Residual OPD" if not stale else "No sample (gap)")
        axs[1].set_title("DM")
        axs[2].set_title("ZWFS")
        for cursor in cursors: cursor.set_xdata([time,time])
        state=LABELS[-1] if stale else r['inferred_state']
        fig.suptitle(f"Beam {row['meta']['beam']} | frame {r['frame']} | {time:.2f} s | {r['source']} | {state}",fontsize=12)
    summary_dt=[float(np.median(np.diff(t))) if len(t)>1 else 1.0]
    return fig,update


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--beam",type=int,default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--dm-step-nm",type=float,default=.1,help="Activity threshold on consecutive DM_OPD differences (default 0.1 nm RMS)")
    parser.add_argument("--window",type=int,default=5,help="Odd centred smoothing window in recorded frames")
    parser.add_argument("--activity-fraction",type=float,default=.6)
    parser.add_argument("--speed",type=float,default=5,help="Playback speed relative to timestamps (default 5x)")
    parser.add_argument("--fps",type=int,default=20)
    parser.add_argument("--no-movie",action="store_true")
    args=parser.parse_args()
    if not args.directory.is_dir(): parser.error("Input directory does not exist")
    if not np.isfinite(args.speed) or args.speed<=0 or args.fps<1: parser.error("Positive speed and fps required")
    if not np.isfinite(args.dm_step_nm) or args.dm_step_nm<0 or not 0<args.activity_fraction<=1: parser.error("Invalid activity thresholds")
    if args.window<1 or args.window%2==0: parser.error("Window must be positive and odd")
    outputs=['metrics.csv','loop_intervals.csv','summary.json','opd_summary.png']+([] if args.no_movie else ['telemetry.mp4'])
    if any((args.output/name).exists() for name in outputs): parser.error("Output files already exist; choose a new --output directory")
    if not args.no_movie and not writers.is_available('ffmpeg'): parser.error("ffmpeg is required for MP4; install it or use --no-movie")
    with ExitStack() as stack:
        rows,run,chunks,duplicates=load_index(stack,args.directory,args.beam,args.run_id)
        records,dt=analyse(rows,args.dm_step_nm,args.window,args.activity_fraction)
        spans=intervals(records,dt)
        summary=dict(run_id=run,beam=args.beam,files=chunks,frames=len(rows),duplicate_frames_removed=duplicates,
                     duration_s=records[-1]['elapsed_s'],missing_frames=sum(r['gap_frames'] for r in records),
                     sources=sorted({r['source'] for r in records}),dm_counter_matches=sum(r['dm_match'] for r in records),
                     camera_counter_matches=sum(r['cam_match'] for r in records),
                     max_opd_sum_error_nm=max(r['sum_error_nm'] for r in records),
                     dm_activity_threshold_nm=args.dm_step_nm,window_frames=args.window,activity_fraction=args.activity_fraction,
                     inference="DM_OPD activity heuristic, not RTC servo state; transitions uncertain by smoothing window. Pokes can mimic closure, converged control can mimic open loop.",
                     intervals=[dict(start_s=a,end_s=b,state=LABELS[s]) for a,b,s in spans])
        args.output.mkdir(parents=True,exist_ok=True)
        save_products(records,spans,args.output,summary)
        fig,update=create_figure(rows,records,spans)
        update(len(rows)-1);fig.savefig(args.output/'opd_summary.png',dpi=150)
        fig.set_dpi(100)
        fig.canvas.draw()  # Resolve layout at the movie's actual DPI before freezing it.
        fig.set_layout_engine('none')
        if not args.no_movie:
            times=np.array([r['elapsed_s'] for r in records])
            movie_times=np.arange(0,times[-1]+dt,args.speed/args.fps)
            indices=np.clip(np.searchsorted(times,movie_times,side='right')-1,0,len(rows)-1)
            writer=FFMpegWriter(fps=args.fps,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','20','-preset','fast','-threads','2'])
            with writer.saving(fig,str(args.output/'telemetry.mp4'),dpi=100):
                for k,(i,t) in enumerate(zip(indices,movie_times)):
                    update(int(i),float(t));writer.grab_frame()
                    if k%100==0: print(f"Movie: {k+1}/{len(indices)} frames",flush=True)
        plt.close(fig)
        print(json.dumps(summary,indent=2));print(f"Results: {args.output}")


if __name__=='__main__':
    main()
