#!/usr/bin/env python3
"""Standalone DM probing and analytic phase fitting (no BaldrApp/config imports).

acquire: clear pupil, baseline, four inner-corner +/- pokes, then modal +/- probes.
fit: offline common phase per illuminated camera pixel, with an affine registration
prior from the corner pokes. Uses a monochromatic, fixed radial reference field.
I(+/-) = A + M**2 B**2 + 2 M sqrt(A) B cos(mu - phi -/+ probe_phase),
where B is reference FIELD amplitude and M exp(i mu) = exp(i theta) - 1.
Both signs are fitted together; phi is the common baseline phase, independent
of the chosen probe basis. OPD output has piston removed.
This is an approximate first-pass model for broadband/Fresnel data.

Requires numpy/scipy/astropy; acquisition also needs pyzmq, xaosim, posix_ipc and
DCS modal_basis.py dependencies. DCS basis amplitudes retain their original scaling.
Keep RTC off and use an otherwise idle probe channel. Only acquire --execute
writes hardware; the original channel and mask position are restored in finally.
No automatic flat correction. SIGKILL cannot restore hardware.

Examples:
  python probe_dm_phase.py acquire --beam 1 --amplitude .02 --modes 0 1 2 3 --output probes.fits
  # Add --execute to perform the acquisition (default mask J3, probe channel 3).
  python probe_dm_phase.py fit probes.fits --output phase_fit.fits
  python analyse_probe_phase.py phase_fit.fits
"""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import sys
import time

import numpy as np
from astropy.io import fits

DEFAULT_DCS=Path('/home/bbarrer/Documents/baldr/dcs/baldr_jcr')
# Material-independent mask definitions: wavelength [m], phase shift [rad],
# DIAMETER [lambda/D]. Physical diameters converted with F#=21.2.
MASKS = {
    name: dict(wavelength=wave*1e-6, theta=np.pi/2, diameter_lamD=diam/(21.2*wave))
    for band, wave, diameters in [('J',1.20,[32,36,44,54,65]), ('H',1.65,[31,37,44,53,68])]
    for name, diam in zip([f'{band}{i}' for i in range(1,6)], diameters)
}


def json_hdu(name,value):
    raw=json.dumps(value,allow_nan=False)
    return fits.BinTableHDU.from_columns([fits.Column(name='JSON',format=f'{len(raw)}A',array=[raw])],name=name)


def read_json(h,name):
    return json.loads(h[name].data['JSON'][0])


def basis_maps(directory,name,indices):
    path=Path(directory)/'modal_basis.py'
    sys.path.insert(0,str(Path(directory).resolve()))
    try:
        spec=importlib.util.spec_from_file_location('_probe_dcs_basis',path)
        module=importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            raise RuntimeError(f'DCS basis dependency missing: {exc.name}. Install it in the environment running this script (modal_basis.py requires aotools).') from exc
        basis=getattr(module,name)()
        modes=basis.modes_on_unit_disk(12,max(indices)+1)[:,indices].T.reshape(-1,12,12)
    finally:
        sys.path.pop(0)
    modes=np.asarray(modes,dtype=float)
    modes[:,0,0]=modes[:,0,-1]=modes[:,-1,0]=modes[:,-1,-1]=0
    if not np.isfinite(modes).all():raise ValueError('Non-finite basis')
    # Preserve DCS scaling (the norm argument in its implementation is unused).
    return modes


class ImageReader:
    def __init__(self,path):
        from xaosim.shmlib import hdr_fmt_aln,mtkeys,all_dtypes
        self.path=Path(path);self.fmt=hdr_fmt_aln;self.keys=mtkeys;self.dtypes=all_dtypes
        self.offset=struct.calcsize(self.fmt);self.fd=os.open(self.path,os.O_RDONLY)
        self.identity=self.stat_id(os.fstat(self.fd))
    @staticmethod
    def stat_id(s):return s.st_dev,s.st_ino
    def header(self):
        if self.stat_id(self.path.stat())!=self.identity:raise RuntimeError(f'SHM replaced: {self.path}')
        raw=os.pread(self.fd,self.offset,0)
        if len(raw)!=self.offset:raise ValueError('Incomplete SHM header')
        return dict(zip(self.keys,struct.unpack(self.fmt,raw)))
    def read(self):
        a=self.header()
        if a['write']:return None
        if a['naxis']!=2 or not 1<=a['atype']<=len(self.dtypes):raise ValueError('Unsupported SHM layout')
        shape=(a['y'],a['x']);dtype=np.dtype(self.dtypes[a['atype']-1]);size=int(np.prod(shape))*dtype.itemsize
        if a['nel']!=np.prod(shape) or self.offset+size>os.fstat(self.fd).st_size:raise ValueError('Invalid SHM size')
        raw=os.pread(self.fd,size,self.offset);again=os.pread(self.fd,size,self.offset);b=self.header()
        if a!=b or raw!=again or b['write']:return None
        data=np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
        if not np.isfinite(data).all():raise ValueError('Non-finite SHM image')
        return data,int(b['cnt0'])
    def stable(self,timeout=10):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            result=self.read()
            if result is not None:return result
            time.sleep(.001)
        raise TimeoutError(f'No stable image: {self.path}')
    def close(self):os.close(self.fd)


def rpc(address,command,timeout=10000):
    import zmq
    ctx=zmq.Context();sock=ctx.socket(zmq.REQ)
    try:
        sock.setsockopt(zmq.LINGER,0);sock.setsockopt(zmq.RCVTIMEO,timeout);sock.setsockopt(zmq.SNDTIMEO,timeout)
        sock.connect(address);sock.send_string(command);reply=sock.recv_string().strip()
        if reply.upper().startswith(('ERR','NACK','ERROR')):raise RuntimeError(reply)
        return reply
    finally:sock.close();ctx.term()


def camera_frames(camera,master,expected,n,settle,timeout):
    deadline=time.monotonic()+timeout
    # Wait for combined DM to reflect this probe before starting camera settling.
    while time.monotonic()<deadline:
        dm=master.read()
        if dm is not None and np.allclose(dm[0],expected,rtol=0,atol=1e-10):break
        time.sleep(.002)
    else:raise TimeoutError('Combined DM did not reach the requested probe; check competing RTC writer')
    last=camera.header()['cnt0'];frames=[];counts=[];timestamps=[];discard=settle
    while len(frames)<n:
        if time.monotonic()>deadline:raise TimeoutError(f'Camera stalled: {len(frames)}/{n} frames')
        dm=master.read()
        if dm is not None and not np.allclose(dm[0],expected,rtol=0,atol=1e-10):
            raise RuntimeError('Combined DM changed unexpectedly: stop the RTC/other DM writers before probing')
        data=camera.read()
        if data is None or data[1]==last:time.sleep(.002);continue
        if data[1]<last:raise RuntimeError('Camera counter reset')
        image,counter=data;last=counter
        if discard:discard-=1;continue
        frames.append(image.astype(float));counts.append(counter);timestamps.append(time.time_ns())
    return np.stack(frames),counts,timestamps


def save_acquisition(path,settings,modes,baseline,combined,clear,shots,records,complete):
    header=fits.Header();header['COMPLETE']=complete;header['BEAM']=settings['beam'];header['CHANNEL']=settings['channel']
    header['AMPL']=settings['amplitude'];header['BASIS']=settings['basis'];header['BUNIT']='DM command units'
    hdus=[fits.PrimaryHDU(header=header),
          json_hdu('SETTINGS',settings),json_hdu('RECORDS',records),fits.ImageHDU(modes,name='MODES'),
          fits.ImageHDU(baseline,name='CHANNEL_BEFORE'),fits.ImageHDU(combined,name='DM_BEFORE')]
    if clear is not None:
        hdus.append(fits.ImageHDU(clear,name='CLEAR_FRAMES'))
        hdus.append(fits.ImageHDU(clear.mean(axis=0),name='CLEAR_MEAN'))
    if shots:
        hdus.append(fits.ImageHDU(np.stack(shots),name='PROBE_FRAMES'))
        hdus.append(fits.ImageHDU(np.stack(shots).mean(axis=1),name='PROBE_MEANS'))
        pairs={}
        for image,record in zip(shots,records):
            if record['sign']:pairs.setdefault((record['cycle'],record['mode']),{})[record['sign']]=image.mean(axis=0)
        differences=[pair[1]-pair[-1] for pair in pairs.values() if 1 in pair and -1 in pair]
        if differences:hdus.append(fits.ImageHDU(np.stack(differences),name='DIFFERENCES'))
    temp=path.with_suffix('.fits.partial')
    with fits.HDUList(hdus) as h:h.writeto(temp,overwrite=True,checksum=True)
    os.replace(temp,path)


def acquire(args):
    if not args.execute:
        print(f'PLAN ONLY: beam {args.beam}, channel {args.channel}, modes {args.modes}, ±{args.amplitude} DM units, {args.frames} frames/sign, {args.cycles} cycles; four inner-corner ±{args.registration_amplitude} registration pokes; mask {args.mask}.\nUse --execute for acquisition; keep the RTC off. No hardware commands sent.')
        return
    modes=basis_maps(args.dcs_dir,args.basis,args.modes)
    corners=[(args.corner_offset,args.corner_offset),(args.corner_offset,11-args.corner_offset),
             (11-args.corner_offset,args.corner_offset),(11-args.corner_offset,11-args.corner_offset)]
    corner_modes=np.zeros((4,12,12))
    for k,(row,col) in enumerate(corners):corner_modes[k,row,col]=1
    if args.output.exists():raise FileExistsError(args.output)
    if args.output.with_suffix('.fits.partial').exists():raise FileExistsError('Partial output already exists')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    initial_mask=rpc(args.mds,f'fpm_whereami {args.beam}')
    if initial_mask!=args.mask:raise ValueError(f'Expected mask {args.mask}, MDS reports {initial_mask!r}')
    settings=dict(beam=args.beam,channel=args.channel,amplitude=args.amplitude,basis=args.basis,
                  mode_indices=args.modes,frames=args.frames,cycles=args.cycles,settle_frames=args.settle_frames,
                  mask=args.mask,mask_parameters=MASKS[args.mask],mask_offset_um=args.mask_offset_um,
                  coupling=args.coupling,opd_per_cmd=args.opd_per_cmd,corner_rows_cols=corners,
                  registration_amplitude=args.registration_amplitude,clear_record=None,restored=False,
                  format_version=2)
    from xaosim.shmlib import shm
    import posix_ipc
    with ExitStack() as stack:
        def reader(name):
            obj=ImageReader(args.shm_dir/name);stack.callback(obj.close);return obj
        source=reader(f'dm{args.beam}disp{args.channel:02d}.im.shm')
        master=reader(f'dm{args.beam}.im.shm');camera=reader(f'baldr{args.beam}.im.shm')
        baseline=source.stable()[0];combined=master.stable()[0]
        if baseline.shape!=(12,12) or baseline.dtype.kind!='f':raise ValueError('Expected float 12x12 DM channel')
        active=np.ones((12,12),bool);active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
        for mode,amplitude in [(m,args.amplitude) for m in modes]+[(m,args.registration_amplitude) for m in corner_modes]:
            for sign in (-1,1):
                trial=combined+sign*amplitude*mode
                if np.any((trial[active]<0)|(trial[active]>1)):raise ValueError('Probe would clip combined DM; reduce amplitude')
        writer=shm(str(source.path),nosem=True);stack.callback(writer.close,erase_file=False)
        if ImageReader.stat_id(os.fstat(writer.fd))!=source.identity:raise RuntimeError('Probe SHM replaced')
        prefix=str(args.shm_dir.resolve()).replace('/','.')
        sem=posix_ipc.Semaphore(f'/{prefix}.dm{args.beam}_sem01',flags=0);stack.callback(sem.close)
        last_written=baseline.copy();mask_out=False;mask_uncertain=False
        clear=None;shots=[];records=[];complete=False
        def write(command):
            nonlocal last_written
            current=source.stable()[0]
            if not np.array_equal(current,last_written):raise RuntimeError('Probe channel modified by another writer')
            last_written=np.asarray(command,dtype=baseline.dtype)
            writer.set_data(last_written);sem.release()
        def on_signal(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        previous=signal.signal(signal.SIGTERM,on_signal);stack.callback(signal.signal,signal.SIGTERM,previous)
        try:
            save_acquisition(args.output,settings,modes,baseline,combined,clear,shots,records,False)
            mask_uncertain=True
            rpc(args.mds,f'moverel BMY{args.beam} {args.mask_offset_um}')
            mask_out=True;mask_uncertain=False
            if rpc(args.mds,f'fpm_whereami {args.beam}')!='':raise RuntimeError('Mask did not move out')
            time.sleep(args.mask_settle_s)
            clear,counts,times=camera_frames(camera,master,combined,args.frames,args.settle_frames,args.timeout)
            settings['clear_record']=dict(counters=counts,time_ns=times)
            mask_uncertain=True
            rpc(args.mds,f'moverel BMY{args.beam} {-args.mask_offset_um}')
            mask_out=False;mask_uncertain=False
            time.sleep(args.mask_settle_s)
            if rpc(args.mds,f'fpm_whereami {args.beam}')!=initial_mask:raise RuntimeError('Mask restore verification failed')
            orders=[('baseline',-1,0,0)]
            orders += [('registration',-2,k,sign) for k in range(4) for sign in (1,-1)]
            for cycle in range(args.cycles):
                for k in range(len(modes)):
                    for sign in ((1,-1) if cycle%2==0 else (-1,1)):orders.append(('modal',cycle,k,sign))
            for kind,cycle,k,sign in orders:
                amplitude=args.registration_amplitude if kind=='registration' else args.amplitude
                mode=corner_modes[k] if kind=='registration' else modes[k]
                delta=sign*amplitude*mode
                write(baseline+delta)
                frames,counts,times=camera_frames(camera,master,combined+(last_written.astype(float)-baseline),args.frames,args.settle_frames,args.timeout)
                shots.append(frames)
                records.append(dict(kind=kind,cycle=cycle,mode=k,
                                    mode_index=args.modes[k] if kind=='modal' else None,sign=sign,
                                    counters=counts,time_ns=times,command=last_written.tolist()))
                save_acquisition(args.output,settings,modes,baseline,combined,clear,shots,records,False)
                print(f'Saved {kind}, index {k}, sign {sign:+d}'+(f', cycle {cycle+1}' if kind=='modal' else ''),flush=True)
            complete=True
        finally:
            failures=[]
            try:write(baseline)
            except Exception as exc:failures.append(f'DM restoration failed: {exc}')
            if mask_uncertain:
                failures.append('Mask command outcome uncertain; inspect mask position before moving it again')
            elif mask_out:
                try:rpc(args.mds,f'moverel BMY{args.beam} {-args.mask_offset_um}')
                except Exception as exc:failures.append(f'Mask restoration failed: {exc}')
            if not mask_uncertain:
                try:
                    if rpc(args.mds,f'fpm_whereami {args.beam}')!=initial_mask:
                        failures.append('Mask name does not match initial state; inspect MDS')
                except Exception as exc:failures.append(f'Mask restoration could not be verified: {exc}')
            settings['restored']=not failures;settings['restoration_errors']=failures
            save_acquisition(args.output,settings,modes,baseline,combined,clear,shots,records,complete and not failures)
            if failures:raise RuntimeError('; '.join(failures))
    print(f'Acquisition saved: {args.output}. Original probe channel and mask restored. Flat and RTC untouched.')


def fit_file(args):
    from scipy.interpolate import RectBivariateSpline
    from scipy.optimize import curve_fit, least_squares
    from scipy.special import j0, j1
    from scipy.sparse import lil_matrix

    if args.output.exists():raise FileExistsError(args.output)
    with fits.open(args.input) as h:
        if not h[0].header.get('COMPLETE',False):raise ValueError('Acquisition incomplete; inspect restoration status')
        settings=read_json(h,'SETTINGS');records=read_json(h,'RECORDS')
        if settings.get('format_version')!=2:
            raise ValueError('This fit requires a new acquisition with four registration pokes (format_version=2). Older Fresnel fit files remain readable by analyse_probe_phase.py.')
        clear=h['CLEAR_MEAN'].data.astype(float)
        frames=h['PROBE_FRAMES'].data.astype(float)
        before=h['CHANNEL_BEFORE'].data.astype(float)
    mask=MASKS[settings['mask']]
    wavelength=mask['wavelength'];theta=mask['theta']
    mu=float(np.angle(np.exp(1j*theta)-1));M=abs(np.exp(1j*theta)-1)
    yy,xx=np.indices(clear.shape)
    border=np.zeros(clear.shape,bool);border[[0,-1],:]=True;border[:,[0,-1]]=True
    background=float(np.median(clear[border])) if args.background is None else args.background
    clear=np.maximum(clear-background,0)
    pupil=clear>args.pupil_threshold*clear.max()
    if pupil.sum()<12:raise ValueError('Too few illuminated pixels; check background/pupil threshold')
    # Geometry of the optical pupil is distinct from the fitted DM centre.
    cx=float((xx[pupil].min()+xx[pupil].max())/2)
    cy=float((yy[pupil].min()+yy[pupil].max())/2)
    radius=float(np.max(np.hypot(xx[pupil]-cx,yy[pupil]-cy)))
    if args.pupil_geometry is not None:cx,cy,radius=args.pupil_geometry
    if radius<=0:raise ValueError('Pupil radius must be positive')
    pupil &= np.hypot(xx-cx,yy-cy)<=radius+1e-6
    if pupil.sum()<12:raise ValueError('Pupil geometry excludes the illuminated region')
    norm=float(np.median(clear[pupil]));A=clear/norm
    means=frames.mean(axis=1)

    # Match the existing registration approach: spline interpolation followed by
    # a local elliptical Gaussian fit to each absolute +/- intensity difference.
    def gaussian(xy,amp,x0,y0,sx,sy,angle,offset):
        x,y=xy;u=(x-x0)*np.cos(angle)+(y-y0)*np.sin(angle)
        v=-(x-x0)*np.sin(angle)+(y-y0)*np.cos(angle)
        return offset+amp*np.exp(-.5*((u/sx)**2+(v/sy)**2))

    corner_images=[];centres=[];gaussian_parameters=[]
    for k in range(4):
        pair={r['sign']:i for i,r in enumerate(records) if r['kind']=='registration' and r['mode']==k}
        delta=means[pair[1]]-means[pair[-1]];corner_images.append(delta)
        image=np.abs(delta)
        py,px=np.unravel_index(np.argmax(np.where(pupil,image,-np.inf)),image.shape)
        x=np.arange(max(0,px-args.centroid_radius),min(image.shape[1],px+args.centroid_radius+1))
        y=np.arange(max(0,py-args.centroid_radius),min(image.shape[0],py+args.centroid_radius+1))
        if min(len(x),len(y))<4:raise ValueError('Registration response too close to image boundary')
        fine_x=np.linspace(x[0],x[-1],len(x)*5);fine_y=np.linspace(y[0],y[-1],len(y)*5)
        fx,fy=np.meshgrid(fine_x,fine_y)
        data=RectBivariateSpline(y,x,image[np.ix_(y,x)])(fine_y,fine_x)
        amplitude=max(float(data.max()-np.median(data)),1e-9)
        popt,_=curve_fit(gaussian,(fx.ravel(),fy.ravel()),data.ravel(),
                        p0=[amplitude,px,py,1,1,0,float(np.median(data))],
                        bounds=([0,x[0],y[0],.2,.2,-np.pi,-np.inf],
                                [np.inf,x[-1],y[-1],len(x),len(y),np.pi,np.inf]),maxfev=10000)
        if np.linalg.norm(popt[1:3]-[px,py])>args.centroid_radius:
            raise ValueError('Gaussian centre moved outside the poke neighbourhood')
        centres.append(popt[1:3]);gaussian_parameters.append(popt)
    centres=np.asarray(centres)
    corners=np.asarray(settings['corner_rows_cols'],float)[:,::-1]-5.5
    corner_design=np.column_stack([corners,np.ones(4)])
    # Known actuator order defines opposite corners (0,3) and (1,2), including flips.
    v=centres[3]-centres[0];w=centres[2]-centres[1]
    crossing=np.linalg.solve(np.column_stack([v,-w]),centres[1]-centres[0])
    centre=centres[0]+crossing[0]*v
    affine=np.column_stack([np.linalg.lstsq(corners,centres-centre,rcond=None)[0].T,centre])
    if np.linalg.cond(affine[:,:2])>10:raise ValueError('Degenerate registration; inspect the corner responses')
    print('Initial DM-to-camera affine matrix:\n',affine,flush=True)

    # Uniform circular-pupil reference amplitude, in units of incident amplitude:
    # b(rho) = integral_0^(pi*d/2) J1(t) J0(rho*t) dt, d = mask DIAMETER in lambda/D.
    # Gauss-Legendre quadrature avoids a propagation/config dependency. Obscurations,
    # scintillation and broadband changes of b are intentionally not modelled.
    nodes,weights=np.polynomial.legendre.leggauss(128)
    cutoff=np.pi*mask['diameter_lamD']/2;t=(nodes+1)*cutoff/2
    rho=np.hypot(xx-cx,yy-cy)/radius
    B=np.sum(j0(rho[...,None]*t)*j1(t)*weights,axis=-1)*cutoff/2
    selected=[i for i,r in enumerate(records) if r['kind']!='registration']
    observed=(means[selected]-background)/norm
    sigma=np.maximum(frames[selected].std(axis=1,ddof=1)/np.sqrt(frames.shape[1]),args.noise_floor)/norm
    commands=np.asarray([np.asarray(records[i]['command'])-before for i in selected])
    active=np.ones((12,12),bool);active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
    ay,ax=np.indices((12,12));actuators=np.column_stack([ax[active]-5.5,ay[active]-5.5])
    commands=commands[:,active]
    coupling=settings['coupling'] if args.coupling is None else args.coupling
    gain=settings['opd_per_cmd'] if args.opd_per_cmd is None else args.opd_per_cmd
    pixels=np.column_stack([xx[pupil],yy[pupil]])
    ap=A[pupil];bp=B[pupil];base=ap+M*M*bp*bp;response=2*M*np.sqrt(ap)*bp
    n=pupil.sum();exposures=len(selected)
    obs=observed[:,pupil];err=sigma[:,pupil]

    def prediction(parameters):
        transform=parameters[:6].reshape(2,3)
        # Evaluate circular Gaussian influences in DM coordinates. The affine
        # transform then carries rotation, scale, shear and parity into the image.
        uv=np.linalg.solve(transform[:,:2],(pixels-transform[:,2]).T).T
        distance=np.sum((uv[:,None,:]-actuators[None,:,:])**2,axis=2)
        influences=np.exp(-distance/coupling**2)
        probe_phase=(2*np.pi*gain/wavelength)*(commands @ influences.T)
        angle=mu-parameters[6:][None,:]-probe_phase
        return base[None,:]+response[None,:]*np.cos(angle),probe_phase

    def residual(parameters):
        model,_=prediction(parameters)
        # A pixel-space prior is independent of how affine coefficients are scaled.
        shift=corner_design @ parameters[:6].reshape(2,3).T-centres
        return np.concatenate([((model-obs)/err).ravel(),(shift/args.registration_prior_px).ravel()])

    initial=np.r_[affine.ravel(),np.zeros(n)]
    _,probe_phase=prediction(initial)
    # Linear sin/cos initialization, using all probes; then select the near-flat branch.
    c=response[None,:]*np.cos(mu-probe_phase)/err
    s=response[None,:]*np.sin(mu-probe_phase)/err
    target=(obs-base)/err
    cc=np.sum(c*c,axis=0);ss=np.sum(s*s,axis=0);cs=np.sum(c*s,axis=0)
    yc=np.sum(target*c,axis=0);ys=np.sum(target*s,axis=0);det=cc*ss-cs*cs
    good=det>1e-10*np.maximum(cc*ss,1e-30)
    initial[6:][good]=np.arctan2((ys*cc-yc*cs)[good],(yc*ss-ys*cs)[good])
    initial[6:]=np.clip(initial[6:],-args.phase_bound*.95,args.phase_bound*.95)
    # Each image pixel depends only on its own phase and the six shared affine terms.
    pattern=lil_matrix((exposures*n+8,n+6),dtype=int);pattern[:,:6]=1
    for k in range(exposures):pattern[k*n+np.arange(n),6+np.arange(n)]=1
    scale=max(float(np.linalg.svd(affine[:,:2],compute_uv=False).min()),1e-3)
    limits=np.tile([.2*scale,.2*scale,args.registration_bound_px],2)
    lower=np.r_[affine.ravel()-limits,np.full(n,-args.phase_bound)]
    upper=np.r_[affine.ravel()+limits,np.full(n,args.phase_bound)]
    result=least_squares(residual,initial,bounds=(lower,upper),jac_sparsity=pattern.tocsr(),
                         x_scale='jac',max_nfev=args.max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    predicted,probes=prediction(result.x)
    phase=np.full(clear.shape,np.nan);phase[pupil]=result.x[6:]
    opd=(phase-np.mean(phase[pupil]))*wavelength/(2*np.pi)
    model=np.full(observed.shape,np.nan);model[:,pupil]=predicted
    phase_information=np.sum((response[None,:]*np.sin(mu-phase[pupil][None,:]-probes)/err)**2,axis=0)
    uncertainty=np.full(clear.shape,np.nan)
    uncertainty[pupil]=np.divide(1,np.sqrt(phase_information),out=np.full(n,np.inf),where=phase_information>0)
    hit=np.abs(phase[pupil])>=args.phase_bound*.99
    bound_image=np.zeros(clear.shape,np.uint8);bound_image[pupil]=hit
    refined=result.x[:6].reshape(2,3)
    affine_bound=bool(np.any(np.minimum(result.x[:6]-lower[:6],upper[:6]-result.x[:6])<.01*limits))
    report=dict(success=bool(result.success),message=result.message,nfev=result.nfev,
                phase_bound_rad=args.phase_bound,phase_bound_pixels=int(hit.sum()),registration_bound_hit=affine_bound,
                residual_rms_normalised=float(np.sqrt(np.mean((obs-predicted)**2))),
                weighted_residual_rms=float(np.sqrt(np.mean(((obs-predicted)/err)**2))),
                opd_rms_nm=float(np.std(opd[pupil])*1e9),background_adu=background,normalisation_adu=norm,
                pupil_geometry=[cx,cy,radius],coupling=coupling,opd_per_cmd=gain,mu_rad=mu,
                registration_prior_px=args.registration_prior_px,
                registration_initial_rms_px=float(np.sqrt(np.mean((corner_design@affine.T-centres)**2))),
                registration_refined_rms_px=float(np.sqrt(np.mean((corner_design@refined.T-centres)**2))),
                assumption='Single-wavelength local intensity law; fixed ideal circular-pupil radial reference amplitude; one shared phase map relative to reference phase zero. OPD piston removed. PHASE_SIGMA is conditional on fixed registration/model, not total uncertainty.',
                input=str(args.input.resolve()))
    header=fits.Header();header['WLREF']=wavelength;header['SUCCESS']=result.success;header['MODEL']='ANALYTIC';header['VERSION']=2
    hdus=[fits.PrimaryHDU(header=header),json_hdu('FIT_REPORT',report),json_hdu('SETTINGS',settings),
          json_hdu('RECORDS',[records[i] for i in selected]),
          json_hdu('FIT_OPTIONS',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})]
    for name,data in [('PHASE_RAD',phase),('OPD',opd),('PUPIL',pupil.astype(np.uint8)),('A',A),('B_AMPLITUDE',B),
                      ('OBSERVED',observed),('MODEL',model),('RESIDUAL',observed-model),('SIGMA',sigma),
                      ('PHASE_SIGMA',uncertainty),('PHASE_BOUND',bound_image),('CORNER_IMAGES',np.asarray(corner_images)),
                      ('CORNER_DM',corners),('CORNER_CENTRES',centres),('CORNER_GAUSSIANS',np.asarray(gaussian_parameters)),
                      ('DIAGONAL_CENTRE',centre),('AFFINE_INITIAL',affine),('AFFINE_REFINED',refined)]:
        hdu=fits.ImageHDU(data,name=name)
        if name=='OPD':hdu.header['BUNIT']='m'
        if name in ('PHASE_RAD','PHASE_SIGMA'):hdu.header['BUNIT']='rad'
        hdus.append(hdu)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fits.HDUList(hdus).writeto(args.output,checksum=True)
    print(json.dumps(report,indent=2));print(f'Fit saved: {args.output}')
    if not result.success or np.any(hit) or affine_bound:
        print('FIT NEEDS REVIEW: convergence or parameter bounds reached; inspect diagnostics.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    commands=parser.add_subparsers(dest='command',required=True)
    a=commands.add_parser('acquire',help='Plan or execute registration and modal probes')
    a.add_argument('--beam',type=int,choices=range(1,5),required=True)
    a.add_argument('--channel',type=int,choices=range(1,5),default=3,help='Dedicated idle probe channel; default 3')
    a.add_argument('--amplitude',type=float,required=True,help='Raw DCS mode coefficient in DM units')
    a.add_argument('--registration-amplitude',type=float,help='Single-actuator poke in DM units; default modal amplitude')
    a.add_argument('--corner-offset',type=int,choices=range(1,6),default=3,help='Inset from 12x12 edge for registration square')
    a.add_argument('--modes',type=int,nargs='+',default=[0,1,2,3],help='Zero-based DCS indices; select spatially diverse modes')
    a.add_argument('--basis',choices=['Zernike','Fourier','FourierModified','Zonal'],default='Zernike')
    a.add_argument('--dcs-dir',type=Path,default=DEFAULT_DCS)
    a.add_argument('--mask',choices=MASKS,default='J3',help='Mask already in the beam; verified through MDS')
    a.add_argument('--coupling',type=float,default=.7,help='Gaussian width in actuator pitches; simulator currently overrides config to .7')
    a.add_argument('--opd-per-cmd',type=float,default=3e-6,help='Metres OPD per DM command, already includes reflection')
    a.add_argument('--frames',type=int,default=10)
    a.add_argument('--cycles',type=int,default=2)
    a.add_argument('--settle-frames',type=int,default=3)
    a.add_argument('--timeout',type=float,default=120)
    a.add_argument('--mask-offset-um',type=float,default=200)
    a.add_argument('--mask-settle-s',type=float,default=1)
    a.add_argument('--mds',default='tcp://127.0.0.1:5555')
    a.add_argument('--shm-dir',type=Path,default=Path('/dev/shm'))
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--execute',action='store_true')
    f=commands.add_parser('fit',help='Offline analytic common-phase fit; no hardware access')
    f.add_argument('input',type=Path);f.add_argument('--output',type=Path,required=True)
    f.add_argument('--background',type=float,help='ADU; default median border of clear image')
    f.add_argument('--pupil-threshold',type=float,default=.15,help='Clear intensity fraction of maximum')
    f.add_argument('--pupil-geometry',type=float,nargs=3,metavar=('CX','CY','RADIUS'),help='Camera pixels; default clear pupil extent')
    f.add_argument('--noise-floor',type=float,default=1,help='Minimum uncertainty on mean intensity, ADU')
    f.add_argument('--centroid-radius',type=int,default=3,help='Gaussian fitting neighbourhood radius, pixels')
    f.add_argument('--registration-prior-px',type=float,default=.2,help='1-sigma penalty per measured corner coordinate')
    f.add_argument('--registration-bound-px',type=float,default=1,help='Maximum centre shift; linear affine terms bounded to 20 percent of minimum initial scale')
    f.add_argument('--phase-bound',type=float,default=np.pi/2,help='Near-flat phase branch bound, radians')
    f.add_argument('--coupling',type=float,help='Override saved Gaussian width/pitch')
    f.add_argument('--opd-per-cmd',type=float,help='Override saved metres OPD per command')
    f.add_argument('--max-nfev',type=int,default=200)
    args=parser.parse_args()
    if args.command=='acquire':
        if args.registration_amplitude is None:args.registration_amplitude=args.amplitude
        if args.frames<2 or args.cycles<1 or args.settle_frames<1:parser.error('frames>=2, cycles>=1, settle-frames>=1 required')
        if any(i<0 for i in args.modes) or len(set(args.modes))!=len(args.modes):parser.error('Modes must be unique nonnegative indices')
        positive=[args.amplitude,args.registration_amplitude,args.timeout,args.mask_offset_um,args.coupling,args.opd_per_cmd]
        if not np.isfinite(args.mask_settle_s) or args.mask_settle_s<0:parser.error('Invalid mask settling time')
    else:
        positive=[args.noise_floor,args.registration_prior_px,args.registration_bound_px,args.phase_bound]
        positive += [v for v in (args.coupling,args.opd_per_cmd) if v is not None]
        if not 0<args.pupil_threshold<1 or args.max_nfev<1 or args.centroid_radius<2:parser.error('Invalid fit settings')
        if args.background is not None and not np.isfinite(args.background):parser.error('Background must be finite')
        if args.pupil_geometry is not None and not np.isfinite(args.pupil_geometry).all():parser.error('Invalid pupil geometry')
    if any(not np.isfinite(v) or v<=0 for v in positive):parser.error('Amplitudes, scales and limits must be positive and finite')
    (acquire if args.command=='acquire' else fit_file)(args)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit('Interrupted; inspect acquisition restoration status before further commands.')
    except (OSError,RuntimeError,ValueError,TimeoutError) as exc:raise SystemExit(f'ERROR: {exc}')
