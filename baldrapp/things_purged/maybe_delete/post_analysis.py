#!/usr/bin/env python3
"""Offline phase-regularisation sweep; never reads or writes live shared memory.

Keep this beside probe_dm_phase_v2.py (used only for initial offline calibration).
Example:
  python post_analysis.py probes.fits --output-dir post_analysis
  python post_analysis.py probes.fits --seed-fit post_analysis/seed_fit.fits --output-dir another_sweep

Registration is re-estimated from individual nonlinear modal exposures and corner subsets.
By default only registration diagnostics are produced; add --fit-phase to run
the smoothing sweep after inspection. Pupil, reference field and atmospheric
errors use the seed fit. The chosen registration is fixed during phase fitting. No DM correction is generated or applied. The seed fit's correction
is NOT the correction for these refits. Defocus is measured afterwards, never a
prior. Defocus is the coefficient of 2*rho**2-1 in a joint least-squares
projection onto piston, x/y tilt, defocus and two astigmatisms over valid pixels.
Its reported RMS is evaluated on those pixels; it is not a DM-unit coefficient.
Registration plots predict the held-out cycle. Phase-sweep residuals are in-sample
diagnostics, not independent validation. The optional phase sweep retains the
simpler fixed reference-field model; gain and training-cycle flux are carried over.

Phase-sweep objective = mean(soft_l1(weighted intensity residual, scale=3))
            + strength * mean((L @ phase)**2).
L is a nearest-neighbour difference (gradient), or a graph Laplacian (curvature),
on valid pupil pixels only. Phase is in radians, pixel spacing is one camera
pixel. Curvature uses the available neighbours at boundaries (graph boundary
condition); its strength is not numerically equivalent to gradient strength.
"""
import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import sparse
from scipy.optimize import least_squares
import probe_dm_phase_v2 as backend


# Registration controls: fixed here to keep the command line small.
REG_GAIN_BOUNDS = (.2, 2.5)  # multiplier of recorded OPD per DM command
REG_FLUX_BOUNDS = (.8, 1.2)  # relative flux; first training cycle anchors unity
REG_FLUX_PRIOR = .05
REG_MAX_NFEV = 100
REG_STARTS_PER_PARITY = 5


def fit_registration(data, records, settings, report, pupil, out):
    """Joint +/- fit: I_jp = flux_cycle * [a_p+b_p cos(delta_jp)+c_p sin(delta_jp)].

    Three shared coefficients per pixel absorb mean intensity and a static mean
    phase/contrast. No per-exposure phase is fitted. Nonlinear probe phases use
    actual commands. Robust variable projection uses fixed-count IRLS; it is an
    approximation to the profiled robust objective. Held-out cycles do not fit
    geometry or pixel coefficients; predictive scores use flux=1, and an optional
    separately labelled flux-adjusted score fits only a scalar on that cycle.
    """
    from itertools import combinations
    yy,xx=np.indices(pupil.shape)
    pixels=np.column_stack([xx[pupil],yy[pupil]])
    active=np.ones((12,12),bool)
    active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
    ay,ax=np.indices((12,12)); actuators=np.column_stack([ax[active]-5.5,ay[active]-5.5])
    ids=np.array([i for i,r in enumerate(records) if r['kind']=='modal'])
    cycles=sorted({records[i]['cycle'] for i in ids})
    modes=sorted({records[i]['mode'] for i in ids})
    if len(modes)<3 or len(cycles)<2:
        raise ValueError('Need at least three paired modes and two cycles (one held out)')
    for cycle in cycles:
        for mode in modes:
            signs=[records[i]['sign'] for i in ids if records[i]['cycle']==cycle and records[i]['mode']==mode]
            if sorted(signs)!=[-1,1]: raise ValueError('Every modal cycle must contain one +/- pair per mode')
    train=ids[[records[i]['cycle']!=cycles[-1] for i in ids]]
    test=ids[[records[i]['cycle']==cycles[-1] for i in ids]]
    train_cycles=cycles[:-1]
    cycle_index=np.array([train_cycles.index(records[i]['cycle']) for i in train])
    commands=np.array([np.asarray(r['command'])-data['CHANNEL_BEFORE'] for r in records])[:,active]
    observed=data['OBSERVED'][:,pupil]; sigma=data['SIGMA'][:,pupil]
    if not np.isfinite(observed[ids]).all() or not np.isfinite(sigma[ids]).all() or np.any(sigma[ids]<=0):
        raise ValueError('Invalid modal observations or uncertainties')
    obs=observed[train]; err=sigma[train]; n=pupil.sum()
    cx,cy,radius=report['pupil_geometry']; pitch=2*radius/backend.MIXED_PUPIL_PITCHES
    corners=data['CORNER_DM']; centres=data['CORNER_CENTRES']
    factor=2*np.pi*report['opd_per_cmd']/backend.MASKS[settings['mask']]['wavelength']
    def transform(v,parity):
        angle,scale,stretch,shear,x,y=v[:6]; c,s=np.cos(angle),np.sin(angle)
        linear=scale*np.array([[c,-s],[s,c]])@np.array([[np.exp(stretch),shear],[0,np.exp(-stretch)]])@np.diag([1,parity])
        return np.column_stack([linear,[x,y]])
    def features(v,parity,indices):
        af=transform(v,parity)
        uv=np.linalg.solve(af[:,:2],(pixels-af[:,2]).T).T
        inf=np.exp(-np.sum((uv[:,None]-actuators[None])**2,axis=2)/report['coupling']**2)
        delta=v[6]*factor*(commands[indices]@inf.T)
        return np.stack([np.ones_like(delta),np.cos(delta),np.sin(delta)],axis=-1)
    def evaluate(v,parity,details=False):
        flux=np.r_[1.,v[7:]]
        x=features(v,parity,train)*flux[cycle_index,None,None]
        design=x/err[...,None]; target=obs/err
        weights=np.ones_like(target)
        # Same soft-L1 data loss as the outer fit, unlike the old OLS profile.
        for _ in range(8):
            gram=np.einsum('epk,epl,ep->pkl',design,design,weights)
            rhs=np.einsum('epk,ep,ep->pk',design,target,weights)
            ridge=1e-10*np.maximum(np.trace(gram,axis1=1,axis2=2),1)
            coef=np.linalg.solve(gram+ridge[:,None,None]*np.eye(3),rhs[...,None])[...,0]
            predicted=np.einsum('epk,pk->ep',x,coef)
            weights=1/np.sqrt(1+(((predicted-obs)/err)/3)**2)
        af=transform(v,parity)
        distance=(corners@af[:,:2].T+af[:,2]-centres)/1.5
        # Corner influence is bounded by soft-L1. Affine geometry has weak priors.
        prior=np.r_[distance.ravel(),v[2]/.15,v[3]/.2,(flux[1:]-1)/REG_FLUX_PRIOR]
        residual=np.r_[((predicted-obs)/err).ravel(),prior]
        return (predicted,coef,af,flux) if details else residual
    starts=[]
    for parity in (-1,1):
        for angle in np.deg2rad(np.arange(0,360,30)):
            for gain in (.4,1.,2.):
                starts.append((np.r_[angle,pitch,0,0,cx,cy,gain,np.ones(len(train_cycles)-1)],parity,'grid'))
        for size in (2,3,4):
            for subset in combinations(range(4),size):
                source=corners[list(subset)]@np.diag([1,parity]); target=centres[list(subset)]
                a=source-source.mean(axis=0); b=target-target.mean(axis=0)
                u,_,vt=np.linalg.svd(a.T@b)
                rotation=vt.T@np.diag([1,np.linalg.det(vt.T@u.T)])@u.T
                scale=np.sum((a@rotation.T)*b)/np.sum(a*a)
                centre=target.mean(axis=0)-scale*rotation@source.mean(axis=0)
                if .6*pitch<scale<1.5*pitch and np.linalg.norm(centre-[cx,cy])<.5*radius:
                    for gain in (.4,1.,2.):
                        starts.append((np.r_[np.arctan2(rotation[1,0],rotation[0,0]),scale,0,0,centre,gain,np.ones(len(train_cycles)-1)],parity,'corners_'+''.join(map(str,subset))))
    score=lambda r: float(np.mean(18*(np.sqrt(1+(r/3)**2)-1)))
    ranked=sorted([(score(evaluate(v,p)),v,p,label) for v,p,label in starts],key=lambda t:t[0])
    chosen=[]
    # Keep both parities and distinct gain/orientation basins in the local search.
    for parity in (-1,1):
        selected=[]
        for item in ranked:
            _,v,p,label=item
            if p!=parity: continue
            if any(abs(np.angle(np.exp(1j*(v[0]-w[1][0]))))<.25 and abs(np.log(v[6]/w[1][6]))<.4 for w in selected): continue
            selected.append(item)
            if len(selected)==REG_STARTS_PER_PARITY: break
        chosen.extend(selected)
    candidates=[]
    # Predict a repeat from earlier cycles of the SAME commanded mode and sign.
    empirical=np.array([np.mean(observed[[i for i in train if records[i]['mode']==records[j]['mode'] and records[i]['sign']==records[j]['sign']]],axis=0) for j in test])
    constant=np.broadcast_to(np.mean(obs,axis=0),observed[test].shape)
    rms=lambda value: float(np.sqrt(np.mean(value**2)))
    for k,(_,v,parity,label) in enumerate(chosen):
        print(f'Registration search {k+1}/{len(chosen)} (parity {parity}, {label})',flush=True)
        lo=np.r_[v[0]-np.pi/2,.6*pitch,-.22,-.3,cx-.5*radius,cy-.5*radius,REG_GAIN_BOUNDS[0],np.full(len(train_cycles)-1,REG_FLUX_BOUNDS[0])]
        hi=np.r_[v[0]+np.pi/2,1.5*pitch,.22,.3,cx+.5*radius,cy+.5*radius,REG_GAIN_BOUNDS[1],np.full(len(train_cycles)-1,REG_FLUX_BOUNDS[1])]
        result=least_squares(evaluate,v,args=(parity,),bounds=(lo,hi),loss='soft_l1',f_scale=3,max_nfev=REG_MAX_NFEV,diff_step=1e-4,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        predicted,coef,af,flux=evaluate(result.x,parity,True)
        test_pred=np.einsum('epk,pk->ep',features(result.x,parity,test),coef)
        w=1/sigma[test]**2
        adjusted_flux=float(np.clip(np.sum(w*test_pred*observed[test])/max(np.sum(w*test_pred**2),1e-20),*REG_FLUX_BOUNDS))
        candidates.append(dict(score=score(result.fun),success=bool(result.success),message=str(result.message),nfev=result.nfev,start=label,parity=parity,parameters=result.x.tolist(),affine=af.tolist(),gain_multiplier=float(result.x[6]),cycle_flux=flux.tolist(),
            bound_hit=bool(np.any(np.minimum(result.x-lo,hi-result.x)<.005*(hi-lo))),
            heldout_rms=rms((test_pred-observed[test])/sigma[test]),heldout_flux_adjusted_rms=rms((adjusted_flux*test_pred-observed[test])/sigma[test]),heldout_adjusted_flux=adjusted_flux,
            corner_residual_px=np.linalg.norm(corners@af[:,:2].T+af[:,2]-centres,axis=1).tolist()))
    candidates.sort(key=lambda r:r['score'])  # Held-out data NEVER select the solution.
    best=candidates[0];v=np.array(best['parameters']);parity=best['parity']
    predicted,coef,affine,flux=evaluate(v,parity,True)
    all_pred=np.einsum('epk,pk->ep',features(v,parity,ids),coef)
    for j,i in enumerate(ids):
        if records[i]['cycle'] in train_cycles: all_pred[j]*=flux[train_cycles.index(records[i]['cycle'])]
    lookup={i:j for j,i in enumerate(ids)}
    differences=[];pred_differences=[];sums=[];pred_sums=[];mode_metrics=[]
    for mode in modes:
        pos=[i for i in test if records[i]['mode']==mode and records[i]['sign']==1][0]
        neg=[i for i in test if records[i]['mode']==mode and records[i]['sign']==-1][0]
        d=observed[pos]-observed[neg]; pd=all_pred[lookup[pos]]-all_pred[lookup[neg]]
        se=np.sqrt(sigma[pos]**2+sigma[neg]**2)
        differences.append(d);pred_differences.append(pd)
        sums.append(observed[pos]+observed[neg]);pred_sums.append(all_pred[lookup[pos]]+all_pred[lookup[neg]])
        mode_metrics.append(dict(mode=mode,measured_rms=rms(d),predicted_rms=rms(pd),residual_rms=rms(d-pd),weighted_residual_rms=rms((d-pd)/se),zero_rms=rms(d/se),fraction_energy_explained=float(1-np.sum((d-pd)**2)/max(np.sum(d*d),1e-20))))
    d=np.asarray(differences);pd=np.asarray(pred_differences)
    explained=float(1-np.sum((d-pd)**2)/max(np.sum(d*d),1e-20))
    alternatives=[r for r in candidates[1:] if np.linalg.norm(np.asarray(r['affine'])-affine)>.5 or abs(np.log(r['gain_multiplier']/best['gain_multiplier']))>.2]
    gap=(alternatives[0]['score']-best['score'])/max(abs(best['score']),1e-12) if alternatives else None
    warnings=[]
    if gap is not None and gap<.05: warnings.append('Competing geometry/gain solutions score within 5%.')
    if explained<.5: warnings.append('Less than half the held-out difference energy is explained.')
    if best['heldout_rms']>=.9*rms((constant-observed[test])/sigma[test]):
        warnings.append('Less than 10% RMS improvement over a constant pixel response on the held-out cycle.')
    if np.sum(np.asarray(best['corner_residual_px'])<2)<2:
        warnings.append('Fewer than two corner predictions agree within two pixels.')
    if not best['success']: warnings.append('Selected optimisation did not converge.')
    if best['bound_hit']: warnings.append('Selected parameters are at a search bound.')
    payload=dict(method='Affine + global gain; joint nonlinear +/- exposures; shared per-pixel mean/cos/sin coefficients; robust IRLS; cycle flux',
        training_cycles=train_cycles,heldout_cycle=cycles[-1],validation_note='Geometry and pixel response use training cycles only; seed pupil and uncertainties still use all frames. Flux-adjusted score fits one held-out scalar and is not blind prediction.',
        gain_bounds=REG_GAIN_BOUNDS,flux_bounds=REG_FLUX_BOUNDS,flux_prior_sigma=REG_FLUX_PRIOR,max_nfev=REG_MAX_NFEV,starts_per_parity=REG_STARTS_PER_PARITY,irls_iterations=8,corner_sigma_px=1.5,heldout_difference_energy_explained=explained,
        baseline_heldout_rms=dict(zero=rms(observed[test]/sigma[test]),constant_pixel_mean=rms((constant-observed[test])/sigma[test]),repeat_mode_sign_mean=rms((empirical-observed[test])/sigma[test])),
        candidates=candidates,mode_metrics=mode_metrics,warnings=warnings,ambiguous=bool(warnings),
        caution='Optimisation convergence is not proof of registration. Review model failure and competing solutions before phase fitting.')
    (out/'registration_candidates.json').write_text(json.dumps(payload,indent=2))
    for warning in warnings: print('WARNING:',warning,flush=True)
    maps=np.full((len(ids),*pupil.shape),np.nan);maps[:,pupil]=all_pred
    fits.HDUList([fits.PrimaryHDU(),backend.json_hdu('REPORT',payload),backend.json_hdu('RECORDS',[records[i] for i in ids]),fits.ImageHDU(maps,name='MODEL'),fits.ImageHDU(data['OBSERVED'][ids],name='OBSERVED'),fits.ImageHDU(data['SIGMA'][ids],name='SIGMA'),fits.ImageHDU(affine,name='AFFINE'),fits.ImageHDU(coef,name='RESPONSE_COEFFICIENTS'),fits.ImageHDU(pupil.astype('uint8'),name='PUPIL')]).writeto(out/'registration_fit.fits',overwrite=True,checksum=True)
    positions=actuators@affine[:,:2].T+affine[:,2]; expected=corners@affine[:,:2].T+affine[:,2]
    fig,axplot=plt.subplots(figsize=(7,6),layout='constrained');axplot.imshow(data['CLEAR_MEAN'],origin='lower');axplot.scatter(*positions.T,s=8,c='red')
    axplot.scatter(*centres.T,marker='x',c='cyan',label='Measured corners');axplot.scatter(*expected.T,marker='s',facecolors='none',edgecolors='yellow',label='Predicted corners')
    for k,(a,b) in enumerate(zip(centres,expected)):
        axplot.plot([a[0],b[0]],[a[1],b[1]],c='yellow');axplot.text(*a,str(k),color='white')
    axplot.legend();axplot.set_title(f'Candidate affine; gain x{v[6]:.3g} — inspect warnings')
    fig.savefig(out/'registration.png',dpi=150);plt.close(fig)
    labels=settings.get('mode_labels',[])
    for filename,meas,pred in [('registration_modes.png',d,pd),('registration_sums.png',np.asarray(sums),np.asarray(pred_sums))]:
        fig,axes=plt.subplots(len(modes),3,figsize=(11,2.6*len(modes)),squeeze=False,layout='constrained')
        for j,mode in enumerate(modes):
            limit=max(np.max(np.abs(meas[j])),np.max(np.abs(pred[j])),1e-9)
            for axplot,values,title in zip(axes[j],[meas[j],pred[j],meas[j]-pred[j]],['Held-out measured','Blind prediction','Residual']):
                image=np.full(pupil.shape,np.nan);image[pupil]=values
                im=axplot.imshow(image,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit);axplot.set_title(title);fig.colorbar(im,ax=axplot)
            axes[j,0].set_ylabel(labels[mode] if mode<len(labels) else f'Mode {mode}')
        fig.savefig(out/filename,dpi=120);plt.close(fig)
    return affine,float(v[6])


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('post_analysis'))
    parser.add_argument('--seed-fit',type=Path,help='Reuse a fit made from this exact acquisition')
    parser.add_argument('--strengths',type=float,nargs='+',default=[0,.01,.1,1,10])
    parser.add_argument('--penalties',choices=['gradient','curvature'],nargs='+',default=['gradient','curvature'])
    parser.add_argument('--phase-bound',type=float,default=np.pi/2,help='Absolute phase bound in radians')
    parser.add_argument('--max-nfev',type=int,default=300)
    parser.add_argument('--fit-phase',action='store_true',help='Also sweep phase regularisation after estimating registration')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if not np.isfinite(args.strengths).all() or min(args.strengths)<0 or not np.isfinite(args.phase_bound) or args.phase_bound<=0 or args.max_nfev<1:
        parser.error('Use finite nonnegative strengths, a positive phase bound and iteration limit')
    runs=[('unregularised',0)] if 0 in args.strengths else []
    runs += [(p,s) for p in dict.fromkeys(args.penalties) for s in dict.fromkeys(args.strengths) if s>0]
    out=args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        parser.error('Output directory is not empty; choose another or use --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    seed=args.seed_fit or out/'seed_fit.fits'
    if seed.resolve()==args.input.resolve():
        parser.error('Seed and acquisition must be different files')
    if args.seed_fit is None:
        options=dict(backend.ACQUISITION_DEFAULTS,**backend.FIT_DEFAULTS)
        options.update(input=args.input,output=seed,overwrite=args.overwrite,coupling=None,opd_per_cmd=None,gain=1.,phase_bound=args.phase_bound,max_nfev=args.max_nfev)
        print('Estimating registration, pupil and noise once; details in seed_fit.log.',flush=True)
        with (out/'seed_fit.log').open('w') as log,contextlib.redirect_stdout(log):
            backend.fit_file(argparse.Namespace(**options))
    with fits.open(seed) as h:
        report=backend.read_json(h,'FIT_REPORT')
        settings=backend.read_json(h,'SETTINGS')
        records=backend.read_json(h,'RECORDS')
        data={x.name:x.data.copy() for x in h if isinstance(x,fits.ImageHDU)}
        wavelength=float(h[0].header['WLREF'])
    # Check exposure identity as well as filename; users may move their data.
    with fits.open(args.input) as h:
        original=backend.read_json(h,'RECORDS')
        selected=[r for r in original if r['kind']!='registration']
        if selected!=records or not np.array_equal(h['CHANNEL_BEFORE'].data,data['CHANNEL_BEFORE']) or not np.array_equal(h['CLEAR_MEAN'].data,data['CLEAR_MEAN']):
            raise ValueError('Seed fit does not match the supplied acquisition')
    warnings=backend.correction_warnings(report)
    for warning in warnings:
        print('Seed diagnostic:',warning,flush=True)
    pupil=data['PUPIL'].astype(bool)
    yy,xx=np.indices(pupil.shape)
    pixels=np.column_stack([xx[pupil],yy[pupil]])
    affine,gain_multiplier=fit_registration(data,records,settings,report,pupil,out)
    registration_path=out/'registration_candidates.json'
    registration=json.loads(registration_path.read_text())
    registration.update(input=str(args.input.resolve()),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),seed=str(seed.resolve()))
    registration_path.write_text(json.dumps(registration,indent=2))
    if not args.fit_phase:
        print('Stopped after registration diagnostics. Inspect these before using --fit-phase.')
        return
    # Optional phase sweep uses the inferred training-cycle flux normalisation.
    flux_by_cycle=dict(zip(registration['training_cycles'],registration['candidates'][0]['cycle_flux']))
    for i,r in enumerate(records):
        if r['kind']=='modal':
            flux=flux_by_cycle.get(r['cycle'],1.)
            data['OBSERVED'][i]/=flux
            data['SIGMA'][i]/=flux
    active=np.ones((12,12),bool)
    active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
    ay,ax=np.indices((12,12))
    actuators=np.column_stack([ax[active]-5.5,ay[active]-5.5])
    uv=np.linalg.solve(affine[:,:2],(pixels-affine[:,2]).T).T
    influences=np.exp(-np.sum((uv[:,None]-actuators[None])**2,axis=2)/report['coupling']**2)
    commands=np.array([np.asarray(r['command'])-data['CHANNEL_BEFORE'] for r in records])[:,active]
    positions=actuators@affine[:,:2].T+affine[:,2]
    fig,axplot=plt.subplots(figsize=(6,5),layout='constrained')
    im=axplot.imshow(data['CLEAR_MEAN'],origin='lower')
    axplot.scatter(positions[:,0],positions[:,1],s=8,c='red')
    axplot.contour(pupil.astype(float),levels=[.5],colors='white',linewidths=.7)
    axplot.set_title('Fixed registration: actuators over clear pupil')
    fig.colorbar(im,ax=axplot,label='ADU')
    fig.savefig(out/'registration_phase_grid.png',dpi=150); plt.close(fig)
    probes=(gain_multiplier*2*np.pi*report['opd_per_cmd']/wavelength)*(commands@influences.T)
    theta=settings.get('mask_parameters',backend.MASKS[settings['mask']])['theta']
    m=abs(np.exp(1j*theta)-1)
    mu=report['mu_rad']
    a=data['A'][pupil]; b=data['B_AMPLITUDE'][pupil]
    base=a+m*m*b*b; response=2*m*np.sqrt(a)*b
    obs=data['OBSERVED'][:,pupil]; sigma=data['SIGMA'][:,pupil]
    if not np.isfinite(obs).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):
        raise ValueError('Invalid observations or uncertainties inside pupil')
    n=int(pupil.sum()); count=obs.size
    index=np.full(pupil.shape,-1); index[pupil]=np.arange(n)
    edges=[]
    for dy,dx in [(0,1),(1,0)]:
        for y,x in zip(*np.nonzero(pupil)):
            if y+dy<pupil.shape[0] and x+dx<pupil.shape[1] and pupil[y+dy,x+dx]:
                edges.append((index[y,x],index[y+dy,x+dx]))
    edges=np.asarray(edges)
    gradient=sparse.csr_matrix((np.tile([1.,-1.],len(edges)),(np.repeat(np.arange(len(edges)),2),edges.ravel())),shape=(len(edges),n))
    operators={'gradient':gradient,'curvature':gradient.T@gradient,'unregularised':gradient}
    # Fit low-order components jointly, so missing pupil pixels do not mix them.
    cx,cy,radius=report['pupil_geometry']
    x=(pixels[:,0]-cx)/radius; y=(pixels[:,1]-cy)/radius
    design=np.column_stack([np.ones(n),x,y,2*(x*x+y*y)-1,x*x-y*y,2*x*y])
    # Initialise phase for the NEW registration, rather than reuse the old map.
    c=response[None]*np.cos(mu-probes)/sigma
    q=response[None]*np.sin(mu-probes)/sigma
    target=(obs-base)/sigma
    cc=np.sum(c*c,axis=0); qq=np.sum(q*q,axis=0); cq=np.sum(c*q,axis=0)
    yc=np.sum(target*c,axis=0); yq=np.sum(target*q,axis=0)
    initial=np.clip(np.arctan2(yq*cc-yc*cq,yc*qq-yq*cq),-args.phase_bound*.999,args.phase_bound*.999)
    manifest=dict(input=str(args.input.resolve()),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),seed=str(seed.resolve()),
                  backend_sha256=hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest(),phase_bound=args.phase_bound,
                  max_nfev=args.max_nfev,registration='fixed from joint exposure registration; see registration_candidates.json',gain_multiplier=gain_multiplier,cycle_flux=flux_by_cycle,warnings=warnings,objective=__doc__,runs=[])
    rows=[]; maps=[]
    for penalty,strength in runs:
        name=f'{penalty}_{strength:g}'
        print('Fitting',name,flush=True)
        start=time.monotonic(); op=operators[penalty]
        weight=np.sqrt(strength*count/max(op.shape[0],1))
        def residual(phi):
            model=base+response*np.cos(mu-phi[None]-probes)
            return np.r_[((model-obs)/sigma).ravel(),weight*(op@phi)]
        def jacobian(phi):
            derivative=response*np.sin(mu-phi[None]-probes)/sigma
            j=sparse.csr_matrix((derivative.ravel(),(np.arange(count),np.tile(np.arange(n),len(obs)))),shape=(count,n))
            return sparse.vstack([j,weight*op],format='csr')
        def loss(z):
            # Robust data likelihood, quadratic (not robustified) smoothness prior.
            value=np.vstack([z.copy(),np.ones_like(z),np.zeros_like(z)])
            t=1+z[:count]
            value[0,:count]=2*(np.sqrt(t)-1)
            value[1,:count]=1/np.sqrt(t)
            value[2,:count]=-.5/t**1.5
            return value
        result=least_squares(residual,initial,jac=jacobian,bounds=(-args.phase_bound,args.phase_bound),loss=loss,f_scale=3.,x_scale='jac',max_nfev=args.max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        phi=result.x; model=base+response*np.cos(mu-phi[None]-probes)
        opd=(phi-phi.mean())*wavelength/(2*np.pi)*1e9
        coefficients=np.linalg.lstsq(design,opd,rcond=None)[0]
        defocus=coefficients[3]*(design[:,3]-design[:,3].mean())
        row=dict(name=name,penalty=penalty,strength=strength,gain_multiplier=gain_multiplier,success=bool(result.success),nfev=result.nfev,
                 seconds=time.monotonic()-start,weighted_residual_rms=float(np.sqrt(np.mean(((model-obs)/sigma)**2))),
                 opd_rms_nm=float(np.std(opd)),defocus_rms_nm=float(np.std(defocus)),defocus_coefficient_nm=float(coefficients[3]),
                 phase_bound_pixels=int(np.sum(np.abs(phi)>=.99*args.phase_bound)),gradient_rms_rad=float(np.sqrt(np.mean((gradient@phi)**2))))
        defocus_map=np.full(pupil.shape,np.nan); defocus_map[pupil]=defocus
        opd_map=np.full(pupil.shape,np.nan); opd_map[pupil]=opd
        rms_map=np.full(pupil.shape,np.nan); rms_map[pupil]=np.sqrt(np.mean(((model-obs)/sigma)**2,axis=0))
        header=fits.Header(dict(MODEL='ANALYTIC',REGFIX=True,REGTYPE=penalty,REGSTR=strength,GAINMULT=gain_multiplier,SUCCESS=bool(result.success),WLREF=wavelength))
        hdus=[fits.PrimaryHDU(header=header),backend.json_hdu('REPORT',{**row,'message':str(result.message),'low_order_coefficients_nm':coefficients.tolist()}),backend.json_hdu('SETTINGS',settings),backend.json_hdu('RECORDS',records)]
        for ext,values,unit in [('OPD',opd_map*1e-9,'m'),('DEFOCUS_OPD',defocus_map*1e-9,'m'),('PHASE_RAD',np.where(pupil,0.,np.nan),'rad'),('MODEL',np.full(data['MODEL'].shape,np.nan),'normalised'),('PUPIL',pupil.astype('uint8'),None),('AFFINE_REFINED',affine,None),('OBSERVED',data['OBSERVED'],'normalised'),('SIGMA',data['SIGMA'],'normalised')]:
            if ext=='PHASE_RAD': values[pupil]=phi
            if ext=='MODEL': values[:,pupil]=model
            hdu=fits.ImageHDU(values,name=ext)
            if unit: hdu.header['BUNIT']=unit
            hdus.append(hdu)
        hdus.append(fits.ImageHDU(data['OBSERVED']-next(h.data for h in hdus if h.name=='MODEL'),name='RESIDUAL'))
        fits.HDUList(hdus).writeto(out/f'{name}.fits',overwrite=args.overwrite,checksum=True)
        rows.append(row); maps.append((opd_map,defocus_map,rms_map))
        manifest['runs'].append(row)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        with (out/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    residual_limit=max(float(np.nanmax(pair[2])) for pair in maps) or 1
    limit=max(float(np.nanmax(np.abs(v))) for pair in maps for v in pair[:2]) or 1
    fig,axes=plt.subplots(len(rows),3,figsize=(12,3*len(rows)),squeeze=False,layout='constrained')
    for axes_row,row,(opd_map,defocus_map,rms_map) in zip(axes,rows,maps):
        for ax,values,title in zip(axes_row,[opd_map,defocus_map,rms_map],['Recovered OPD [nm]','Defocus component [nm]','Weighted residual RMS']):
            im=ax.imshow(values,origin='lower',cmap='viridis' if title.startswith('Weighted') else 'RdBu_r',**(dict(vmin=0,vmax=residual_limit) if title.startswith('Weighted') else dict(vmin=-limit,vmax=limit)))
            ax.set_title(title); fig.colorbar(im,ax=ax)
        axes_row[0].set_ylabel(row['name']+(' [NOT CONVERGED]' if not row['success'] else ''))
    fig.savefig(out/'comparison.png',dpi=140); plt.close(fig)
    print(f'Saved {len(rows)} refits, summary.csv and comparison.png in {out}. No DM commands generated or applied.')


if __name__=='__main__':
    main()
