#!/usr/bin/env python3
"""Experimental offline Fresnel refit of recorded probes (no DM/SHM access).

Uses BaldrApp's actual nonlinear field propagation, including the probe-dependent
reference field, configured relay, cold stop and wavelength-dependent phase mask.
Requires the existing seed FITS for pupil/noise estimates and optionally a
registration_candidates.json. Registration is FIXED, never assumed validated.
Default static basis: recorded Mixed defocus and two astigmatisms. Fits total
static aberration relative to acquisition baseline, not the injected term alone.

Example (run inside the BaldrApp environment):
  python post_analysis_fresnel.py probes.fits --seed-fit seed_fit.fits \
      --registration registration_candidates.json --output-dir fresnel_fit

All cycles except the last train the fit; the last is held out. For speed repeated
identical commands share one deterministic prediction, but each exposure retains
its own weight. The model has NO atmospheric marginalisation: repeat-cycle errors
account for scatter, not the nonlinear mean bias of atmospheric phase. Config is
an approximation, saved in effective_config.json. No correction command is saved.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

# Support running this file directly from the simulator folder.
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from baldrapp.common import baldr_core as core
import probe_dm_phase_v2 as backend


def make_forward(config_path, settings, report, pupil, affine, commands, basis, n_wavelengths):
    """Calibrate detector orientation optically, then map DM to measured pixels."""
    z=core.init_zwfs_from_json(config_path)
    z.dm.current_cmd=np.zeros(140)
    zero=np.zeros_like(z.grid.pupil_mask)
    amp=z.grid.pupil_mask.astype(float)
    ny,nx=amp.shape; yy,xx=np.indices(amp.shape)
    cy0,cx0=(ny-1)/2,(nx-1)/2
    def propagate(opd, amplitude, wavelength, theta=None):
        return core.get_frame_fresnel(zero,amplitude,opd,z,detector=None,
            include_shotnoise=False,wavelength=wavelength,
            theta=(z.optics.theta*z.optics.wvl0/wavelength if theta is None else theta),
            phasemask_diameter=z.optics.mask_diam*z.optics.wvl0/wavelength,return_intermediates=True)
    # Three translated amplitude patches measure the relay's inversion and scale.
    positions=np.array([[cx0,cy0],[cx0+z.grid.N/4,cy0],[cx0,cy0+z.grid.N/4]])
    centroids=[]
    for x,y in positions:
        patch=np.exp(-((xx-x)**2+(yy-y)**2)/(2*(z.grid.N/14)**2))
        result=propagate(zero,patch,z.optics.wvl0,theta=0.)
        image=result['intensity'];total=image.sum()
        centroids.append([np.sum(xx*image)/total,np.sum(yy*image)/total])
    relay=np.linalg.lstsq(np.c_[positions,np.ones(3)],centroids,rcond=None)[0].T
    if np.linalg.cond(relay[:,:2])>10: raise ValueError('Degenerate relay sampling calibration')
    image_centre=relay[:,:2]@np.array([cx0,cy0])+relay[:,2]
    cx,cy,radius=report['pupil_geometry']
    pixels_per_camera=z.fresnel_relay.D_detector_predicted/result['dx_detector']/(2*radius)
    # Camera -> canonical detector -> input grid -> BaldrApp wave coordinates.
    input_linear=np.linalg.solve(relay[:,:2],pixels_per_camera*affine[:,:2])
    input_offset=np.linalg.solve(relay[:,:2],image_centre+pixels_per_camera*(affine[:,2]-[cx,cy])-relay[:,2])
    wave=z.grid.wave_coord
    spacing=np.array([wave.x[1]-wave.x[0],wave.y[1]-wave.y[0]])
    transform=np.c_[spacing[:,None]*input_linear,spacing*input_offset+[wave.x[0],wave.y[0]]]
    # Carry Gaussian influences through the full affine in DM coordinates.
    # The legacy registration helper derives width from x spacing alone, which
    # collapses near a 90-degree rotation; avoid that assumption in this fitter.
    uv=np.linalg.solve(transform[:,:2],np.stack([wave.X-transform[0,2],wave.Y-transform[1,2]]).reshape(2,-1)).T
    dmcoords=np.array([z.dm2wavespace_registration.dm_actuator_to_coord[i] for i in range(140)])
    influences=np.exp(-np.sum((uv[:,None,:]-dmcoords[None,:,:])**2,axis=2)/z.dm.actuator_coupling_factor**2)
    command_opds=(commands@influences.T*z.dm.opd_per_cmd).reshape(-1,ny,nx)
    basis_opds=(basis@influences.T*z.dm.opd_per_cmd).reshape(-1,ny,nx)
    basis_opds*=amp>0  # Saved static OPD is zero outside the illuminated pupil.
    waves=np.asarray(z.spectrum.wavelengths);weights=np.asarray(z.spectrum.weights_normalized)
    if n_wavelengths and n_wavelengths<len(waves):
        # Group adjacent spectral quadrature nodes, preserving group weight.
        groups=np.array_split(np.arange(len(waves)),n_wavelengths)
        reduced=np.array([np.sum(waves[g]*weights[g])/np.sum(weights[g]) for g in groups])
        weights=np.array([weights[g].sum() for g in groups]);waves=reduced
    weights=weights/weights.sum()
    rows,cols=np.indices(pupil.shape)
    sample_points=[]
    # 2x2 subpixel integration; output scaling is calibrated against clear flux.
    for dy in (-.25,.25):
        for dx in (-.25,.25):
            sample_points.append([image_centre[1]+pixels_per_camera*(rows+dy-cy),image_centre[0]+pixels_per_camera*(cols+dx-cx)])
    def sample(image):
        return sum(map_coordinates(image,points,order=1,mode='constant',cval=0.) for points in sample_points)/4
    clear=sum(w*sample(propagate(zero,amp,wl,theta=0.)['intensity']) for wl,w in zip(waves,weights))
    normalization=float(np.median(clear[pupil]))
    if normalization<=0: raise ValueError('No simulated clear-pupil flux')
    calls=0
    def forward(parameters):
        nonlocal calls
        static=np.einsum('i,ijk->jk',parameters[:len(basis)],basis_opds)
        gain,flux=parameters[-2:]
        images=[]
        for command in command_opds:
            opd=gain*(static+command)
            image=sum(w*sample(propagate(opd,amp,wl)['intensity']) for wl,w in zip(waves,weights))
            images.append(flux*image/normalization)
        calls+=1
        if calls%20==0: print(f'Forward evaluations: {calls}',flush=True)
        return np.asarray(images)
    metadata=dict(mask_chromatic_model='theta and angular diameter scale as reference wavelength / wavelength; material dispersion omitted',influences='Gaussian in DM coordinates, full affine mapping',wavelengths_m=waves.tolist(),weights=weights.tolist(),dm_to_wave=transform.tolist(),input_to_detector=relay.tolist(),
        image_centre=image_centre.tolist(),detector_pixels_per_camera_pixel=pixels_per_camera,
        detector_subsamples=4,normalization=normalization,static_basis_units='recorded DM mode coefficients',
        note='Relay mapping calibrated with three mask-out amplitude patches. Check clear pupil alignment before interpreting fits.')
    return forward,basis_opds,clear/normalization,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input',type=Path)
    parser.add_argument('--seed-fit',required=True,type=Path)
    parser.add_argument('--registration',type=Path)
    parser.add_argument('--candidate',type=int,default=0)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'fake_configs/baldr_config.json')
    parser.add_argument('--output-dir',type=Path,default=Path('diagnostics/fresnel_fit'))
    parser.add_argument('--modes',type=int,nargs='+',default=[0,1,2],help='Recorded MODES indices used to represent static aberration')
    parser.add_argument('--wavelengths',type=int,default=3,help='Spectral quadrature groups; 0 uses all configured nodes')
    parser.add_argument('--regularization',type=float,default=.01,help='Mean data loss + alpha*sum((coefficient/.1)**2)')
    parser.add_argument('--max-nfev',type=int,default=20)
    parser.add_argument('--evaluate-only',action='store_true',help='Predict with zero static aberration, gain=flux=1')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if args.candidate<0 or args.wavelengths<0 or args.max_nfev<1 or not np.isfinite(args.regularization) or args.regularization<0:
        parser.error('Invalid candidate, wavelengths, iteration limit or regularization')
    out=args.output_dir
    protected={v.resolve() for v in (args.input,args.seed_fit,args.config,args.registration) if v is not None}
    if any((out/name).resolve() in protected for name in ('fit.fits','effective_config.json','report.json','clear_alignment.png','differences.png','sums.png','static_opd.png')):
        parser.error('Output paths must not overwrite input files')
    if out.exists() and any(out.iterdir()) and not args.overwrite: parser.error('Output directory not empty; choose another or use --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    with fits.open(args.seed_fit) as h:
        settings=backend.read_json(h,'SETTINGS');records=backend.read_json(h,'RECORDS');report=backend.read_json(h,'FIT_REPORT')
        data={x.name:x.data.copy() for x in h if isinstance(x,fits.ImageHDU)}
    with fits.open(args.input) as h:
        original=backend.read_json(h,'RECORDS')
        if [r for r in original if r['kind']!='registration']!=records or not np.array_equal(h['CLEAR_MEAN'].data,data['CLEAR_MEAN']) or not np.array_equal(h['CHANNEL_BEFORE'].data,data['CHANNEL_BEFORE']):
            raise ValueError('Seed and acquisition do not match')
        modes=h['MODES'].data.copy()
    if not args.modes or min(args.modes)<0 or max(args.modes)>=len(modes) or len(set(args.modes))!=len(args.modes): parser.error('Invalid static mode indices')
    affine=data['AFFINE_REFINED'];registration_warnings=['Using seed affine; registration has not been independently validated.']
    if args.registration:
        registration=json.loads(args.registration.read_text());affine=np.asarray(registration['candidates'][args.candidate]['affine'])
        registration_warnings=registration.get('warnings',[])
        if registration.get('ambiguous'): registration_warnings=registration_warnings+['Supplied registration is flagged ambiguous.']
    if affine.shape!=(2,3) or not np.isfinite(affine).all() or np.linalg.cond(affine[:,:2])>10: raise ValueError('Invalid affine')
    config=json.loads(args.config.read_text())
    # Explicitly record changes to the approximate simulator configuration.
    mask=settings.get('mask_parameters',backend.MASKS[settings['mask']])
    config['optics'].update(wvl0=mask['wavelength'],theta=mask['theta'],theta_mode='constant',mask_diam=mask['diameter_lamD'],mask_diam_mode='lambda_over_D')
    config['dm']['opd_per_cmd']=settings['opd_per_cmd']
    # Preserve simulator coupling convention; it need not equal the old analytic
    # fit's Gaussian-width convention. Save both for inspection.
    (out/'effective_config.json').write_text(json.dumps(config,indent=2))
    active=np.ones((12,12),bool);active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
    pupil=data['PUPIL'].astype(bool)
    ids=np.array([i for i,r in enumerate(records) if r['kind']=='modal'])
    cycles=sorted({records[i]['cycle'] for i in ids})
    if len(cycles)<2: raise ValueError('Need at least two modal cycles for held-out assessment')
    train=ids[[records[i]['cycle']!=cycles[-1] for i in ids]];test=ids[[records[i]['cycle']==cycles[-1] for i in ids]]
    commands=np.array([np.asarray(records[i]['command'])-data['CHANNEL_BEFORE'] for i in ids])[:,active]
    unique,inverse=np.unique(commands,axis=0,return_inverse=True)
    lookup={i:j for j,i in enumerate(ids)}
    train_map=np.array([inverse[lookup[i]] for i in train]);test_map=np.array([inverse[lookup[i]] for i in test])
    forward,basis_opds,clear,metadata=make_forward(out/'effective_config.json',settings,report,pupil,affine,unique,modes[args.modes][:,active],args.wavelengths)
    obs=data['OBSERVED'][:,pupil];sigma=data['SIGMA'][:,pupil]
    if not np.isfinite(obs[ids]).all() or not np.isfinite(sigma[ids]).all() or np.any(sigma[ids]<=0): raise ValueError('Invalid data or sigma')
    count=obs[train].size; n=len(args.modes)
    initial=np.r_[np.zeros(n),1.,1.]
    low=np.r_[np.full(n,-.2),.2,.5];high=np.r_[np.full(n,.2),2.5,2.]
    def residual(parameters):
        prediction=forward(parameters)[:,pupil]
        return np.r_[((prediction[train_map]-obs[train])/sigma[train]).ravel(),np.sqrt(args.regularization*count)*parameters[:n]/.1]
    def loss(z):
        # Keep coefficient regularisation quadratic, while data use soft-L1.
        result=np.vstack([z.copy(),np.ones_like(z),np.zeros_like(z)])
        t=1+z[:count];result[0,:count]=2*(np.sqrt(t)-1);result[1,:count]=1/np.sqrt(t);result[2,:count]=-.5/t**1.5
        return result
    start=time.monotonic();print(f'{len(unique)} distinct commands, {len(train)} training exposures, {len(test)} held out; {n} static modes.',flush=True)
    initial_prediction=forward(initial)
    if args.evaluate_only:
        parameters=initial;success=False;message='Evaluation only; no optimisation';nfev=1
    else:
        result=least_squares(residual,initial,bounds=(low,high),loss=loss,f_scale=3,x_scale=np.r_[np.full(n,.03),.3,.2],max_nfev=args.max_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        parameters=result.x;success=bool(result.success);message=str(result.message);nfev=result.nfev
    prediction=forward(parameters)
    rms=lambda a:float(np.sqrt(np.mean(a*a)))
    empirical=np.array([np.mean(obs[[i for i in train if records[i]['mode']==records[j]['mode'] and records[i]['sign']==records[j]['sign']]],axis=0) for j in test])
    measured_diff=[];predicted_diff=[];measured_sum=[];predicted_sum=[]
    mode_ids=sorted({records[i]['mode'] for i in test})
    for mode in mode_ids:
        positive=next(i for i in test if records[i]['mode']==mode and records[i]['sign']==1);negative=next(i for i in test if records[i]['mode']==mode and records[i]['sign']==-1)
        ip=prediction[inverse[lookup[positive]]];im=prediction[inverse[lookup[negative]]]
        measured_diff.append(data['OBSERVED'][positive]-data['OBSERVED'][negative]);predicted_diff.append(ip-im)
        measured_sum.append(data['OBSERVED'][positive]+data['OBSERVED'][negative]);predicted_sum.append(ip+im)
    d=np.array(measured_diff)[:,pupil];pd=np.array(predicted_diff)[:,pupil]
    explained=float(1-np.sum((d-pd)**2)/max(np.sum(d*d),1e-20))
    warnings=list(registration_warnings)
    if explained<.5:warnings.append('Less than half the held-out difference energy explained.')
    if not success:warnings.append(message)
    if np.any(np.minimum(parameters-low,high-parameters)<.005*(high-low)):warnings.append('A fitted parameter reached a bound.')
    summary=dict(success=success,message=message,nfev=nfev,seconds=time.monotonic()-start,static_modes=args.modes,static_coefficients_dm=parameters[:n].tolist(),gain_multiplier=float(parameters[-2]),flux=float(parameters[-1]),
        heldout_cycle=cycles[-1],heldout_weighted_rms=rms((prediction[test_map][:,pupil]-obs[test])/sigma[test]),
        initial_heldout_weighted_rms=rms((initial_prediction[test_map][:,pupil]-obs[test])/sigma[test]),repeat_baseline_weighted_rms=rms((empirical-obs[test])/sigma[test]),
        heldout_difference_energy_explained=explained,warnings=warnings,registration_affine=affine.tolist(),sampling=metadata,regularization=args.regularization,
        assumptions=['Approximate supplied optical config; spectrum/pupil/relay are not inferred from on-sky data.','No atmosphere marginalisation, only empirical repeat-cycle uncertainty.','Seed noise and pupil use all cycles; validation is not fully independent.','Static aberration limited to selected recorded modes. No DM correction produced.'],
        input=str(args.input.resolve()),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),config_source=str(args.config.resolve()),seed_fit=str(args.seed_fit.resolve()),registration_file=str(args.registration.resolve()) if args.registration else None,candidate=args.candidate,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(summary,indent=2))
    static_opd=parameters[-2]*np.einsum('i,ijk->jk',parameters[:n],basis_opds)
    fits.HDUList([fits.PrimaryHDU(),backend.json_hdu('REPORT',summary),backend.json_hdu('RECORDS',[records[i] for i in ids]),fits.ImageHDU(prediction[inverse],name='MODEL'),fits.ImageHDU(data['OBSERVED'][ids],name='OBSERVED'),fits.ImageHDU(data['OBSERVED'][ids]-prediction[inverse],name='RESIDUAL'),fits.ImageHDU(static_opd,header=fits.Header({'BUNIT':'m','COMMENT':'Padded input-pupil OPD; zero outside illuminated pupil.'}),name='STATIC_OPD'),fits.ImageHDU(pupil.astype('uint8'),name='PUPIL'),fits.ImageHDU(clear,name='SIM_CLEAR')]).writeto(out/'fit.fits',overwrite=args.overwrite,checksum=True)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    support=np.any(basis_opds!=0,axis=0)
    im=axes[0].imshow(np.where(support,static_opd*1e9,np.nan),origin='lower',cmap='RdBu_r')
    axes[0].set_title('Fitted static OPD (input pupil)');fig.colorbar(im,ax=axes[0],label='nm')
    axes[1].bar(np.arange(n),parameters[:n]);axes[1].set_xticks(np.arange(n),[str(i) for i in args.modes])
    axes[1].set_xlabel('Recorded mode index');axes[1].set_ylabel('Static coefficient [DM units]')
    axes[1].set_title('Model estimate, not a validated correction')
    fig.savefig(out/'static_opd.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,image,title in zip(axes,[data['A'],clear],['Measured clear (seed normalisation)','Simulated clear']):
        im=ax.imshow(image,origin='lower');ax.contour(pupil,levels=[.5],colors='red');ax.set_title(title);fig.colorbar(im,ax=ax)
    fig.savefig(out/'clear_alignment.png',dpi=140);plt.close(fig)
    for filename,measured,predicted in [('differences.png',measured_diff,predicted_diff),('sums.png',measured_sum,predicted_sum)]:
        fig,axes=plt.subplots(len(mode_ids),3,figsize=(11,2.5*len(mode_ids)),squeeze=False,layout='constrained')
        for row,mode,meas,pred in zip(axes,mode_ids,measured,predicted):
            limit=max(np.max(np.abs(meas[pupil])),np.max(np.abs(pred[pupil])),1e-9)
            for ax,image,title in zip(row,[meas,pred,meas-pred],['Held-out measured','Blind prediction','Residual']):
                im=ax.imshow(np.where(pupil,image,np.nan),origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit);ax.set_title(title);fig.colorbar(im,ax=ax)
            row[0].set_ylabel(settings.get('mode_labels',[str(i) for i in range(len(modes))])[mode])
        fig.savefig(out/filename,dpi=120);plt.close(fig)
    print(json.dumps(summary,indent=2))
    print('Saved offline Fresnel results:',out)


if __name__=='__main__':
    main()
