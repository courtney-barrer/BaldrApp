#!/usr/bin/env python3
"""Plot analytic probe-fit results (also reads previous broadband Fresnel fits).

python analyse_probe_phase.py phase_fit.fits
python analyse_probe_phase.py phase_fit.fits --save-prefix figures/beam1 --no-show

Read-only: numpy, astropy and matplotlib only; no hardware or BaldrApp imports.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--save-prefix',type=Path)
    parser.add_argument('--no-show',action='store_true')
    args=parser.parse_args()
    with fits.open(args.input) as h:
        report=json.loads(h['FIT_REPORT'].data['JSON'][0])
        data={x.name:np.array(x.data) for x in h if isinstance(x,fits.ImageHDU)}
        wavelength=h[0].header['WLREF']
        records=json.loads(h['RECORDS'].data['JSON'][0]) if 'RECORDS' in h else None
    print(json.dumps(report,indent=2))
    pupil=data['PUPIL']>0
    opd=data['OPD'] if 'OPD' in data else data['TOTAL_OPD']
    opd=np.where(pupil,opd*1e9,np.nan)
    phase=np.where(pupil,data['PHASE_RAD'],np.nan)
    observed=data['OBSERVED'];model=data['MODEL'];residual=observed-model
    sigma=data['SIGMA'];standardised=residual/sigma
    camera_pupil=pupil if pupil.shape==observed.shape[1:] else observed[0]>.15*np.max(observed[0])
    rms=np.sqrt(np.nanmean(residual[:,camera_pupil]**2,axis=1))
    print(f'Wavelength: {wavelength*1e6:.3f} um; piston-removed OPD RMS: {np.nanstd(opd):.3f} nm')
    print('Per-exposure residual RMS:',rms)
    figures={}
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    figures['phase']=fig
    third=data['PHASE_SIGMA'] if 'PHASE_SIGMA' in data else np.sqrt(np.nanmean(residual**2,axis=0))
    for ax,array,title in zip(axes,[opd,phase,np.where(pupil if third.shape==pupil.shape else camera_pupil,third,np.nan)],
                            ['Piston-removed OPD [nm]','Fitted phase [rad]',
                             'Conditional phase uncertainty [rad]' if 'PHASE_SIGMA' in data else 'Residual RMS']):
        im=ax.imshow(array,origin='lower',cmap='RdBu_r' if ax!=axes[2] else 'viridis')
        ax.set_title(title);fig.colorbar(im,ax=ax)
    if 'PHASE_BOUND' in data and np.any(data['PHASE_BOUND']):
        y,x=np.where(data['PHASE_BOUND']);axes[1].scatter(x,y,c='black',s=12,marker='x',label='Phase bound');axes[1].legend()

    if 'CORNER_IMAGES' in data:
        fig,axes=plt.subplots(2,2,figsize=(9,8),layout='constrained');figures['registration']=fig
        design=np.column_stack([data['CORNER_DM'],np.ones(4)])
        initial=design@data['AFFINE_INITIAL'].T;refined=design@data['AFFINE_REFINED'].T
        for k,ax in enumerate(axes.flat):
            im=ax.imshow(data['CORNER_IMAGES'][k],origin='lower',cmap='RdBu_r')
            ax.plot(*data['CORNER_CENTRES'][k],'rx',label='Gaussian centre')
            ax.plot(*initial[k],'wo',mfc='none',label='Initial affine')
            ax.plot(*refined[k],'g+',label='Refined affine')
            ax.plot(*data['DIAGONAL_CENTRE'],'y+',label='Diagonal intersection')
            ax.set_title(f'Registration poke {k}');fig.colorbar(im,ax=ax)
        axes.flat[0].legend(fontsize=8)
        print('Initial affine:\n',data['AFFINE_INITIAL'],'\nRefined affine:\n',data['AFFINE_REFINED'])
        print('Measured minus refined corner positions [pixels]:\n',data['CORNER_CENTRES']-refined)

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');figures['diagnostics']=fig
    axes[0].plot(rms,'o-');axes[0].set(xlabel='Exposure index',ylabel='Residual RMS (saved intensity units)')
    values=standardised[:,camera_pupil].ravel();values=values[np.isfinite(values)]
    axes[1].hist(values,bins=50);axes[1].set(xlabel='Residual / measurement uncertainty',ylabel='Count')

    fig,axes=plt.subplots(1,4,figsize=(15,4));fig.subplots_adjust(bottom=.23,wspace=.5);figures['exposure']=fig
    intensity_limit=np.nanpercentile(observed[:,camera_pupil],[1,99])
    residual_limit=max(float(np.nanpercentile(np.abs(residual[:,camera_pupil]),99)),1e-12)
    images=[]
    for ax,array,title,limits in zip(axes,[observed,model,residual,standardised],
                                    ['Observed','Model','Residual','Residual / sigma'],
                                    [intensity_limit,intensity_limit,[-residual_limit,residual_limit],[-5,5]]):
        images.append(ax.imshow(array[0],origin='lower',vmin=limits[0],vmax=limits[1],cmap='RdBu_r'))
        ax.set_title(title);fig.colorbar(images[-1],ax=ax,fraction=.046)
    exposure_fig=fig
    title=fig.suptitle('Exposure 0')
    def update(value):
        k=int(value)
        for im,array in zip(images,[observed,model,residual,standardised]):im.set_data(array[k])
        label=f'Exposure {k}'
        if records:
            r=records[k];label+=f" — {r['kind']}, mode {r.get('mode_index')}, sign {r['sign']:+d}, cycle {r['cycle']+1}"
        title.set_text(label);exposure_fig.canvas.draw_idle()
    if len(observed)>1:
        slider=Slider(fig.add_axes([.2,.07,.6,.035]),'Exposure',0,len(observed)-1,valinit=0,valstep=1)
        slider.on_changed(update)
    update(0)
    if records:
        pairs={}
        for i,r in enumerate(records):
            if r['kind']=='modal':pairs.setdefault((r['cycle'],r['mode_index']),{})[r['sign']]=i
        pairs=[(key,pair) for key,pair in pairs.items() if 1 in pair and -1 in pair]
        if pairs:
            differences=np.array([observed[v[1]]-observed[v[-1]] for _,v in pairs])
            fitted_differences=np.array([model[v[1]]-model[v[-1]] for _,v in pairs])
            diff_residual=differences-fitted_differences
            difference_fig,diff_axes=plt.subplots(1,3,figsize=(12,4))
            difference_fig.subplots_adjust(bottom=.23,wspace=.5);figures['differences']=difference_fig
            limit=max(float(np.nanpercentile(np.abs(differences[:,camera_pupil]),99)),1e-12)
            error_limit=max(float(np.nanpercentile(np.abs(diff_residual[:,camera_pupil]),99)),1e-12)
            diff_images=[]
            for ax,arr,label,lim in zip(diff_axes,[differences,fitted_differences,diff_residual],
                                       ['Measured I+ - I-','Fitted I+ - I-','Difference residual'],[limit,limit,error_limit]):
                im=ax.imshow(arr[0],origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim)
                diff_images.append(im);ax.set_title(label);difference_fig.colorbar(im,ax=ax,fraction=.046)
            diff_title=difference_fig.suptitle('')
            def update_difference(value):
                k=int(value)
                for im,arr in zip(diff_images,[differences,fitted_differences,diff_residual]):im.set_data(arr[k])
                cycle,mode=pairs[k][0];diff_title.set_text(f'Mode {mode}, cycle {cycle+1}')
                difference_fig.canvas.draw_idle()
            if len(pairs)>1:
                difference_slider=Slider(difference_fig.add_axes([.2,.07,.6,.035]),'Pair',0,len(pairs)-1,valinit=0,valstep=1)
                difference_slider.on_changed(update_difference)
            update_difference(0)
    if 'B_AMPLITUDE' in data:
        b=data['B_AMPLITUDE'];b=b[0] if b.ndim==3 else b
        fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained');figures['reference']=fig
        for ax,array,label in zip(axes,[data.get('A',data['PUPIL']),b],['Clear pupil intensity A','Reference amplitude B']):
            im=ax.imshow(array,origin='lower');ax.set_title(label);fig.colorbar(im,ax=ax)
    if args.save_prefix:
        args.save_prefix.parent.mkdir(parents=True,exist_ok=True)
        for name,fig in figures.items():
            path=Path(f'{args.save_prefix}_{name}.png');fig.savefig(path,dpi=150);print(f'Saved {path}')
    if not args.no_show:plt.show()
    else:plt.close('all')


if __name__=='__main__':main()
