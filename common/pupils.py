'''
Aperture/pupil utility functions largely copied from pyZELDA 
'''

import numpy as np
import scipy.ndimage as ndimage


def coordinates(dim, size, diameter=False, strict=False, center=(), cpix=False, normalized=True, outside=np.nan, polar=True):
    '''
    Returns rho,theta coordinates defining a circular aperture
    
    Parameters
    ----------
    dim : int
        Size of the output array
    
    size : int
        Size of the disk, representing either the radius (default) or the diameter
    
    diameter : bool, optional
        Specify if input size is the diameter. Default is 'False'
    
    strict : bool optional
        If set to Trye, size must be strictly less than (<), instead of less
        or equal (<=). Default is 'False'
    
    center : sequence, optional
        Specify the center of the disc. Default is '()', i.e. the center of the array
    
    cpix : bool optional
        If set to True, the disc is centered on pixel at position (dim//2, dim//2).
        Default is 'False', i.e. the disc is centered between 4 pixels
    
    normalized : bool optional
        Determines if the rho coordinates are normalized to 1. Default is True
    
    outside : float, optional
        Value used to fill the array outside of the aperture. Default is np.nan
    
    polar : bool, optional
        Return polar coordinates. Default is True
    
    Returns
    -------
    rho, theta : arrays
        Arrays containing the rho and theta coordinates of the aperture
    '''
    
    if (diameter is True):
        rad = size/2
    else:
        rad = size

    if (len(center) == 0):
        if (cpix is False):
            cx = (dim-1) / 2
            cy = (dim-1) / 2
        else:
            cx = dim // 2
            cy = dim // 2
    elif (len(center) == 2):
        cx = center[0]
        cy = center[1]
    else:
        raise ValueError('Error, you must pass 2 values for center')
        return None

    x = np.arange(dim, dtype=np.float64) - cx
    y = np.arange(dim, dtype=np.float64) - cy
    xx, yy = np.meshgrid(x, y)
    
    rho = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    
    if (strict is True):
        msk = (rho < rad)
    else:
        msk = (rho <= rad)

    xx[np.logical_not(msk)] = outside
    yy[np.logical_not(msk)] = outside
    rho[np.logical_not(msk)] = outside
    theta[np.logical_not(msk)] = outside

    if (normalized is True):
        xx  = xx / rad
        yy  = yy / rad
        rho = rho / rad

    if (polar is True):
        return rho, theta
    else:
        return xx, yy
    

def disc_obstructed(dim, size, obs, **kwargs):
    '''
    Create a numerical array containing a disc with central obstruction
    
    Parameters
    ----------
    dim : int
        Size of the output array
    
    size : int
        Size of the disk, representing either the radius (default) or the diameter
    
    obs : float
        Fractional size occupied by the central obstruction
    
    All other parameters are the same as for the disc() function and are described there.
    
    Returns
    -------
    disc : array
        An array containing a disc with central obstruction
    '''

    if (obs < 0) or (obs > 1):
        raise ValueError('obs value must be within [0,1]')
        return None
    
    ap_out = disc(dim, size, **kwargs)
    
    if obs == 0:
        return ap_out
    
    ap_in = disc(dim, size*obs, **kwargs)
    
    return ap_out - ap_in


def annulus(dim, inner_size, outer_size, **kwargs):
    '''
    Create a numerical array containing a disc with central obstruction
    
    Parameters
    ----------
    dim : int
        Size of the output array
    
    inner_size : int
        Inner size of the annulus, representing either the radius (default) or the diameter

    outer_size : int
        Outer size of the annulus, representing either the radius (default) or the diameter

    All other parameters are the same as for the disc() function and are described there.
    
    Returns
    -------
    disc : array
        An array containing a disc with central obstruction
    '''

    if (inner_size >= outer_size):
        raise ValueError('Inner radius must be smaller than outer radius')
        return None

    # outer ring
    ap_out = disc(dim, outer_size, **kwargs)

    # inner ring
    if inner_size <= 0:
        ap_final = ap_out
    else:
        ap_in = disc(dim, inner_size, **kwargs)
        ap_final = ap_out - ap_in
        
    return ap_final


def disc(dim, size, diameter=False, strict=False, center=(), cpix=False, invert=False, mask=False):
    '''
    Create a numerical array containing a disc.
    
    Parameters
    ----------
    dim : int
        Size of the output array
    
    size : int
        Size of the disk, representing either the radius (default) or the diameter
    
    diameter : bool, optional
        Specify if input size is the diameter. Default is 'False'
    
    strict : bool optional
        If set to Trye, size must be strictly less than (<), instead of less
        or equal (<=). Default is 'False'
    
    center : sequence, optional
        Specify the center of the disc. Default is '()', i.e. the center of the array
    
    cpix : bool optional
        If set to True, the disc is centered on pixel at position (dim//2, dim//2).
        Default is 'False', i.e. the disc is centered between 4 pixels
    
    invert : bool, optinal
        Specify if the disc must be inverted. Default is 'False'
    
    mask : bool, optional
        Specify if return value is a masked array or a numerical array. Default
        is 'False'
        
    Returns
    -------
    disc : array
        An array containing a disc with the specified parameters
    '''
    
    if (diameter is True):
        rad = size/2
    else:
        rad = size

    if (len(center) == 0):
        if (cpix is False):
            cx = (dim-1) / 2
            cy = (dim-1) / 2
        else:
            cx = dim // 2
            cy = dim // 2
    elif (len(center) == 2):
        cx = center[0]
        cy = center[1]
    else:
        raise ValueError('Error, you must pass 2 values for center')
        return None

    x = np.arange(dim, dtype=np.float64) - cx
    y = np.arange(dim, dtype=np.float64) - cy
    xx, yy = np.meshgrid(x, y)
    
    rr = np.sqrt(xx**2 + yy**2)

    if (strict is True):
        msk = (rr < rad)
    else:
        msk = (rr <= rad)
    
    if (invert is True):
        msk = np.logical_not(msk)

    if (mask is True):
        return msk
    else:
        d = np.zeros((dim, dim))
        d[msk] = 1
    
    return d


def _rotate_interp(array, alpha, center, mode='constant', cval=0):
    '''
    Rotation around a provided center

    This is the only way to be sure where exactly is the center of rotation.

    '''
    dtype = array.dtype
    dims  = array.shape
    alpha_rad = -np.deg2rad(alpha)

    x, y = np.meshgrid(np.arange(dims[1], dtype=dtype), np.arange(dims[0], dtype=dtype))

    xp = (x-center[0])*np.cos(alpha_rad) + (y-center[1])*np.sin(alpha_rad) + center[0]
    yp = -(x-center[0])*np.sin(alpha_rad) + (y-center[1])*np.cos(alpha_rad) + center[1]

    rotated = ndimage.map_coordinates(array, [yp, xp], mode=mode, cval=cval, order=3)
    
    return rotated


def _rotate_spider_interp(array, alpha0, center0, alpha1, center1):
    '''
    Rotation around a provided center

    This is the only way to be sure where exactly is the center of rotation.

    '''
    dtype = array.dtype
    dims  = array.shape
    alpha0_rad = -np.deg2rad(alpha0)
    alpha1_rad = -np.deg2rad(alpha1)

    x, y = np.meshgrid(np.arange(dims[1], dtype=dtype), np.arange(dims[0], dtype=dtype))

    x0 =  (x-center0[0])*np.cos(alpha0_rad) + (y-center0[1])*np.sin(alpha0_rad) + center0[0]
    y0 = -(x-center0[0])*np.sin(alpha0_rad) + (y-center0[1])*np.cos(alpha0_rad) + center0[1]

    x1 =  (x0-center1[0])*np.cos(alpha1_rad) + (y0-center1[1])*np.sin(alpha1_rad) + center1[0]
    y1 = -(x0-center1[0])*np.sin(alpha1_rad) + (y0-center1[1])*np.cos(alpha1_rad) + center1[1]
    
    rotated = ndimage.map_coordinates(array, [y1, x1], mode='constant', cval=0, order=3)
    
    return rotated


def vlt_pupil(dim, diameter, spiders_thickness=0.008, spiders_orientation=0, 
              dead_actuators=None,dead_actuator_diameter=0.025, strict=False, cpix=False):
    '''Very Large Telescope theoretical pupil with central obscuration and spiders

    Parameters
    ----------
    dim : int
        Size of the output array
    
    diameter : int
        Diameter the disk

    spiders_thickness : float
        Thickness of the spiders, in fraction of the pupil
        diameter. Default is 0.008

    spiders_orientation : float
        Orientation of the spiders. The zero-orientation corresponds
        to the orientation of the spiders when observing in ELEV
        mode. Default is 0

    dead_actuators : array
        Position of dead actuators in the pupil, given in fraction of
        the pupil size. The default values are for SPHERE dead
        actuators but any other values can be provided as a Nx2 array.

    dead_actuator_diameter : float
        Size of the dead actuators mask, in fraction of the pupil
        diameter. This is the dead actuators of SPHERE. Default is
        0.025

    strict : bool optional
        If set to Trye, size must be strictly less than (<), instead of less
        or equal (<=). Default is 'False'
    
    cpix : bool optional
        If set to True, the disc is centered on pixel at position (dim//2, dim//2).
        Default is 'False', i.e. the disc is centered between 4 pixels
    

    Returns
    -------
    pup : array
        An array containing a disc with the specified parameters

    '''

    # central obscuration (in fraction of the pupil)
    obs  = 1100/8000

    # spiders
    if spiders_thickness > 0:
        # adds some padding on the borders
        tdim = dim+50

        # dimensions
        cc = tdim // 2
        spdr = int(max(1, spiders_thickness*dim))
            
        ref = np.zeros((tdim, tdim))
        ref[cc:, cc:cc+spdr] = 1
        spider1 = _rotate_interp(ref, -5.5, (cc, cc+diameter/2))

        ref = np.zeros((tdim, tdim))
        ref[:cc, cc-spdr+1:cc+1] = 1
        spider2 = _rotate_interp(ref, -5.5, (cc, cc-diameter/2))
        
        ref = np.zeros((tdim, tdim))
        ref[cc:cc+spdr, cc:] = 1
        spider3 = _rotate_interp(ref, 5.5, (cc+diameter/2, cc))
        
        ref = np.zeros((tdim, tdim))
        ref[cc-spdr+1:cc+1, :cc] = 1
        spider4 = _rotate_interp(ref, 5.5, (cc-diameter/2, cc))

        spider0 = spider1 + spider2 + spider3 + spider4

        spider0 = _rotate_interp(spider1+spider2+spider3+spider4, 45+spiders_orientation, (cc, cc))
        
        spider0 = 1 - spider0
        spider0 = spider0[25:-25, 25:-25]
    else:
        spider0 = np.ones(dim)

    # main pupil
    pup = disc_obstructed(dim, diameter, obs, diameter=True, strict=strict, cpix=cpix)

    # add spiders
    pup *= spider0
    
    # dead actuators
    if (dead_actuator_diameter > 0) and (dead_actuators is not None):
        dead_actuators = np.array(dead_actuators)
        xarr = dead_actuators[0]
        yarr = dead_actuators[1]
        for i in range(len(xarr)):
            cx = xarr[i] * diameter + dim/2
            cy = yarr[i] * diameter + dim/2

            dead = disc(dim, dead_actuator_diameter*diameter, center=(cx, cy), invert=True)

            pup *= dead

    return (pup >= 0.5).astype(int)



def sphere_saxo_pupil(dim=240):
    '''SPHERE pupil in the SAXO geometry

    Parameters
    ----------
    dim : int
        Size of the output array. Default is 384
    
    Returns
    -------
    pup : array
        An array containing a disc with the specified parameters

    '''

    # fixed diameter
    diameter = 240

    if dim < diameter:
        raise ValueError('Image dimensions cannot be smaller than 384 pixels')
    
    # main pupil
    pup = disc_obstructed(diameter, diameter, 0.14, diameter=True, strict=False, cpix=False)
    
    return pup



    
    
    
def AT_pupil(dim, diameter, spiders_thickness=0.008, strict=False, cpix=False):
    '''Auxillary Telescope theoretical pupil with central obscuration and spiders
    
    function adapted from pyzelda..
    
    
    Parameters
    ----------
    dim : int
        Size of the output array (pixels)
    
    diameter : int
        Diameter the disk (pixels)
    spiders_thickness : float
        Thickness of the spiders, in fraction of the pupil
        diameter. Default is 0.008
    spiders_orientation : float
        Orientation of the spiders. The zero-orientation corresponds
        to the orientation of the spiders when observing in ELEV
        mode. Default is 0
    dead_actuators : array
        Position of dead actuators in the pupil, given in fraction of
        the pupil size. The default values are for SPHERE dead
        actuators but any other values can be provided as a Nx2 array.
    dead_actuator_diameter : float
        Size of the dead actuators mask, in fraction of the pupil
        diameter. This is the dead actuators of SPHERE. Default is
        0.025
    strict : bool optional
        If set to Trye, size must be strictly less than (<), instead of less
        or equal (<=). Default is 'False'
    
    cpix : bool optional
        If set to True, the disc is centered on pixel at position (dim//2, dim//2).
        Default is 'False', i.e. the disc is centered between 4 pixels
    
    Returns
    -------
    pup : array
        An array containing a disc with the specified parameters
    '''

    # central obscuration (in fraction of the pupil)
    obs  = 0.13/1.8
    spiders_orientation = 0

    pp1 = 2.5
    # spiders
    if spiders_thickness > 0:
        # adds some padding on the borders
        tdim = dim+50

        # dimensions
        cc = tdim // 2
        spdr = int(max(1, spiders_thickness*dim))
            
        ref = np.zeros((tdim, tdim))
        ref[cc:, cc:cc+spdr] = 1
        spider1 = _rotate_interp(ref, -pp1 , (cc, cc+diameter/2))

        ref = np.zeros((tdim, tdim))
        ref[:cc, cc-spdr+1:cc+1] = 1
        spider2 = _rotate_interp(ref, -pp1 , (cc, cc-diameter/2))
        
        ref = np.zeros((tdim, tdim))
        ref[cc:cc+spdr, cc:] = 1
        spider3 = _rotate_interp(ref, pp1 , (cc+diameter/2, cc))
        
        ref = np.zeros((tdim, tdim))
        ref[cc-spdr+1:cc+1, :cc] = 1
        spider4 = _rotate_interp(ref, pp1 , (cc-diameter/2, cc))

        spider0 = spider1 + spider2 + spider3 + spider4

        spider0 = _rotate_interp(spider1+spider2+spider3+spider4, 45+spiders_orientation, (cc, cc))
        
        spider0 = 1 - spider0
        spider0 = spider0[25:-25, 25:-25]
    else:
        spider0 = np.ones(dim)

    # main pupil
    pup = disc_obstructed(dim, diameter, obs, diameter=True, strict=strict, cpix=cpix)

    # add spiders
    pup *= spider0

    return (pup >= 0.5).astype(int)


def pick_pupil(pupil_geometry, dim, diameter ):
        
    if pupil_geometry == 'AT':
        pup = AT_pupil(dim = dim, diameter = diameter) 
    elif pupil_geometry == 'UT':
        pup = vlt_pupil(dim = dim, diameter =  diameter, dead_actuator_diameter=0) 
    elif pupil_geometry == 'disk':
        pup = disc( dim = dim, size = diameter//2) 
    else :
        print('no valid geometry defined (try pupil_geometry == disk, or UT, or AT\nassuming disk pupil')
        pup = disc( dim = dim, size = diameter//2) 

    return(pup)
