


# ZWFS initialization 
z = 

# grid 
dim = 
N = 
dt = 0.001 # s

# atmosphere 
r0 = 0.1 #cm
L0 = 25 #m

# first stage AO 
Nmodes_removed = 0

# vibrations 
mode_indicies = [0, 1]
spectrum_type = ['1/f', '1/f']
opd = [50e-9, 50e-9]
vibration_frequencies = []

# create phase screen 

# function to remove N modes from the phase screen

# create vibration spectrum 

# create time series of the vibration spectrum

# for a non-aberrated system 
#   calculate the non-observable parameters 
#       - optical gain (b)
#   simulate the observable parameters
#       - ZWFS signal with and without phasemask
#   find a way to estimate b0. Problem is that we don't know the internal aberrations.
#   The general model is:
#       I0 - N0 = abs(psi_A)**2 + abs(psi_R)**2 + 2 * abs(psi_A) * abs(psi_R) * np.cos(  mu ) - abs(psi_A)**2    
#   Problem statement:
#   We have measurements of two images (intensities in a NxM camera defined by x,y pixel coordinates)  I0(x,y), N0(x,y). 
#   I0(x,y) is the intensity with a phase shifting phase mask inserted at focus and N0(x,y) is the intensity without 
#   the phasemask which is essentailly a uniformly illuminated pupil. The difference  of these two quantities follows the model 
#       delta(x,y)  = I0(x,y) - N0(x,y) 
#                   = abs(psi_R(x,y)) * ( abs(psi_R(x,y)) + 2 * sqrt(N0(x,y)) * np.cos( phi0(x,y) - mu ) ) 
#   We know mu (which has no spatial dependence) by design but need to estimate phi0(x,y) and psiR(x,y). 
#   A reasonable prior for phi0(x,y) is a zero mean Guassian with some variance sigma^2_phi. 
#   We also know N0(x,y) is zero outside the pupil (i.e. x,y not in P) - which allows direct sampling of  
#       delta(x,y not in P) =  abs(psi_R(x,y not in P))^2
#   We also have a theoretical model of psi_R(x,y) across all regions that 
#       psi_R(x,y) = A * iFFT( Pi(x,y) * FFT( N0 * exp(1j phi0 (x,y) ) ) where iFFT and FFT are the Fourier transforms, 
#   and A is a known (by construction) parameter and Pi(x,y) defines the phaseshifting region on the mask (0 if no phase shift applie, 1 otherwise).
 


#  I0 - N0  = abs(psi_A)**2 + abs(psi_R)**2 + 2 * abs(psi_A) * abs(psi_R) * np.cos(  mu ) - abs(psi_A)**2
#           = abs(psi_R) * ( abs(psi_R) + 2 * abs(psi_A) * np.cos( phi0 - mu ) ) 
#  => 0     = abs(psi_R)**2 + 2 * abs(psi_A) * abs(psi_R) * np.cos( phi0 - mu )  - (I0 - N0)   
# quadratic in abs(psi_R). a= 1, b = 2 * abs(psi_A) * np.cos( phi0 - mu ),  c = -(I0 - N0). 
# All measured besides phi0 (internal aberrations). initially assume phi0 = 0.
# we could also estimate phi0 via minimization of the residuals by leaving phi0 as free variable
# then minimize the residuals to find the best fit phi0.
# min_phi0 (delta_I_m - delta_i_theory) 
# we could use a centered normal distribution for the prior on phi0 to avoid phase wrapping.

#a= 1, b = 2 * abs(psi_A) * np.cos( lambda + phi0 - mu ),  c = -(I0 - N0). 

abs(psi_R)_theory - 


 
# simulate simulate the ZWFS signal with no-aberrations, es 
# iterate phase screens with vibrations
#   for each phase screen,
#       remove N modes
#       add any modal vibrations
#       calculate the non-observable parameters 
#       - strehl 
#       - optical gain (b)
#       simulate the observable parameters
#       - ZWFS signal
#   goal is to build a model that predicts non-observable parameters from observable parameters
#   baseline model for Strehl is a linear combination of ZWFS intensity in sub-regions of the image  
#   baseline model for the optical gain is then b = sqrt(S) * b0. Where b0 is the optical gain without aberrations  