import numpy as np
import matplotlib.pyplot as plt
import os 
import sys
import pandas as pd
from scipy.optimize import curve_fit


# Define Cauchy's equation
def cauchy_eqn(wavelength, A, B, C):
    return A + B / wavelength**2 + C / wavelength**4


def fit_cauchy_eqn_to_data(df, savefig = None):
    # Example dataframe (replace with your actual df)
    # df = pd.read_csv('path_to_your_data.csv')  # If your data is stored in a CSV
    # Assuming your data is already in a dataframe 'df'
    wavelengths = df['Wavelength(um)'].values  # Extracting wavelength values
    n_measured = df['n'].values  # Extracting refractive index values

    # Perform the curve fitting
    popt, pcov = curve_fit(cauchy_eqn, wavelengths, n_measured)

    # Extract the fitted coefficients
    A_fit, B_fit, C_fit = popt

    # Generate the fitted curve
    n_fitted = cauchy_eqn(wavelengths, A_fit, B_fit, C_fit)

    # Plot the measured data vs the fitted curve
    plt.plot(wavelengths, n_measured, 'b-', label='Measured Data')
    plt.plot(wavelengths, n_fitted, 'r--', label=f'Fitted Cauchy Eqn\nA={A_fit:.4f}, B={B_fit:.4e}, C={C_fit:.4e}')
    plt.xlabel('Wavelength [um]')
    plt.ylabel('Refractive Index n')
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight', dpi=200)  
    plt.show()


df = pd.read_csv('data/Exposed_Ma-N_1405_optical_constants.txt', sep='\s+', header=1)
df['Wavelength(um)'] =  df['Wavelength(nm)'] * 1e-3
df_900nm_cut = df[df['Wavelength(nm)'] > 900]
fit_cauchy_eqn_to_data( df, savefig = '/home/benja/Documents/BALDR/figures/Exposed_Ma-N_1405_cauchy_fit.png' )

fit_cauchy_eqn_to_data( df_900nm_cut, savefig = '/home/benja/Documents/BALDR/figures/Exposed_Ma-N_1405_cauchy_fit_wave900-1600.png' )