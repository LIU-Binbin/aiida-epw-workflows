# -*- coding: utf-8 -*-
"""Calculators for the superconductivity analysis."""

import numpy
from numpy.typing import ArrayLike
import scipy
from aiida.engine import calcfunction
from aiida.orm import ArrayData, XyData, Float
from scipy.interpolate import interp1d

meV_to_Kelvin = 11.604518121550082


def allen_dynes(lambo, omega_log, mu_star):
    """Calculate the Allen-Dynes critical temperature Tc."""
    if lambo - mu_star * (1 + 0.62 * lambo) < 0:
        return 0
    else:
        return omega_log * numpy.exp(-1.04 * (1 + lambo) / (lambo - mu_star * (1 + 0.62 * lambo))) / 1.2


def calculate_lambda_omega(frequency: ArrayLike, spectrum: ArrayLike) -> tuple:
    """Calculate lambda and omega_log from the parsed a2F spectrum.

    :param frequency: Frequency array on which the a2F spectrum is defined [meV].
    :param spectrum: a2F spectral function values.

    :returns: Tuple of the calculated lambda and omega_log values.
    """
    lambda_ = 2 * scipy.integrate.trapezoid(spectrum / frequency, frequency)  # unitless
    omega_log = numpy.exp(2 / lambda_ * scipy.integrate.trapezoid(spectrum / frequency * numpy.log(frequency), frequency))  # eV
    omega_log = omega_log * meV_to_Kelvin

    return lambda_, omega_log

@calcfunction
def calculate_Allen_Dynes_tc(a2f: ArrayData, mustar = 0.13) -> Float:
    w        = a2f.get_array('frequency')
    # Here we preassume that there are 10 smearing values for a2f calculation
    spectral = a2f.get_array('a2f')[:, 9]
    mev2K    = 11.604525006157

    _lambda  = 2*numpy.trapz(numpy.divide(spectral, w), x=w)

    # wlog =  np.exp(np.average(np.divide(alpha, w), weights=np.log(w)))
    wlog     =  numpy.exp(2/_lambda*numpy.trapz(numpy.multiply(numpy.divide(spectral, w), numpy.log(w)), x=w))

    Tc = wlog/1.2*numpy.exp(-1.04*(1+_lambda)/(_lambda-mustar*(1+0.62*_lambda))) * mev2K


    return Float(Tc)

def check_convergence(
    a2f_tcs,
    convergence_threshold
    ):
    """Check if the convergence is reached."""
    if len(a2f_tcs) < 3:
        return (False, 'Not enough data to check convergence.')
    else:
        subsequent_differences = [
            abs((prev_allen_dynes - new_allen_dynes)/new_allen_dynes)
            for prev_allen_dynes, new_allen_dynes in zip(a2f_tcs[:-1], a2f_tcs[1:])
        ]

        if all([difference < convergence_threshold for difference in subsequent_differences[-1:]]):
            return (True, a2f_tcs[-1])
        else:
            return (False, None)

def _calculate_iso_tc(max_eigenvalue, allow_extrapolation=False):
    if max_eigenvalue[:, 1].max() < 1.0:
        return 0.0
    elif max_eigenvalue[:, 1].min() > 1.0:
        if allow_extrapolation:
            print("This Tc is estimated from the extrapolation of the max eigenvalues. Please check whether it's reliable.")
            f_extrapolate = interp1d(
                max_eigenvalue[:, 1],
                max_eigenvalue[:, 0],
                kind='linear',          # Can be 'linear', 'quadratic', etc. for interpolation
                bounds_error=False,     # Do not raise an error for out-of-bounds values
                fill_value="extrapolate" # Extrapolate using a line from the last two points
                )
            return float(f_extrapolate(1.0))
        else:
            return numpy.nan
    else:
        return float(interp1d(max_eigenvalue[:, 1], max_eigenvalue[:, 0])(1.0))


@calcfunction
def calculate_iso_tc(max_eigenvalue: XyData) -> Float:
    return Float(_calculate_iso_tc(max_eigenvalue.get_array('max_eigenvalue')))

