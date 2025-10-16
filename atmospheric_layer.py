import numpy as np
import warnings
import scipy.special as spc


def vonKarmanPSD(f, r0, L0, λ=0.5e-6):
    """Calculate the von Karman PSD."""
    rad2nm = λ / 2.0 / np.pi
    cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
    PSD = r0**(-5/3)*cte*(f**2 + 1/L0**2)**(-11/6)
    PSD[ PSD.shape[0]//2, PSD.shape[1]//2 ] = 0  # Remove piston
    return PSD * rad2nm


def SimpleBoiling(f, c):
    return f * 2*c # linear radial gradient map


class Layer:
    def __init__(self, weight, altitude, wind_speed, wind_direction, boiling_factor, PSD_spatial, PSD_temporal):
        warnings.warn('Layer height is not yet implemented.')
        self.weight = weight
        self.altitude = altitude
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.boiling_factor = boiling_factor
        self.PSD_spatial_func = PSD_spatial
        self.PSD_temporal_func = PSD_temporal
        self.PSD_spatial = None
        self.PSD_temporal = None
