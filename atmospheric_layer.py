import numpy as np
import warnings
import scipy.special as spc


def vonKarmanPSD(f, r0, L0, λ=500):
    """Calculate the von Karman PSD."""
    rad2nm = λ / 2.0 / np.pi
    
    cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
    PSD = r0**(-5/3)*cte*(f**2 + 1/L0**2)**(-11/6)
    PSD[ PSD.shape[0]//2, PSD.shape[1]//2 ] = 0  # Remove piston
    return PSD * rad2nm**2


def vonKarmanPSDDynamic(f, t, r0, L0, λ=500):
    """Time-variable version of the von Karman PSD that creates 3D cubes with temporal dimension."""
    rad2nm = λ / 2.0 / np.pi
    
    if not callable(r0): raise TypeError("r0 must be a function or lambda")
    if not callable(L0): raise TypeError("L0 must be a function or lambda")
    
    # Ensure t is an array
    # t = np.atleast_1d(t)
    
    cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
    
    # Vectorize the functions and compute for all time values at once
    r0_vals = r0(t)
    L0_vals = L0(t)
    
    # if GPU_
    
    # Broadcast computation: f is (spatial_x, spatial_y), time values are (len(t),)
    f_expanded  = f[..., np.newaxis]  # Shape: (spatial_x, spatial_y, 1)
    r0_expanded = r0_vals[np.newaxis, np.newaxis, :]  # Shape: (1, 1, len(t))
    L0_expanded = L0_vals[np.newaxis, np.newaxis, :]  # Shape: (1, 1, len(t))
    
    # Vectorized computation for all time slices
    PSD = r0_expanded**(-5/3) * cte * (f_expanded**2 + 1/L0_expanded**2)**(-11/6)
    
    # Remove piston for each time slice
    PSD[PSD.shape[0]//2, PSD.shape[1]//2, :] = 0
    
    return PSD * rad2nm**2


def SimpleBoiling(f, c):
    return f * 2*c # linear radial gradient map


def SimpleBoilingDynamic(f, t, c):
    """Time-variable version of SimpleBoiling that creates 3D cubes with temporal dimension."""
    
    if not callable(c): raise TypeError("c must be a function or lambda")
    
    # Ensure t is an array
    # t = np.atleast_1d(t)
    
    # Vectorize the function and compute for all time values at once
    c_vals = c(t)
    
    # Broadcast computation: f is (spatial_x, spatial_y), time values are (len(t),)
    f_expanded = f[..., np.newaxis]  # Shape: (spatial_x, spatial_y, 1)
    c_expanded = c_vals[np.newaxis, np.newaxis, :]  # Shape: (1, 1, len(t))
    
    # Vectorized computation for all time slices
    result = f_expanded * 2 * c_expanded
    
    return result


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
