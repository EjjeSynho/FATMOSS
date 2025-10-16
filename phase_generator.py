import numpy as np
from misc import *
import scipy.special as spc
from interpolate import Interpolator
import gc
from atmospheric_layer import Layer


class PhaseScreensGenerator:
    def __init__(self, D, dx, dt, batch_size=100, n_cascades=3, double_precision=False, seed=None, debug_flag=False) -> None:

        if double_precision:
            self.datafloat     = xp.float64
            self.datacomplex   = xp.complex128
            self.datafloat_cpu = np.float64
        else:
            self.datafloat     = xp.float32
            self.datacomplex   = xp.complex64
            self.datafloat_cpu = np.float32

        self.n_cascades = n_cascades
        self.batch_size = batch_size # [step]
        self.wvl_atmo   = 500.0 # [nm]

        self.debug = debug_flag

        self.screens_all_layers = None

        # Initialize random number generator with defined seed for reproducibility
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # Seeded random number generator
        
        self.dx = dx # [m/pixel]
        self.dt = dt # [s/step]

        factor = 3**(n_cascades-1)
        N_min = np.ceil(D / self.dx / factor)
        N_min += 1 - N_min % 2 # Make sure that the minimal size in the cascade is odd so it has a central piston pixel
        self.N = int(N_min * factor) # [pixels] Number of grid points; dividable by 3 since each cascade zooms in 3X
                
        self.GeneratePSDGrids() # Generate frequencies cascades
        
        # Initial random realization of PSD at t = 0 - using seeded RNG
        self.init_noise = xp.array(self.rng.normal(size=(self.N, self.N, 1)), dtype=self.datafloat)
        # Spatially-dependant phase retardation of the PSD's realisation phase, used to simulate boiling
        self.random_retardation = xp.asarray(self.rng.uniform(0, 1, size=(self.N,self.N,1)), dtype=self.datafloat) * dt

        self.current_batch_id = 0
        
        # Generate tip/tilt modes to simulate directional moving of the frozen flow
        coords  = np.linspace(0, self.N-1, self.N) - self.N//2 + 0.5 * (1-self.N%2)
        [xx,yy] = np.meshgrid( coords, coords, copy=False)
        center_grid = lambda x: x / (self.N//2 - 0.5*(1-self.N%2)) / 2.0
        self.tip  = xp.array(center_grid(xx)[..., None], dtype=self.datafloat) # 1/N-th of it shifts phase screen by 1 pixel
        self.tilt = xp.array(center_grid(yy)[..., None], dtype=self.datafloat)
        
        if GPU_flag:
            self.mempool = cp.get_default_memory_pool()
            self.pinned_mempool = cp.get_default_pinned_memory_pool()

        self.layers = []
        self.n_layers = 0
        self.interpolator = Interpolator()


    def __str__(self):
        """String representation of the PhaseScreensGenerator object."""
        info = []
        info.append(f"PhaseScreensGenerator Configuration:")
        info.append(f"  Grid size: {self.N}x{self.N} pixels")
        info.append(f"  Pixel size: {self.dx:.3f} m/pixel")
        info.append(f"  Time step: {self.dt:.3f} s")
        info.append(f"  Batch size: {self.batch_size} screens")
        info.append(f"  Number of cascades: {self.n_cascades}")
        # info.append(f"  Wavelength: {self.wvl_atmo} nm")
        info.append(f"  Data precision: {'double' if self.datafloat == xp.float64 else 'single'}")
        info.append(f"  Random seed: {self.seed}")
        info.append(f"  Current batch ID: {self.current_batch_id}")
        info.append(f"  Debug mode: {self.debug}")
        info.append(f"  GPU acceleration: {GPU_flag}")
        
        info.append(f"\nAtmospheric Layers ({len(self.layers)}):")
        if self.layers:
            for i, layer in enumerate(self.layers):
                info.append(f"  Layer {i+1}: weight={layer.weight:.3f}, "
                           f"wind_speed={layer.wind_speed:.1f} m/s, "
                           f"wind_direction={layer.wind_direction:.1f}°, "
                           f"boiling_factor={layer.boiling_factor:.3f}")
        else:
            info.append("  No layers added")
        
        return "\n".join(info)


    def __repr__(self):
        """Concise representation of the PhaseScreensGenerator object."""
        return (f"PhaseScreensGenerator(N={self.N}, dx={self.dx:.3f}, dt={self.dt:.3f}, "
                f"batch_size={self.batch_size}, n_cascades={self.n_cascades}, "
                f"layers={len(self.layers)})")

    
    def reset_random_state(self, new_seed=None):
        """
        Reset the random number generator state.
        
        Parameters:
        -----------
        new_seed : int or None
            New seed for the random number generator. If None, uses the original seed.
        """
        seed_to_use = new_seed if new_seed is not None else self.seed
        self.rng = np.random.default_rng(seed_to_use)
        
        # Regenerate the random arrays with new seed
        self.init_noise = xp.array(self.rng.normal(size=(self.N, self.N, 1)), dtype=self.datafloat)
        self.random_retardation = xp.asarray(self.rng.uniform(0, 1, size=(self.N, self.N, 1)), dtype=self.datafloat) * self.dt


    def deallocate_gpu_memory(self, full_cleanup=True):
        """
        Deallocate GPU memory and clean up resources.
        
        Parameters:
        -----------
        full_cleanup : bool
            If True, performs complete cleanup including internal arrays.
            If False, only cleans up cached results but keeps core arrays.
        """
        if not GPU_flag:
            return  # Nothing to do if not using GPU
        
        try:
            # Clean up cached results and intermediate arrays
            if hasattr(self, 'screens_all_layers') and self.screens_all_layers is not None:
                if hasattr(self.screens_all_layers, 'device'):
                    del self.screens_all_layers
                self.screens_all_layers = None
            
            
            if hasattr(self, 'phase_buf'):
                for item in self.phase_buf:
                    if hasattr(item, 'device'):
                        del item
                delattr(self, 'phase_buf')
        
            
            if full_cleanup:
                # Clean up core arrays (use with caution - will require regeneration)
                if hasattr(self, 'init_noise') and hasattr(self.init_noise, 'device'):
                    del self.init_noise
                    self.init_noise = None
                
                if hasattr(self, 'random_retardation') and hasattr(self.random_retardation, 'device'):
                    del self.random_retardation
                    self.random_retardation = None
            
            # Force garbage collection and memory pool cleanup
            if hasattr(self, 'mempool'):
                self.mempool.free_all_blocks()
            
            if hasattr(self, 'pinned_mempool'):
                self.pinned_mempool.free_all_blocks()
            
            # Additional cleanup
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            # Force CUDA synchronization
            cp.cuda.Device().synchronize()
            
            if self.debug:
                print("GPU memory successfully deallocated.")
                
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Error during GPU memory deallocation: {e}")


    def __del__(self):
        """
        Destructor to ensure GPU memory is cleaned up when object is destroyed.
        """
        try:
            self.deallocate_gpu_memory(full_cleanup=True)
        except:
            pass  # Ignore errors during cleanup


    def GeneratePSDGrids(self):
        arrays_shape = [self.N, self.N, self.n_cascades]
        self.fx = np.zeros(arrays_shape,    dtype=self.datafloat_cpu)
        self.fy = np.zeros(arrays_shape,    dtype=self.datafloat_cpu)
        self.f  = np.zeros(arrays_shape,    dtype=self.datafloat_cpu)
        self.df = np.zeros(self.n_cascades, dtype=self.datafloat_cpu)
        
        select_middle = lambda N: np.s_[N//2-N//6:N//2+N//6+N%2, N//2-N//6:N//2+N//6+N%2, :] # selects the middle 1/3 quandrant of the image
        self.crops = [np.s_[0:self.N, 0:self.N, :]] # 1st crop contains the whole image

        for i in range(self.n_cascades):
            self.crops.append(select_middle(self.N // 3**i))
            dx_ = self.dx * 3**i # every next cascade zooms out frequencies by 3 to capture more low frequencies
            self.fx[...,i], self.fy[...,i], self.f[...,i], self.df[...,i] = self.freq_array(self.N, dx_) # [1/m]
        _ = self.crops.pop(-1)


    def GeneratePSDCascade(self, PSD_spatial_func, boiler_func):
        # Pre-allocate arrays
        arrays_shape = [self.N, self.N, self.n_cascades]
        PSDs_temporal = np.zeros(arrays_shape, dtype=self.datafloat_cpu) # Describes the temporal evolution of the PSD
        PSDs_spatial  = np.zeros(arrays_shape, dtype=self.datafloat_cpu) # Contains spatial frequencies

        for i in range(self.n_cascades):
            PSDs_spatial [...,i] = PSD_spatial_func(self.f[...,i]) * self.df[i]**2 # PSD spatial [nm^2/m^2]
            PSDs_temporal[...,i] = boiler_func(self.f[...,i]) # TODO: [??/??] units

        # Masks are used to supress the spatial frequencies that belong to other cascades
        mask_outer = mask_circle( self.N, self.N/2, centered=True )
        mask_inner = mask_circle( self.N, self.N/6, centered=True )

        PSDs_spatial [...,:-1] *= (mask_outer-mask_inner)[..., np.newaxis]
        PSDs_spatial [..., -1] *=  mask_outer # The last cascade has the most information about the low frequencies

        PSDs_temporal[...,:-1] *= (mask_outer-mask_inner)[..., np.newaxis]
        PSDs_temporal[..., -1] *=  mask_outer
        
        return PSDs_spatial, PSDs_temporal


    def freq_array(self, N, dx):
        """Generate spatial frequency arrays."""
        df = 1 / (N*dx)  # Spatial frequency interval [1/m]
        fx = (np.arange(-N//2, N//2, 1) + N%2) * df
        fy = (np.arange(-N//2, N//2, 1) + N%2) * df
        fx, fy = np.meshgrid(fx, fy, indexing='ij')
        return fx, fy, np.sqrt(fx**2 + fy**2), df


    def generate_phase_screen(self, PSD, N):
        """Generates single phase screen from PSD."""
        random_phase = np.exp(2j * np.pi * self.rng.random((N, N)))
        complex_spectrum = np.fft.ifftshift(np.sqrt(PSD) * random_phase)
        phase_screen = np.fft.ifft2(complex_spectrum) * PSD.size
        phase_screen_nm = np.real(phase_screen)  
        return phase_screen_nm


    def screens_from_PSD_and_phase(self, PSD, complex_phase):
        """Generate phase screens batch from PSD and random phase."""
        dimensions = complex_phase.shape
        # PSD_realizations = xp.zeros(dimensions, dtype=self.datacomplex)
        phase_batch = xp.zeros(dimensions, dtype=self.datafloat)

        PSD_ = xp.atleast_3d(xp.array(PSD, dtype=self.datafloat))
        PSD_realizations = xp.sqrt(PSD_) * complex_phase
        
        # Perform batched FFT
        if GPU_flag:
            plan = gfft.get_fft_plan(PSD_realizations, axes=(0,1), value_type='C2C') # for batched, C2C, 2D transform
            phase_batch = cp.real( gfft.ifft2(gfft.ifftshift(PSD_realizations, axes=(0,1)), axes=(0,1), plan=plan) * PSD.size )
        else:
            phase_batch = np.real( np.fft.ifft2(np.fft.ifftshift(PSD_realizations, axes=(0,1)), axes=(0,1)) * PSD.size)

        return phase_batch


    def AddLayer(self, layer: Layer):
        # Generate PSDs for the layer
        layer.PSD_spatial, layer.PSD_temporal = self.GeneratePSDCascade(layer.PSD_spatial_func, layer.PSD_temporal_func)
        self.layers.append(layer)
        self.n_layers = len(self.layers)


    def GenerateScreensBatch(self, PSD_spatial, PSD_temporal, wind_speed, wind_direction, boiling_factor):
        # Frames IDs with an additional temporal shift to simulate virtually infinite temporal evolution
        t = xp.arange(self.batch_size, dtype=self.datafloat) + self.current_batch_id * self.batch_size
        t = t[None, None, :] # [1 x 1 x batch_size]

        # Generate cascaded phase screens (W x H x batch_size x N_cascades)
        screens_batch = xp.zeros([self.N, self.N, self.batch_size, self.n_cascades], dtype=self.datafloat)
        raw_batch     = xp.zeros([self.N, self.N, self.batch_size, self.n_cascades], dtype=self.datafloat)
        phase_batch = []
        
        if self.debug:
            raw_batch_cropped = []

        for i in range(self.n_cascades):
            # Due to zooming out the lower frequencies, the wind speed must to be slowed down
            V = xp.array(wind_speed / self.dx * self.dt, dtype=self.datafloat) # [pixels/step]
            Vx_ = V * xp.cos(xp.deg2rad(xp.array(wind_direction, dtype=self.datafloat))) / 3**i
            Vy_ = V * xp.sin(xp.deg2rad(xp.array(wind_direction, dtype=self.datafloat))) / 3**i
            # TODO: implement 2π wrapping
            PSD_temporal_ = xp.array(PSD_temporal[...,i][..., None], self.datafloat)
            evolution = self.tip*Vx_ + self.tilt*Vy_ + self.random_retardation * boiling_factor * PSD_temporal_
            phi = t * evolution + self.init_noise
            random_complex_phase = xp.exp(2j*xp.pi * phi)
            
            phase_buffer = self.screens_from_PSD_and_phase(PSD_spatial[...,i], random_complex_phase)
        
            screens_batch[...,i] = self.interpolator.zoom(phase_buffer[self.crops[i]], i)
            phase_batch.append(random_complex_phase) #.copy() )

            if self.debug:
                raw_batch[...,i] =  phase_buffer.copy()
                raw_batch_cropped.append( phase_buffer[self.crops[i]] )
            
        screens_batch[...,-1] -= screens_batch[...,-1].mean(axis=(0,1), keepdims=True) # remove piston, just in case

        if self.debug:        
            raw_batch[...,-1] -= raw_batch[...,-1].mean(axis=(0,1), keepdims=True) # remove piston again
            self.buffa = screens_batch#.copy()
            self.raw_batch = raw_batch#.copy()
            self.raw_batch_cropped = raw_batch_cropped
            self.batch_interp = screens_batch#.copy()
            self.phase_buf = phase_batch
        
        screens_batch = screens_batch.sum(axis=-1) # sum all cascades to W x H x N_screens

        if GPU_flag:
            self.mempool.free_all_blocks()
            self.pinned_mempool.free_all_blocks() # clear GPU memory

        return screens_batch
    
    
    def UpdateScreensBatch(self, batch_id=None, split_layers=False):
        batch_id = 0 if batch_id is None else batch_id

        if batch_id != self.current_batch_id or self.screens_all_layers is None:
            self.current_batch_id = batch_id
            self.screens_all_layers = self.GenerateScreens(split_layers=split_layers)

        return self.screens_all_layers


    def GetScreenByTimestep(self, id):
        batch_id  = id // self.batch_size
        screen_id = id % self.batch_size
        batch = self.UpdateScreensBatch(batch_id)
        
        return batch[..., screen_id]
    
    
    def reset(self):
        self.current_batch_id = 0
        self.screens_all_layers = None
        self.reset_random_state(new_seed=self.seed)
        self.deallocate_gpu_memory(full_cleanup=False)
    
    
    def GenerateScreens(self, split_layers=False):
        if len(self.layers) == 0:
            raise ValueError("No atmospheric layers have been added. Please add at least one layer before generating screens.")
        
        if split_layers:
            screens_batch = [
                self.GenerateScreensBatch(
                    layer.PSD_spatial,
                    layer.PSD_temporal,
                    layer.wind_speed,
                    layer.wind_direction,
                    layer.boiling_factor
                ) * layer.weight for layer in self.layers
            ]
            return screens_batch
        
        else:
            screens_batch = 0
            for layer in self.layers:
                screens_batch += self.GenerateScreensBatch(
                    layer.PSD_spatial,
                    layer.PSD_temporal,
                    layer.wind_speed,
                    layer.wind_direction,
                    layer.boiling_factor
                ) * layer.weight
                
            return screens_batch