import numpy as np
import warnings
from misc import *
import scipy.special as spc


class Layer:
    def __init__(self, phase_generator, weight, r0, L0, wind_speed, wind_direction, boiling_factor):
        warnings.warn('Layer height is not yet properly implemented.')
        self.phase_generator = phase_generator
        self.weight = weight
        self.r0 = r0
        self.L0 = L0
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.boiling_factor = boiling_factor
        self.screens_batch = None
        self.PSDs, self.PSD_temporal = self.phase_generator.GeneratePSDCascade(r0, L0)
    
    def GenerateScreensBatch(self):
        self.screens_batch = self.phase_generator.GenerateScreensBatch(
            self.PSDs,
            self.PSD_temporal,
            self.wind_speed,
            self.wind_direction,
            self.boiling_factor
        )
        return self.screens_batch


class CascadedPhaseGenerator:
    def __init__(self, D, dx, dt, batch_size=100, n_cascades=3, double_precision=False) -> None:

        if double_precision:
            self.datafloat     = xp.float64
            self.datacomplex   = xp.complex128
            self.datafloat_cpu = np.float64
        else:
            self.datafloat     = xp.float32
            self.datacomplex   = xp.complex64
            self.datafloat_cpu = np.float32

        self.n_cascades  = n_cascades
        self.num_screens = batch_size #[step]
        self.wvl_atmo    = 500.0 # [nm]

        self.screens_all_layers = None

        # Initialize cascade parameters
        factor = 3**(n_cascades-1)

        self.dx = dx # [m/pixel]

        N_min = np.ceil(D / dx / factor)
        N_min += 1 - N_min % 2 # Make sure that the minimal size in the cascade is odd so it has a central piston pixel
        self.N = int(N_min * factor) # [pixels] Number of grid points; Make sure size is subdividable by 3, because each cascade zooms 3X
        
        self.dt = dt # [s/step]
        
        self.GeneratePSDGrids()
        
        # Initial random realization of PSD at t=0
        rng = np.random.default_rng() # random generation on the CPU is faster
        self.init_noise = xp.array( rng.normal(size=(self.N, self.N, 1)), dtype=self.datafloat )
        # Spatially-dependant phase retardation of the PSD's realisation phase, used to simulate boiling
        self.random_retardation = xp.asarray(np.random.uniform(0, 1, size=(self.N,self.N,1)), dtype=self.datafloat) * dt

        self.boiler = lambda f: f * 2*dx # just radial linear gradient map with 1.0 at the N/2 distance from the center for temporal evolution

        self.current_batch_id = 0
        
        if GPU_flag:
            self.mempool = cp.get_default_memory_pool()
            self.pinned_mempool = cp.get_default_pinned_memory_pool()

        self.layers = []


    def AddLayer(self, weight, r0, L0, wind_speed, wind_direction, boiling_factor):
        self.layers.append(Layer(self, weight, r0, L0, wind_speed, wind_direction, boiling_factor))


    def GeneratePSDGrids(self):
        arrays_shape = [self.N, self.N, self.n_cascades]
        self.fx = np.zeros(arrays_shape,    dtype=self.datafloat_cpu)
        self.fy = np.zeros(arrays_shape,    dtype=self.datafloat_cpu)
        self.f  = np.zeros(arrays_shape,    dtype=self.datafloat_cpu)
        self.df = np.zeros(self.n_cascades, dtype=self.datafloat_cpu)
        
        select_middle = lambda N: np.s_[N//2-N//6:N//2+N//6+N%2, N//2-N//6:N//2+N//6+N%2, :]
        self.crops = [np.s_[0:self.N, 0:self.N, :]] # 1st one selects the whole image

        for i in range(self.n_cascades):
            self.crops.append(select_middle(self.N // 3**i)) # selects thw middle 1/3 quandrant of the image
            dx_ = self.dx * 3**i # every next cascade zooms out frequencies by 3 to capture more low frequencies
            self.fx[...,i], self.fy[...,i], self.f[...,i], self.df[...,i] = self.freq_array(self.N, dx_) # [1/m]
        _ = self.crops.pop(-1)


    def GeneratePSDCascade(self, r0, L0):
        rad2nm = self.wvl_atmo / 2.0 / np.pi # [nm/rad]
        
        # fx_, fy_, f_, df_ = self.freq_array(N, dx) # [1/m]
        # PSD_test = self.vonKarmanPSD(f_, r0, L0) * df_**2 * rad2nm**2 # [nm^2/m^2]

        # Pre-allocate arrays
        arrays_shape = [self.N, self.N, self.n_cascades]
        PSD_temporal = np.zeros(arrays_shape, dtype=self.datafloat_cpu) # Describes the temporal evolution of the PSD
        PSDs = np.zeros(arrays_shape, dtype=self.datafloat_cpu) # Contains spatial frequencies

        for i in range(self.n_cascades):
            PSDs[...,i] = self.vonKarmanPSD(self.f[...,i], r0, L0) * self.df[i]**2 * rad2nm**2 # PSD spatial [nm^2/m^2]
            PSD_temporal[...,i] = self.boiler(self.f[...,i]) # PSD temporal [??/??]

        # Masks are used to supress the spatial frequencies that belong to other cascades
        mask_outer = mask_circle( self.N, self.N/2, centered=True )
        mask_inner = mask_circle( self.N, self.N/6, centered=True )

        PSDs[...,:-1] *= (mask_outer-mask_inner)[..., np.newaxis]
        PSDs[...,-1]  *=  mask_outer # The last cascade has the most information about the low frequencies

        PSD_temporal[...,:-1] *= (mask_outer-mask_inner)[..., np.newaxis]
        PSD_temporal[...,-1]  *=  mask_outer
        
        return PSDs, PSD_temporal


    def freq_array(self, N, dx):
        """Generate spatial frequency arrays."""
        df = 1 / (N*dx)  # Spatial frequency interval [1/m]
        fx = (np.arange(-N//2, N//2, 1) + N%2) * df
        fy = (np.arange(-N//2, N//2, 1) + N%2) * df
        fx, fy = np.meshgrid(fx, fy, indexing='ij')
        return fx, fy, np.sqrt(fx**2 + fy**2), df


    def vonKarmanPSD(self, k, r0, L0):
        """Calculate the von Karman PSD."""
        cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
        PSD = r0**(-5/3)*cte*(k**2 + 1/L0**2)**(-11/6)
        PSD[ PSD.shape[0]//2, PSD.shape[1]//2 ] = 0  # Avoid division by zero at the origin
        return PSD


    def generate_phase_screen(self, PSD, N):
        """Generates single phase screen from PSD."""
        random_phase = np.exp(2j * np.pi * np.random.rand(N, N))
        complex_spectrum = np.fft.ifftshift(np.sqrt(PSD) * random_phase)
        phase_screen = np.fft.ifft2(complex_spectrum) * PSD.size
        phase_screen_nm = np.real(phase_screen)  
        return phase_screen_nm


    def screens_from_PSD_and_phase(self, PSD, PSD_phase):
        """Generate phase screens batch from PSD and random phase."""
        dimensions = PSD_phase.shape
        # PSD_realizations = xp.zeros(dimensions, dtype=self.datacomplex)
        phase_batch = xp.zeros(dimensions, dtype=self.datafloat)

        PSD_ = xp.atleast_3d(xp.array(PSD, dtype=self.datafloat))
        PSD_realizations = xp.sqrt(PSD_) * PSD_phase

        # Perform batched FFT
        if GPU_flag:
            plan = gfft.get_fft_plan(PSD_realizations, axes=(0,1), value_type='C2C') # for batched, C2C, 2D transform
            phase_batch = cp.real( gfft.ifft2(gfft.ifftshift(PSD_realizations, axes=(0,1)), axes=(0,1), plan=plan) * PSD.size )
        else:
            phase_batch = np.real( np.fft.ifft2(np.fft.ifftshift(PSD_realizations, axes=(0,1)), axes=(0,1)) * PSD.size)

        return phase_batch


    def zoomX3(self, x, iters=0, interp_order=3):
        """Scales up the resolution of the phase screens stack 'x' by a factor of 3."""	
        if iters > 0:
            zoom_factor = (3**iters, 3**iters, 1)
            return zoom(x, zoom_factor, order=interp_order)
        else:
            return x


    def zoomX3_FFT(self, x, iters):
        if iters > 0:
            factor = 3**iters
            original_height, original_width, num_screens = x.shape
            new_height = int(original_height * factor)
            new_width  = int(original_width  * factor)

            fft_batch = xp.fft.fft2(x, axes=(0, 1))
            fft_shifted_batch = xp.fft.fftshift(fft_batch, axes=(0, 1))

            padded_fft_batch = xp.zeros((new_height, new_width, num_screens), dtype=xp.complex64)
            pad_height_start = (new_height - original_height) // 2
            pad_width_start  = (new_width  - original_width)  // 2

            padded_fft_batch[
                pad_height_start : pad_height_start + original_height,
                pad_width_start  : pad_width_start  + original_width,
            :] = fft_shifted_batch

            ifft_shifted_batch = xp.fft.ifftshift(padded_fft_batch, axes=(0, 1))
            interpolated_batch = xp.fft.ifft2(ifft_shifted_batch, axes=(0, 1))

            return xp.real(interpolated_batch) * factor**2
        else:
            return x


    def GenerateScreensBatch(self, PSDs, PSD_temporal, wind_speed, wind_direction, boiling_factor):
        N = self.N
        dt = self.dt
        num_screens = self.num_screens
        n_cascades  = self.n_cascades

        # Frames IDs with an additional temporal shift to simulate virtually infinite temporal  evolution
        screen_id = xp.arange(self.num_screens, dtype=self.datafloat) + self.current_batch_id * self.num_screens
        screen_id = screen_id[None, None, :] # 1 x 1 x N_screens

        # Generate tip/tilt modes to simulate directional moving of the frozen flow
        coords  = np.linspace(0, N-1, N) - N//2 + 0.5 * (1-N%2)
        [xx,yy] = np.meshgrid( coords, coords, copy=False)
        center_grid = lambda x: x / (N//2 - 0.5*(1-N%2)) / 2.0
        tip  = xp.array( center_grid(xx)[..., None], dtype=self.datafloat ) # 1/N-th of it shifts phase screen by 1 pixel
        tilt = xp.array( center_grid(yy)[..., None], dtype=self.datafloat )

        # Generate cascaded phase screens (W x H x N_screens x N_cascades)
        screens_batch = xp.zeros([N, N, num_screens, n_cascades], dtype=self.datafloat)

        for i in range(n_cascades):
            # Due to zooming out the lower frequencies, the wind speed must to be slowed down
            V = xp.array(wind_speed / self.dx * dt, dtype=self.datafloat) # [pixels/step]
            Vx_ = V * xp.cos(xp.deg2rad(xp.array(wind_direction, dtype=self.datafloat))) / 3**i
            Vy_ = V * xp.sin(xp.deg2rad(xp.array(wind_direction, dtype=self.datafloat))) / 3**i

            PSD_temporal_ = xp.array(PSD_temporal[...,i][..., None], self.datafloat)
            evolution = tip*Vx_ + tilt*Vy_ + self.random_retardation * boiling_factor * PSD_temporal_
            random_phase = xp.exp(2j*xp.pi * (screen_id*evolution + self.init_noise) )
            
            screens_batch[...,i] = self.zoomX3( self.screens_from_PSD_and_phase(PSDs[...,i], random_phase)[self.crops[i]], i )
            # screens_batch[...,i] = self.zoomX3_FFT( self.screens_from_PSD_and_phase(PSDs[...,i], random_phase)[self.crops[i]], i )
            
        screens_batch[...,-1] -= screens_batch[...,-1].mean(axis=(0,1), keepdims=True) # remove piston again
        screens_batch = screens_batch.sum(axis=-1) # sum all cascades to W x H x N_screens

        if GPU_flag:
            self.mempool.free_all_blocks()
            self.pinned_mempool.free_all_blocks() # clear GPU memory

        return screens_batch
    
    
    def UpdateScreensBatch(self, batch_id=None):
        batch_id = 0 if batch_id is None else batch_id

        if batch_id != self.current_batch_id or self.screens_all_layers is None:
            self.current_batch_id = batch_id
            buf = [layer.GenerateScreensBatch() * layer.weight for layer in self.layers]
            self.screens_all_layers = xp.stack(buf, axis=-1).sum(axis=-1)

        return self.screens_all_layers


    def GetScreenByTimestep(self, id):
        batch_id  = id // self.num_screens
        screen_id = id % self.num_screens
        batch = self.UpdateScreensBatch(batch_id)
        
        return batch[..., screen_id]