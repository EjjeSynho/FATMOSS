from misc import xp, zoom

'''
Periodic & smooth image decomposition

Originally based on: https://github.com/jacobkimmel/ps_decomp

References
----------
Periodic Plus Smooth Image Decomposition
Moisan, L. J Math Imaging Vis (2011) 39: 161.
doi.org/10.1007/s10851-010-0227-1
'''


'''
fft_plans_f,  fft_plans_d  = {}, {}
ifft_plans_f, ifft_plans_d = {}, {}

data_type    = cp.float64
complex_type = cp.complex128

for i in range(screen_generator.n_cascades):
    img_size   = screen_generator.N // 3**i
    shape_plan = (img_size, img_size, screen_generator.num_screens)
    fft_plans_d  [img_size] = gfft.get_fft_plan(cp.empty(shape_plan, dtype=cp.float64), axes=(0, 1))
    ifft_plans_d [img_size] = gfft.get_fft_plan(cp.empty(shape_plan, dtype=cp.complex128), axes=(0, 1), value_type='C2C')
    fft_plans_f  [img_size] = gfft.get_fft_plan(cp.empty(shape_plan, dtype=cp.float32), axes=(0, 1))
    ifft_plans_f [img_size] = gfft.get_fft_plan(cp.empty(shape_plan, dtype=cp.complex64), axes=(0, 1), value_type='C2C')
    
    print(f"FFT plan for {img_size}x{img_size} images created.")

def get_plan(N, dtype=cp.float64):
    if   dtype == cp.float64:
        return fft_plans_d[N]
    elif dtype == cp.float32:
        return fft_plans_f[N]
    elif dtype == cp.complex128:
        return ifft_plans_d[N]
    elif dtype == cp.complex64:
        return ifft_plans_f[N]
'''


class Interpolator:
    
    def __init__(self):
        pass
    
       
    def __u2v_batch(self, u):
        '''Converts a batch of images `u` into images `v` in an optimized manner.

        Parameters
        ----------
        u : xp.ndarray
            [W, H, N] image batch.

        Returns
        -------
        v : xp.ndarray
            [W, H, N] image batch, zeroed except for the outermost rows and cols.
        '''
        # Initialize v directly with the required computations for edges
        v = xp.zeros_like(u)

        # Compute differences for top and bottom edges across all batches
        v[0,  :, :] = u[-1, :, :] - u[0,  :, :]
        v[-1, :, :] = u[0,  :, :] - u[-1, :, :]

        # Compute differences for left and right edges across all batches
        v[:, 0,  :] += u[:, -1, :] - u[:,  0, :]
        v[:, -1, :] += u[:,  0, :] - u[:, -1, :]

        return v


    def __v2s_batch(self, v_hat):
        '''Computes the maximally smooth component of `u`, `s` from `v`, for a batch in an optimized manner.

        Parameters
        ----------
        v_hat : xp.ndarray
            [W, H, N] DFT of v batch.
        '''
        M, N, _ = v_hat.shape
        
        q = xp.arange(M).astype(v_hat.dtype)
        r = xp.arange(N).astype(v_hat.dtype)

        den = (2 * xp.cos((2 * xp.pi * q[:, None]) / M) + 
            2 * xp.cos((2 * xp.pi * r[None, :]) / N) - 4)

        s = xp.nan_to_num(v_hat / den[:, :, None])

        # Set the DC component to 0 for all images in the batch
        s[0, 0, :] = 0

        return s


    def periodic_smooth_decomp_batch(self, I):
        '''Performs periodic-smooth image decomposition on a batch of images using FFT plans.

        Parameters
        ----------
        I : xp.ndarray
            [W, H, N] image batch.

        Returns
        -------
        P : xp.ndarray
            [W, H, N] batch, periodic portion.
        S : xp.ndarray
            [W, H, N] batch, smooth portion.
        '''
        u = I.astype(xp.float64)
        v = self.__u2v_batch(u)  # Assuming the optimized version of u2v_batch is used

        # Use the plan for FFT and IFFT operations
        # with get_plan(v.shape[0], v.dtype):
        v_fft = xp.fft.fftn(v, axes=(0, 1))
            
        s = self.__v2s_batch(v_fft)
        
        # with get_plan(s.shape[0], s.dtype):
        s_i = xp.fft.ifftn(s, axes=(0, 1))
            
        s_f = xp.real(s_i)
        p = u - s_f  # u = p + s
        
        return p, s_f


    def zoom_interp(self, x, iters=0, interp_order=3):
        """Scales up the resolution of the phase screens stack 'x' by a factor of 3."""	
        if iters > 0:
            zoom_factor = (3**iters, 3**iters, 1)
            return zoom(x, zoom_factor, order=interp_order)
        else:
            return x


    def zoom_FFT(self, x, iters=0):  
        if iters == 0:
            return x
        
        factor = 3**iters
        zoom_factor = (factor, factor, 1)

        original_height, original_width, num_screens = x.shape
        new_height = int(original_height * factor)
        new_width  = int(original_width  * factor)

        x_p, x_s = self.periodic_smooth_decomp_batch(x)

        # plan = get_plan(original_width, x_p.dtype)
        # fft_shifted_batch = gfft.fftshift(gfft.fft2(x_p, axes=(0, 1), plan=plan), axes=(0, 1))
        fft_shifted_batch = xp.fft.fftshift(xp.fft.fft2(x_p, axes=(0, 1)), axes=(0, 1))

        padded_fft_batch = xp.zeros((new_height, new_width, num_screens), dtype=xp.complex64)
        pad_height_start = (new_height - original_height) // 2
        pad_width_start  = (new_width  - original_width)  // 2

        padded_fft_batch[
            pad_height_start : pad_height_start + original_height,
            pad_width_start  : pad_width_start  + original_width,
        :] = fft_shifted_batch

        # plan = get_plan(new_width, padded_fft_batch.dtype)
        # X_p  = gfft.ifft2(gfft.ifftshift(padded_fft_batch, axes=(0, 1)), axes=(0, 1), plan=plan)
        X_p = xp.fft.ifft2(xp.fft.ifftshift(padded_fft_batch, axes=(0, 1)), axes=(0, 1))
        X_s = zoom(x_s, zoom_factor, order=1)
        return xp.real(X_p) * factor**2 + X_s
