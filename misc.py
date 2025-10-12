import json
import numpy as np
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import rescale


with open('settings.json') as f:
    config = json.load(f)

global GPU_flag
GPU_flag = config['use_GPU']


if GPU_flag:
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import zoom
        import cupyx.scipy.fft as gfft
        
    except ImportError:
        warnings.warn('CuPy is not installed. Using NumPy backend.')
        GPU_flag = False
        xp = np
        from scipy.ndimage import zoom
    else:
        xp = cp
else:
    xp = np
    from scipy.ndimage import zoom

import matplotlib.pyplot as plt
from PIL import Image


def SaveVideo(frames_batch, output_video_path='output/screens.mp4'):
    from matplotlib import cm
    from skimage.transform import rescale
    from tqdm import tqdm
    import warnings

    try:
        import cv2
    except ImportError:
        warnings.warn('OpenCV is not installed. Cannot write video.')
        return

    colormap = cm.viridis
    scale_factor = 2
    normalizer = xp.abs(frames_batch).max()

    height, width, layers = frames_batch.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
    video  = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width*scale_factor, height*scale_factor))

    print('Writing video...')
    for i in tqdm(range(layers)):
        buf = (frames_batch[..., i] + normalizer) / 2 / normalizer
        if GPU_flag:
            buf = cp.asnumpy(buf)
        buf = rescale(buf, scale_factor, order=1)
        frame = (colormap(buf) * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()
    

def SaveGIF(array, duration=1e3, scale=1, path='test.gif', colormap=plt.cm.viridis):
    # If the input is an array or a tensor, we need to convert it to a list of PIL images first
    if type(array) == np.ndarray:
        gif_anim = []
        array_ = array.copy()

        if array.shape[0] != array.shape[1] and array.shape[1] == array.shape[2]:
            array_ = array_.transpose(1,2,0)

        for layer in np.rollaxis(array_, 2):
            buf = layer/layer.max()
            if scale != 1.0:
                buf = rescale(buf, scale, order=0)
            gif_anim.append( Image.fromarray(np.uint8(colormap(buf)*255)) )
    else:
        # In this case, we can directly save the list of PIL images
        gif_anim = array

    # gif_anim[0].save(path, save_all=True, append_images=gif_anim[1:], optimize=False, quality=100, duration=duration, loop=0)
    gif_anim[0].save(path, save_all=True, append_images=gif_anim[1:], optimize=False, compress_level=0, duration=duration, loop=0)


def SaveGIF_RGB(images_stack, duration=1e3, downscale=4, path='test.gif'):
    gif_anim = []
    
    def remove_transparency(img, bg_colour=(255, 255, 255)):
        alpha = img.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", img.size, bg_colour + (255,))
        bg.paste(img, mask=alpha)
        return bg
    
    for layer in tqdm(images_stack):
        im = Image.fromarray(np.uint8(layer*255))
        gif_anim.append( remove_transparency(im) )
        gif_anim[0].save(path, save_all=True, append_images=gif_anim[1:], optimize=True, duration=duration, loop=0)

    
def mask_circle(N, r, center=(0,0), centered=True):
    """Generates a circular mask of radius r in a grid of size N."""
    factor = 0.5 * (1-N%2)
    if centered:
        coord_range = np.linspace(-N//2+N%2+factor, N//2-factor, N)
    else:
        coord_range = np.linspace(0, N-1, N)
    xx, yy = np.meshgrid(coord_range-center[1], coord_range-center[0])
    pupil_round = np.zeros([N, N], dtype=np.int32)
    pupil_round[np.sqrt(yy**2+xx**2) < r] = 1
    return pupil_round


def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt( (x-center[0])**2 + (y-center[1])**2 )
    r = r.astype('int')
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile
   
 
def PSD_to_phase(phase_batch): 
    N_, _, num_screens = phase_batch.shape
    hanning_window = (np.hanning(N_).reshape(-1, 1) * np.hanning(N_))[..., None]
    hanning_window = xp.array(hanning_window, dtype=phase_batch.dtype) * 1.6322**2 # Corrective factor

    if phase_batch.dtype == xp.float32:
        datacomplex = xp.complex64
    else:
        datacomplex = xp.complex128

    temp_mean = xp.zeros([N_, N_], dtype=phase_batch.dtype)
    FFT_batch = xp.zeros([N_, N_, num_screens], dtype=datacomplex)

    if GPU_flag:
        plan = gfft.get_fft_plan(FFT_batch, axes=(0,1), value_type='C2C')

        FFT_batch = gfft.fftshift(
            gfft.fft2(gfft.fftshift(phase_batch*hanning_window, axes=(0,1)), axes=(0,1), plan=plan) / N_**2, axes=(0,1)        
        )
    else:
        FFT_batch = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(phase_batch*hanning_window, axes=(0,1)), axes=(0,1)) / N_**2, axes=(0,1)        
        )
        
    temp_mean = FFT_batch.mean(axis=(0,1), keepdims=True)
    return 2 * xp.mean( xp.abs(FFT_batch-temp_mean)**2, axis=2 )


'''
def PSD_to_phase_advanced(phase_batch):
    N_, _, N_screens = phase_batch.shape
    batch_size = 32
    pad_size   = N_//2+N_%2
    N_batches  = N_screens // batch_size

    hanning_window = cp.array( (np.hanning(N_).reshape(-1, 1) * np.hanning(N_))[..., None], dtype=datafloat ) * 1.6322**2

    temp_mean = cp.zeros([pad_size*2+N_, pad_size*2+N_], dtype=datafloat)
    FFT_batch = cp.zeros([pad_size*2+N_, pad_size*2+N_, batch_size], dtype=datacomplex)
    variance_batches = cp.zeros([pad_size*2+N_, pad_size*2+N_], dtype=datafloat)

    plan = get_fft_plan(FFT_batch, axes=(0,1), value_type='C2C') # for batched, C2C, 2D transform

    pad_dims = ( (pad_size, pad_size), (pad_size,pad_size), (0, 0) )

    for i in range(N_batches):
        buf = cp.pad( phase_batch[..., i*batch_size:(i+1)*batch_size]*hanning_window, pad_dims, 'constant', constant_values=(0,0))
        FFT_batch = gfft.fftshift(
            gfft.fft2(gfft.fftshift(buf, axes=(0,1)), axes=(0,1), plan=plan) / N_**2, axes=(0,1)        
        )
        temp_mean = FFT_batch.mean(axis=(0,1), keepdims=True)
        variance_batches += 2 * cp.mean( cp.abs(FFT_batch-temp_mean)**2, axis=2 )
        variance_batches = variance_batches / N_batches

    return cupyx.scipy.ndimage.zoom(variance_batches, N_/(pad_size*2+N_), order=3)
'''




'''
N_new = screen_generator.N
_,_,f_0, df_0 = screen_generator.freq_array(N_new, dx=r0/3.0)

rad2nm = 500.0 / 2.0 / np.pi # [nm/rad]

stats = []

masko = mask_circle(N_new, N_new//8, centered=True)
PSD_testo = screen_generator.vonKarmanPSD(f_0, r0, L0) * df_0**2 * rad2nm**2 * (1-masko)
select_middle = lambda N: np.s_[N//2-N//6:N//2+N//6+N%2, N//2-N//6:N//2+N//6+N%2]
croppah = select_middle(N_new//3)

for i in tqdm(range(200)):
    testo = screen_generator.generate_phase_screen(PSD_testo, N_new)
    stats.append(testo[croppah])

stats = np.dstack(stats)

# plt.imshow(testo)


phase_screens_batch = cp.array(stats, dtype=cp.float32)
# phase_screens_batch = stats

def zoomax3(phase_screens_batch, iters):
    if iters > 0:
        factor = 3**iters
        original_height, original_width, num_screens = phase_screens_batch.shape
        new_height = int(original_height * factor)
        new_width  = int(original_width  * factor)

        fft_batch = xp.fft.fft2(phase_screens_batch, axes=(0, 1))
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
        return phase_screens_batch

resulto = zoomax3(phase_screens_batch, 2)
resulto -= resulto.mean(axis=(0,1), keepdims=True)
# resulto = cp.array(resulto, dtype=cp.float32)

plt.imshow(testo)
plt.show()
plt.imshow(resulto[...,0].get())
plt.show()

PSDec = PSD_to_phase(resulto)


plt.imshow(np.log(PSDec.get()))
'''