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


def SaveVideo(frames_batch, output_video_path):
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
    
    # Automatically detect if array is CuPy or NumPy
    if GPU_flag and hasattr(frames_batch, '__array_module__') and frames_batch.__array_module__.__name__ == 'cupy':
        normalizer = cp.abs(frames_batch).max()
        is_cupy = True
    else:
        normalizer = np.abs(frames_batch).max()
        is_cupy = False

    height, width, layers = frames_batch.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
    video  = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width*scale_factor, height*scale_factor))

    print('Writing video...')
    for i in tqdm(range(layers)):
        buf = (frames_batch[..., i] + normalizer) / 2 / normalizer
        if is_cupy:
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


def PrintGPUInfo():
    if GPU_flag:
        print(f"GPU Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(f"GPU Memory Total: {cp.cuda.runtime.memGetInfo()[1] / 1024**3:.1f} GB")
        print(f"GPU Memory Free:  {cp.cuda.runtime.memGetInfo()[0] / 1024**3:.1f} GB")
        print(f"GPU Memory Used:  {(cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0]) / 1024**3:.1f} GB")
    else:
        print("GPU not available or not enabled")


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

