import json
import time
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from skimage.transform import rescale
from IPython.display import HTML
from typing import Dict, Any, Optional
from PIL import Image

with open('settings.json') as f:
    config = json.load(f)

global GPU_flag
GPU_flag = config['use_GPU']

import numpy as np
# Attempt enabling the GPU acceleration
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


rad2mas  = 3600 * 180 * 1000 / np.pi
rad2arc  = rad2mas / 1000
deg2rad  = np.pi / 180
asec2rad = np.pi / 180 / 3600


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
    """ Saves a GIF animation from a 3D array or a list of PIL images. """
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


def RenderVideo(   
    screens_sequence,
    max_frames=200,
    interval=50,
    frame_step=1,
    start=0,
    dt=1.0,
    title=''
):
    '''
    Description:
        Creates an animated visualization of atmospheric turbulence phase screens.

    Parameters:
        screens_sequence — 3D NumPy array (H, W, T) containing phase screens over time.
        max_frames — maximum number of frames to include in the animation (default: 200).
        interval — delay between frames in milliseconds, controls playback speed (default: 50).
        frame_step — step size for selecting every N-th frame from the sequence (default: 1).
        start — starting frame index within the sequence (default: 0).
        dt — time step between original frames, used for labeling in seconds (default: 1.0).
        title — plot title.

    Returns:
        HTML — an embeddable HTML animation object (for Jupyter notebooks).
    '''

    # Build the list of frame indices we intend to show (start, step)
    all_indices = np.arange(start, screens_sequence.shape[2], frame_step)
    frame_indices = all_indices[:max_frames]
    n_frames = len(frame_indices)
    
    if n_frames == 0:
        raise ValueError("No frames selected: check 'start' and 'frame_step'.")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    # Initialize the image
    vmin = np.percentile(screens_sequence, 1)
    vmax = np.percentile(screens_sequence, 99.999)
    im = ax.imshow(
        screens_sequence[:, :, frame_indices[0]],
        cmap='viridis',
        animated=True,
        vmin=vmin,
        vmax=vmax
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Phase [nm]', rotation=270, labelpad=20)

    # Labels
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Pixels')

    if not title == '':
        title += '\n'

    # Animation function
    def animate(i):
        idx = frame_indices[i]
        im.set_array(screens_sequence[:, :, idx])
        # i counts shown frames (0..n_frames-1); idx is the original frame number
        ax.set_title(title +
            f'Frame {i}/{n_frames-1}  |  Original frame {idx}  (t={idx*dt:.3f}s)',
            fontsize=14, pad=20
        )
        return [im]

    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=n_frames,
                                  interval=interval, blit=True, repeat=True)

    plt.tight_layout()
    plt.close(fig)  # Prevent static display

    return HTML(ani.to_jshtml())


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
    # Check if input is numpy
    if not hasattr(phase_batch, 'device') and GPU_flag:
        phase_batch = xp.array(phase_batch)
    
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



def BenchmarkScreensGenerator(
    screen_generator,
    iters: int,
    use_tqdm: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run screen_generator over `iters` timesteps, stack results into a 3D array,
    and measure timing on CPU or GPU.

    Parameters
    ----------
    screen_generator : object with .GetScreenByTimestep(i) -> 2D array-like
        Must also provide attributes `n_layers` and `n_cascades` (ints) for stats
        (defaults to 1 if missing).
    iters : int
        Number of timesteps to generate.
    use_tqdm : bool, default True
        If True, wraps the loop with tqdm for a progress bar.

    Returns
    -------
    result : dict
        {
          "screens": np.ndarray (H, W, iters) on host memory,
          "total_time_ms": float,
          "time_per_screen_ms": float,
          "time_per_screen_per_layer_ms": float,
          "time_per_screen_per_layer_per_cascade_ms": float,
        }
    """
    if xp is None:
        raise ValueError("Please pass `xp` as numpy or cupy.")

    if GPU_flag and cp is None:
        raise ValueError("GPU_flag=True requires `cp` (CuPy) to be provided.")

    try:
        from tqdm import tqdm as _tqdm
    except Exception:
        _tqdm = lambda x, **_: x  # fallback no-op

    prog = _tqdm(range(iters)) if use_tqdm else range(iters)

    if GPU_flag:
        start_evt = cp.cuda.Event()
        end_evt   = cp.cuda.Event()

    total_time_ms = 0.0
    screens_list = []

    for i in prog:
        if GPU_flag:
            start_evt.record()
        else:
            t0 = time.perf_counter()

        screens_list.append(screen_generator.GetScreenByTimestep(i))

        if GPU_flag:
            end_evt.record()
            end_evt.synchronize()
            total_time_ms += cp.cuda.get_elapsed_time(start_evt, end_evt)  # ms
        else:
            total_time_ms += (time.perf_counter() - t0) * 1000.0

    # Stack along the third axis to shape (H, W, iters)
    screens_stack = xp.dstack(screens_list)

    # Ensure host (NumPy) output for downstream tools/animation
    if hasattr(screens_stack, "get"):
        screens_host = screens_stack.get()
    else:
        screens_host = screens_stack

    n_layers   = getattr(screen_generator, "n_layers", 1) or 1
    n_cascades = getattr(screen_generator, "n_cascades", 1) or 1

    result = {
        "screens": screens_host,
        "total_time_ms": total_time_ms,
        "time_per_screen_ms": total_time_ms / iters,
        "time_per_screen_per_layer_ms": total_time_ms / (iters * n_layers),
        "time_per_screen_per_layer_per_cascade_ms": total_time_ms / (iters * n_layers * n_cascades),
    }
    
    if verbose:
        print(f"Total elapsed time: {result['total_time_ms']/1e3:.1f} s")
        print(f"Time per screen: {result['time_per_screen_ms']:.1f} ms")
        print(f"Per layer: {result['time_per_screen_per_layer_ms']:.1f} ms")
        print(f"Per layer per cascade: {result['time_per_screen_per_layer_per_cascade_ms']:.1f} ms")
    return result


def mask_circle(N, r, center=(0,0), centered=True):
    """Generates a circular mask of radius r in a grid of size N."""
    factor = 0.5 * (1-N%2)
    if centered:
        coord_range = xp.linspace(-N//2+N%2+factor, N//2-factor, N)
    else:
        coord_range = xp.linspace(0, N-1, N)
    xx, yy = xp.meshgrid(coord_range-center[1], coord_range-center[0])
    pupil_round = xp.zeros([N, N], dtype=xp.int32)
    pupil_round[xp.sqrt(yy**2+xx**2) < r] = 1
    return pupil_round


circular_pupil = lambda resolution: mask_circle(N=resolution, r=resolution/2, centered=True)
