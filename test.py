#%%
%reload_ext autoreload
%autoreload 2

import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from phase_generator import *
from atmospheric_layer import *
from misc import *

# Parameters
D  = 8.0  # Size of the phase screen [m]
# D  = 39.0  # Size of the phase screen [m]
r0 = 0.125 # Fried parameter [m]
L0 = 25.0 # Outer scale [m]

dx = r0 / 2 # Spatial sampling interval [m/pixel], make sure r0 is Nyquist sampled
dt = 0.001 # Time step [s/step]

simulation_scenarios = {
    'frozen_flow': {
        'wind_speed': 40,
        'wind_direction': 90,
        'boiling_factor': 0
    },
    'boiling': {
        'wind_speed': 0,
        'wind_direction': 90,
        'boiling_factor': 1500
    },
    'mixed': {
        'wind_speed': 40,
        'wind_direction': 90,
        'boiling_factor': 400
    },
}

# scenario = 'mixed'
# scenario = 'boiling'
scenario = 'frozen_flow'

wind_speed     = simulation_scenarios[scenario]['wind_speed']     # [m/s]
wind_direction = simulation_scenarios[scenario]['wind_direction'] # [degree]
boiling_factor = simulation_scenarios[scenario]['boiling_factor'] # [a.u], need to figure them out

#%%
screen_generator = PhaseScreensGenerator(D, dx, dt, batch_size=100, n_cascades=3, seed=142)
layer1 = Layer(1.0, 0, wind_speed, wind_direction, boiling_factor, lambda f: vonKarmanPSD(f, r0, L0), lambda f: SimpleBoiling(f, screen_generator.dx))
# layer2 = Layer(0.2, 0, wind_speed, wind_direction+90, boiling_factor, lambda f: vonKarmanPSD(f, r0, L0), lambda f: SimpleBoiling(f, screen_generator.dx))
# layer3 = Layer(0.2, 0, wind_speed, wind_direction+120, boiling_factor, lambda f: vonKarmanPSD(f, r0, L0), lambda f: SimpleBoiling(f, screen_generator.dx))
# layer4 = Layer(0.2, 0, wind_speed, wind_direction-30, boiling_factor, lambda f: vonKarmanPSD(f, r0, L0), lambda f: SimpleBoiling(f, screen_generator.dx))
# layer5 = Layer(0.2, 0, wind_speed, wind_direction-90, boiling_factor, lambda f: vonKarmanPSD(f, r0, L0), lambda f: SimpleBoiling(f, screen_generator.dx))
screen_generator.AddLayer(layer1)
# screen_generator.AddLayer(layer2)
# screen_generator.AddLayer(layer3)
# screen_generator.AddLayer(layer4)
# screen_generator.AddLayer(layer5)


print(screen_generator)


#%%
if GPU_flag:
    start = cp.cuda.Event()
    end   = cp.cuda.Event()

    total_time = 0
screens_sequence = []

iters = 1000

for i in tqdm(range(iters)):
    if GPU_flag:
        start.record()
    else:
        start = time.time()
        
    screens_sequence.append(screen_generator.GetScreenByTimestep(i))
    
    if GPU_flag:
        end.record()
        end.synchronize()
        total_time += cp.cuda.get_elapsed_time(start, end)  # Time in [ms]
    else:
        end = time.time()
        total_time += (end-start) * 1000

screens_sequence = xp.dstack(screens_sequence).get()

print(f"Total elapsed time: {total_time/1e3:.1f} [s]")
print(f"Time per screen: {total_time/iters:.1f} [ms]")
print(f"Time per screen per layer: {total_time/(iters*screen_generator.n_layers):.1f} [ms]")
print(f"Time per screen per layer per cascade: {total_time/(iters*screen_generator.n_layers*screen_generator.n_cascades):.1f} [ms]")

screen_generator.deallocate_gpu_memory(full_cleanup=True)

#%%
SaveVideo(screens_sequence, f'phase_screens_{scenario}.mp4')


#%%
for i in range(0,100,10):
    # plt.imshow(screens_sequence[...,i].get())
    plt.imshow(screens_sequence[...,i])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"C:/Users/akuznets/Desktop/poster_buf/phase_screen_{i}_CL.png", dpi=300)


#%%
from misc import SaveGIF_RGB, SaveGIF
from PIL import Image
from scipy.ndimage import zoom as zoomer
from interpolate import Interpolator
import os

save_folda = "C:/Users/akuznets/Desktop/poster_buf/"
interp = Interpolator()

A = screen_generator.raw_batch[...,0,:].get()
B = [ x[...,0].get() for x in screen_generator.raw_batch_cropped ]
C = screen_generator.batch_interp[...,0,:].get()
D = screen_generator.raw_batch_cropped[-1]

D_p, D_s = interp.periodic_smooth_decomp_batch(D)
D_s, D_p = D_s[...,0].get(), D_p[...,0].get()

result = np.dstack([screen_generator.batch_interp[...,0,i].get() for i in range(screen_generator.n_cascades)]).sum(axis=-1)

def to_gray_image(x):
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 255
    return x.astype(np.uint8)

def to_cm_image(x):
    colormap = plt.cm.viridis
    x = to_gray_image(x)
    return Image.fromarray(np.uint8(colormap(x)*255))

for i in range(A.shape[-1]):
    to_cm_image(A[...,i]).save(save_folda + f"cascade_n{i}.png")

for i in range(len(B)):
    z = B[0].shape[0] // B[i].shape[0]
    zoom_factor = (z, z)
    y = zoomer(B[i], zoom_factor, order=0)
    to_cm_image(y).save(save_folda + f"crop_n{i}.png")

for i in range(A.shape[-1]):
    to_cm_image(C[...,i]).save(save_folda + f"interp_n{i}.png")
    
to_cm_image(result).save(save_folda+"cascade_sum.png")

z = B[0].shape[0] // D_p.shape[0]
zoom_factor = (z, z)

to_cm_image( zoomer(D_p, zoom_factor, order=0) ).save(save_folda + "periodic.png")
to_cm_image( zoomer(D_s, zoom_factor, order=0) ).save(save_folda + "smooth.png")

to_cm_image( zoomer(D_p, zoom_factor, order=3) ).save(save_folda + "periodic_interp.png")
to_cm_image( zoomer(D_s, zoom_factor, order=3) ).save(save_folda + "smooth_interp.png")


phases   = screen_generator.phase_buf[-2]
phases_R = to_gray_image(phases.real.get())
phases_G = to_gray_image(phases.imag.get())
phases_B = np.zeros_like(phases_R)

phases_RGB   = np.stack([phases_R, phases_G, phases_B], axis=-2)
phases_stack = [phases_RGB[...,i] for i in range(phases_RGB.shape[-1])]

if not os.path.exists(save_folda+'pure_phase.gif'):
    SaveGIF_RGB(phases_stack, duration=1e0, downscale=1, path=save_folda+'pure_phase.gif')

PSD_sample = screen_generator.vonKarmanPSD(screen_generator.f[...,0], r0, L0)

PSD_sample = np.log10(PSD_sample)
PSD_sample[PSD_sample.shape[0]//2, PSD_sample.shape[1]//2] = PSD_sample.max()
PSD_sample -= PSD_sample.min()
PSD_sample /= PSD_sample.max()
plt.imsave(save_folda + "PSD_sample.png", PSD_sample, cmap='gray')

PSD_sqrt = np.sqrt(PSD_sample)
PSD_sqrt = (PSD_sqrt - PSD_sqrt.min()) / (PSD_sqrt.max() - PSD_sqrt.min())
PSD_sqrt = PSD_sqrt[..., None, None]
phases_RGB_PSD = phases_RGB * PSD_sqrt

phases_stack_PSD = [ phases_RGB_PSD[...,i]/255 for i in range(phases_RGB.shape[-1]) ]

if not os.path.exists(save_folda + 'PSD_and_phase.gif'):
    SaveGIF_RGB(phases_stack_PSD, duration=1e0, downscale=1, path=save_folda+'PSD_and_phase.gif')

SaveGIF_RGB(phases_stack_PSD, duration=1e0, downscale=1, path=save_folda+'PSD_and_phase.gif')  
    
#%%
from misc import PSD_to_phase

colors_ = ['tab:orange', 'tab:green', 'tab:purple', 'tab:pink', 'tab:olive', 'tab:cyan']
colors_ = [ np.array(plt.get_cmap('tab10')(i)[:-1]) for i in range(len(colors_)) ]

N = PSD_sample.shape[0]
N_ = N // 3

mask_basis = mask_circle(N, N/2, centered=True )
mask_outer = mask_circle(N, N_/2, centered=True )
mask_inner = mask_circle(N, N_/6, centered=True )

mask_2 = mask_outer - mask_inner
mask_3 = mask_basis - mask_outer
mask_1 = mask_inner

mask_1 = mask_1[..., None] * colors_[0][None, None, :]
mask_2 = mask_2[..., None] * colors_[1][None, None, :]
mask_3 = mask_3[..., None] * colors_[2][None, None, :]

PSD_sample_RGB = PSD_sample[...,None] * mask_1 + PSD_sample[...,None] * mask_2 + PSD_sample[...,None] * mask_3

plt.imsave(save_folda + "PSD_sample_RGB.png", PSD_sample_RGB**0.65)

#%%
PSDs_cascade = []
rad2nm = 500.0 / 2.0 / np.pi # [nm/rad]

for i in range(screen_generator.n_cascades):
    phase_scr = screen_generator.raw_batch[...,i]
    PSD_ = PSD_to_phase(phase_scr)
    PSD_ = PSD_ * rad2nm**2 * screen_generator.df[...,i]**2 * 9**(2*(i-1))
    PSD_[PSD_ < xp.median(PSD_)/4] = 1e-12 # Just for more beautiful display
    if GPU_flag:
        PSDs_cascade.append(PSD_.get())
    else:
        PSDs_cascade.append(PSD_)

oversampling_factor = 17
_,_,f_over, df_over = screen_generator.freq_array(screen_generator.N*oversampling_factor, dx)

rad2nm = 500.0 / 2.0 / np.pi # [nm/rad]

PSD_ultimate = screen_generator.vonKarmanPSD(f_over, r0, L0) * df_over**2 * rad2nm**2 * oversampling_factor**2

PSD_out = PSD_to_phase(screens_sequence)

if GPU_flag:
    PSD_out = cp.asnumpy(PSD_out)

def radialize_PSD(PSD, grid, label='', plot=True, fill=False, **kwargs):
    PSD_profile  = radial_profile(PSD, (PSD.shape[0]//2, PSD.shape[1]//2))[:PSD.shape[1]//2]
    grid_profile = grid[grid.shape[0]//2, grid.shape[1]//2:-1]
    if plot:
        plt.plot(grid_profile, PSD_profile, label=label, **kwargs)
    if fill:
        plt.fill_between(grid_profile, PSD_profile, alpha=0.3, **kwargs)

    return PSD_profile

colors_ = ['tab:orange', 'tab:green', 'tab:purple', 'tab:pink', 'tab:olive', 'tab:cyan']

#%
for i in range(screen_generator.n_cascades):
    _ = radialize_PSD(PSDs_cascade[i], screen_generator.f[...,i], label=f'Reconstructed cascade #{i+1}', plot=True, fill=True, color=colors_[i])

_ = radialize_PSD(PSD_ultimate, f_over, 'Ultimate PSD', linewidth=1.5, color='black', linestyle='dashed')
_ = radialize_PSD(PSD_out, screen_generator.f[...,0], 'Reconstructed PSD', linewidth=3)

plt.grid()
plt.title('Large-scale PSD vs. PSD cascade reconstruction')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-3, 1e8])
plt.xlim([5e-3, 15])
plt.xlabel('Spatial frequency [1/m]')
plt.ylabel(r'PSD $[nm^2 / \hspace{0.5} m^2]$')
# plt.show()

plt.savefig(save_folda + "PSD_reconstruction_reconst.pdf", dpi=300)

# %%
from cupyx.scipy.signal import correlate2d

screens_sequence = cp.asarray(screens_sequence)

corr = correlate2d(screens_sequence[..., 0], screens_sequence[..., 1], mode="full", boundary="fill")

# Plot the correlation result
plt.imshow(corr.get(), cmap='viridis')
plt.colorbar()

#%% 
# Example shapes

# Create two batches (H, W, N)
A = screens_sequence[..., 100:200]  # First N screens
B = screens_sequence[..., 101:201]  # Second N screens

# 1) Full cross-correlation maps per image (output (H+H-1, W+W-1, N))
corr_maps = correlate2d_fft_batched(A, B, return_numpy=True)
print("corr_maps shape:", corr_maps.shape)

#%%
corr_ = corr_maps.mean(axis=-1)
plt.imshow(corr_, cmap='viridis')
plt.colorbar()

# Empty cupy cache
cp._default_memory_pool.free_all_blocks()
cp.get_default_memory_pool().used_bytes()

#%%
A_ = A[...,::10]

vmin=A_.min()
vmax=A_.max()

# Print as horizontal subplots
fig, axs = plt.subplots(1, A_.shape[-1], figsize=(20, 5))
for i in range(A_.shape[-1]):
    axs[i].imshow(A_[...,i].get(), cmap='inferno', vmin=vmin, vmax=vmax)
    axs[i].axis('off')
plt.savefig(f"C:/Users/akuznets/Desktop/poster_buf/phase_screens_line_{scenario}.pdf", dpi=300)

