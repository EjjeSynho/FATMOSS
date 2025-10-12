#%%
%reload_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from phase_generator import *

# Parameters
D  = 8.0  # Size of the phase screen [m]
r0 = 0.1  # Fried parameter [m]
L0 = 25.0 # Outer scale [m]

dx = r0 / 3.0 # Spatial sampling interval [m/pixel], make sure r0 is Nyquist sampled
dt = 0.001 # Time step [s/step]

wind_speed = 40*0 # [m/s]
wind_direction = 45 # [degree]
boiling_factor = 1500 # [a.u], need to figure them out

screen_generator = CascadedPhaseGenerator(D, dx, dt, batch_size=config['batch_size'], n_cascades=3)
screen_generator.AddLayer(1.0, r0, L0, wind_speed, wind_direction, boiling_factor)
# screen_generator.AddLayer(0.5, r0/2., L0, wind_speed*2, wind_direction*4, boiling_factor)

#%%
if GPU_flag:
    start = cp.cuda.Event()
    end   = cp.cuda.Event()

total_time = 0
screens_cascade = []

iters = 1000

for i in tqdm(range(iters)):
    if GPU_flag:
        start.record()
    else:
        start = time.time()
        
    screens_cascade.append(screen_generator.GetScreenByTimestep(i))
    
    if GPU_flag:
        end.record()
        end.synchronize()
        total_time += cp.cuda.get_elapsed_time(start, end)  # Time in [ms]
    else:
        end = time.time()
        total_time += (end-start) * 1000

screens_cascade = xp.dstack(screens_cascade)

print(f"Total elapsed time: {np.round(total_time/1e3,1)} [s]")
print(f"Time per screen: {np.round(total_time/iters).astype(xp.uint32)} [ms]")

#%%

SaveVideo(screens_cascade)


#%%

screens_cascade.shape


#%%
from misc import SaveGIF_RGB, SaveGIF
from PIL import Image
from scipy.ndimage import zoom as zoomer
from interpolate import Interpolator
import os

save_folda = "C:/Users/akuznets/Desktop/presa_buf/"
interp = Interpolator()

A = screen_generator.buffa_raw[...,0,:].get()
B = [ x[...,0].get() for x in screen_generator.buffa_raw_cropped ]
C = screen_generator.buffa_interp[...,0,:].get()
D = screen_generator.buffa_raw_cropped[-1]

D_p, D_s = interp.periodic_smooth_decomp_batch(D)
D_s, D_p = D_s[...,0].get(), D_p[...,0].get()

result = np.dstack([screen_generator.buffa_interp[...,0,i].get() for i in range(screen_generator.n_cascades)]).sum(axis=-1)

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

np.save(save_folda + "screens_closed_loop.npy", screens_cascade.get())


#%%
PSDs_cascade = []
rad2nm = 500.0 / 2.0 / np.pi # [nm/rad]

for i in range(screen_generator.n_cascades):
    phase_scr = screen_generator.buffa_raw[...,i]
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

PSD_out = PSD_to_phase(screens_cascade)

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

plt.savefig(save_folda + "PSD_reconstruction_reconst.png", dpi=300)

# %%
