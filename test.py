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

wind_speed = 40 # [m/s]
wind_direction = 15 # [degree]
boiling_factor = 500 # [a.u], need to figure them out

screen_generator = CascadedPhaseGenerator(D, dx, dt, batch_size=config['batch_size'], n_cascades=4)
screen_generator.AddLayer(1.0, r0, L0, wind_speed, wind_direction, boiling_factor)
# screen_generator.AddLayer(0.5, r0/2., L0, wind_speed*2, wind_direction*4, boiling_factor)

#%%
if GPU_flag:
    start = cp.cuda.Event()
    end   = cp.cuda.Event()

total_time = 0
screens_cascade = []

for i in tqdm(range(1000)):
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
print(f"Time per screen: {np.round(total_time/screen_generator.num_screens).astype(xp.uint16)} [ms]")

#%%
PSDs_cascade = []
rad2nm = 500.0 / 2.0 / np.pi # [nm/rad]

for i in range(screen_generator.n_cascades):
    phase_scr = screen_generator.buffa_raw[...,i]
    PSD_ = PSD_to_phase(phase_scr)
    PSD_ = PSD_ * rad2nm**2 * screen_generator.df[...,i]**2 * 9**(2*(i-1))
    PSD_[PSD_ < xp.median(PSD_)/4] = 1e-12
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

for i in range(screen_generator.n_cascades):
    _ = radialize_PSD(PSDs_cascade[i], screen_generator.f[...,i], label=f'Cascade #{i+1}', plot=True, fill=True, color=colors_[i])

_ = radialize_PSD(PSD_ultimate, f_over, 'Ultimate PSD', linewidth=1.5, color='black', linestyle='dashed')
_ = radialize_PSD(PSD_out, screen_generator.f[...,0], 'Reconstructed PSD', linewidth=2)

plt.grid()
plt.title('Large-scale PSD vs. PSD cascade reconstruction')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-3, 1e8])
plt.xlim([5e-3, 15])
plt.xlabel('Spatial frequency [1/m]')
plt.ylabel(r'PSD $[nm^2 / \hspace{0.5} m^2]$')
plt.show()
# %%
