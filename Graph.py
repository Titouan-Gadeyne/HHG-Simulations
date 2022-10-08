import numpy as np
import matplotlib.pyplot as plt

def ShowField(field, grid):
    x, y = grid[0][0], grid[1].T[0] # retrieve axes from meshgrid
    Lx, Ly = x[-1]-x[0], y[-1]-y[0]
    Nx, Ny = len(x), len(y)

    extent=[x[0], x[-1], y[0], y[-1]]
    aspect = Lx/Ly

    fig, ax = plt.subplots(1, 2, tight_layout=True)
    ax[0].imshow(np.abs(field), extent=extent, aspect=aspect, vmin=0, cmap='hot')
    ax[1].imshow(np.angle(field), extent=extent, aspect=aspect, vmin=-np.pi, vmax=np.pi, cmap='hsv')

    return fig


