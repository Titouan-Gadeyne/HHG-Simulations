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

def Extent(grid):
    x, y = grid[0][0], grid[1].T[0] # retrieve axes from meshgrid
    Lx, Ly = x[-1]-x[0], y[-1]-y[0]
    Nx, Ny = len(x), len(y)
    extent=[x[0], x[-1], y[0], y[-1]]
    return extent

def ShowHHG(IRfield, XUV_NF, XUV_FF, grid_NF, grid_FF):
    extent_NF = Extent(grid_NF)
    extent_FF = Extent(grid_FF)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(np.abs(IRfield), extent=extent_NF, aspect='equal', vmin=0, cmap='hot')
    ax[1].imshow(np.abs(XUV_NF), extent=extent_NF, aspect='equal', vmin=0, cmap='Purples_r')
    ax[2].imshow(np.abs(XUV_FF), extent=extent_FF, aspect='equal', vmin=0, cmap='gist_stern')

    return fig

def OrdersLineout(XUV_FF, grid_FF, q, theta):
    p_list = np.linspace(0, q+1, q+2)
    angle_list = theta*p_list/q
    x, y = grid_FF[0][0], grid_FF[1].T[0]
    lineouts = [ np.abs(XUV_FF[np.argmin(np.abs(y-a))]) for a in angle_list ]

    fig, ax = plt.subplots(1, 1)

    for i in range(len(lineouts)):
        ax.plot(x, lineouts[i])

    return fig


