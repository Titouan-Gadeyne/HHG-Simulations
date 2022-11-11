import numpy as np
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

    fig, ax = plt.subplots(1, 3, figsize=(10, 8))
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

class Phase2D():
    # To put a 2D colormap on a 2D complex field
    def colorize(self, z, mode='Amplitude'):

        if mode=='Amplitude':
            r = (np.abs(z) / np.nanmax(np.abs(z)))
        elif mode=='Intensity':
            r = (np.abs(z) / np.nanmax(np.abs(z)))**2
        arg = np.angle(z) 

        h = (arg + np.pi)  / (2 * np.pi) + 0.5
        l = r/2 # linear brightness variation, peaks at 1/2
        s = 0.8

        c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
        c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
        c = c.transpose(1,2,0)
        return c

    def key_inset(self, mode, ax):
        amp = np.linspace(0, 1, 100)
        phase = np.linspace(-np.pi, np.pi, 100)
        z = np.outer(amp, np.exp(1j*phase))
        key = self.colorize(z=z.T, mode=mode)

        ins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(1.05, 0.15, .08, .7),
                   bbox_transform=ax.transAxes, loc=2, borderpad=0)
        ins.imshow(key, extent=[0, 1, -np.pi, np.pi])
        ins.tick_params(left=False, right=True, labelleft=False, labelright=True)
        ins.set_xticks([0, 1], [0, 1])
        ins.set_yticks([-np.pi, 0, np.pi], [r"-$\pi$", "$0$", r"$\pi$"])
        if mode=="Amplitude":
            ins.set_xlabel(r'$|E|$')
        elif mode=="Intensity":
            ins.set_xlabel(r'$|E|^2$')
        ins.set_ylabel(r'$\arg{E}$')
        ins.yaxis.set_label_position("right")

        return ins

    def ShowFieldPhase(self, field, grid, mode):
        x, y = grid[0][0], grid[1].T[0] # retrieve axes from meshgrid
        Lx, Ly = x[-1]-x[0], y[-1]-y[0]
        Nx, Ny = len(x), len(y)

        extent=[x[0], x[-1], y[0], y[-1]]
        aspect = Lx/Ly

        # Create 2D colormap reference
        x = np.linspace(0, 1, 100).T
        y = np.exp(1j*np.linspace(-np.pi, np.pi, 100))
        map = np.outer(x, y)

        fig = plt.figure()

        gs = gridspec.GridSpec(12, 13)
        gs.update(wspace=0.05)

        ax0 = plt.subplot(gs[:, 0:12])
        ax1 = plt.subplot(gs[4:, 12])

        ax = [ax0, ax1]


        ax[0].imshow(self.colorize(z=field, mode=mode), extent=extent, aspect=aspect)
        ax[1].imshow(self.colorize(z=map, mode=mode), extent=[0, 1, -np.pi, np.pi], aspect='auto')

        ax[0].set_xlabel('x (m)')
        ax[0].set_ylabel('y (m)')

        ax[1].set_xlabel('Amplitude')
        ax[1].set_ylabel('Phase')
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        ax[1].set_xticks([0, 1], [0, 1])
        ax[1].set_yticks([-np.pi, np.pi], [r'-$\pi$', r'$\pi$'])

        return fig