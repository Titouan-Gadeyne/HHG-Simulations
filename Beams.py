import numpy as np

from scipy.special import genlaguerre, hermite


class Beam():
    def __init__(self, w0, lam):
        self.w0 = w0
        self.lam = lam
        self.k = 2*np.pi/self.lam
        self.zR = np.pi*self.w0**2/self.lam 

    "Transverse mode formulae"
    # Gaussian beam expressions
    def R(self, z):
        if z!=0:
            return z*(1 + (self.zR/z)**2)
        else:
            return np.inf # infinite radius of curvature at focus

    def Gouy(self, z):
        return np.arctan(z/self.zR)

    def w(self, z):
        return self.w0 * np.sqrt(1 + (z/self.zR)**2)

    def GBeam(self, x, y, z):
        r = np.sqrt(x**2 + y**2)
        return self.w0/self.w(z) * np.exp(-r**2/self.w(z)**2) * np.exp(1j*( self.k*z + self.k*r**2/(2*self.R(z)) - self.Gouy(z) ) )

    # Laguerre-Gaussian beam
    def LGBeam(self, x, y, z, l, p):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        C = np.sqrt( 2*np.math.factorial(p) / (np.pi + np.math.factorial(p+np.abs(l)) ) ) # normalization factor
        return self.GBeam(x, y, z) \
                * C * genlaguerre(p, np.abs(l))(2*r**2/self.w(z)**2) * np.exp(-1j*l*theta) * (r*np.sqrt(2)/self.w(z))**np.abs(l) * np.exp(-1j*self.Gouy(z)*(np.abs(l)+2*p))

    # Hermite-Gaussian beam
    def HGBeam(self, x, y, z, m, n):
        r = np.sqrt(x**2 + y**2)
        return self.GBeam(x, y, z) \
                * hermite(m)(np.sqrt(2)*x/self.w(z)) * hermite(n)(np.sqrt(2)*y/self.w(z)) * np.exp(-1j*self.Gouy(z)*(m+n))
    
    "Propagation"
    # Spherical phase factor
    def SphFactor(self, x, y, z):
        return np.exp((x**2 + y**2)*1j*self.k/(2*z))

    # Fresnel propagation
    def Fresnel(self, field_0, grid_0, d):

        x, y = grid_0[0][0], grid_0[1].T[0] # retrieve axes from meshgrid
        Lx, Ly = x[-1]-x[0], y[-1]-y[0]
        Nx, Ny = len(x), len(y)

        field = field_0 * self.SphFactor(*grid_0, d)

        field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field))) # fft assumes origin of axis at top left corner, need to fftshift beforehand
        field = field * (Lx/(Nx-1)) * (Ly/(Ny-1)) # correct for the sampling rate (difference between continuous and discrete FT)

        # Compute the conjugate x and y axis. Their width is appx (N lam z / L)
        FT_x_axis = np.fft.fftshift(np.fft.fftfreq(Nx, Lx/(Nx-1)))*self.lam*d
        FT_y_axis = np.fft.fftshift(np.fft.fftfreq(Ny, Ly/(Ny-1)))*self.lam*d
        grid_d = np.meshgrid(FT_x_axis, FT_y_axis)

        field_d = field / (1j*self.lam*d) * np.exp(1j*self.k*d) * self.SphFactor(*grid_d, d)
        
        return field_d, grid_d

    # Fraunhofer propagation, equivalent to Fresnel without the spherical wavefront factors
    def Fraunhofer(self, field_0, grid_0, d):

        x, y = grid_0[0][0], grid_0[1].T[0] # retrieve axes from meshgrid
        Lx, Ly = x[-1]-x[0], y[-1]-y[0]
        Nx, Ny = len(x), len(y)

        field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_0))) # fft assumes origin of axis at top left corner, need to fftshift beforehand
        field = field * (Lx/(Nx-1)) * (Ly/(Ny-1)) # correct for the sampling rate (difference between continuous and discrete FT)

        # Compute the conjugate x and y axis. Their width is appx (N lam z / L)
        FT_x_axis = np.fft.fftshift(np.fft.fftfreq(Nx, Lx/(Nx-1)))*self.lam*d
        FT_y_axis = np.fft.fftshift(np.fft.fftfreq(Ny, Ly/(Ny-1)))*self.lam*d
        grid_d = np.meshgrid(FT_x_axis, FT_y_axis)

        field_d = field / (1j*self.lam*d) * np.exp(1j*self.k*d)
        
        return field_d, grid_d

    # Adds a phase factor to simulate passing through a lens
    def Lens(self, field, grid, f):
        return field * self.SphFactor(*grid, -f) # minus sign to have converging lens for f>0

    "Grid generators"
    # Returns a meshgrid with desired L and N
    def Grid(self, Lx, Ly, Nx, Ny):
        x_axis = np.linspace(-Lx/2, Lx/2, Nx) 
        y_axis = np.linspace(-Ly/2, Ly/2, Ny) 
        return np.meshgrid(x_axis, y_axis)

    # For L and N the desired width and resolution of the image at distance d, returns the grid to use at z=0
    # For use with Fraunhofer propagation, use d=1 to specify L as a divergence angle
    def FocusGrid(self, Lx, Ly, Nx, Ny, d):
        lx = self.lam*d*(Nx-1)**2/(Nx*Lx)
        ly = self.lam*d*(Ny-1)**2/(Ny*Ly)

        x_axis = np.linspace(-lx/2, lx/2, Nx) 
        y_axis = np.linspace(-ly/2, ly/2, Ny) 

        return np.meshgrid(x_axis, y_axis)

class Mask():
    def __init__(self):
        pass

    def Iris(self, x, y, R):
        r = np.sqrt(x**2 + y**2)
        return np.where((r<R), 1, 0)

    def ZeroPi(self, x, y):
        phase = np.sign(y)*np.pi/2 + np.pi/2
        return np.exp(1j*phase)

