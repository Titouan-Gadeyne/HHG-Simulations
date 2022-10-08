import numpy as np

from scipy.special import genlaguerre, hermite


class Beam():
    def __init__(self, w0, lam):
        self.w0 = w0
        self.lam = lam
        self.k = 2*np.pi/self.lam
        self.zR = np.pi*self.w0**2/self.lam 

    # Gaussian beam expressions:
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
    


"PROPAGATION"

#def SphFactor(x, y, z): # Spherical phase factor in Fresnel propagation
#    return np.exp((x**2 + y**2)*1j*k/(2*z))

#def Propagate_Fresnel(field_0, z, grid):
    
#    # Multiply by the first exponential factor
#     field = field_0 * SphFactor(*grid, z)

#     # Perform the 2D FFT
#     field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field))) # fft assumes origin of axis at top left corner, need to fftshift beforehand
#     field = field * (Lx/(N-1)) * (Ly/(N-1)) # correct for the sampling rate (related to difference between continuous and discrete FT)

#     # Compute the conjugate x and y axis. Their width is appx (N lam z / L)
#     FT_x_axis = np.fft.fftshift(np.fft.fftfreq(N, Lx/(N-1)))*lam*z
#     FT_y_axis = np.fft.fftshift(np.fft.fftfreq(N, Ly/(N-1)))*lam*z
#     FT_grid = np.meshgrid(FT_x_axis, FT_y_axis)

#     # Multiply by the second exponential factor
#     field_z = field / (1j*lam*z) * np.exp(1j*k*z) * SphFactor(*FT_grid, z)
    
#     return field_z, FT_grid

# def Propagate_Fraunhofer(field_0, z, grid): # equivalent to Fresnel without the spherical wavefront factors

#     # Perform the 2D FFT
#     field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_0))) # fft assumes origin of axis at top left corner, need to fftshift beforehand
#     field = field * (Lx/(N-1)) * (Ly/(N-1)) # correct for the sampling rate (related to difference between continuous and discrete FT)

#     # Compute the conjugate x and y axis. Their width is appx (N lam z / L)
#     FT_x_axis = np.fft.fftshift(np.fft.fftfreq(N, Lx/(N-1)))*lam*z
#     FT_y_axis = np.fft.fftshift(np.fft.fftfreq(N, Ly/(N-1)))*lam*z
#     FT_grid = np.meshgrid(FT_x_axis, FT_y_axis)

#     # Multiply by the second exponential factor
#     field_z = field / (1j*lam*z) * np.exp(1j*k*z)
    
#     return field_z, FT_grid