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
        if type(z) == np.ndarray:
            return np.where( (z!=0) , z*(1 + (self.zR/z)**2), np.inf)
        else:
            if z!=0:
                return z*(1 + (self.zR/z)**2)
            else:
                return np.inf

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
                * hermite(m)(np.sqrt(2)*x/self.w(z)) * hermite(n)(np.sqrt(2)*y/self.w(z)) * np.exp(-1j*self.Gouy(z)*(m+n)) \
                * 1/np.sqrt(2**(m+n-1)*np.math.factorial(m)*np.math.factorial(n)*np.pi) # Scaling for superpositions

    # SU(2) coherent states
    def SU2(self, x, y, z, alpha, beta, phi, n0, m0, l0, p, q, N):
        S=0
        for K in range(0, N+1):
            S += np.sqrt(np.math.comb(N, K)) * np.exp(1j*K*phi) * self.HLG(x, y, z, alpha, beta, n0+p*K, m0+q*K, l0)
        return S * np.sqrt(1/(2**N))

    def HLG(self, x, y, z, alpha, beta, n, m, l):
        S=0
        for k in range(0, n+m+1):
            S += np.exp(1j*k*alpha) * self.Wigner_d(beta, n, m, k) * self.HGBeam(x, y, z, k, n+m-k)
        return S * np.exp(1j*(n+m)*alpha/2)
    
    def Wigner_d(self, beta, n, m, k):
        S = 0
        for v in range(max(0, k-n), min(m, k)+1):
            S += ((-1)**v) * (np.cos(beta/2)**(m+k-2*v)) * (np.sin(beta/2)**(n-k+2*v)) / (np.math.factorial(v)*np.math.factorial(m-v)*np.math.factorial(k-v)*np.math.factorial(n-k+v))
        return S * np.sqrt(float(np.math.factorial(k)*np.math.factorial(n+m-k)*np.math.factorial(n)*np.math.factorial(m)))

    
    
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

    def PhaseGrating(self, x, y, field, theta):
        return field * np.exp(-1j*2*np.pi*np.sin(theta)*y/self.lam)

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


class Pulse():
    def __init__(self, w0, lam, wt):
        self.w0 = w0
        self.lam = lam
        self.k = 2*np.pi/self.lam
        self.zR = np.pi*self.w0**2/self.lam
        self.wt = wt
        self.omega = 2*np.pi*3e8/lam

    "Spatio-Temporal pulse profiles"

    def LG_STOV(self, x, y, t, l, p):
        T = t *self.w0 / self.wt
        r = np.sqrt(x**2 + T**2)
        theta = np.arctan2(T, x)

        C = np.sqrt( 2*np.math.factorial(p) / (np.pi + np.math.factorial(p+np.abs(l)) ) ) # normalization factor

        return np.exp(-(x**2+y**2)/self.w0**2) * np.exp(-t**2/self.wt**2) * np.exp(-1j*self.omega*t) \
                * C * genlaguerre(p, np.abs(l))(2*(r**2/self.w0**2)) * np.exp(-1j*l*theta) * (r*np.sqrt(2)/self.w0)**np.abs(l)

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

    def Grating(self, field_0, grid_0):

        x, t = grid_0[0][0], grid_0[1].T[0] # retrieve axes from meshgrid
        Lx, Lt = x[-1]-x[0], t[-1]-t[0]
        Nx, Nt = len(x), len(t)

        field = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field_0, axes=-1), axis=-1), axes=-1) # fft assumes origin of axis at top left corner, need to fftshift beforehand
        field = field * (Lx/(Nx-1)) * (Lt/(Nt-1)) # correct for the sampling rate (difference between continuous and discrete FT)

        # Compute the conjugate x and y axis. Their width is appx (N lam z / L)
        FT_t_axis = np.fft.fftshift(np.fft.fftfreq(Nt, Lt/(Nt-1)))
        grid_d = np.meshgrid(x, FT_t_axis)

        field_d = field

        return field_d, grid_d


    "Grid generators"
    # Returns a meshgrid with desired L and N
    def Grid(self, Lx, Ly, Nx, Ny):
        x_axis = np.linspace(-Lx/2, Lx/2, Nx) 
        y_axis = np.linspace(-Ly/2, Ly/2, Ny) 
        return np.meshgrid(x_axis, y_axis)


    # Returns a meshgrid with desired L and N
    def ST_Grid(self, Lx, Ly, Lt, Nx, Ny, Nt):
        x_axis = np.linspace(-Lx/2, Lx/2, Nx) 
        y_axis = np.linspace(-Ly/2, Ly/2, Ny) 
        t_axis = np.linspace(-Lt/2, Lt/2, Nt) 
        return np.meshgrid(x_axis, y_axis, t_axis, indexing='ij')


class Mask():
    def __init__(self):
        pass

    def Iris(self, x, y, R):
        r = np.sqrt(x**2 + y**2)
        return np.where((r<R), 1, 0)

    def ZeroPi(self, x, y):
        phase = np.sign(y)*np.pi/2 + np.pi/2
        return np.exp(1j*phase)


def Tilt_beam(x, y, z, angle): # returns xyz coordinates for a cut of the beam at z=0 propagating at an angle
    return x, y, z+y*np.tan(-angle)

def Offset_beam(x, y, z, x0, y0): # returns xyz coordinates for a cut of the beam at z=0, offset from the center
    return x+x0, y+y0, z