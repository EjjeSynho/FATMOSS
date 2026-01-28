from misc import *
import math

def mask_circle(N, r, center=(0,0), centered=True, xp=np):
    factor = 0.5 * (1-N%2)
    if centered:
        coord_range = xp.linspace(-N//2+N%2+factor, N//2-factor, N)
    else:
        coord_range = xp.linspace(0, N-1, N)
    xx, yy = xp.meshgrid(coord_range-center[1], coord_range-center[0])
    pupil_round = xp.zeros([N, N], dtype=xp.int32)
    pupil_round[xp.sqrt(yy**2+xx**2) < r] = 1
    return pupil_round


class Zernike:
    def __init__(self, modes_num=1):
        
        self.nModes = modes_num
        self.modesFullRes = None
        self.pupil = None

        self.modes_names = [
            'Tip', 'Tilt', 'Defocus', 'Astigmatism (X)', 'Astigmatism (+)',
            'Coma vert', 'Coma horiz', 'Trefoil vert', 'Trefoil horiz',
            'Sphere', 'Secondary astig (X)', 'Secondary astig (+)',
            'Quadrofoil vert', 'Quadrofoil horiz',
            'Secondary coma horiz', 'Secondary coma vert',
            'Secondary trefoil horiz', 'Secondary trefoil vert',
            'Pentafoil horiz', 'Pentafoil vert'
        ]
        self.gpu = GPU_flag  


    @property
    def gpu(self):
        return self.__gpu

    @gpu.setter
    def gpu(self, var):
        if var:
            self.__gpu = True
            if hasattr(self, 'modesFullRes'):
                if not hasattr(self.modesFullRes, 'device'):
                    self.modesFullRes = cp.array(self.modesFullRes, dtype=cp.float32)
        else:
            self.__gpu = False
            if hasattr(self, 'modesFullRes'):
                if hasattr(self.modesFullRes, 'device'):
                    self.modesFullRes = self.modesFullRes.get()


    def zernikeRadialFunc(self, n, m, r):
        """
        Fucntion to calculate the Zernike radial function

        Parameters:
            n (int): Zernike radial order
            m (int): Zernike azimuthal order
            r (ndarray): 2-d array of radii from the centre the array

        Returns:
            ndarray: The Zernike radial function
        """
        xp = cp if self.gpu else np

        R = xp.zeros(r.shape)
        # Can cast the below to "int", n,m are always *both* either even or odd
        for i in range(0, int((n-m)/2) + 1):
            R += xp.array(r**(n - 2 * i) * (((-1)**(i)) *
                            math.factorial(n-i)) / (math.factorial(i) *
                            math.factorial(int(0.5 * (n+m) - i)) *
                            math.factorial(int(0.5 * (n-m) - i))),
                            dtype='float')
        return R


    def zernIndex(self, j):
        n = int((-1.0 + np.sqrt(8*(j-1)+1))/2.)
        p = (j-(n*(n+1))/2.)
        k = n % 2
        m = int((p+k)/2.)*2 - k

        if m != 0:
            if j % 2 == 0: s = 1
            else:  s = -1
            m *= s

        return [n, m]


    def rotate_coordinates(self, angle, X, Y):
            xp = cp if self.gpu else np
            angle_rad = np.radians(angle)

            rotation_matrix = xp.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])

            coordinates = xp.vstack((X, Y))
            rotated_coordinates = xp.dot(rotation_matrix, coordinates)
            rotated_X, rotated_Y = rotated_coordinates[0, :], rotated_coordinates[1, :]

            return rotated_X, rotated_Y
        

    def computeZernike(self, resolution, normalize_unit=False, angle=None, transposed=False):
        """
        Function to calculate the Zernike modal basis

        Parameters:
            tel (Telescope): A telescope object, needed mostly to extract pupil data 
            normalize_unit (bool): Sets the regime for normalization of Zernike modes
                                   it's either the telescope's pupil or a unit circle  
        """
        xp = cp if self.gpu else np

        self.pupil = mask_circle(N=resolution, r=resolution/2, xp=xp)


        X, Y = xp.where(self.pupil == 1)
        X = (X-resolution//2+0.5*(1-resolution%2)) / resolution
        Y = (Y-resolution//2+0.5*(1-resolution%2)) / resolution
        
        if transposed:
            X, Y = Y, X
        
        if angle is not None and angle != 0.0:
            X, Y = self.rotate_coordinates(angle, X, Y)
        
        R = xp.sqrt(X**2 + Y**2)
        R /= R.max()
        theta = xp.arctan2(Y, X)

        self.modesFullRes = xp.zeros([resolution**2, self.nModes])

        for i in range(1, self.nModes+1):
            n, m = self.zernIndex(i+1)
            if m == 0:
                Z = xp.sqrt(n+1) * self.zernikeRadialFunc(n, 0, R)
            else:
                if m > 0: # j is even
                    Z = xp.sqrt(2*(n+1)) * self.zernikeRadialFunc(n, m, R) * xp.cos(m*theta)
                else:   #i is odd
                    m = abs(m)
                    Z = xp.sqrt(2*(n+1)) * self.zernikeRadialFunc(n, m, R) * xp.sin(m*theta)
            
            Z -= Z.mean()
            Z /= Z.std()

            self.modesFullRes[xp.where(xp.reshape(self.pupil, resolution*resolution)>0), i-1] = Z
            
        self.modesFullRes = xp.reshape( self.modesFullRes, [resolution, resolution, self.nModes] )
        
        if self.gpu: # if GPU is used, return a GPU-based array
            self.modesFullRes = xp.array(self.modesFullRes, dtype=xp.float32)


    def modeName(self, index):
        if index < 0:
            return('Incorrent index!')
        elif index >= len(self.modes_names):
            return('Z ' + str(index+2))
        else:
            return(self.modes_names[index])


    # Generate wavefront shape corresponding to given model coefficients and modal basis 
    def wavefrontFromModes(self, tel, coefs_inp):
        xp = cp if self.gpu else np

        coefs = xp.array(coefs_inp).flatten()
        coefs[xp.where(xp.abs(coefs)<1e-13)] = xp.nan
        valid_ids = xp.where(xp.isfinite(coefs))[0]

        if self.modesFullRes is None:
            print('Warning: Zernike modes were not computed! Calculating...')
            self.nModes = xp.max(xp.array([coefs.shape[0], self.nModes]))
            self.computeZernike(tel)

        if self.nModes < coefs.shape[0]:
            self.nModes = coefs.shape[0]
            print('Warning: vector of coefficients is too long. Computiong additional modes...')
            self.computeZernike(tel)

        return self.modesFullRes[:,:,valid_ids] @ coefs[valid_ids]

    def Mode(self, coef):
        return self.modesFullRes[:,:,coef]
