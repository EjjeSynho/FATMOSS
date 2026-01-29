from misc import *
import math


class Zernike:
    def __init__(self, pupil, modes_num=10):
        
        self.N_modes = modes_num
        self.modes = None
        self.pupil = pupil
        
        self.modes_names = [
            'Tip', 'Tilt', 'Defocus', 'Astigmatism (X)', 'Astigmatism (+)',
            'Coma vert', 'Coma horiz', 'Trefoil vert', 'Trefoil horiz',
            'Sphere', 'Secondary astig (X)', 'Secondary astig (+)',
            'Quadrofoil vert', 'Quadrofoil horiz',
            'Secondary coma horiz', 'Secondary coma vert',
            'Secondary trefoil horiz', 'Secondary trefoil vert',
            'Pentafoil horiz', 'Pentafoil vert'
        ]
        # self.gpu = GPU_flag
        self.GenerateBasis(rotation_angle=None, transposed=False) # Compute default basis


    def _zernike_radial_func(self, n, m, r):
        """
        Fucntion to calculate the Zernike radial function

        Parameters:
            n (int): Zernike radial order
            m (int): Zernike azimuthal order
            r (ndarray): 2-d array of radii from the centre the array

        Returns:
            ndarray: The Zernike radial function
        """
        # xp = cp if self.gpu else np

        R = xp.zeros(r.shape)
        # Can cast the below to "int", n,m are always *both* either even or odd
        for i in range(0, int((n-m)/2) + 1):
            R += xp.array(r**(n - 2 * i) * (((-1)**(i)) *
                            math.factorial(n-i)) / (math.factorial(i) *
                            math.factorial(int(0.5 * (n+m) - i)) *
                            math.factorial(int(0.5 * (n-m) - i))),
                            dtype='float')
        return R


    def _zern_index(self, j):
        n = int((-1.0 + np.sqrt(8*(j-1)+1))/2.)
        p = (j-(n*(n+1))/2.)
        k = n % 2
        m = int((p+k)/2.)*2 - k

        if m != 0:
            if j % 2 == 0: s = 1
            else:  s = -1
            m *= s

        return [n, m]


    def _rotate_coordinate_grids(self, angle, X, Y):
            # xp = cp if self.gpu else np
            angle_rad = np.radians(angle)

            rotation_matrix = xp.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad),  np.cos(angle_rad)]
            ])

            coordinates = xp.vstack((X, Y))
            rotated_coordinates = xp.dot(rotation_matrix, coordinates)
            rotated_X, rotated_Y = rotated_coordinates[0, :], rotated_coordinates[1, :]

            return rotated_X, rotated_Y
        

    def GenerateBasis(self, rotation_angle=None, transposed=False):
        """
        Function to calculate the Zernike modal basis.

        Parameters:
            rotation_angle (float): Angle in degrees to rotate the coordinate grid before calculating the Zernike modes
            transposed (bool): If True, swap X and Y coordinates before calculating the Zernike modes
            
        Returns:
            None: The Zernike modes are stored in self.modes as a 3D array of shape (resolution, resolution, N_modes)
        """
        
        if self.pupil is None:
            raise ValueError("Pupil mask is not defined.")
            
        assert self.pupil.ndim == 2, "Pupil mask must be a 2D array."
        assert self.pupil.shape[0] == self.pupil.shape[1], "Pupil mask must be square."
        
        resolution = self.pupil.shape[0]

        X, Y = xp.where(self.pupil == 1)
        X = (X-resolution//2+0.5*(1-resolution%2)) / resolution
        Y = (Y-resolution//2+0.5*(1-resolution%2)) / resolution
        
        if transposed:
            X, Y = Y, X
        
        if rotation_angle is not None and np.abs(rotation_angle) > 1e-10:
            X, Y = self._rotate_coordinate_grids(rotation_angle, X, Y)
        
        R = xp.sqrt(X**2 + Y**2)
        R /= R.max()
        theta = xp.arctan2(Y, X)

        self.modes = xp.zeros([resolution**2, self.N_modes])

        for i in range(1, self.N_modes+1):
            n, m = self._zern_index(i+1)
            if m == 0:
                Z = xp.sqrt(n+1) * self._zernike_radial_func(n, 0, R)
            else:
                if m > 0: # j is even
                    Z = xp.sqrt(2*(n+1)) * self._zernike_radial_func(n, m, R) * xp.cos(m*theta)
                else:   #i is odd
                    m = abs(m)
                    Z = xp.sqrt(2*(n+1)) * self._zernike_radial_func(n, m, R) * xp.sin(m*theta)
            
            Z -= Z.mean()
            Z /= Z.std()

            self.modes[xp.where(xp.reshape(self.pupil, resolution*resolution)>0), i-1] = Z
            
        self.modes = xp.reshape( self.modes, [resolution, resolution, self.N_modes] )
        
        # if GPU_flag:
        self.modes = xp.array(self.modes, dtype=xp.float32) # if GPU is used, initialize a GPU-based array


    def Mode(self, coef):
        return self.modes[:,:,coef]


    def ModeName(self, index):
        if index < 0:
            return('Incorrent index!')
        elif index >= len(self.modes_names):
            return('Z ' + str(index+2))
        else:
            return(self.modes_names[index])


    def WavefrontFromModes(self, coefs):
        """ Generate wavefront shape corresponding to given model coefficients and modal basis. """
        coefs_ = xp.array(coefs).flatten()
        coefs_[xp.where(xp.abs(coefs_)<1e-13)] = xp.nan
        valid_ids = xp.where(xp.isfinite(coefs_))[0] # NaNs is coefficient vector are ignored

        if self.modes is None:
            print('Warning: Zernike modes were not computed! Calculating...')
            self.N_modes = xp.max(xp.array([coefs_.shape[0], self.N_modes]))
            self.GenerateBasis()

        if self.N_modes < coefs_.shape[0]:
            self.N_modes = coefs_.shape[0]
            print('Warning: vector of coefficients is too long. Computing additional modes...')
            self.GenerateBasis()

        return self.modes[:,:,valid_ids] @ coefs_[valid_ids]


    def ProjectWavefront(self, wavefront):
        """ Project given wavefront onto the Zernike modal basis and return the coefficients. """
        
        wavefront_ = xp.atleast_3d(xp.array(wavefront))
        
        if self.modes is None:
            print('Warning: Zernike modes were not computed! Calculating...')
            self.GenerateBasis()

        M_flat = self.modes.reshape(-1, self.modes.shape[2])
        screens_flat = wavefront_.reshape(-1, wavefront_.shape[2])
        modal_coefs = M_flat.T @ screens_flat / self.pupil.sum()
            
        return modal_coefs