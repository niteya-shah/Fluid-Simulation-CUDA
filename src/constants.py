import numpy as np


class Constant:
    def __init__(self):
        self.time = np.float32(0.005)
        self.mass = np.float32(65)
        self.h = np.float32(16)
        self.k = np.float32(2000)  # Ideal Gas Constant
        self.threshold = np.float32(0.05)
        self.eta = np.float32(250)  # viscosity coefficient
        self.rest_density = np.float32(1000)  # Rest Density of water
        self.sigma = np.float32(0.07)  # surface tension coefficient
        self.eps = np.float32(0.1)
        self.Width = np.float32(10)
        self.damping = np.float32(-0.5)
        self.WPoly6_const = np.float32(
            315 * 1/(np.power(self.h, 9) * 64 * np.pi))
        self.grad_WPoly6_const = np.float32(
            945 * 1/(np.power(self.h, 9) * 32 * np.pi))
        self.lap_WPoly6_const = np.float32(
            945 * 1/(np.power(self.h, 9) * 32 * np.pi))
        self.Wspiky_const = np.float32(15 * 1/(np.power(self.h, 6) * np.pi))
        self.grad_Wspiky_const = np.float32(- 45 *
                                            1/(np.power(self.h, 6) * np.pi))
        self.Wviscosity_const = np.float32(
            15 * 1/(np.power(self.h, 3) * np.pi * 2))
        self.lap_Wviscosity_const = np.float32(
            45 * 1/(np.power(self.h, 6) * np.pi))
        self.num_particles = 1000
        self.num_neighbhours = 500
