import numpy as np

class Particle:
    def __init__(self):
        r = np.zeros(3)
        vel = np.zeros(3)



class Constant:
    def __init__(self):
        self.mass = np.float32(0.00020543)
        self.h = np.float32(0.2)
        self.k = np.float32(8.3144621) # Ideal Gas Constant
        self.eta = np.float32(0.00089) #viscosity coefficient
        self.rest_density = np.float32(1000.0) #Rest Density of water
        self.sigma = np.float32(0.0728) #surface tension coefficient
        self.WPoly6_const = np.array(315 * 1/(np.power(self.h,9) * 64 * np.pi)).astype(np.float32)
        self.grad_WPoly6_const = np.array(945 * 1/(np.power(self.h,9) * 32 * np.pi)).astype(np.float32)
        self.lap_WPoly6_const = np.array(945 * 1/(np.power(self.h,9) * 8 * np.pi)).astype(np.float32)
        self.Wspiky_const = np.array(15 * 1/(np.power(self.h, 6) * np.pi)).astype(np.float32)
        self.grad_Wspiky_const = np.array(45 * 1/(np.power(self.h, 6) * np.pi)).astype(np.float32)
        self.Wviscosity_const = np.array(15 * 1/(np.power(self.h, 3) * np.pi * 2)).astype(np.float32)
        self.lap_Wviscosity_const = np.array(45 * 1/(np.power(self.h, 5) * np.pi)).astype(np.float32)
