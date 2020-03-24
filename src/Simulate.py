import pycuda.autoinit  # NOQA
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import numpy as np
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import time

from constants import Constant

with open("src/source.cu", "r") as file:
    source = file.read()

module_calculate_consts = SourceModule(source)  # NOQA

const = Constant()
nn = NearestNeighbors(n_neighbors=const.num_neighbhours)
timer = np.zeros([100, 7])
calc_density = module_calculate_consts.get_function("calc_density")
calc_forces = module_calculate_consts.get_function("calc_forces")
update_pos = module_calculate_consts.get_function("update_pos")
r1 = (np.random.rand(const.num_particles, 3).astype(np.float32) -
      np.array([0.5, 0.5, 0.5]).astype(np.float32)) * 10
v = np.zeros([const.num_particles, 3]).astype(np.float32)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for iteration in range(100):
    timer[iteration][0] = time.time()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.scatter3D(r1.T[0], r1.T[1], r1.T[2])
    plt.draw()
    plt.pause(0.000001)
    ax.cla()
    timer[iteration][1] = time.time()
    nn.fit(r1)
    neighbors = nn.kneighbors(
        r1, return_distance=False).flatten().astype(np.int32)

    timer[iteration][2] = time.time()
    r_dash = r1.flatten().astype(np.float32)
    v_n = v.flatten().astype(np.float32)
    density = np.zeros(
        [const.num_particles, const.num_neighbhours]
    ).flatten().astype(np.float32)
    force = np.zeros(
        [const.num_particles, const.num_neighbhours, 3]).astype(np.float32)
    color_field_lap_val = np.zeros(
        [const.num_particles, const.num_neighbhours]).astype(np.float32)
    color_field_grad_val = np.zeros(
        [const.num_particles, const.num_neighbhours, 3]).astype(np.float32)

    timer[iteration][3] = time.time()
    calc_density(drv.Out(density), drv.In(r_dash), drv.In(neighbors),
                 drv.In(const.mass), drv.In(const.h),
                 drv.In(const.WPoly6_const),
                 block=(const.num_neighbhours, 1, 1),
                 grid=(const.num_particles, 1))

    Density = density.reshape(
        [const.num_particles, const.num_neighbhours]).sum(axis=1)

    timer[iteration][4] = time.time()
    calc_forces(drv.Out(force), drv.Out(color_field_lap_val),
                drv.Out(color_field_grad_val), drv.In(r_dash),
                drv.In(Density), drv.In(v_n), drv.In(neighbors),
                drv.In(const.h), drv.In(const.eta), drv.In(const.mass),
                drv.In(const.rest_density), drv.In(const.k),
                drv.In(const.grad_WPoly6_const),
                drv.In(const.lap_WPoly6_const), drv.In(const.Wspiky_const),
                drv.In(const.grad_Wspiky_const),
                drv.In(const.Wviscosity_const),
                drv.In(const.lap_Wviscosity_const),
                block=(const.num_neighbhours, 1, 1), grid=(const.num_particles,
                                                           1))

    force = force.reshape(
        [const.num_particles, const.num_neighbhours, 3]).sum(axis=1)
    color_field_grad_val = color_field_grad_val.reshape(
        [const.num_particles, const.num_neighbhours, 3]).sum(axis=1)
    color_field_lap_val = color_field_lap_val.reshape(
        [const.num_particles, const.num_neighbhours]).sum(axis=1)
    force = force.flatten().astype(np.float32)
    color_field_grad_val = color_field_grad_val.flatten().astype(np.float32)

    timer[iteration][5] = time.time()
    update_pos(drv.InOut(r_dash), drv.InOut(v_n), drv.In(force),
               drv.In(const.threshold), drv.In(const.mass), drv.In(const.time),
               drv.In(const.sigma), drv.In(const.Width), drv.In(const.damping),
               drv.In(const.eps), drv.In(color_field_lap_val),
               drv.In(color_field_grad_val),
               block=(1, 1, 1), grid=(const.num_particles, 1))

    r1 = r_dash.reshape(const.num_particles, 3)
    v = v_n.reshape(const.num_particles, 3)
    timer[iteration][6] = time.time()

timer = timer - timer.T[0].T.reshape(100, 1)  # NOQA
normalised = timer/(np.sum(timer, axis=1) + 1e-16).reshape(100, 1)
average_value = np.sum(normalised, axis=0)/np.sum(normalised, axis=0).sum()
plt.bar(["Plotting", "Nearst neighbors", "Preparation", "Density Calculation",
         "Force Calculation", "Position Update"], average_value[1:])
