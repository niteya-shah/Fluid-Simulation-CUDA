%matplotlib qt
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

class Constant:
    def __init__(self):
        self.time = np.float32(0.005)
        self.mass = np.float32(65)
        self.h = np.float32(16)
        self.k = np.float32(2000) # Ideal Gas Constant
        self.threshold = np.float32(0.05)
        self.eta = np.float32(250) #viscosity coefficient
        self.rest_density = np.float32(1000) #Rest Density of water
        self.sigma = np.float32(0.07) #surface tension coefficient
        self.eps = np.float32(0.1)
        self.Width = np.float32(10)
        self.damping = np.float32(-0.5)
        self.WPoly6_const = np.float32(315 * 1/(np.power(self.h,9) * 64 * np.pi))
        self.grad_WPoly6_const = np.float32(945 * 1/(np.power(self.h,9) * 32 * np.pi))
        self.lap_WPoly6_const = np.float32(945 * 1/(np.power(self.h,9) * 32 * np.pi))
        self.Wspiky_const = np.float32(15 * 1/(np.power(self.h, 6) * np.pi))
        self.grad_Wspiky_const = np.float32(- 45 * 1/(np.power(self.h, 6) * np.pi))
        self.Wviscosity_const = np.float32(15 * 1/(np.power(self.h, 3) * np.pi * 2))
        self.lap_Wviscosity_const = np.float32(45 * 1/(np.power(self.h, 6) * np.pi))
        self.num_particles = 1000
        self.num_neighbhours = 500

module_calculate_consts = SourceModule("""

__device__ float magnitude(const float *r) {
  return pow((r[0] * r[0]) + (r[1] * r[1]) + (r[2] * r[2]), 0.5);
}

__device__ float magnitude_withId(const float *r) {
  const int i = threadIdx.x;
  return pow((r[3 * i] * r[3 * i]) + (r[3 * i + 1] * r[3 * i + 1]) + (r[3 * i + 2] * r[3 * i + 2]), 0.5);
}

__device__ float clamp(float x, float max_v, float min_v)
{
    return max(min_v, min(max_v, x));
}

__device__ float WPoly6(const float mag, const float *h,
                       const float *WPoly6_const) {
  if (mag < *h && mag > 0) {
    float inner_val = ((*h * *h) - (mag * mag));
    return inner_val * inner_val * inner_val * *WPoly6_const;
  }
  else
    return 0;
}

__device__ void grad_WPoly6(float *grad, float *r, const float mag,
                            const float *h, const float *grad_WPoly6_const) {
    float inner_val = ((*h * *h) - (mag * mag));
    inner_val = inner_val * inner_val * *grad_WPoly6_const * (1/(mag + 0.000001f));
    grad[0] = inner_val * r[0];
    grad[1] = inner_val * r[1];
    grad[2] = inner_val * r[2];
}

__device__ float lap_WPoly6(const float *h, const float mag,
                            const float *lap_WPoly6_const) {
  float inner_val = ((*h * *h) - (mag * mag));
  return *lap_WPoly6_const * inner_val * ((3 * *h * *h) - (7 * mag * mag));
}

__device__ float Wspiky(const float *h, const float mag,
                        const float *Wpiky_const) {
  if (mag < *h && mag > 0) {
    return *Wpiky_const * (*h - mag) * (*h - mag) * (*h - mag);
  } else {
    return 0;
  }
}

__device__ void grad_WSpiky(float *grad, const float mag, const float *r,
                            const float *h, const float *grad_Wspiky_const) {
  float inner_val = *grad_Wspiky_const * (*h - mag) * (*h - mag) * (1 / (mag + 0.0000001f));
  grad[0] = r[0]  * inner_val;
  grad[1] = r[1]  * inner_val;
  grad[2] = r[2]  * inner_val;
}

__device__ float Wviscosity(const float *h, const float mag,
                            const float *Wviscosity_const) {
  float inner_val = ((-mag * mag * mag) / (2 * *h * *h * *h)) +
                    ((mag * mag) / (*h * *h)) + (*h / (2 * mag + 0.0000001f)) - 1;
  return inner_val * *Wviscosity_const;
}

__device__ float lap_Wviscosity(const float mag, const float *h,
                                const float *lap_Wviscosity_const) {
  return *lap_Wviscosity_const * (*h - mag);
}

__device__ void pressure_force(float *press_force, const float *mass,const float mag,
                               const float *r, const float *h,
                               const float *density_p, const float *density_n,
                               const float *rest_density, const float *k,
                               const float *grad_Wspiky_const) {
  const int i = threadIdx.x;
  float grad[3];
  grad_WSpiky(grad, mag, r, h, grad_Wspiky_const);
  float const_val =
      *mass * *k * (*density_p + density_n[i] - 2 * *rest_density) * 1/(2 * density_n[i]);
  press_force[0] = const_val * grad[0];
  press_force[1] = const_val * grad[1];
  press_force[2] = const_val * grad[2];
}

__device__ void viscosity_force(float *visc_force, const float *eta,
                                const float *mass, const float mag,
                                const float *h, const float *density_n,
                                const float *vel_n, const float *vel_p,
                                const float *lap_Wviscosity_const) {
  const int i = threadIdx.x;
  float const_val = *eta * *mass * (1 / (density_n[i] +  0.0000001f)) *
                    lap_Wviscosity(mag, h, lap_Wviscosity_const);
  visc_force[0] = const_val * (vel_n[3 * i] - vel_p[0]);
  visc_force[1] = const_val * (vel_n[3 * i + 1] - vel_p[1]);
  visc_force[2] = const_val * (vel_n[3 * i + 2] - vel_p[2]);
}

__device__ void color_field_grad(float *color_field_grad_val, const float *mass,
                                 const float *density_n, float *r,
                                 const float *h, const float mag,
                                 const float *grad_WPoly6_const) {
  const int i = threadIdx.x;
  float inner_val = *mass * 1 / (density_n[i] +  0.0000001f);
  float grad_Wpoly[3];
  grad_WPoly6(grad_Wpoly, r, mag, h, grad_WPoly6_const);
  color_field_grad_val[3 * i] = grad_Wpoly[0] * inner_val;
  color_field_grad_val[3 * i + 1] = grad_Wpoly[1] * inner_val;
  color_field_grad_val[3 * i + 2] = grad_Wpoly[2] * inner_val;
}

__device__ void color_field_lap(float *color_field_lap_val, const float *mass,
                                const float *density_n, const float *r,
                                const float *h, const float mag,
                                const float *lap_WPoly6_const) {
  const int i = threadIdx.x;
  color_field_lap_val[i] =
      *mass * 1 / (density_n[i] +  0.0000001f) * lap_WPoly6(h, mag, lap_WPoly6_const);
}

__global__ void calc_density(float *density,const float *r1,const int *neighbors,const float *mass,const float *h,const float *WPoly6_const)
{
      float r_dash[3];
      const int i = threadIdx.x;
      const int j = blockIdx.x;
      const int n = j * blockDim.x + i;
      r_dash[0] = r1[j] - r1[3 * neighbors[n]];
      r_dash[1] = r1[j + 1] - r1[3 * neighbors[n] + 1];
      r_dash[2] = r1[j + 2] - r1[3 * neighbors[n] + 2];
      const float mag = magnitude(r_dash);
      density[n] = WPoly6(mag, h, WPoly6_const) * *mass;
}

__global__ void calc_forces(
    float *force, float *color_field_lap_val, float *color_field_grad_val,
    const float *r, const float *r1, const float *h, const float *eta,
    const float *mass, const float *density_p, const float *density_n,
    const float *rest_density, const float *k, float *vel_n, float *vel_p,
    const float *grad_WPoly6_const, const float *lap_WPoly6_const,
    const float *Wspiky_const, const float *grad_Wspiky_const,
    const float *Wviscosity_const, const float *lap_Wviscosity_const) {
  const int i = threadIdx.x;
  float r_dash[3], press_force[3], visc_force[3];
  r_dash[0] = r[0] - r1[3 * i];
  r_dash[1] = r[1] - r1[3 * i + 1];
  r_dash[2] = r[2] - r1[3 * i + 2];
  float mag = magnitude(r_dash);
  if(mag > 0.000001f && mag < *h){
  pressure_force(press_force, mass, mag, r_dash, h, density_p, density_n,
                 rest_density, k, grad_Wspiky_const);
  viscosity_force(visc_force, eta, mass, mag, h, density_n, vel_n, vel_p,
                  lap_Wviscosity_const);
  color_field_grad(color_field_grad_val, mass, density_n, r_dash, h, mag,
                   grad_WPoly6_const);
  color_field_lap(color_field_lap_val, mass, density_n, r_dash, h, mag,
                  lap_WPoly6_const);
  force[3 * i] = clamp(press_force[0] + visc_force[0], 200, -200);
  force[3 * i + 1] = clamp(press_force[1] + visc_force[1], 200, -200);
  force[3 * i + 2] = clamp(press_force[2] + visc_force[2], 200, -200);
  }
  else
  {
    force[3 * i] = 0;
    force[3 * i + 1] = 0;
    force[3 * i + 2] = 0;
    color_field_grad_val[3 * i] = 0;
    color_field_grad_val[3 * i + 1] = 0;
    color_field_grad_val[3 * i + 2] = 0;
    color_field_lap_val[i] = 0;
  }
}

__global__ void update_pos(float *r, float *vel_p, float *force,
                           const float *threshold, const float *mass,
                           const float *time, const float *sigma,const float *Width, const float *damping,const float *eps,
                           const float *color_field_lap_val,
                           const float *color_field_grad_val) {
  const int i = threadIdx.x;
  float gradient_length = magnitude_withId(color_field_grad_val);
  float force_surface[3] = {0,0,0};

  if (gradient_length >= *threshold) {
    float const_val = - 1 * *sigma * color_field_lap_val[i] * (1/(gradient_length + 0.0000001f));
    force_surface[0] = const_val * color_field_grad_val[3 * i];
    force_surface[1] = const_val * color_field_grad_val[3 * i + 1];
    force_surface[2] = const_val * color_field_grad_val[3 * i + 2];
  }

  vel_p[3 * i] += clamp((force[3 * i] + force_surface[0]) * (1 / (*mass)) * *time, 200, -200);
  vel_p[3 * i + 1] += clamp((force[3 * i + 1] + force_surface[1]) * (1 / (*mass)) * *time, 200, -200);
  vel_p[3 * i + 2] += clamp((((force[3 * i + 2] + force_surface[2]) * 1 / (*mass) ) - 10 * 12000) * *time, 200, -200);

  r[3 * i] += vel_p[3 * i] * *time;
  r[3 * i + 1] += vel_p[3 * i + 1] * *time;
  r[3 * i + 2] += vel_p[3 * i + 2] * *time;

  if(r[3 * i] < -*Width)
  {
    r[3 * i] = -*Width + *eps;
    vel_p[3 * i] = *damping * vel_p[3 * i];
  }
  if(r[3 * i + 1] < -*Width)
  {
    r[3 * i + 1] = -*Width + *eps;
    vel_p[3 * i + 1] = *damping * vel_p[3 * i + 1];
  }
  if(r[3 * i + 2] < -*Width)
  {
    r[3 * i + 2] = -*Width + *eps;
    vel_p[3 * i + 2] = *damping * vel_p[3 * i + 2];
  }
  if(r[3 * i] > *Width)
  {
    r[3 * i] = *Width - *eps;
    vel_p[3 * i] = *damping * vel_p[3 * i];
  }
  if(r[3 * i + 1] > *Width)
  {
    r[3 * i + 1] = *Width - *eps;
    vel_p[3 * i + 1] = *damping * vel_p[3 * i + 1];
  }
  if(r[3 * i + 2] > *Width)
  {
    r[3 * i + 2] = *Width * 10 - *eps;
    vel_p[3 * i + 2] = *damping * vel_p[3 * i + 2];
  }

}

""")

const = Constant()
nn = NearestNeighbors(n_neighbors = const.num_neighbhours)

calc_density = module_calculate_consts.get_function("calc_density")
calc_forces = module_calculate_consts.get_function("calc_forces")
update_pos = module_calculate_consts.get_function("update_pos")
r1 = (np.random.rand(const.num_particles,3).astype(np.float32) - np.array([0.5,0.5,0.5]).astype(np.float32)) * 10
v = np.zeros([const.num_particles,3]).astype(np.float32)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for iteration in range(40):
    plt.xlim(-10,10)
    plt.ylim(-10, 10)
    ax.set_zlim(-10,10)
    ax.scatter3D(r1.T[0], r1.T[1], r1.T[2])
    plt.draw()
    plt.pause(0.000001)
    ax.cla()
    nn.fit(r1)
    neighbors = nn.kneighbors(r1, return_distance=False)
    neighbors_dash = neighbors.flatten().astype(np.int32)
    r_dash = r1.flatten().astype(np.float32)
    density = np.zeros([const.num_particles,const.num_neighbhours]).flatten().astype(np.float32)
    calc_density(drv.Out(density), drv.In(r_dash),drv.In(neighbors),drv.In(const.mass) ,drv.In(const.h), drv.In(const.WPoly6_const),block=(const.num_neighbhours,1,1), grid=(const.num_particles,1))
    Density = density.reshape([const.num_particles, const.num_neighbhours]).sum(axis = 1)
    print("dens")
    force = np.zeros([const.num_particles,3]).astype(np.float32)
    color_field_lap_val = np.zeros([const.num_particles]).astype(np.float32)
    color_field_grad_val = np.zeros([const.num_particles,3]).astype(np.float32)
    print("bfor force")
    for i in range(const.num_particles):
        force_temp = np.zeros([const.num_neighbhours,3]).flatten().astype(np.float32)
        color_field_lap_val_temp = np.zeros([const.num_neighbhours]).astype(np.float32)
        color_field_grad_val_temp = np.zeros([const.num_neighbhours,3]).flatten().astype(np.float32)
        r = r1[i].astype(np.float32)
        r_dash = r1[neighbors[i]].flatten().astype(np.float32)
        density_p = np.float32(Density[i])
        v_n = v[neighbors[i]].flatten().astype(np.float32)
        v_p = v[i].astype(np.float32)
        calc_forces(drv.Out(force_temp), drv.Out(color_field_lap_val_temp), drv.Out(color_field_grad_val_temp), drv.In(r), drv.In(r1), drv.In(const.h), drv.In(const.eta), drv.In(const.mass), drv.In(density_p), drv.In(Density), drv.In(const.rest_density), drv.In(const.k), drv.In(v_n), drv.In(v_p), drv.In(const.grad_WPoly6_const),drv.In(const.lap_WPoly6_const),drv.In(const.Wspiky_const),drv.In(const.grad_Wspiky_const),drv.In(const.Wviscosity_const),drv.In(const.lap_Wviscosity_const), block=(const.num_neighbhours,1,1), grid=(1,1))
        force[i] = force_temp.reshape([const.num_neighbhours,3]).sum(axis = 0)
        color_field_lap_val[i] = color_field_lap_val_temp.sum()
        color_field_grad_val[i] = color_field_grad_val_temp.reshape([const.num_neighbhours,3]).sum(axis = 0)
    print("after force")
    r_dash = r1.flatten().astype(np.float32)
    v_n = v.flatten().astype(np.float32)
    force = force.flatten().astype(np.float32)
    color_field_grad_val = color_field_grad_val.flatten().astype(np.float32)
    print("update_pos")
    update_pos(drv.InOut(r_dash), drv.InOut(v_n), drv.In(force), drv.In(const.threshold),drv.In(const.mass),drv.In(const.time) ,drv.In(const.sigma),drv.In(const.Width),drv.In(const.damping),drv.In(const.eps) ,drv.In(color_field_lap_val), drv.In(color_field_grad_val), block=(const.num_particles,1,1), grid=(1,1))
    print("final")
    r1 = r_dash.reshape(const.num_particles,3)
    v = v_n.reshape(const.num_particles,3)
