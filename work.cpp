__global__ void WPoly6(float *val, const float *r, const float *h,
                       const float *WPoly6_const) {
  const int i = threadIdx.x;
  if (r < h && r > 0) {
    float inner_val = ((*h * *h) - (r[i] * r[i]));
    inner_val = inner_val * inner_val * inner_val;
    val[i] += *WPoly6_const * inner_val;
  }
}

__device__ float magnitude(const float *r) {
  return pow(r[0] * r[0] + r[1] * r[1] + r[2] * r[2], 0.5);
}

__device__ float magnitude_withId(const float *r) {
  const int i = threadIdx.x;
  return pow(r[i] * r[i] + r[i + 1] * r[i + 1] + r[i + 2] * r[i + 2], 0.5);
}

__device__ void grad_WPoly6(float *grad, float *r, const float mag,
                            const float *h, const float *grad_WPoly6_const) {
  if (mag > 0) {
    float inner_val = ((*h * *h) - (mag * mag));
    inner_val = inner_val * inner_val * *grad_WPoly6_const;
    grad[0] = inner_val * r[0];
    grad[1] = inner_val * r[1];
    grad[2] = inner_val * r[2];
  } else {
    grad[0] = 0;
    grad[0] = 0;
    grad[0] = 0;
  }
}

__device__ float lap_WPoly6(const float *h, const float mag,
                            const float *lap_WPoly6_const) {
  float inner_val = ((*h * *h) - (mag * mag));
  return *lap_WPoly6_const * inner_val * ((mag * mag) - (0.75 * inner_val));
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
  float inner_val = *grad_Wspiky_const * (*h - mag) * (*h - mag);
  grad[0] = r[0] * 1 / mag * inner_val;
  grad[1] = r[1] * 1 / mag * inner_val;
  grad[2] = r[2] * 1 / mag * inner_val;
}

__device__ float Wviscosity(const float *h, const float mag,
                            const float *Wviscosity_const) {
  float inner_val = ((-mag * mag * mag) / (2 * *h * *h * *h)) +
                    ((mag * mag) / (*h * *h)) + (*h / (2 * mag)) - 1;
  return inner_val * *Wviscosity_const;
}

__device__ float lap_Wviscosity(const float mag, const float *h,
                                const float *lap_Wviscosity_const) {
  return *lap_Wviscosity_const * (1 - mag / *h);
}

__device__ void pressure_force(float *press_force, const float *mass, float mag,
                               const float *r, const float *h,
                               const float *density_p, const float *density_n,
                               const float *rest_density, const float *k,
                               const float *grad_Wspiky_const) {
  const int i = threadIdx.x;
  float grad[3];
  grad_WSpiky(grad, mag, r, h, grad_Wspiky_const);
  float const_val =
      *mass * *k * (*density_n + density_n[i] - 2 * *rest_density);
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
  float const_val = *eta * *mass * 1 / (*density_n) *
                    lap_Wviscosity(mag, h, lap_Wviscosity_const);
  visc_force[0] = const_val * (vel_n[i] - vel_p[i]);
  visc_force[1] = const_val * (vel_n[i + 1] - vel_p[i + 1]);
  visc_force[2] = const_val * (vel_n[i + 2] - vel_p[i + 2]);
}

__device__ void color_field_grad(float *color_field_grad_val, const float *mass,
                                 const float *density_n, float *r,
                                 const float *h, const float mag,
                                 const float *grad_WPoly6_const) {
  const int i = threadIdx.x;
  float inner_val = *mass * 1 / (*density_n);
  float grad_Wpoly[3];
  grad_WPoly6(grad_Wpoly, r, mag, h, grad_WPoly6_const);
  color_field_grad_val[i] = grad_Wpoly[0] * inner_val;
  color_field_grad_val[i + 1] = grad_Wpoly[1] * inner_val;
  color_field_grad_val[i + 2] = grad_Wpoly[2] * inner_val;
}

__device__ void color_field_lap(float *color_field_lap_val, const float *mass,
                                const float *density_n, const float *r,
                                const float *h, const float mag,
                                const float *lap_WPoly6_const) {
  const int i = threadIdx.x;
  color_field_lap_val[i] =
      *mass * 1 / (*density_n) * lap_WPoly6(h, mag, lap_WPoly6_const);
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
  r_dash[0] = r[i] - r1[i];
  r_dash[1] = r[i + 1] - r1[i + 1];
  r_dash[2] = r[i + 2] - r1[i + 2];
  float mag = magnitude(r_dash);
  pressure_force(press_force, mass, mag, r_dash, h, density_p, density_n,
                 rest_density, k, grad_Wspiky_const);
  viscosity_force(visc_force, eta, mass, mag, h, density_n, vel_n, vel_p,
                  lap_Wviscosity_const);
  color_field_grad(color_field_grad_val, mass, density_n, r_dash, h, mag,
                   grad_WPoly6_const);
  color_field_lap(color_field_lap_val, mass, density_n, r_dash, h, mag,
                  lap_WPoly6_const);
  force[i] = press_force[0] + visc_force[0];
  force[i + 1] = press_force[1] + visc_force[1];
  force[i + 2] = press_force[2] + visc_force[2];
}

__global__ void update_pos(float *r, float *vel_p, float *force,
                           const float *threshold, const float *mass,
                           const float *time, const float *sigma,
                           const float *color_field_lap_val,
                           const float *color_field_grad_val) {
  const int i = threadIdx.x;
  float gradient_length = magnitude_withId(color_field_grad_val);
  if (gradient_length >= *threshold) {
    float const_val = -*sigma * color_field_lap_val[i];
    force[i] += const_val * color_field_grad_val[i];
    force[i + 1] += const_val * color_field_grad_val[i + 1];
    force[i + 2] += const_val * color_field_grad_val[i + 2];
  }
  force[i + 2] += 9.8f;
  vel_p[i] += force[i] * 1 / (*mass) * *time;
  vel_p[i + 1] += force[i + 1] * 1 / (*mass) * *time;
  vel_p[i + 2] += force[i + 2] * 1 / (*mass) * *time;

  r[i] += vel_p[i] * *time;
  r[i + 1] += vel_p[i + 1] * *time;
  r[i + 2] += vel_p[i + 2] * *time;
}
