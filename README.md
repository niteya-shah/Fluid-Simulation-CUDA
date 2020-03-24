# Fluid-Simulation-CUDA
Many visual applications display fluid systems. Most of these systems have been generated
artificially using Fluid Simulation, a technique whereby many miniature fluid particles
interact and behave according to Newtonian Physics to create realistic simulations.
Calculating thousands of interactions for thousands of millions of particles is a very
compute intensive task , and requires massive parallelism to solve efficiently. With the rise
in use of GPUs as parallel computer which have thousands of cores, parallel computations
like fluid simulation have become feasible to compute in real time. To build upon this Idea
is the physical model of SPH. SPH or Smoothed Particle Hydrodynamics is massively
parrelisable algorithm designed for simulating fluid motion in real time by making
approximations that allow it to outperform other models at the cost of some accuracy while
still performing good enough to produce realistic simulations.

## SPH
Smoothed-particle hydrodynamics (SPH) is a computational method used for simulating the
mechanics of continuum media, such as solid mechanics and fluid flows. It was developed
by Gingold and Monaghan and Lucy in 1977, initially for astrophysical problems. It has
been used in many fields of research, including astrophysics, ballistics, volcanology, and
oceanography. It is a meshfree Lagrangian method (where the coordinates move with the
fluid), and the resolution of the method can easily be adjusted with respect to variables such
as density.
Smoothed-particle hydrodynamics is being increasingly used to model fluid motion as well.
This is due to several benefits over traditional grid-based techniques. First, SPH guarantees
conservation of mass without extra computation since the particles themselves represent
mass.
Second, SPH computes pressure from weighted contributions of neighboring particles
rather than by solving linear systems of equations

## Structure

<img src="https://github.com/niteya-shah/Fluid-Simulation-CUDA/blob/master/Results/Structure.png">

## Simulation

<img src="https://github.com/niteya-shah/Fluid-Simulation-CUDA/blob/master/Results/Simulation.png">

## Results

<img src="https://github.com/niteya-shah/Fluid-Simulation-CUDA/blob/master/Results/result.png">
