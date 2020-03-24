# Launcher Trajectory Optimization Module (LaTOM)

This package has been developed by **Alberto FOSSA'** and **Giuliana Elena
MICELI** as part of the MAE2 Research Project **"Optimal Control of Trajectory
of a Reusable Launcher in OpenMDAO/dymos"**

## Authors

* Alberto FOSSA'
* Giuliana Elena MICELI

## Installation

* Refer to [install.md](docs/source/installation/install.md) for the installation
instructions
* Refer to [environment.md](docs/source/installation/environment.md) to properly
setup the OpenMDAO/dymos environment

## Documentation

Refer to [https://mid2supaero.github.io/LaTOM/](https://mid2supaero.github.io/LaTOM/)
for the package documentation

## Contents

This directory contains the source code to find the most fuel efficient transfer
trajectory from the Moon surface to a specified Low Lunar Orbit (LLO) and back
or from an LLO to an Highly Elliptical Orbit (HEO).

The optimization can be performed in the following cases:

1. two-dimensional ascent trajectories:
  - constant thrust
  - variable thrust
  - variable thrust and constrained minimum safe altitude

2. two-dimensional descent trajectories:
  - constant thrust
  - constant thrust and constrained vertical landing
  - variable thrust
  - variable thrust and constrained minimum safe altitude

3. two-dimensional LLO to HEO transfers:
  - single phase trajectory with variable thrust
  - three-phases trajectory with two powered phases at constant thrust and
  intermediate coasting arc
  - single phase escape burn with constant thrust

## Run a simulation

There are two ways for running a simulation and display the results:

* perform a new optimization to obtain your own solution starting from custom
values for the different trajectory parameters (multiple examples are already
included)

* load a structure stored in one of the three *data* directories that contains
the results already obtained for an optimal transfer trajectory and simply
display those results

In either case do the following:

1. open one of the scripts in the *scripts* subdirectory
2. read the list that describes the different possibilities and choose the
appropriate settings
3. optionally define your own parameters to perform a new optimization
4. run the script and wait for the results to be displayed

## References

Gray, Justin S., et al. ‘OpenMDAO: An Open-Source Framework for
Multidisciplinary Design, Analysis, and Optimization’. Structural and
Multidisciplinary Optimization, vol. 59, no. 4, Apr. 2019, pp. 1075–104.
doi:10.1007/s00158-019-02211-z.

Hendricks, Eric S., et al. ‘Simultaneous Propulsion System and Trajectory
Optimization’. 18th AIAA/ISSMO Multidisciplinary Analysis and Optimization
Conference, American Institute of Aeronautics and Astronautics, 2017.
doi:10.2514/6.2017-4435.

HSL, A Collection of Fortran Codes for Large Scale Scientific Computation.
http://www.hsl.rl.ac.uk/.

Perez, Ruben E., et al. ‘PyOpt: A Python-Based Object-Oriented Framework for
Nonlinear Constrained Optimization’. Structural and Multidisciplinary
Optimization, vol. 45, no. 1, Jan. 2012, pp. 101–18. DOI.org (Crossref),
doi:10.1007/s00158-011-0666-3.

Wächter, Andreas, and Lorenz T. Biegler. ‘On the Implementation of an
Interior-Point Filter Line-Search Algorithm for Large-Scale Nonlinear
Programming’. Mathematical Programming, vol. 106, no. 1, Mar. 2006, pp. 25–57.
doi:10.1007/s10107-004-0559-y.
