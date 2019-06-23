## Optimal control of trajectory of a reusable launcher in OpenMDAO/dymos

# Source code

This directory contains the source code to find the most fuel efficient transfer trajectory from the Moon surface to a specified Low Lunar Orbit (LLO) and back.

The optimization can be performed in the following cases:

1. two-dimensional ascent trajectories:
  - constant thrust
  - constant thrust and constrained vertical take-off
  - variable thrust
  - variable thrust and constrained minimum safe altitude
  
2. two-dimensional descent trajectories:
  - constant thrust
  - constant thrust and constrained vertical landing
  - variable thrust

3. three-dimensional ascent trajectories:
  - constant thrust
  - variable thrust

## Installation

1. verify the required dependencies listed under **Dependencies**
2. download the src folder
3. do not modify the directory structure to maintain consistency with the defined relative paths
4. run one of the scripts in the *Mains* subdirectory as described under **Run a simulation**

## Run a simulation

There are two ways for running a simulation and display the results:

* perform a new optimiation to obtain your own solution starting from custom values for the different trajectory parameters (multiple examples are already included)

* load a structure stored in one of the three *data* directories that contains the results already obtained for an optimal transfer trajectory and simply display those results

In either case do the following:

1. open one of the scripts in the *Mains* subdirectory
2. read the list that describes the different possibilities and choose the appropriate value for the variable *ch*
3. optionally define your own parameters to perform a new optimization
4. run the script and wait for the results to be displayed

## Dependencies

The scripts have the following dependencies:

* Python 3.7.3 +
* numpy 1.16.4 +
* scipy 1.2.1 +
* matplotlib 3.1.0 +
* openmdao 2.7.1 +
* dymos 0.13.0 +

Additional dependencies to use the NLP solver IPOPT instead of SLSQP (recommended, required in most of the cases)

* pyoptsparse
* pyipopt
* IPOPT 3.12.13

## Notes

To correctly link the NLP solver IPOPT with your OpenMDAO installation the following steps have to be performed:

1. [compile IPOPT from source](https://coin-or.github.io/Ipopt/INSTALL.html) enabling the *--disable-linear-solver-loader* option in the configuration step
2. [complile pyipopt](https://github.com/xuy/pyipopt) modifying the *setup.py* script to detect your IPOPT installation
3. [compile pyoptsparse](https://github.com/mdolab/pyoptsparse)
4. enter your pyoptsparse installation folder and edit the file *pyoptsparse/pyIPOPT/pyIPOPT.py* replacing the line *from . import pyipoptcore* with *from pyipopt import pyipoptcore*
5. OpenMDAO shold now be able to wrap you IPOPT installation

