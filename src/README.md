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

1. download the src folder
2. do not modify the directory structure to maintain consistency with the defined relative paths
3. run one of the scripts in the *Mains* subdirectory as described under **Run a simulation**

## Run a simulation

There are two ways for running a simulation and display the results:

* perform a new optimiation to obtain your own solution starting from custom values for the different trajectory parameters (multiple examples are already included)

* load a structure stored in one of the three *data* directories that contains the results already obtained for an optimal transfer trajectory and simply display those results

In either case do the following:

1. open one of the scripts in the *Mains* subdirectory
2. read the list that describes the different possibilities and choose the appropriate value for the variable *ch*
3. optionally define your own parameters to perform a new optimization
4. run the script and wait for the results to be displayed
