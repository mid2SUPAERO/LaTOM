
# Installation

All the scripts were successfully run and tested on Ubuntu 18.04 LTS with the packages listed below.

## Dependencies

#### Required packages:

* Python 3.7.3 +
* numpy 1.16.4 +
* scipy 1.2.1 +
* matplotlib 3.1.0 +
* openmdao 2.8.0 +
* dymos 0.13.0 +

#### Additional packages to use the NLP solver IPOPT instead of SLSQP (recommended, required in most of the cases):

* pyoptsparse
* pyipopt
* IPOPT 3.12

## Installation

1. verify the required dependencies
2. download the src folder
3. do not modify the directory structure to maintain consistency with the defined relative paths
4. run a simulation as described in [README.md](README.md)

## Notes

To correctly link the NLP solver IPOPT with your OpenMDAO installation do the followings:

1. If you are an academic or a student, you should obtain the [HSL subroutines](http://hsl.rl.ac.uk/ipopt) for better performances
2. [compile IPOPT from source](https://coin-or.github.io/Ipopt/INSTALL.html) enabling the ```--disable-linear-solver-loader``` option in the configuration step
3. [complile pyipopt](https://github.com/xuy/pyipopt) modifying the ```setup.py``` script to detect your IPOPT installation
4. [compile pyoptsparse](https://github.com/mdolab/pyoptsparse)
5. enter your pyoptsparse installation folder and edit the file ```pyoptsparse/pyIPOPT/pyIPOPT.py``` replacing the line
```python
from . import pyipoptcore
```
with
```python
from pyipopt import pyipoptcore
```
6. OpenMDAO shold now be able to wrap your IPOPT installation

Detailed instructions to properly setup your environment can be found in [environment_setup.md](environment_setup.md)
