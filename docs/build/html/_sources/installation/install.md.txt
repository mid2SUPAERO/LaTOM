# Package Installation

@author Alberto FOSSA'

This *step-by-step* guide will help you to install **LaTOM**,
Launcher Trajectory Optimization Module.

## Dependencies

* [OpenMDAO/dymos environment](environment.md)
* [smt][smt] Surrogate Modelling Toolbox

## Installation

Clone the package repository:

```
git clone https://github.com/mid2SUPAERO/LaTOM.git
```

Enter the top level folder, activate your environment and install the package in development mode:

```
(myenv) python setup.py bdist_wheel
(myenv) pip install -e ./
```

The package is successfully installed.

## Documentation

#### Additional Dependencies

* sphinx 2.4.3 +
* sphinx-autopackagesummary 1.2 +
* sphinx-markdown-tables 0.0.9 +
* sphinx_rtd_theme 0.4.3 +
* graphviz 2.40.1 +
* recommonmark 0.6.0 +

#### Documentation Generation

Install the additional dependencies listed above:

```
(myenv) conda install sphinx sphinx_rtd_theme graphviz recommonmark
(myenv) pip install sphinx-autopackagesummary -markdown-tables
```

Enter the ```/docs``` folder and issue the following commands:

```
(myenv) make clean
(myenv) make html
```

Documentation can be reviewed in ```/docs/build/html/index.html```

[smt]: <https://smt.readthedocs.io/en/latest/>
