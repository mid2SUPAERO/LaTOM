# Trajectory Optimization in OpenMDAO/dymos
# Installation guide

This *step-by-step* guide will help you to setup an [Anaconda][anaconda] environment ready to perform Multidisciplinary Design Optimization (MDO) and Trajectory Optimization using the Python-based libraries [OpenMDAO][openmdao] and [dymos][dymos].

## Dependencies to be installed

#### Python packages

* Python 3.7.3 +
* numpy 1.16.4 +
* scipy 1.2.1 +
* matplotlib 3.1.0 +
* openmdao 2.8.0 +
* dymos 0.13.0 +
* pyoptsparse (SNOPT optional)
* pyipopt

#### Additional software

* IPOPT 3.12

#### Optional licensed packages

* [HSL 2001][hsl_ipopt] subroutines, free for academic purposes upon request (one working day delay)
* [SNOPT][snopt] sparse nonlinear optimizer

## Requirements

* system with 64-bit Linux OS (Ubuntu 18.04 LTS recommended)
* temporarly root access using ```sudo``` (step 1 only)

## Installation procedure

The setup of the whole environment is splitted in the following steps:

1. Installation of the MPI, BLAS and Lapack libraries through the OS package manager
2. Compilation from source of the large-scale nonlinear optimizer [IPOPT][ipopt]
3. Installation of the [Anaconda][anaconda] platform
4. Creation of a dedicated ```conda``` environment
5. Installation of [OpenMDAO][openmdao] and [dymos][dymos] packages with all their dependencies in the dedicated environment
6. Compilation from source of the [pyipopt][pyipopt] package and linkage with the compiled IPOPT library
7. Compilation from source of the [pyOptSparse][pyoptsparse] package (SNOPT optional) and linkage with pyipopt 

## 1. Installation of MPI, BLAS and Lapack libraries

The MPI libraries are needed to run your optimization problems in parallel, thus reducing the CPU time to perform your studies. On the other side, BLAS and Lapack are two libraries to perform linear algebra calculations used internally by IPOPT and SNOPT.

All the instructions included in this chapter are valid for Debian-based OS that use the ```apt``` package manager and were successfully tested on Ubuntu 18.04 LTS. If your machine comes with a different OS you should look for similar commands suitable for its specific package manager.

#### 1.1 Installation of the GNU compiler and other system packages

The GNU compilers and the other packages mentioned here are usually already installed in your machine. You should carefully check for each of them and perform the installation of the missed ones if any.

```
$ sudo apt-get install gcc g++ gfortran make git patch wget
```

#### 1.2 Installation of MPI libraries

Both the binaries and the development files are required and they can be installed with the following command:

```
$ sudo apt-get install openmpi-bin libopenmpi-dev
```

#### 1.3 Installation of BLAS and Lapack libraries

Once more, both the binaries and the development files are required and can be installed issuing the command below:

```
$ sudo apt-get install libblas-dev libblas3 liblapack-dev liblapack3 libopenblas-dev libopenblas-base
```

## 2. Compilation from source of IPOPT


#### HSL 2011 subroutines

As pointed out in the requirements, if you use IPOPT for academic purposes you are strongly recommended to obtain the HSL 2011 subroutines since they perform much more better than the other options. The full suite has to be requested as follows:

- go on the [HSL for IPOPT][hsl_ipopt] webpage
- under **Coin-HSL Full (Stable)** click on **Source**
- In the request page select **Personal Academic License**, then fill and submit the form
- Within one working day you should receive an email with a link to download the source code as a compressed archive

#### Download of the required source code

Once you have obtained the HSL subroutines or you decide to use a different linear solver you can proceed downloading the IPOPT source code and perform the installation. Open a new terminal window and type the following command:

```
$ git clone --recursive -b stable/3.12 https://github.com/coin-or/Ipopt.git CoinIpopt
```

This will directly download the latest stable release in the directory ```~/CoinIpopt```.
When the download is complete, enter the new directory to download the different dependencies as follows:

```
$ cd CoinIpopt
$ cd ThirdParty/ASL
$ ./get.ASL
```

This will download the ASL libraries to directly interface IPOPT with the AMPL modelling language. Then proceed with:

```
$ cd ../Metis
$ ./get.Metis
```

To download the matrix ordering algorithm Metis.

If you have obtained the HSL source code unpack the archive, move and rename the corresponding directory such that it becomes ```~/CoinIpopt/ThirdParty/HSL/coinhsl```.

> If you haven't requested the HSL subroutines you have to download the Mumps linear solver to replace the ones provided by the former libraries.
> Tod do so, enter the directory ```~/CoinIpopt/ThirdParty/Mumps``` and issue the following command: ```$ ./get.Mumps```.

At this point you have obtained all the required source code and you can proceed with the configuration and installation steps.

#### Configuration and installation

First, move back to the directory ```~/CoinIpopt``` and create a separate folder where the compiled code will be placed. Then move in the newely created directory:

```
$ mkdir build
$ cd build
```

After that run the ```configure``` script adding the proper compilers flags to enable parallel computation at runtime and disable the **linear solver loader** option that will prevent you to link IPOPT with the pyipopt package. For that purpose issue the following command, all on the same line:

```
$ ../configure ADD_CFLAGS=-fopenmp ADD_FFLAGS=-fopenmp ADD_CXXFLAGS=-fopenmp --disable-linear-solver-loader
```

If the last output is ```Main configuration of Ipopt successful``` you can proceed building the code. In your terminal window type the following:

```
$ make
$ make test
```

This will build the code and perform a series of test on different examples. If all the test are successful you can complete the installation procedure issuing the command below:

```
$ make install
```

#### Verify your IPOPT installation

At this stage you can verify if the IPOPT binaries were correctly built entering the ```bin``` directory and checking the presence of the executable ```ipopt```. From your ```~/CoinIpopt/build``` directory type the following:

```
$ cd bin
$ ipopt
```

You should get an output similar to this one:

```
No stub!
usage: ipopt [options] stub [-AMPL] [<assignment> ...]

Options:
    --  {end of options}
	-=  {show name= possibilities}
	-?  {show usage}
	-bf {read boundsfile f}
	-e  {suppress echoing of assignments}
	-of {write .sol file to file f}
	-s  {write .sol file (without -AMPL)}
	-v  {just show version}
```

You can also run a simple example included in the source code to verify the convergence of the algorithm:

```
$ cd ../../Ipopt/test
$ ../../build/bin/ipopt mytoy.nl
```

The last two output lines should be similar to these ones:

```
EXIT: Optimal Solution Found.
Ipopt 3.12.13: Optimal Solution Found
```

Once you have successfully installed IPOPT you can jump to the next chapter to setup your ```conda``` environment.

#### Common troubleshooting

> If the configuration step fails to locate your BLAS and Lapack libraries, you can download their source code inside your ```~/CoinIpopt``` directory and let IPOPT to compile them at the same time. For that purpose, after obtaining the Metis source code with the ```./get.Metis``` command type the following:

```
$ cd ../Blas
$ ./get.Blas
$ cd ../Lapack
$ ./get.Lapack
```

> This will download a local copy of the BLAS and Lapack libraries and IPOPT will use them. You can now proceed adding the source code for the linear solver of your choice and perform the configuration and installation steps as described above.
For other issues please refer to the [official IPOPT documentation][ipopt_install] availabe online.

## 3. Installation of the Anaconda platform

If you already have Anaconda installed on your system, you can directly jump to the next chapter. Otherwise follow the instructions below to install and setup the platform.

First, download the **Anaconda Distribution Linux Installer** with **Python 3.7** from the [official Anaconda website][anaconda_download].
Once the download is complete enter the corresponding folder and type:

```
$ bash Anaconda-latest-Linux-x86_64.sh
```

Follow the prompts on the installer screen and accept the default configurations. This will install Anaconda in your ```$HOME``` folder and does not require root permissions.
Once the installation is complete, you will be asked to initialize ```conda```. Answer ```yes``` and wait for the initialization to be performed. Finally, close and re-open your terminal for changes to take effect.
At this point you may notice that the ```conda``` ```base```environment is automatically activated every time a new terminal window is opened. The prompt line should look like this:

```
(base) $
```

To deactivate ```conda```, type:

```
(base) $ conda deactivate
```

And to reactivate:

```
$ conda activate
```

To prevent ```conda``` from automatically activating the ```base``` environment issue the following command:

```
(base) $ conda config --set auto_activate_base False
```

You can now move on the next chapter and create your dedicated ```conda``` environment.

## 4. Creation of a dedicated conda environment

To create a new ```conda``` environment open a terminal window and issue the following commands replacing ```myenv``` with the desired environment name:

```
$ conda activate
(base) $ conda create --name myenv python=3.7
```

This will create a new ```conda``` environment with Python 3.7 and other basics packages such as ```pip``` already included.
The new environment can be activated and deactivated with the same commands seen for the ```base``` environment:

```
$ conda activate myenv
(myenv) $ conda deactivate myenv
```

## 5. Installation of OpenMDAO, dymos and relative dependencies

Assuming you have created a dedicated ```conda``` environment named ```myenv```, you can proceed with the installation of OpenMDAO, dymos and their required dependencies. First of all, open a terminal window and activate your environment:

```
$ conda activate myenv
```

Then follow the steps below to download and install the different packages.

#### 5.1 Installation of Numpy, Scipy, Matplotlib, Spyder, Swig

Numpy and Scipy are two Python packages to perform mathematical operations on large set of arrays, while  Matplotlib is a widely-used library to visualize your results. Finally, Spyder is an easy-to-use interactive Python IDE and Swig a package to wrap C and C++ code within Python. With your environment active issue the following:

```
(myenv) $ conda install numpy scipy matplotlib spyder swig
```

And answer ```yes``` when prompted.

#### 5.2 Installation of mpi4py

mpi4py provides a Python wrapper for the MPI libraries. The last version of this package is not yet released on the official conda channels and has to be installed using ```pip```:

```
(myenv) $ pip install mpi4py
```

#### 5.3 Installation of OpenMDAO and dymos

As in the previous case, these packages have to be downloaded and installed using ```pip``` since they are not deployed on the official conda channels. They can be obtained issuing the followings:

```
(myenv) $ pip install openmdao
(myenv) $ pip install git+https://github.com/OpenMDAO/dymos.git
```

You can now launch Spyder and start your own projects, but you wan't be able to link IPOPT and SNOPT with your OpenMDAO installation since the lasts require pyipopt and pyOptSparse to be properly compiled as described in the following chapters.

## 6. Compilation from source of pyipopt

#### Download of the required source code

The [pyipopt][pyipopt] package provides a Python wrapper for your IPOPT installation allowing OpenMDAO to call the NLP solver while performing the user-defined MDO. This package can be installed followig the instructions provided in this section.

With your environment active, move to a convenient directory and clone the source code from its GitHub repository:

```
(myenv) $ git clone https://github.com/xuy/pyipopt.git
```

#### Adjust IPOPT installation path

Once the download is completed enter the corresponding folder and edit the ```setup.py``` script to reflect the configuration of your system:

```
(myenv) $ cd pyipopt
(myenv) $ spyder setup.py &
```

With the ```setup.py``` script open in your Spyder IDE, launch a new terminal window and enter the directory where IPOPT was compiled. After that retrieve its absolute path with the command ```pwd```:

```
$ cd CoinIpopt/build
$ pwd
```

You should obtain the following output where ```username``` is your actual username:

```
/home/username/CoinIpopt/build
```

At this point copy the output string and move back to Spyder to customize the ```setup.py``` script.

To tell pyipopt where to catch your IPOPT installation replace the three lines:

```
IPOPT_DIR = '/usr/local/'
IPOPT_LIB = get_ipopt_lib()
IPOPT_INC = os.path.join(IPOPT_DIR, 'include/coin/')
```

with:

```
IPOPT_DIR = '/home/username/CoinIpopt/build/'
IPOPT_LIB = '/home/username/CoinIpopt/build/lib'
IPOPT_INC = '/home/username/CoinIpopt/build/include/coin/'
```

Where the strings between ```''``` have to reflect the output of ```pwd``` copied before.

#### Adjust IPOPT libraries included in your installation

To specify which libraries you included during the IPOPT installation modify the corresponding arguments passed to the ```Extension``` class:

1. IPOPT compiled with system-wide BLAS and Lapack libraries (**without** running the scripts ```get.Blas``` and ```get.Lapack``` as described under **Common troubleshooting**) and the HSL subroutines:

```
pyipopt_extension = Extension(
        'pyipoptcore',
        FILES,
        extra_link_args=['-Wl,--rpath','-Wl,'+ IPOPT_LIB],
        library_dirs=[IPOPT_LIB],
        libraries=[
            'ipopt',
            'coinhsl',
            'coinmetis',
            'dl','m',
            ],
        include_dirs=[numpy_include, IPOPT_INC],
        )
```

2. IPOPT compiled running the scripts ```get.Blas``` and ```get.Lapack``` to build a local copy of the BLAS and Lapack libraries and the HSL subroutines:

```
pyipopt_extension = Extension(
        'pyipoptcore',
        FILES,
        extra_link_args=['-Wl,--rpath','-Wl,'+ IPOPT_LIB],
        library_dirs=[IPOPT_LIB],
        libraries=[
            'ipopt', 'coinblas',
            'coinhsl',
            'coinmetis',
            'coinlapack','dl','m',
            ],
        include_dirs=[numpy_include, IPOPT_INC],
        )
```

3. IPOPT compiled with system-wide BLAS and Lapack libraries (**without** running the scripts ```get.Blas``` and ```get.Lapack``` as described under **Common troubleshooting**) and the Mumps linear solver (running the script ```get.Mumps```):

```
pyipopt_extension = Extension(
        'pyipoptcore',
        FILES,
        extra_link_args=['-Wl,--rpath','-Wl,'+ IPOPT_LIB],
        library_dirs=[IPOPT_LIB],
        libraries=[
            'ipopt',
            'coinmumps',
            'coinmetis',
            'dl','m',
            ],
        include_dirs=[numpy_include, IPOPT_INC],
        )
```

4. IPOPT compiled running the scripts ```get.Blas``` and ```get.Lapack``` to build a local copy of the BLAS and Lapack libraries and the Mumps linear solver (running the script ```get.Mumps```):

```
pyipopt_extension = Extension(
        'pyipoptcore',
        FILES,
        extra_link_args=['-Wl,--rpath','-Wl,'+ IPOPT_LIB],
        library_dirs=[IPOPT_LIB],
        libraries=[
            'ipopt', 'coinblas',
            'coinmumps',
            'coinmetis',
            'coinlapack','dl','m',
            ],
        include_dirs=[numpy_include, IPOPT_INC],
        )
```

Notice that in each case the line ```extra_link_args=['-Wl,--rpath','-Wl,'+ IPOPT_LIB]``` has to be uncommented. Then save your changes and close Spyder.

#### Installation

From the terminal window with ```myenv``` activated and inside the pyipopt directory issue the following command to install the package:

```
(myenv) $ python setup.py install --user
```

This will install the package under ```~/.local/lib/python3.7/site-packages/pyipopt``` where you should find the  file ```pyipoptcore.cpython-37m-x86_64-linux-gnu.so```. If it is not the case, pyipot was not able to properly link your IPOPT installation and the entire procedure has to be repeated after identifying the issues responsible for the failure.

## 7. Compilation from source of pyOptSparse (SNOPT optional)

[pyOptSparse][pyoptsparse] provides a convinient interface between OpenMDAO and a wide suite of optimizers with enhanced MPI capabilities. It is required to wrap both IPOPT and SNOPT using the ```pyOptSparseDriver``` class.

#### Download of the required source code

To install the package, open a terminal window and activate your environment, than clone the source code from its [official GitHub repository][pyoptsparse_repo]:

```
(myenv) $ git clone https://github.com/mdolab/pyoptsparse.git
```

If you have purchased a license for SNOPT and obtained the source code copy all the ```.f``` files except ```snopth.f``` into the folder ```pyoptsparse/pySNOPT/source```. If not, simply move to the next installation instructions.

#### Installation

To install the package come back to the terminal window, enter the root directory ```pyoptsparse``` cloned before and run the ```setup.py``` script:

```
(myenv) $ cd pyoptsparse
(myenv) $ python setup.py install --user
```

This will install pyOptSparse under ```~/.local/lib/python3.7/site-packages/pyoptsparse```.

#### IPOPT wrapping

At this point the package is unable to wrap the nonlinear solver IPOPT and you have to modify the ```pyIPOPT.py``` script to tell where the library ```pyipoptcore.cpython-37m-x86_64-linux-gnu.so``` is located and which linear solvers are available in your IPOPT installation.

Firstly, enter the directory in which ```pyIPOPT.py``` is located and open it in your Spyder IDE:

```
(myenv) $ cd
(myenv) $ cd .local/lib/python3.7/site-packages/pyoptsparse/pyIPOPT
(myenv) $ spyder pyIPOPT.py &
```

Secondly, replace the line:

```
    from . import pyipoptcore
```

with:

```
    from pyipopt import pyipoptcore
```

Thirdly, scroll the ```def_opts``` dictionary inside the ```IPOPT``` class until you find the following lines:

```
    # Linear Solver.
    'linear_solver' : [str, "ma27"],
    'linear_system_scaling' : [str, "none"],  # Had been "mc19", but not always available.
    'linear_scaling_on_demand' : [str, "yes"],
```

1. If you have compiled IPOPT with the HSL subroutines, you can try the different solvers included in your IPOPT installation replacing ```"ma27"``` with either ```"ma57"```, ```"ma86"``` or ```"ma97"``` to determine which is the best for your problem. Moreover, replace ```"none"``` with ```"mc19"``` to use the corresponding scaling routines.
2. If you select the linear solver ```"ma57"```, scroll the dictionary until you reach the MA57 specific options and uncomment the line ```'ma57_automatic_scaling' : [str, "yes"]``` to allow the solver to use automatic scaling.
3. If you compiled IPOPT with the Mumps linear solver, simply replace ```"ma27"``` with ```"mumps"``` and let the linear system scaling set to ```"none"```.

You can then modify all the other default settings included in the ```def_opts``` dictionary after carefully reading the corresponding [official documentation][ipopt_opts]. Finally save your changes and close the file.

At this point the environment setup is complete and you should be able to perform your MDO analysis and trajectory optimization using the OpenMDAO and dymos libraries.

#### Common troubleshooting

> If you are compiling pyOptSparse with SNOPT, the installation could fail if the ```gfortran``` compiler is not able to locate the BLAS and Lapack libraries installed system-wide. In this case you can explicitly tell where these libraries are located as follows:

> Open a new terminal window and identify where BLAS and Lapack are installed:

```
$ locate libblas.so
```

> One of the output should be similar to the following line:

```
/usr/lib/x86_64-linux-gnu/libblas.so
```

> Similarly, for Lapack:

```
$ locate liblapack.so
/usr/lib/x86_64-linux-gnu/liblapack.so
```

> Copy the absolute path of the two directories in which ```libblas.so``` and ```liblapack.so``` are located and re-run the ```setup.py``` script adding the following flags:

```
python setup.py config_fc --fcompiler=gnu95 --f77flags='-L/usr/lib/x86_64-linux-gnu' --f90flags='-L/usr/lib/x86_64-linux-gnu' install --user
```

> Then come back to the regular installation instructions to modify the ```pyIPOPT.py``` script.

## Useful Links

Anaconda homepage: https://www.anaconda.com/
dymos documentation: https://openmdao.github.io/dymos/
dymos GitHub repository: https://github.com/OpenMDAO/dymos
HSL for IPOPT: http://www.hsl.rl.ac.uk/ipopt/
IPOPT homepage: https://coin-or.github.io/Ipopt/index.html
IPOPT installation instructions: https://coin-or.github.io/Ipopt/INSTALL.html
IPOPT options: https://www.coin-or.org/Ipopt/documentation/node40.html
OpenMDAO homepage: https://openmdao.org/
OpenMDAO GitHub repository: https://github.com/OpenMDAO/OpenMDAO
pyipopt GitHub repository: https://github.com/xuy/pyipopt
pyOptSparse documentation: http://mdolab.engin.umich.edu/docs/packages/pyoptsparse/doc/index.html
pyOptSparse GitHub repository: https://github.com/mdolab/pyoptsparse
SNOPT license: http://www.sbsi-sol-optimize.com/asp/sol_snopt.htm


[anaconda]: <https://www.anaconda.com/>
[anaconda_download]: <https://www.anaconda.com/distribution/>
[dymos]: <https://openmdao.github.io/dymos/>
[dymos_repo]: <https://github.com/OpenMDAO/dymos>
[hsl_ipopt]: <http://www.hsl.rl.ac.uk/ipopt/>
[ipopt]: <https://coin-or.github.io/Ipopt/index.html>
[ipopt_install]: <https://coin-or.github.io/Ipopt/INSTALL.html>
[ipopt_opts]: <https://www.coin-or.org/Ipopt/documentation/node40.html>
[openmdao]: <https://openmdao.org/>
[openmdao_repo]: <https://github.com/OpenMDAO/OpenMDAO>
[pyipopt]: <https://github.com/xuy/pyipopt>
[pyoptsparse]: <http://mdolab.engin.umich.edu/docs/packages/pyoptsparse/doc/index.html>
[pyoptsparse_repo]: <https://github.com/mdolab/pyoptsparse>
[snopt]: <http://www.sbsi-sol-optimize.com/asp/sol_snopt.htm>
