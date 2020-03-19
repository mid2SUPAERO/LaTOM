"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

Defines useful constants
"""

# physical constants
g0 = 9.80665

# variables names
states_2d = ['r', 'theta', 'u', 'v', 'm']
states_rates_2d = ['rdot', 'thetadot', 'udot', 'vdot', 'mdot']
states_3d = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm']
states_rates_3d = ['xdot', 'ydot', 'zdot', 'vxdot', 'vydot', 'vzdot', 'mdot']
controls_3d = ['T', 'ux', 'uy', 'uz']

# variables to be excluded from the recording options
rec_excludes = ['*.control_rates:*', '*.control_states.*', '*.control_values:*', '*.continuity_comp.*', '*.dt_dstau*',
                '*.final_jump:*', '*.initial_jump:*', '*.interleave_comp.*', '*.rhs_disc.*', '*.rhs_col.*',
                '*.state_interp.*', '*.t_initial*', '*.time_phase*', '*.traj_parameters:*', '*++*', '*-+*', '*+-*',
                '*--*']
