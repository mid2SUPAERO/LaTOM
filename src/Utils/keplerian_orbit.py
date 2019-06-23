#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors:
    Alberto Fossa'
    Giuliana Miceli

This script defines the class KeplerianOrbit that implements the required methods to compute
the spacecraft state vector knowing the 6 classical orbital elements and vice-versa in the
restricted two-body problem framework 
"""

import numpy as np

class KeplerianOrbit():
    
    def __init__(self, mu):
        
        """
        initialize a KeplerianOrbit class instance specifying the central body standard gravitational parameter
        """
        
        self.mu = mu #central body standard gravitational parameter (m^3/s^2)
        self.eps = np.finfo(float).eps #smallest number such that 1.0 + eps != 1.0
        
    def set_sv(self, R, V):
        
        """
        define the spacecraft state vector and compute its classical orbital elements
        """
        
        self.R = R #position vector R=[x, y, z] (m)
        self.V = V #velocity vector V=[vx, vy, vz] (m/s)
        
        self.compute_coe()
        
    def get_sv(self):
        
        """
        retrieve the spacecraft state vector
        """
        
        R = self.R
        V = self.V
        
        return R, V
        
    def set_coe(self, a, e, i, raan, w, ta, angle_unit='rad'):
        
        """
        define the spacecraft classical orbital elements and compute its state vector
        """
        
        self.a = a #semimajor axis (m)
        self.e = e #eccentricity
        
        if angle_unit == 'rad':
            self.i = i #inclination (rad)
            self.raan = raan #righ ascension of the ascending node (rad)
            self.w = w #argument of perigee (rad)
            self.ta = ta #true anomaly (rad)
        elif angle_unit == 'deg':
            deg2rad = np.pi/180 #one degree in radians
            self.i = i*deg2rad #inclination (rad)
            self.raan = raan*deg2rad #righ ascension of the ascending node (rad)
            self.w = w*deg2rad #argument of perigee (rad)
            self.ta = ta*deg2rad #true anomaly (rad)
        else:
            print('\nInvalid angle unit! Choose one between rad and deg\n')
            
        self.compute_sv()
            
    def get_coe(self, angle_unit='rad'):
        
        """
        retrieve the spacecraft classical orbital elements
        """
        
        a = self.a #semimajor axis (m)
        e = self.e #eccentricity
        
        if angle_unit == 'rad':
            i = self.i #inclination (rad)
            raan = self.raan #righ ascension of the ascending node (rad)
            w = self.w #argument of perigee (rad)
            ta = self.ta #true anomaly (rad)
        elif angle_unit == 'deg':
            rad2deg = 180/np.pi #one radian in degrees
            i = self.i*rad2deg #inclination (deg)
            raan = self.raan*rad2deg #righ ascension of the ascending node (deg)
            w = self.w*rad2deg #argument of perigee (deg)
            ta = self.ta*rad2deg #true anomaly (deg)
        else:
            print('\nInvalid angle unit! Choose one between rad and deg\n')
            
        return a, e, i, raan, w, ta
    
    def set_coe_vector(self, coe, angle_unit='rad'):
        
        """
        define the spacecraft classical orbital elements as the vector (a, e, i, raan, w, ta)
        and compute its state vector
        """
        
        self.coe = coe
        self.set_coe(self.coe[0], self.coe[1], self.coe[2], self.coe[3], self.coe[4], self.coe[5], angle_unit=angle_unit)
        
    def get_coe_vector(self, angle_unit='rad'):
        
        """
        retrieve the spacecraft classical orbital elements as the vector (a, e, i, raan, w, ta)
        """
        
        a, e, i, raan, w, ta = self.get_coe(angle_unit)
        coe = np.array([a, e, i, raan, w, ta])
        
        return coe
    
    def compute_angular_momentum(self):
        
        """
        compute the spacecraft specific angular momentum vector from its state vector
        """
        
        self.H = np.cross(self.R, self.V) #angular momentum vector (m^2/s)
        
    def compute_eccentricity_vector(self):
        
        """
        compute the spacecraft eccentricity vector from its state vector
        """
        
        self.E = np.cross(self.V, self.H)/self.mu - self.R/self.r #eccentricity vector
    
    def get_angular_momentum(self):
        
        """
        retrieve the spacecraft specific angular momentum vector
        """
        
        H = self.H #specific angular momentum vector (m^2/s)
        
        return H
        
    def get_eccentricity_vector(self):
        
        """
        retrieve the spacecraft eccentricity vector
        """
        
        E = self.E #eccentricity vecotr
        
        return E
    
    def R1(self, a):
    
        """
        elementary rotation matrix around X axis
        """
        
        R = np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])
        
        return R

    def R3(self, a):
    
        """
        elementary rotation matrix around Z axis
        """
        
        R = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
        
        return R
    
    def equatorial2perifocal(self):
        
        """
        rotation matrix between inertial, body-centred equatorial reference frame and perifocal reference frame
        """
        
        Q = self.R3(self.w)@self.R1(self.i)@self.R3(self.raan)
        
        return Q
        
    def perifocal2equatorial(self):
        
        """
        rotation matrix between perifocal reference frame and  inertial, body-centred equatorial reference frame
        """
        
        Q = np.matrix.transpose(self.equatorial2perifocal())
        
        return Q
            
    def compute_coe(self):
        
        """
        compute the spacecraft classical orbital elements, specific angular momentum vector and eccentricity vector
        from its state vector
        """
        
        self.r = np.linalg.norm(self.R, 2) #orbit radius (m)
        self.vr = np.dot(self.R, self.V)/self.r #radial velocity (m/s)
        
        self.compute_angular_momentum() #angular momentum vector (m^2/s)
        self.h = np.linalg.norm(self.H, 2) #angular momentum magnitude (m^2/s)
    
        self.i = np.arccos(self.H[2]/self.h) #inclination (rad)
        
        K = np.array([0, 0, 1]) #unit vector along inertial Z axis
        N = np.cross(K, self.H) #node line vector
        n = np.linalg.norm(N, 2) #node line vector magnitude
            
        #right ascension of the ascending node (rad)
        if self.i>=self.eps: #inclined orbit
            self.raan = np.arccos(N[0]/n)
            if N[1]<0.0:
                self.raan = 2*np.pi -self.raan
        else: #equatorial orbit
            self.raan = 0.0
        
        self.compute_eccentricity_vector() #eccentricity vector
        self.e = np.linalg.norm(self.E, 2) #eccentricity
    
        #argument of perigee (rad)
        if self.e>=self.eps: #elliptical orbit
            if self.i>=self.eps: #inclined orbit
                self.w = np.arccos(np.dot(N, self.E)/(n*self.e))
                if self.E[2]<0.0:
                    self.w = 2*np.pi - self.w
            else: #equatorial orbit
                self.w = np.arccos(self.E[0]/self.e)
                if self.E[1]<0.0:
                    self.w = 2*np.pi - self.w
        else: #circular orbit
            self.w = 0.0
        
        #true anomaly (rad)
        if self.e>=self.eps: #elliptical orbit
            self.ta = np.arccos(np.dot(self.E, self.R)/(self.e*self.r))
            if self.vr<0.0:
                self.ta = 2*np.pi - self.ta
        else: #circular orbit
            if self.i>=self.eps: #inclined orbit
                self.ta = np.arccos(np.dot(N, self.R)/(n*self.r))
                if self.R[2]<0.0:
                    self.ta = 2*np.pi - self.ta
            else: #equatorial orbit
                self.ta = np.arccos(self.R[0]/self.r)
                if self.R[1]<0.0:
                    self.ta = 2*np.pi - self.ta
        
        self.a = (self.h**2/self.mu)/(1 - self.e**2) #semimajor axis (m)
        
    def compute_sv(self):
        
        """
        compute the spacecraft state vector, specific angular momentum vector and eccentricity vector
        from its classical orbital elements
        """
        
        self.h = (self.mu*self.a*(1 - self.e**2))**0.5 #angular momentum magnitude (m^2/s)
        self.r = (self.h**2/self.mu)/(1 + self.e*np.cos(self.ta)) #distance from central body (m)
        
        r_per = self.r*np.array([np.cos(self.ta), np.sin(self.ta), 0.0]) #position vector in perifocal reference frame (m)
        v_per = (self.mu/self.h)*np.array([-np.sin(self.ta), self.e + np.cos(self.ta), 0.0]) #velocity vector in perifocal reference frame (m/s)
        
        Q_per2equ = self.perifocal2equatorial() #rotation matrix from perfocal to equatorial reference frame
        
        self.R = Q_per2equ@r_per #position vector in equatorial reference frame (m)
        self.V = Q_per2equ@v_per #velocity vector in equatorial reference frame (m/s)
        
        self.compute_angular_momentum()
        self.compute_eccentricity_vector()