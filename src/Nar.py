#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 23:56:09 2023

@author: utente
"""
import numpy as np
import torch


# Pr = 2/3

# T_ref=273.15
# d_ref=4.04e-10
# m_ref=6.6335209e-26
# omega=0.25 

# R = 208.1
# gamma=5/3

# Reference viscosity  ----->  T_ref=273.15, d_ref=3.76e-10, m=6.6335209e-26, omega=0.25 --> Argon
Mu_ref = lambda m, T_ref, d_ref, omega : 15*np.sqrt(m*2*np.pi*T_ref*1.380649e-23)/(2*(5-2*omega)*(7-2*omega)*np.pi*d_ref**2)

# MeanFreePath - Hard sphere
def Lam(p,T,d): 
    kb= 1.380649e-23
    return kb * 0.5*(T[:-1]+T[1:])/(np.sqrt(2)*np.pi*0.5*(p[:-1]+p[1:])*d**2)

# MeanFreePath  - Variable Hard Sphere
def LamVHS(p,T,mu,R): 
    return 0.5*(mu[:-1] + mu[1:])/(0.5*(p[:-1]+p[1:])) * np.sqrt(np.pi*R*0.5*(T[:-1]+T[1:])/2)

# pp = 250
# tt = 2500
# pp = 6.67
# tt = 300

# p, T = np.array([pp,pp]), np.array([tt,tt])

# l = Lam(p, T, d_ref)



# mu0 = Mu_ref(m_ref, T_ref, d_ref, omega)
# kT0 = mu0*gamma/(gamma-1)*R/Pr


# mu   = mu0*((T/T_ref)**omega)
# kT   = kT0*((T/T_ref)**omega)

# l2 = LamVHS(p,T,mu,R)
