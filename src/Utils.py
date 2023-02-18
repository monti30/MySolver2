#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:33:12 2023

@author: utente
"""
import numpy as np
import sys 
import os
import torch
from scipy.interpolate import splev, splrep, LinearNDInterpolator
from scipy.optimize    import newton_krylov

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

# Matrix derivatives
def ddx_M(u,x):
    D=          torch.zeros(u.size(), dtype= torch.float64)
    D[1:-1,:]=    (u[2:,:] - u[:-2,:])/(x[2:,:] - x[:-2,:])
    D[0,:]=       (u[1,:]  - u[0,:])  /(x[1,:]  - x[0,:])
    D[-1,:]=      (u[-1,:] - u[-2,:]) /(x[-1,:] - x[-2,:])
    
    return D

def ddy_M(u,y):
    D=          torch.zeros(u.size(), dtype= torch.float64)
    D[:,1:-1]=    (u[:,2:] - u[:,:-2])/(y[:,2:] - y[:,:-2])
    D[:,0]=       (u[:,1]  - u[:,0])  /(y[:,1]  - y[:,0])
    D[:,-1]=      (u[:,-1] - u[:,-2]) /(y[:,-1] - y[:,-2])
    
    return D

# Return values on dir faces
def facevalX(u,elem, pad=True):
    if pad:
        return elem.alfa_x * u[:-1,1:-1] + (1-elem.alfa_x) *u[1:,1:-1]
    else:
        return elem.alfa_x[1:-1,:]*u[:-1,:] + (1 - elem.alfa_x[1:-1,:])*u[1:,:]
    
def facevalY(u,elem, pad=True):
    if pad:
        return elem.alfa_y * u[1:-1,:-1] + (1- elem.alfa_y) *u[1:-1,1:]
    else:
        return elem.alfa_y[:,1:-1]*u[:,:-1] + (1 - elem.alfa_y[:,1:-1])*u[:,1:]

# GreenGauss Gradients
def ddxGGM(u,elem):
    u_fx = elem.alfa_x * u[:-1,1:-1] + (1-elem.alfa_x) *u[1:,1:-1]
    dudx = (u_fx[1:,:] -u_fx[:-1,:])/torch.DoubleTensor(elem.dx_m[1:-1,1:-1]) #per fare le derivate comprese di boundary devo prendere tutti i dy_m

    return dudx

def ddyGGM(u,elem):
    
    u_fy = elem.alfa_y * u[1:-1,:-1] + (1- elem.alfa_y) *u[1:-1,1:]
    dudy = (u_fy[:,1:] -u_fy[:, :-1])/torch.DoubleTensor(elem.dy_m[1:-1,1:-1]) #per fare le derivate comprese di boundary devo prendere tutti i dy_m

    return dudy

# Derivatives of a vector-like solution
def ddx(u,dom):
    if not type(u)==type(torch.ones((1))):
        u = torch.DoubleTensor(u)
        
    D=          torch.zeros(u.size(), dtype= torch.float64)
            
    D[dom.ID_int]=          (u[dom.ID_int        +1] - u[dom.ID_int        -1])/(dom.X[dom.ID_int        +1] - dom.X[dom.ID_int        -1])
    D[dom.ID_up[1:-1]]=     (u[dom.ID_up[1:-1]   +1] - u[dom.ID_up[1:-1]   -1])/(dom.X[dom.ID_up[1:-1]   +1] - dom.X[dom.ID_up[1:-1]   -1])
    D[dom.ID_down[1:-1]]=   (u[dom.ID_down[1:-1] +1] - u[dom.ID_down[1:-1] -1])/(dom.X[dom.ID_down[1:-1] +1] - dom.X[dom.ID_down[1:-1] -1])
    
    D[dom.ID_left]=         (u[dom.ID_left+1] - u[dom.ID_left   ])/(dom.X[dom.ID_left+1] - dom.X[dom.ID_left   ])
    D[dom.ID_right]=        (u[dom.ID_right ] - u[dom.ID_right-1])/(dom.X[dom.ID_right ] - dom.X[dom.ID_right-1])
    
    return D

def ddy(u,dom):
    if not type(u)==type(torch.ones((1))):
        u = torch.DoubleTensor(u)
        
    D=          torch.zeros(u.size(), dtype= torch.float64)
            
    D[dom.ID_int]=          (u[dom.ID_int           +dom.Nx] - u[dom.ID_int         -dom.Nx])/(dom.Y[dom.ID_int         +dom.Nx] - dom.Y[dom.ID_int         -dom.Nx])
    D[dom.ID_left[1:-1]]=   (u[dom.ID_left[1:-1]    +dom.Nx] - u[dom.ID_left[1:-1]  -dom.Nx])/(dom.Y[dom.ID_left[1:-1]  +dom.Nx] - dom.Y[dom.ID_left[1:-1]  -dom.Nx])
    D[dom.ID_right[1:-1]]=  (u[dom.ID_right[1:-1]   +dom.Nx] - u[dom.ID_right[1:-1] -dom.Nx])/(dom.Y[dom.ID_right[1:-1] +dom.Nx] - dom.Y[dom.ID_right[1:-1] -dom.Nx])
    
    D[dom.ID_down]=         (u[dom.ID_down          +dom.Nx] - u[dom.ID_down               ])/(dom.Y[dom.ID_down        +dom.Nx] - dom.Y[dom.ID_down                ])
    D[dom.ID_up]=           (u[dom.ID_up                   ] - u[dom.ID_up          -dom.Nx])/(dom.Y[dom.ID_up                 ] - dom.Y[dom.ID_up           -dom.Nx])
    
    return D

# Return the index above a certain value
def index(xGrid, x):
    j=None
    for i in range(len(xGrid)):
        if xGrid[i] >= x:
            j= i
            break
    return j 

# Geometric progression grid
def geomGrid(x0,xf, Nx, p):
    grid = []
    Ne = Nx-1
    if not (p==1 or p==1.):
        dx0 = (xf - x0)*(1 - p)/(1-p**Ne)
             
        for i in range(Ne+1):
            grid += [(1-p**i)/(1-p)*dx0 + x0]
    
    else:
        dx0 = (xf - x0)/Ne
        for i in range(Ne+1):
            grid += [dx0*i + x0]
    
    print('1st cell dimension is',dx0)
    return np.array(grid)
    
    
def getInternalField(q_):
    temp = []
    i=0
    for var in q_:
        temp += [var[1:-1,1:-1]]
        i+=1
    return temp
        