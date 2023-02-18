#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 20:41:45 2023

@author: utente
"""
import numpy as np
import sys 
import os
import torch
import torch.nn.functional as F
from scipy.interpolate import splev, splrep, LinearNDInterpolator
import matplotlib.pyplot as plt

def getInitialCondition(flow,dom):
        
    x_BC=   []
    y_BC=   []
    
    p_BC=   []
    u_BC=   []
    v_BC=   []
    T_BC=   []
    
    for ID in range(dom.nPoints):
        if dom.getBCMarker(ID)=='WALL':
            p_BC+=   [flow.p1]
            u_BC+=   [0.]
            v_BC+=   [0.]
            T_BC+=   [flow.Tw]
            
        elif dom.getBCMarker(ID)=='SYM_up':
            p_BC+=   [flow.p1]
            u_BC+=   [flow.u1]
            v_BC+=   [0.]
            T_BC+=   [flow.T1]
    
        elif dom.getBCMarker(ID)=='INLET':
            p_BC+=   [flow.p1]
            u_BC+=   [flow.u1]
            v_BC+=   [0.]
            T_BC+=   [flow.T1]
    
        elif dom.getBCMarker(ID)=='SYM_down':
            p_BC+=   [flow.p1]
            u_BC+=   [flow.u1]
            v_BC+=   [0.]
            T_BC+=   [flow.T1]
        
        elif dom.getBCMarker(ID)=='OUTLET':
            p_BC+=   [flow.p1]
            u_BC+=   [flow.u1]
            v_BC+=   [0.]
            T_BC+=   [flow.T1]
    
        else:
            continue
        
        x_BC += [dom.getPoint(ID)[0]]     
        y_BC += [dom.getPoint(ID)[1]]
    
    p = LinearNDInterpolator(list(zip(x_BC, y_BC)), p_BC)
    u = LinearNDInterpolator(list(zip(x_BC, y_BC)), u_BC)
    v = LinearNDInterpolator(list(zip(x_BC, y_BC)), v_BC)
    T = LinearNDInterpolator(list(zip(x_BC, y_BC)), T_BC)
    
    X, Y = dom.X_mesh, dom.Y_mesh
    
    P = p(X,Y)
    U = u(X,Y)
    V = v(X,Y)
    TT= T(X,Y)
    
    fig, ax = plt.subplots()
    cp =ax.contourf(X,Y, U, 10)
    cbar = fig.colorbar(cp)

    plt.show()
    
    #Comservative variables
    rho=    (P/flow.R/TT)
    rhoU=   (rho*U).reshape((len(dom.x)*len(dom.y)), order='F')
    rhoV=   (rho*V).reshape((len(dom.x)*len(dom.y)), order='F')
    rhoE=   (rho*(0.5*(U**2 +V**2) + flow.Cv*TT)).reshape((len(dom.x)*len(dom.y)), order='F')
    
    fig, ax = plt.subplots()
    cp = ax.contourf(X,Y,rho, 10)
    cbar = fig.colorbar(cp)
    
    
    rho=    rho.reshape((len(dom.x)*len(dom.y)), order='F')
    

    q0 = np.concatenate( (rho, rhoU, rhoV, rhoE) )

    return q0 



def getFreeStreamFV(flow, dom, elem):
        
    p = np.ones((elem.nElem))*flow.p1
    u = np.ones((elem.nElem))*flow.u1
    v = np.ones((elem.nElem))*0.
    T = np.ones((elem.nElem))*flow.T1


    
    # for ID in range(dom.nPoints):
    #     if dom.getBCMarker(ID)=='WALL':
    #         p[ID]=   flow.p1
    #         u[ID]=   0.
    #         v[ID]=   0.
    #         T[ID]=   flow.Tw
            
    #     elif dom.getBCMarker(ID)=='SYM_up':
    #         p[ID]=   flow.p1
    #         u[ID]=   flow.u1
    #         v[ID]=   0.
    #         T[ID]=   flow.T1
    
    #     elif dom.getBCMarker(ID)=='INLET':
    #         p[ID]=   flow.p1
    #         u[ID]=   flow.u1
    #         v[ID]=   0.
    #         T[ID]=   flow.T1
            
    #     elif dom.getBCMarker(ID)=='SYM_down':
    #         p[ID]=   flow.p1
    #         u[ID]=   flow.u1
    #         v[ID]=   0.
    #         T[ID]=   flow.T1
        
    #     elif dom.getBCMarker(ID)=='OUTLET':
    #         p[ID]=   flow.p1
    #         u[ID]=   flow.u1
    #         v[ID]=   0.
    #         T[ID]=   flow.T1
            
    #     else:
    #         continue

    
    #Comservative variables
    rho=    (p/flow.R/T)
    rhoU=   (rho*u)#.reshape((len(dom.x)*len(dom.y)), order='F')
    rhoV=   (rho*v)#.reshape((len(dom.x)*len(dom.y)), order='F')
    rhoE=   (rho*(0.5*(u**2 +v**2) + flow.Cv*T))#.reshape((len(dom.x)*len(dom.y)), order='F')

    rho=    rho
    

    q0 = np.concatenate( (rho, rhoU, rhoV, rhoE) )

    return q0 

def getFromCSV(path,flow):
    my_data = np.genfromtxt(path, delimiter=',')
    
    u = my_data[:,3]
    v = my_data[:,4]
    p = my_data[:,5]
    T = my_data[:,6]

    #Comservative variables
    rho=    (p/flow.R/T)
    rhoU=   (rho*u)#.reshape((len(dom.x)*len(dom.y)), order='F')
    rhoV=   (rho*v)#.reshape((len(dom.x)*len(dom.y)), order='F')
    rhoE=   (rho*(0.5*(u**2 +v**2) + flow.Cv*T))#.reshape((len(dom.x)*len(dom.y)), order='F')    
    
    q0 = np.concatenate( (rho, rhoU, rhoV, rhoE) )
    
    return q0

def getAdjFromCSV(path,elem):
    my_data = np.genfromtxt(path, delimiter=',')
    
    rhoA  = my_data[:,0]
    rhoUA = my_data[:,1]
    rhoVA = my_data[:,2]
    rhoEA = my_data[:,3]

    #Comservative variables
    rhoA=     torch.DoubleTensor(rhoA .reshape((elem.Nx-2,elem.Ny-2), order='F')) 
    rhoUA=    torch.DoubleTensor(rhoUA.reshape((elem.Nx-2,elem.Ny-2), order='F')) 
    rhoVA=    torch.DoubleTensor(rhoVA.reshape((elem.Nx-2,elem.Ny-2), order='F')) 
    rhoEA=    torch.DoubleTensor(rhoEA.reshape((elem.Nx-2,elem.Ny-2), order='F')) 
    
    
    
    return rhoA, rhoUA, rhoVA, rhoEA

def getTargetFromCSV(path,elem,flow):
    my_data = np.genfromtxt(path, delimiter=',')
    
    u  = my_data[:,3]
    v  = my_data[:,4]
    p  = my_data[:,5]
    T  = my_data[:,6]
    
    #Comservative variables
    rhoT=    (p/flow.R/T)
    rhoUT=   (rhoT*u)#.reshape((len(dom.x)*len(dom.y)), order='F')
    rhoVT=   (rhoT*v)#.reshape((len(dom.x)*len(dom.y)), order='F')
    rhoET=   (rhoT*(0.5*(u**2 +v**2) + flow.Cv*T))#.reshape((len(dom.x)*len(dom.y)), order='F')    
    
    #Comservative variables
    rhoT=     torch.DoubleTensor(rhoT .reshape((elem.Nx,elem.Ny), order='F')) 
    rhoUT=    torch.DoubleTensor(rhoUT.reshape((elem.Nx,elem.Ny), order='F')) 
    rhoVT=    torch.DoubleTensor(rhoVT.reshape((elem.Nx,elem.Ny), order='F')) 
    rhoET=    torch.DoubleTensor(rhoET.reshape((elem.Nx,elem.Ny), order='F')) 
    
    
    
    return rhoT[1:-1,1:-1], rhoUT[1:-1,1:-1], rhoVT[1:-1,1:-1], rhoET[1:-1,1:-1]
