#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 00:14:01 2023

@author: utente
"""

from Utils import *
import torch.nn.functional as F

# Return stress tensor
def Tau(dudx,dvdx, dudy,dvdy, flow):
    #Computation of stress tensor    
    tauxx= flow.mu*(4./3.*dudx - 2./3.*dvdy)    
    tauyy= flow.mu*(4./3.*dvdy - 2./3.*dudx)    
    tauxy= flow.mu*(dudy + dvdx)
    
    return tauxx, tauyy, tauxy

def ViscousFluxes(dudx, dudy):
    pass

def LocalTimeSteps(rho, u,v,p,T, flow,elem):
    C=4.
    
    c = torch.sqrt( flow.gamma*flow.R*T )
    
    Lx  = (torch.abs(u[1:-1,1:-1] .detach()) + c[1:-1,1:-1] .detach())*elem.dy_m[1:-1,1:-1]
    Ly  = (torch.abs(v[1:-1,1:-1] .detach()) + c[1:-1,1:-1] .detach())*elem.dx_m[1:-1,1:-1]
    
    A = ((((4./3. - flow.gamma)/rho )*( 4./3./rho > flow.gamma/ rho) + flow.gamma/rho)*(flow.mu/flow.Pr)).detach() [1:-1,1:-1]
    
    Lxv = A*elem.dy_m[1:-1,1:-1]/elem.dx_m[1:-1,1:-1]
    Lyv = A*elem.dx_m[1:-1,1:-1]/elem.dy_m[1:-1,1:-1]
    
    dtMax = elem.CFL*elem.dx_m[1:-1,1:-1] *elem.dy_m[1:-1,1:-1] /(Lx + Ly + C*(Lxv + Lyv))
    dtMax = F.pad(dtMax, pad=(1,1,1,1))
    return dtMax[1:-1,1:-1]

def updateBC_cat(list_var,elem,flow):
    u, v, p, T = list_var
    
    #Padding for boundary condition imposition
    u = F.pad(u ,pad=(1, 1, 1, 1),value=0.)
    v = F.pad(v ,pad=(1, 1, 1, 1),value=0.)
    p = F.pad(p ,pad=(1, 1, 1, 1),value=0.)
    T = F.pad(T ,pad=(1, 1, 1, 1),value=0.)
    
    #Inlet
    #ones = torch.ones((1,elem.Ny), dtype=torch.float64)
    p[0,:] = flow.p1#*ones
    u[0,:] = flow.u1#*ones
    v[0,:] = 0.     #*ones
    T[0,:] = flow.T1#*ones
    
    #SymBottom
    p[1:elem.xS,0] =    p[1:elem.xS,1] #.detach()
    u[1:elem.xS,0] =    u[1:elem.xS,1] #.detach()
    v[1:elem.xS,0] =    0.
    T[1:elem.xS,0] =    T[1:elem.xS,1] #.detach()

    #Wall     
    p[elem.xS:,0] =    p[elem.xS:,1] #.detach()
    u[elem.xS:,0] =    0.
    v[elem.xS:,0] =    0.
    T[elem.xS:,0] =    flow.Tw 
    
    #OutRight
    p[-1,1:]  = p[-2,1:] .detach()
    u[-1,1:]  = u[-2,1:] .detach()
    v[-1,1:]  = v[-2,1:] .detach()
    T[-1,1:]  = T[-2,1:] .detach()
    
    #OutUP
    p[1:,-1]  = p[1:,-2] .detach()
    u[1:,-1]  = u[1:,-2] .detach()
    v[1:,-1]  = v[1:,-2] .detach()
    T[1:,-1]  = T[1:,-2] .detach()
    
    return u, v, p, T
       
def grad(u,v,T,elem, flow, BC=True):
    #Gradients - cell centered
    dudx = ddxGGM(u,elem)
    dudy = ddyGGM(u,elem)
    
    dvdx = ddxGGM(v,elem)
    dvdy = ddyGGM(v,elem)
    
    dTdx = ddxGGM(T,elem)
    dTdy = ddyGGM(T,elem)
    

    if BC:
        #Build BCs
        #Padding for boundary condition imposition
        dudx = F.pad(dudx ,pad=(1, 1, 1, 1),value=0.)
        dvdx = F.pad(dvdx ,pad=(1, 1, 1, 1),value=0.)
        dTdx = F.pad(dTdx ,pad=(1, 1, 1, 1),value=0.)
        
        dudy = F.pad(dudy ,pad=(1, 1, 1, 1),value=0.)
        dvdy = F.pad(dvdy ,pad=(1, 1, 1, 1),value=0.)
        dTdy = F.pad(dTdy ,pad=(1, 1, 1, 1),value=0.)
        
        #d_dX ---------------------------------------------------------------------------
        #Dir 
        dudx[0,:] = (u[1,:] - u[0,:])/elem.dx_m[0,0]/2
        dvdx[0,:] = (v[1,:] - v[0,:])/elem.dx_m[0,0]/2
        dTdx[0,:] = (T[1,:] - T[0,:])/elem.dx_m[0,0]/2

        #OutRIGHT
        dudx[-1,1:]  = 0.
        dvdx[-1,1:]  = 0.
        dTdx[-1,1:]  = 0.
        
        #OutUP
        dudx[1:,-1]   = dudx[1:,-2].detach()
        dvdx[1:,-1]   = dvdx[1:,-2].detach()
        dTdx[1:,-1]   = dTdx[1:,-2].detach()
        
        #SymBottom
        dudx[1:elem.xS,0]   = dudx[1:elem.xS,1].detach()
        dvdx[1:elem.xS,0]   = dvdx[1:elem.xS,1].detach()
        dTdx[1:elem.xS,0]   = dTdx[1:elem.xS,1].detach()

        #Wall
        dudx[elem.xS:,0]   = 0.
        dvdx[elem.xS:,0]   = 0.
        dTdx[elem.xS:,0]   = 0. 

        
        
        #d_dY ---------------------------------------------------------------------------
        #Inlet
        #Dir
        dudy[0,:] = 0.
        dvdy[0,:] = 0.
        dTdy[0,:] = 0.

        #OutRIGHT
        dudy[-1,1:]  = dudy[-2,1:].detach()
        dvdy[-1,1:]  = dvdy[-2,1:].detach()
        dTdy[-1,1:]  = dTdy[-2,1:].detach()

        #OutUp
        dudy[1:,-1]  = 0.
        dvdy[1:,-1]  = 0.
        dTdy[1:,-1]  = 0.
        
        #SymBottom
        dudy[1:elem.xS,0]   =  0.
        dvdy[1:elem.xS,0]   =  0.
        dTdy[1:elem.xS,0]   =  0.

        #Wall
        dudy[elem.xS:,0]   =  ((u[elem.xS:,1] - u[elem.xS:,0])/(0.5*(elem.dy_m[elem.xS,0] + elem.dy_m[elem.xS,1]))) #.detach()
        dvdy[elem.xS:,0]   =  0.
        dTdy[elem.xS:,0]   =  ((T[elem.xS:,1] - T[elem.xS:,0])/(0.5*(elem.dy_m[elem.xS,0] + elem.dy_m[elem.xS,1])) ) #.detach()

            
    return dudx,dvdx,dTdx, dudy,dvdy,dTdy
