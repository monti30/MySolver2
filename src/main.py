#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:48:22 2023

@author: utente
"""

import numpy as np
import sys 
import os
import torch
import torch.nn.functional as F
import pickle
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import splev, splrep, LinearNDInterpolator
from torch.autograd import Variable

from Init import *
from Driver import *
import IC
from Utils import *
import RHS
import Flux
import GlobalSettings as GS

plt.close('all')
# %% INSTANCES

Argon   = flowModel('Argon')
flow    = flowProperties(Argon,  M1,T1,p1,Tw)
flow.printAirModel()

dom     = domain(xS, x,y, nIter,dt)
dom.dtA = dtA

elem    = centroids(dom)
elem.xS = dom.xS

elem.localTimeStepping = localTimeStepping
if localTimeStepping: elem.CFL = CFL

# %% INITIAL CONDITION ------------------------------------------------------------------------------------------

#from external file:
restartDir = myCase.restartDir
restartFile = myCase.restartFile
restart_file_path = str(restartDir) + '/' + str(restartFile)


print('\nRestart:', Restart,'\nRestart Adjoint:',RestartAdj)

if not Restart:
    initialSol = solution(IC.getFreeStreamFV(flow, dom, elem), dom, elem, flow)
    print('Solution initialized from BC ...\n')

# Restart solution-----------------------------------------------------------
if Restart:
    try:
        initialSol = solution(IC.getFromCSV(myCase.restartFile,flow), dom, elem, flow)
        # initialSol = pickle.load(open(restart_file_path,'rb'))
        myCase.HaveTarget = False
    except:
        try:
            initialSol, Sol_T = pickle.load(open(restart_file_path,'rb'))
            
        except:
            raise Exception("Error: Could not load restart file")


# %% TARGET DATA ----------------------------------------------------------------------------------------------------
haveTarget = True
if haveTarget:
    rhoT, rhoUT, rhoVT, rhoET = IC.getTargetFromCSV('./TestAdjoint/solfM5T-2.csv'.format(int(flow.M1)),elem,flow)


# %% NEURAL NETWORK MODEL


if True: #myCase.train:
#Creating dummy model
    torch.manual_seed(1000)
    flow.model = GS.Theta(1)
    
    # Imposing start values
    flow.model.fc2.weight.data = torch.tensor([[ 2. ]])
    flow.model.fc2.bias.data = torch.tensor([0.,])
    
    # Imposing only weight as a parameter
    flow.model.fc2.bias.requires_grad = False
    print(flow.model.fc2.weight.requires_grad)
    
    # Convert everything to Double
    for name, param in flow.model.fc2.named_parameters():
        param.data = param.data.type(torch.DoubleTensor)
        print(name, param.data)
    
    
    
                            
    dummyModel_name=    'model_dummy'
    dummyModel_dir=     myCase.outputDir + '/' + 'dummyModel'
    dummyModel_path=    dummyModel_dir + '/' + dummyModel_name
    
    # Load the model if it exists
    if os.path.exists(dummyModel_path):
        flow.model.load_state_dict( torch.load( dummyModel_path ) )
        print('\n --> Loaded model ' + dummyModel_path)

    # Make the directory if it doesn't
    if not os.path.exists(dummyModel_dir):
        os.makedirs(dummyModel_dir)        


# Optimizer for dummy model --------------
    flow.optimizer = torch.optim.SGD(flow.model.parameters(), lr=myCase.LR, momentum=0.9)
    
    dummyOptim_name=    'optim_dummy'
    dummyOptim_path=    dummyModel_dir + '/' + dummyOptim_name
    
    # # Load the optimizer if it exists
    # if os.path.exists(dummyOptim_path):
    #     flow.optimizer.load_state_dict( torch.load( dummyOptim_path ))
    #     LR = 1e-8 #flow.optimizer.param_groups[0]['lr']
    #     print(LR)
    
    for param_group in flow.optimizer.param_groups:
        param_group['lr'] = 1e-4
    
    if RestartAdj:
        for param_group in flow.optimizer.param_groups:
            param_group['lr'] = np.genfromtxt('solfAM{}.csv'.format(int(flow.M1)), delimiter=',')[-1,-1]

        print(' --> Loaded optim {}, LR={}\n'.format(dummyOptim_path,flow.optimizer.param_groups[0]['lr']))
        
    #Scheduler - ExpDecay
    flow.scheduler = torch.optim.lr_scheduler.ExponentialLR(flow.optimizer, gamma=0.9)        
    LRs = [flow.optimizer.param_groups[0]["lr"],]
    

# %% INITIALIZATION OF TORCH VECTORS --------------------------------------------------------------------------------
sol=        solution(copy.deepcopy(torch.DoubleTensor(initialSol.q)), dom, elem, flow)
flow.nvar=   4

# Initialization ----------------------------------------------------------------------------------------------------
# Get conservative variables
rho, rhoU, rhoV, rhoE = sol.reshape2matrix()
rho, rhoU, rhoV, rhoE = getInternalField([rho, rhoU, rhoV, rhoE])

# Computing primitive variables
p = (flow.gamma - 1)*(rhoE - 0.5*(rhoU*rhoU + rhoV*rhoV)/rho)
u = rhoU/rho
v = rhoV/rho
T = (rhoE - 0.5*(rhoU*rhoU + rhoV*rhoV)/rho)/rho/flow.Cv

# Enforcing BC on primitive variables ---> NECESSARY FOR ADJOINT COMPARED WITH 0 SOLUTION
u, v, p, T = RHS.updateBC_cat([u, v, p, T],elem,flow)
    
# Store initial solution ---------------------------------------------------------------------------------------------
p0 = copy.deepcopy(p)
u0 = copy.deepcopy(u)
v0 = copy.deepcopy(v)
T0 = copy.deepcopy(T)
rho0=    copy.deepcopy(p/flow.R/T)
rhoU0=   copy.deepcopy(rho0*u)
rhoV0=   copy.deepcopy(rho0*v)
rhoE0=   copy.deepcopy(rho0*(0.5*(u**2 +v**2) + flow.Cv*T))


# Initialization of adjoint variables --------------------------------------------------------------------------------
rhoA  = torch.zeros( rho.shape ,dtype=torch.float64) 
rhoUA = torch.zeros( rho.shape ,dtype=torch.float64) 
rhoVA = torch.zeros( rho.shape ,dtype=torch.float64) 
rhoEA = torch.zeros( rho.shape ,dtype=torch.float64) 

qdot1A = torch.zeros( rhoA.shape ,dtype=torch.float64)
qdot2A = torch.zeros( rhoA.shape ,dtype=torch.float64)
qdot3A = torch.zeros( rhoA.shape ,dtype=torch.float64)
qdot4A = torch.zeros( rhoA.shape ,dtype=torch.float64)

# Restart adj var from file
if RestartAdj:
    rhoA, rhoUA, rhoVA, rhoEA = IC.getAdjFromCSV('solfAM{}.csv'.format(int(flow.M1)),elem)


# %% INTEGRATION ------------------------------------------------------------------------------------------------

Train = myCase.train
dt = dom.dt

if Train:
    # Evaluating loss function
    loss0 =           ((0.5*(rho0 [1:-1,1:-1] - rhoT )**2 / torch.max(rhoT ) +
                        0.5*(rhoU0[1:-1,1:-1] - rhoUT)**2 / torch.max(rhoUT) + 
                        0.5*(rhoV0[1:-1,1:-1] - rhoVT)**2 / torch.max(rhoVT) +
                        0.5*(rhoE0[1:-1,1:-1] - rhoET)**2 / torch.max(rhoET)          )/(4*elem.nElem)).sum()



if not elem.localTimeStepping:
        #CFL
        CFL = min([(flow.u1 + flow.c1)*dom.dt/min(min(elem.dx_m[:,0]), min(elem.dy_m[0,:])),( flow.c1)*dom.dt/min(elem.dy_m[0,:])] )
        print('\nCFL:',CFL,'\n\n')
else:
    print('\nCFL:',elem.CFL,'\n\n')

# Header
print('iter         res(rho)       res(rhoU)      res(rhoE)   |   res(rhoA)      res(rhoUA)      res(rhoEA)\n -----------------------------------------------------------------------')

# EULER's METHOD ------------------------------------------------------------------------------------------------
for i in range(nIter):  
    # BOUNDARY CONDITIONS ---------------------------------------------------------------------------------------
    # Conservative variables --> Graphs' leaves    
    q1 = Variable(rho,  requires_grad=Train)
    q2 = Variable(rhoU, requires_grad=Train)
    q3 = Variable(rhoV, requires_grad=Train)
    q4 = Variable(rhoE, requires_grad=Train)
        
    # Primitive var
    u = q2/q1
    v = q3/q1
    p = (flow.gamma - 1)*(q4 - 0.5*(q2*q2 + q3*q3)/q1)
    T = (q4 - 0.5*(q2*q2 + q3*q3)/q1)/q1/flow.Cv
    
    # Padded with boundaries --> Cat
    u, v, p, T = RHS.updateBC_cat([u,v,p,T], elem, flow)
    rhop = p/(flow.R * T)
    
    # ADVECTIVE FLUXES -------------------------------------------------------------------------------------------
    Frho, FrhoU, FrhoV, FrhoE,  Grho, GrhoU, GrhoV, GrhoE = Flux.AUSM(u,v,rhop,p,T,flow,elem)
    
    # Advective source terms
    dfrho      = (Frho [1:,:] - Frho [:-1,:])/torch.Tensor(elem.dx_m[1:-1,1:-1])   + (Grho [:, 1:] - Grho [:,:-1])/torch.Tensor(elem.dy_m[1:-1,1:-1]) 
    dfrhoU     = (FrhoU[1:,:] - FrhoU[:-1,:])/torch.Tensor(elem.dx_m[1:-1,1:-1])   + (GrhoU[:, 1:] - GrhoU[:,:-1])/torch.Tensor(elem.dy_m[1:-1,1:-1])
    dfrhoV     = (FrhoV[1:,:] - FrhoV[:-1,:])/torch.Tensor(elem.dx_m[1:-1,1:-1])   + (GrhoV[:, 1:] - GrhoV[:,:-1])/torch.Tensor(elem.dy_m[1:-1,1:-1])
    dfrhoE     = (FrhoE[1:,:] - FrhoE[:-1,:])/torch.Tensor(elem.dx_m[1:-1,1:-1])   + (GrhoE[:, 1:] - GrhoE[:,:-1])/torch.Tensor(elem.dy_m[1:-1,1:-1])
    # ------------------------------------------------------------------------------------------------------------    

    # VISCOUS FLUXES ---------------------------------------------------------------------------------------------
    # Computing gradients with enforced BCs
    dudx,dvdx,dTdx, dudy,dvdy,dTdy = RHS.grad(u,v,T,elem,flow, BC=True)
    
    # Viscous model - Power law
    flow.mu   = (1. + Train*flow.model(torch.DoubleTensor([1,]))) *flow.mu0*((T/flow.T_ref)**flow.omega) 
    flow.kT   = flow.mu*(flow.Cv + flow.R)/flow.Pr
    
    # Stresses
    tau = RHS.Tau(dudx,dvdx, dudy,dvdy, flow) # Derivatives evaluated at cell-center

    # Visc fluxes: X-Direction
    FrhoU_v = facevalX(tau[0] , elem)
    FrhoV_v = facevalX(tau[2] , elem)
    FrhoE_v = FrhoU_v*facevalX(u, elem) + FrhoV_v*facevalX(v, elem) + facevalX(flow.kT*dTdx,elem)  

    # Visc fluxes: Y-Direction
    GrhoU_v = facevalY(tau[2] , elem)
    GrhoV_v = facevalY(tau[1] , elem)
    GrhoE_v = GrhoU_v*facevalY(u, elem) + GrhoV_v*facevalY(v, elem) + facevalY(flow.kT*dTdy,elem)
    
    # Viscous source terms 
    dfrhoU_v     = (FrhoU_v[1:,:] - FrhoU_v[:-1,:])/torch.Tensor(elem.dx_m[1:-1,1:-1])   + (GrhoU_v[:, 1:] - GrhoU_v[:,:-1])/torch.Tensor(elem.dy_m[1:-1,1:-1])
    dfrhoV_v     = (FrhoV_v[1:,:] - FrhoV_v[:-1,:])/torch.Tensor(elem.dx_m[1:-1,1:-1])   + (GrhoV_v[:, 1:] - GrhoV_v[:,:-1])/torch.Tensor(elem.dy_m[1:-1,1:-1])
    dfrhoE_v     = (FrhoE_v[1:,:] - FrhoE_v[:-1,:])/torch.Tensor(elem.dx_m[1:-1,1:-1])   + (GrhoE_v[:, 1:] - GrhoE_v[:,:-1])/torch.Tensor(elem.dy_m[1:-1,1:-1])

    # Right-hand-side ------------------------------ 7.868271515018883---------------------------------------------------------------
    qdot1 = -dfrho
    qdot2 = -dfrhoU + dfrhoU_v
    qdot3 = -dfrhoV + dfrhoV_v
    qdot4 = -dfrhoE + dfrhoE_v 
    
    
    # Local time stepping
    if elem.localTimeStepping:
        dt = RHS.LocalTimeSteps(rhop.detach(), u.detach(),v.detach(),p.detach(),T.detach(), flow, elem)
            
    # ---------------------------------------------------------------------------------------------------------------
    # ADJOINT -------------------------------------------------------------------------------------------------------
    
    # Adjoint RHS
    if Train:

        h = ( torch.sum( qdot1 * rhoA ) +
              torch.sum( qdot2 * rhoUA) +
              torch.sum( qdot3 * rhoVA) +
              torch.sum( qdot4 * rhoEA) )
        
        h.backward()                                                        #27s vs 11s fo 1000 iterations with backward on
        
        
        ## dqA/dt = qA * dF/dq + dJ/dq
        qdot1A = (q1.grad.data ).detach()  + (q1.detach() - rhoT )/torch.max(rhoT)/elem.nElem  
        qdot2A = (q2.grad.data ).detach()  + (q2.detach() - rhoUT)/torch.max(rhoUT)/elem.nElem  
        qdot3A = (q3.grad.data ).detach()  + (q3.detach() - rhoVT)/torch.max(rhoVT)/elem.nElem  
        qdot4A = (q4.grad.data ).detach()  + (q4.detach() - rhoET)/torch.max(rhoET)/elem.nElem  
        
        rhoA  += dt *qdot1A.detach() #[1:,:]
        rhoUA += dt *qdot2A.detach() #[1:,:]
        rhoVA += dt *qdot3A.detach() #[1:,:]
        rhoEA += dt *qdot4A.detach() #[1:,:]
        
        # Compute the gradients        
        grad = flow.model.fc2.weight.grad.data.item()
        
        if i%100 and True: flow.optimizer.step()
        flow.optimizer.zero_grad() #Reset the gradients
    # -------------------------------------------------------------------------------------------------------------
    # UPDATING SOLUTION -------------------------------------------------------------------------------------------------------------
    
    rho  += dt*qdot1. detach()
    rhoU += dt*qdot2. detach()
    rhoV += dt*qdot3. detach()
    rhoE += dt*qdot4. detach()


    
    # -------------------------------------------------------------------------------------------------------------   
    # -------------------------------------------------------------------------------------------------------------
    
    # SAVE INTERMEDIATE SOLUTION ----------------------------------------------------------------------------------
    if i%100==0: 
        # Residuals
        print("\n##{:}\t\t{:10.4e}\t\t{:10.4e}\t\t{:10.4e}\t\t{:10.4e}\t\t{:10.4e}\t\t{:10.4e}".format(i, 
                torch.max(abs(qdot1)*dt).item(),
                torch.max(abs(qdot2)*dt).item(),
                torch.max(abs(qdot4)*dt).item(),
                torch.max(abs(qdot1A)*dt).item(),
                torch.max(abs(qdot2A)*dt).item(),
                torch.max(abs(qdot4A)*dt).item(),   ))
        
        # Print grad
        if Train: 
            # Evaluating loss function
            loss =            ((0.5*(q1 - rhoT )**2 / torch.max(rhoT ) +
                                0.5*(q2 - rhoUT)**2 / torch.max(rhoUT) + 
                                0.5*(q3 - rhoVT)**2 / torch.max(rhoVT) +
                                0.5*(q4 - rhoET)**2 / torch.max(rhoET)          )/(4*elem.nElem)).sum()
            
            print('dJdTheta:', '{:10.4e}'.format(grad),
                  '\t\tTheta:','{:10.4e}'.format(flow.model.fc2.weight.item()),
                  '\t\tLoss:', '{:10.4e}'.format((loss/loss0).item() ))
            


        
        # Save intermidiate CSV
        Z = np.zeros( elem.X.size )
        CSV = np.c_[elem.X,elem.Y,Z, 
                    (u.detach() .numpy()).reshape((len(elem.X)), order='F'),
                    (v.detach() .numpy()).reshape((len(elem.X)), order='F'), 
                    (p.detach() .numpy()).reshape((len(elem.X)), order='F'), 
                    (T.detach() .numpy()).reshape((len(elem.X)), order='F')]
        
        np.savetxt("solM{}.csv".format(int(flow.M1)), CSV, delimiter=",", header= 'x{},y{},z0,u,v,p,T'.format(elem.Nx,elem.Ny))
        
    
        # ------------------------------------------------------------------------------------------------------------- 
        # SCHEDULER ---------------------------------------------------------------------------------------------------
        # Learning rate schedule
        if Train:
            if (loss/loss0 < myCase.threshObj):
                myCase.threshObj *= 0.65
                
                flow.scheduler.step()
                LRs.append(flow.optimizer.param_groups[0]["lr"])
                print('\nNew LR:',LRs[-1], '--------------------------------\n')

# %%
# Converting to numpy array
p = p.detach().numpy()
u = u.detach().numpy()                                      
v = v.detach().numpy()                                             
T = T.detach().numpy() 

# %% POSTPROCESS --------------------------------------------------------------------------------------------------
if Plot:
    
    # Save CSV
    Z = np.zeros( elem.X.size )
    CSV = np.c_[elem.X,elem.Y,Z, 
                u.reshape((len(elem.X)), order='F'), 
                v.reshape((len(elem.X)), order='F'), 
                p.reshape((len(elem.X)), order='F'), 
                T.reshape((len(elem.X)), order='F'), 
                (dudx.detach() .numpy()).reshape((len(elem.X)), order='F')]
    
    np.savetxt("solfM{}.csv".format(
        int(flow.M1)), CSV, delimiter=",", header= 'x{},y.{},z0,u,v,p,T,dudx'.format(elem.Nx,elem.Ny))

    # Save Adj CSV
    if myCase.train:
        # Adjoint vars are only internal field
        CSVA = np.c_[rhoA.numpy() .reshape((rhoA.shape[0]*rhoA.shape[1]), order='F'), 
                    rhoUA.numpy() .reshape((rhoA.shape[0]*rhoA.shape[1]), order='F'), 
                    rhoVA.numpy() .reshape((rhoA.shape[0]*rhoA.shape[1]), order='F'), 
                    rhoEA.numpy() .reshape((rhoA.shape[0]*rhoA.shape[1]), order='F'),
                    LRs[-1]*np.ones((rhoA.shape[0]*rhoA.shape[1]))]
        
        np.savetxt("solfAM{}.csv".format(
            int(flow.M1)), CSVA, delimiter=",", header= 'rhoA,rhoUA,rhoVA,rhoEA,LR'.format(elem.Nx,elem.Ny))


    # Plotting
    XeMesh = elem.X.reshape((u.shape), order='F')
    YeMesh = elem.Y.reshape((u.shape), order='F')
    
    fig, ax = plt.subplots(1,1)
    cp = ax.contourf(XeMesh,YeMesh, u, levels=1000)
    cbar = fig.colorbar(cp)
    plt.show()
    
# %%


