#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 23:23:50 2023

@author: utente
"""

from main import *

# c = torch.sqrt( flow.gamma*flow.R*T )
# cstar = torch.sqrt( 2.*flow.gamma*flow.R*T[:,-2:]/(flow.gamma+1)  +  (u[:,-2:]**2 + v[:,-2:]**2)*(flow.gamma-1)/(flow.gamma+1) )

# UL =    v  [1:-1 , -3:-1]
# UR =    v  [1:-1 , -2:  ]
# VL =    u  [1:-1 , -3:-1]
# VR =    u  [1:-1 , -2:  ]
# rhoL=   rho[1:-1 , -3:-1]
# rhoR=   rho[1:-1 , -2:  ]
# pL=     p  [1:-1 , -3:-1]
# pR=     p  [1:-1 , -2:  ]
# TL=     T  [1:-1 , -3:-1]
# TR=     T  [1:-1 , -2:  ]
# cL=     c  [1:-1 , -3:-1]
# cR=     c  [1:-1 , -2:  ]
# hL=     flow.gamma*flow.R*TL/(flow.gamma-1)  
# hR=     flow.gamma*flow.R*TR/(flow.gamma-1)  
# EL=    (flow.Cv*TL + 0.5*(UL**2 + VL**2))
# ER=    (flow.Cv*TR + 0.5*(UR**2 + VR**2))



Restart=True
RestartAdj=True

print('\nRestart:', Restart,'\nRestart Adjoint:',RestartAdj)



# Restart solution 1 --> mu =1.05mu-----------------------------------------------------------
if Restart:
    try:
        initialSol = solution(IC.getFromCSV('TestAdjoint/solfM5-1.csv',flow), dom, elem, flow)
        # initialSol = pickle.load(open(restart_file_path,'rb'))
        myCase.HaveTarget = False
    except:
        try:
            initialSol, Sol_T = pickle.load(open(restart_file_path,'rb'))
            
        except:
            raise Exception("Error: Could not load restart file")

sol=        solution(copy.deepcopy(torch.DoubleTensor(initialSol.q)), dom, elem, flow)
rho1, rhoU1, rhoV1, rhoE1 = sol.reshape2matrix()

# Restart solution 2 --> mu =1.01mu-----------------------------------------------------------
if Restart:
    try:
        initialSol = solution(IC.getFromCSV('TestAdjoint/solfM5-2.csv',flow), dom, elem, flow)
        # initialSol = pickle.load(open(restart_file_path,'rb'))
        myCase.HaveTarget = False
    except:
        try:
            initialSol, Sol_T = pickle.load(open(restart_file_path,'rb'))
            
        except:
            raise Exception("Error: Could not load restart file")

sol=        solution(copy.deepcopy(torch.DoubleTensor(initialSol.q)), dom, elem, flow)
rho2, rhoU2, rhoV2, rhoE2 = sol.reshape2matrix()

# Restart solution 3 --> mu =1.005mu-----------------------------------------------------------
if Restart:
    try:
        initialSol = solution(IC.getFromCSV('TestAdjoint/solfM5-3.csv',flow), dom, elem, flow)
        # initialSol = pickle.load(open(restart_file_path,'rb'))
        myCase.HaveTarget = False
    except:
        try:
            initialSol, Sol_T = pickle.load(open(restart_file_path,'rb'))
            
        except:
            raise Exception("Error: Could not load restart file")

sol=        solution(copy.deepcopy(torch.DoubleTensor(initialSol.q)), dom, elem, flow)
rho3, rhoU3, rhoV3, rhoE3 = sol.reshape2matrix()

# Restart solution 4 --> mu =1.001mu-----------------------------------------------------------
if Restart:
    try:
        initialSol = solution(IC.getFromCSV('TestAdjoint/solfM5-4.csv',flow), dom, elem, flow)
        # initialSol = pickle.load(open(restart_file_path,'rb'))
        myCase.HaveTarget = False
    except:
        try:
            initialSol, Sol_T = pickle.load(open(restart_file_path,'rb'))
            
        except:
            raise Exception("Error: Could not load restart file")

sol=        solution(copy.deepcopy(torch.DoubleTensor(initialSol.q)), dom, elem, flow)
rho4, rhoU4, rhoV4, rhoE4 = sol.reshape2matrix()

# Restart solution 5 --> mu =1.0001mu-----------------------------------------------------------
if Restart:
    try:
        initialSol = solution(IC.getFromCSV('TestAdjoint/solfM5-5.csv',flow), dom, elem, flow)
        # initialSol = pickle.load(open(restart_file_path,'rb'))
        myCase.HaveTarget = False
    except:
        try:
            initialSol, Sol_T = pickle.load(open(restart_file_path,'rb'))
            
        except:
            raise Exception("Error: Could not load restart file")

sol=        solution(copy.deepcopy(torch.DoubleTensor(initialSol.q)), dom, elem, flow)
rho5, rhoU5, rhoV5, rhoE5 = sol.reshape2matrix()

# Restart solution 6 --> mu =1.001mu----40000iter----------------------------------------------
if Restart:
    try:
        initialSol = solution(IC.getFromCSV('TestAdjoint/solfM5-6.csv',flow), dom, elem, flow)
        # initialSol = pickle.load(open(restart_file_path,'rb'))
        myCase.HaveTarget = False
    except:
        try:
            initialSol, Sol_T = pickle.load(open(restart_file_path,'rb'))
            
        except:
            raise Exception("Error: Could not load restart file")

sol=        solution(copy.deepcopy(torch.DoubleTensor(initialSol.q)), dom, elem, flow)
rho6, rhoU6, rhoV6, rhoE6 = sol.reshape2matrix()

# Restart solution 7 --> mu =1.0005mu----40000iter----------------------------------------------
if Restart:
    try:
        initialSol = solution(IC.getFromCSV('TestAdjoint/solfM5-7.csv',flow), dom, elem, flow)
        # initialSol = pickle.load(open(restart_file_path,'rb'))
        myCase.HaveTarget = False
    except:
        try:
            initialSol, Sol_T = pickle.load(open(restart_file_path,'rb'))
            
        except:
            raise Exception("Error: Could not load restart file")

sol=        solution(copy.deepcopy(torch.DoubleTensor(initialSol.q)), dom, elem, flow)
rho7, rhoU7, rhoV7, rhoE7 = sol.reshape2matrix()

# Restart solution 8 --> mu =1.0001mu----40000iter----------------------------------------------
if Restart:
    try:
        initialSol = solution(IC.getFromCSV('TestAdjoint/solfM5-7.csv',flow), dom, elem, flow)
        # initialSol = pickle.load(open(restart_file_path,'rb'))
        myCase.HaveTarget = False
    except:
        try:
            initialSol, Sol_T = pickle.load(open(restart_file_path,'rb'))
            
        except:
            raise Exception("Error: Could not load restart file")

sol=        solution(copy.deepcopy(torch.DoubleTensor(initialSol.q)), dom, elem, flow)
rho8, rhoU8, rhoV8, rhoE8 = sol.reshape2matrix()

haveTarget = True
if haveTarget:
    rhoT, rhoUT, rhoVT, rhoET = IC.getTargetFromCSV('./TestAdjoint/solfM{}T-2.csv'.format(int(flow.M1)),elem,flow)


#Reference loss function mu =1.001
loss0 =           ((0.5*(rho4 [1:-1,1:-1] - rhoT )**2 / torch.max(rhoT ) +
                    0.5*(rhoU4[1:-1,1:-1] - rhoUT)**2 / torch.max(rhoUT) + 
                    0.5*(rhoV4[1:-1,1:-1] - rhoVT)**2 / torch.max(rhoVT) +
                    0.5*(rhoE4[1:-1,1:-1] - rhoET)**2 / torch.max(rhoET)          )/(4*elem.nElem)).sum()

loss1 =           ((0.5*(rho1 [1:-1,1:-1] - rhoT )**2 / torch.max(rhoT ) +
                    0.5*(rhoU1[1:-1,1:-1] - rhoUT)**2 / torch.max(rhoUT) + 
                    0.5*(rhoV1[1:-1,1:-1] - rhoVT)**2 / torch.max(rhoVT) +
                    0.5*(rhoE1[1:-1,1:-1] - rhoET)**2 / torch.max(rhoET)          )/(4*elem.nElem)).sum()

loss2 =           ((0.5*(rho2 [1:-1,1:-1] - rhoT )**2 / torch.max(rhoT ) +
                    0.5*(rhoU2[1:-1,1:-1] - rhoUT)**2 / torch.max(rhoUT) + 
                    0.5*(rhoV2[1:-1,1:-1] - rhoVT)**2 / torch.max(rhoVT) +
                    0.5*(rhoE2[1:-1,1:-1] - rhoET)**2 / torch.max(rhoET)          )/(4*elem.nElem)).sum()

loss3 =           ((0.5*(rho3 [1:-1,1:-1] - rhoT )**2 / torch.max(rhoT ) +
                    0.5*(rhoU3[1:-1,1:-1] - rhoUT)**2 / torch.max(rhoUT) + 
                    0.5*(rhoV3[1:-1,1:-1] - rhoVT)**2 / torch.max(rhoVT) +
                    0.5*(rhoE3[1:-1,1:-1] - rhoET)**2 / torch.max(rhoET)          )/(4*elem.nElem)).sum()

loss4 =           ((0.5*(rho4 [1:-1,1:-1] - rhoT )**2 / torch.max(rhoT ) +
                    0.5*(rhoU4[1:-1,1:-1] - rhoUT)**2 / torch.max(rhoUT) + 
                    0.5*(rhoV4[1:-1,1:-1] - rhoVT)**2 / torch.max(rhoVT) +
                    0.5*(rhoE4[1:-1,1:-1] - rhoET)**2 / torch.max(rhoET)          )/(4*elem.nElem)).sum()

loss5 =           ((0.5*(rho5 [1:-1,1:-1] - rhoT )**2 / torch.max(rhoT ) +
                    0.5*(rhoU5[1:-1,1:-1] - rhoUT)**2 / torch.max(rhoUT) + 
                    0.5*(rhoV5[1:-1,1:-1] - rhoVT)**2 / torch.max(rhoVT) +
                    0.5*(rhoE5[1:-1,1:-1] - rhoET)**2 / torch.max(rhoET)          )/(4*elem.nElem)).sum()

loss6 =           ((0.5*(rho6 [1:-1,1:-1] - rhoT )**2 / torch.max(rhoT ) +
                    0.5*(rhoU6[1:-1,1:-1] - rhoUT)**2 / torch.max(rhoUT) + 
                    0.5*(rhoV6[1:-1,1:-1] - rhoVT)**2 / torch.max(rhoVT) +
                    0.5*(rhoE6[1:-1,1:-1] - rhoET)**2 / torch.max(rhoET)          )/(4*elem.nElem)).sum()

loss7 =           ((0.5*(rho7 [1:-1,1:-1] - rhoT )**2 / torch.max(rhoT ) +
                    0.5*(rhoU7[1:-1,1:-1] - rhoUT)**2 / torch.max(rhoUT) + 
                    0.5*(rhoV7[1:-1,1:-1] - rhoVT)**2 / torch.max(rhoVT) +
                    0.5*(rhoE7[1:-1,1:-1] - rhoET)**2 / torch.max(rhoET)          )/(4*elem.nElem)).sum()

loss8 =           ((0.5*(rho8 [1:-1,1:-1] - rhoT )**2 / torch.max(rhoT ) +
                    0.5*(rhoU8[1:-1,1:-1] - rhoUT)**2 / torch.max(rhoUT) + 
                    0.5*(rhoV8[1:-1,1:-1] - rhoVT)**2 / torch.max(rhoVT) +
                    0.5*(rhoE8[1:-1,1:-1] - rhoET)**2 / torch.max(rhoET)          )/(4*elem.nElem)).sum()

print('dJdTheta1', (loss1/0.05).item())
print('dJdTheta2', (loss2/0.01).item())
print('dJdTheta3', (loss3/0.005).item())
print('dJdTheta4', (loss4/0.001).item())
print('dJdTheta5', (loss5/0.0001).item())

print('dJdTheta6', (loss6/0.001).item())
print('dJdTheta7', (loss7/0.0005).item())
print('dJdTheta8', (loss8/0.0001).item())
