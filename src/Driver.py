#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 19:34:25 2023

@author: utente
"""
from Init import *
import numpy as np
# %% SYS --------------------------------------------------------------------------------------------------------------#

Mach_trained= 0.5 #int(sys.argv[2])


CaseName        = 'Blasius'
OutputDir       = "../Output"+ CaseName
Mach            = 10. #int(sys.argv[1]) #9

RunDir          = '.'

RestartAdj      = False
Restart         = False
RestartDir      = OutputDir
RestartFile     = 'TestAdjoint/solfM{}-9.csv'.format(int(Mach))
#RestartFile     = 'solfM{}.csv'.format(int(Mach))

ModelDir        = OutputDir + '/' + 'NN_Model_M{}'.format(Mach_trained)     #'NN_Model'
TargetDir       = '../Train_Data'
LR              = 0.001 #2.5e-6
Train    = False
UseModel = False


myCase = Case(CaseName, 
              Mach,
              
              RunDir, 

              Restart,
              RestartDir,
              RestartFile, 

              ModelDir, 

              Train=Train, 
                  
              targetDir=TargetDir, 
              LR=LR,

              outputDir=OutputDir,
              useModel=UseModel)
#print(myCase)

myCase.threshObj = 0.8
print('# Mach_trained: ', Mach_trained,'#\n')
print(myCase)
num_inputs,H = 0,0


# %% PROBLEM PARAMETERS -------------------------------------------------------
M1      = Mach          # External/midpoint Mach number
T1      = 300.0         # External/midpoint temperature [K]
p1      = 6.667         # External/midpoint pressure [Pa]
Tw      = 370.0         # Wall temperature [K]


# %% DOMAIN -----------------------------------------------------------------
xS = 0.0 # X after which the wall starts
Lx = 0.5 # Length of the wall
Ly = 0.2 # Height

Nx = 70   # Number of points of the wall
Ny = 100
passox = 1.07
passoy = 1.05

# Grid assembling - x 
print('xRight:')
xRight = geomGrid(0., Lx,   Nx,              passox)

print('xLeft')
xLeft  =-np.flip(geomGrid(0., 0.02, 12, 1.2*passox))[:-1]

x = np.concatenate([xLeft,xRight])

# Grid - y
print('y')
y = geomGrid(0., Ly, Ny, passoy) 
print()

# TIME INTEGRATION --------------------------------------------------------
nIter   = 40000
dt      = 0.25e-7
dtA     = 0.2e-7

localTimeStepping = True #This overwrite dt 
if localTimeStepping:
    CFL = 0.8
    print('CFL:',CFL,'\n\n')




# %% POSTPROCESS
Plot = True
