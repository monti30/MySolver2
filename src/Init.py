#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 18:27:59 2023

@author: utente
"""

import numpy as np
import sys 
import os
import torch
import torch.nn.functional as F
import pickle
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import splev, splrep, LinearNDInterpolator


import IC
from Utils import *
import RHS
import Flux
import Nar


# %% CLASS DEFINITION ---------------------------------------------------------------------------------------------------#
class Case:
    def __init__(self,caseName, Mach, runDir, Restart=False, restartDir=None,
                 restartFile=None,
                 modelDir=None,Train=False,targetDir=None,LR=0.0,outputDir = '../Output', useModel= 'True'):
        
        self.caseName       = caseName
        self.Mach           = Mach
        self.runDir         = runDir
        self.restart        = Restart
        self.restartDir     = restartDir
        self.restartFile    = restartFile
        self.modelDir       = modelDir
        self.train          = Train
        self.targetDir      = targetDir
        self.LR             = LR
        self.outputDir      = outputDir
        self.useModel       = useModel
        
    def __str__(self):
        return 'Case name: ' + str(self.caseName) + '\nMach: ' + str(self.Mach) + '\n' +'\n---------------------'
    
class flowModel():
    def __init__(self, airModel):
        
        # Problem parameters (Air) ---------------------------------------------
        if airModel =='Air':
            self.airModel = airModel
            self.gamma = 1.4
            self.R     = 287.0
            self.Cv    = 2.5*self.R 
            self.mu0   = 1.716e-5  # Air
            self.kT0   = 0.0241 
            
        # Problem parameters (Air) ---------------------------------------------    
        elif airModel =='Argon':
            self.airModel = airModel
            self.gamma = 5/3;     
            self.R     = 208.120556724;              #8314/39.948;
            self.Cv    = 1.5*self.R;
            self.mu0   = 2.125e-5;   # Argon
            self.kT0   = 0.0163;
        
        else:
            self.airModel = airModel
            print('Specify: gamma, R, Cv, mu0, kT0')
        
    def printAirModel(self):
        print('\n############################')
        for elem in vars(self):
            print( elem,'=',vars(self)[elem])
        print('############################\n')


class flowProperties():
    def __init__(self,airModel,M1,T1,p1,Tw):    
        # Problem parameters 
        self.name=  airModel.airModel
        self.gamma= airModel.gamma
        self.R=     airModel.R
        self.mu0=   airModel.mu0
        self.kT0=   airModel.kT0
        self.Cv=    airModel.Cv
        
        omegaModel = True
        if self.name =='Argon' and omegaModel:
            self.Pr    = 2/3
            self.T_ref = 273.15
            self.d_ref = 4.04e-10
            self.m_ref = 6.6335209e-26
            self.omega = 0.25 
            
            self.mu0 = Nar.Mu_ref(self.m_ref, self.T_ref, self.d_ref, self.omega)
            self.kT0 = self.mu0*(self.Cv + self.R)/self.Pr
    
        #Initial condition
        self.M1=    M1          #Mach at freestream
        self.T1=    T1          #Temperature freestream
        self.p1=    p1          #Pressure at freestream
        self.Tw=    Tw          #Temperature at the wall

        #Derived initiaò variables
        self.rho1=  p1/T1/self.R
        self.c1=    np.sqrt(self.R * self.gamma * T1)
        self.u1=    M1*self.c1
        

        
        # print('\n############################\nProblem parmeters:',self.name ,'\n')
        # for elem in vars(self):
        #     print( elem,'=',vars(self)[elem])
        # print('\n############################\n')
        
    def printAirModel(self):
        print('\n############################\nProblem parmeters:',self.name ,'\n')
        for elem in vars(self):
            print( elem,'=',vars(self)[elem])
            
        print('\n############################\n')
        
class domain():
    def __init__(self, xS, x, y, Nt,dt):    
        #SPACE -----------------------------------------------------------------------


        #Total number of points: Nx*Ny
        self.x = x
        self.y = y
        
        self.Lx=    self.x[-1] -self.x[0]
        self.Ly=    self.y[-1] -self.y[0]
        
        self.Nx=    len(self.x)
        self.Ny=    len(self.y)
        
        self.xS = index(self.x, xS)
        # self.x  = self.x  -0.03
        


        
        #ID-callable points - x-running denomination
        self.Y_mesh, self.X_mesh = np.meshgrid(self.y, self.x)

        self.X = self.X_mesh.reshape((len(self.x)*len(self.y)), order='F')
        self.Y = self.Y_mesh.reshape((len(self.x)*len(self.y)), order='F')
        
        self.IDmin = 0
        self.IDmax = self.Nx*self.Ny - 1
        self.IDs = np.arange(self.IDmin,self.IDmax+1)
        self.nPoints = self.Nx*self.Ny  
        
        
        self.dx0 = (self.x[1]-self.x[0])    # Nx - 1: Elements number
        self.dy0 = (self.y[1]-self.y[0])    # Ny - 1: Elements number
        
        self.kk1  = np.arange(0             ,   self.nPoints)
        self.kk2  = np.arange(1*self.nPoints, 2*self.nPoints)
        self.kk3  = np.arange(2*self.nPoints, 3*self.nPoints)
        self.kk4  = np.arange(3*self.nPoints, 4*self.nPoints)
        
        #TIME ------------------------------------------------------------------------
        self.dt=    dt
        self.Nt=    Nt
        

        
        self.getInternalIDs()
        self.getLeftIDs()
        self.getRightIDs()
        self.getUpIDs()
        self.getDownIDs()
        
    def getPoint(self, ID):
        return self.X[ID], self.Y[ID]
        
    def getAdiacent(self, ID):
        IDup = ID + self.Nx
        IDdown = ID - self.Nx
        IDright = ID + 1
        IDleft = ID - 1
        
        IDs = [IDup, IDdown, IDright, IDleft]
        nodeConnec = []
        
        if ID % self.Nx == 0:
            IDs.pop(-1)
    
        if (ID + 1) % self.Nx == 0:
            IDs.pop(-2)
            
        for i in range(len(IDs)):
    
            if (0 <= IDs[i] < self.Nx*self.Ny):
                nodeConnec += [IDs[i]]
                
        return nodeConnec
    
    def getBCMarker(self,ID):
        #Left        
        if ID % self.Nx == 0:
            return 'INLET'
        
        #Down left
        elif 0<ID<1:
            return 'SYM_down'

        #Up
        elif (self.Nx*self.Ny - self.Nx) <= ID < (self.Nx*self.Ny):
            return 'SYM_up'
        
        #Down right
        elif 0<=ID<self.Nx:
            return 'WALL'        

        #Right
        elif (ID + 1) % self.Nx == 0:
            return 'OUTLET'
        
        #Domain
        else:
            return 'FLOW'
    
    def getInternalIDs(self):
        self.ID_int = []        
        for ID in range(self.nPoints):
            if self.getBCMarker(ID)=='FLOW':
                self.ID_int += [ID] 
        self.ID_int = np.array(self.ID_int)
    
    def getLeftIDs(self):
        self.ID_left = []        
        for ID in range(self.nPoints):
            if self.getBCMarker(ID)=='INLET' or ID==(self.Nx*(self.Ny -1)) or ID==0:
                self.ID_left += [ID] 
        self.ID_left = np.array(self.ID_left)
    
    def getRightIDs(self):
        self.ID_right = []        
        for ID in range(self.nPoints):
            if self.getBCMarker(ID)=='OUTLET' or ID==(self.Nx-1) or ID== (self.Nx*self.Ny -1):
                self.ID_right += [ID] 
        self.ID_right = np.array(self.ID_right)

    def getUpIDs(self):
        self.ID_up = []        
        for ID in range(self.nPoints):
            if self.getBCMarker(ID)=='SYM_up' or ID==(self.Nx*(self.Ny -1)) or ID== (self.Nx*self.Ny -1):
                self.ID_up += [ID] 
        self.ID_up = np.array(self.ID_up)        
    
    def getDownIDs(self):
        self.ID_down = []        
        for ID in range(self.nPoints):
            if self.getBCMarker(ID)=='SYM_down' or self.getBCMarker(ID)=='WALL' or ID==0 or ID== (self.Nx-1):
                self.ID_down += [ID] 
        self.ID_down = np.array(self.ID_down)        
        
        
    def plotMesh(self):
        plt.close(1100)
        fig = plt.figure(1100)
        for i in range(self.nPoints):
            for k in range(len(self.getAdiacent(i))):
                x0,y0 = self.getPoint(i) 
                x1,y1 = self.getPoint(self.getAdiacent(i)[k])
                
                if self.getBCMarker(i)=='WALL':
                    plt.plot([x0,x1], [y0,y1], color='red' ,marker='o')#, markerfacecolor='red')
                
                elif self.getBCMarker(i)=='SYM_up' or self.getBCMarker(i)=='SYM_down':
                    plt.plot([x0,x1], [y0,y1], color='blue' ,marker='o')#, markerfacecolor='red')
                
                else:
                    plt.plot([x0,x1], [y0,y1], color='black' ,marker='o')#, markerfacecolor='red')
        return fig #plt.show()

class centroids():
    def __init__(self,dom):    
        
        self.Nx = dom.Nx-1
        self.Ny = dom.Ny-1
        self.dom= dom
        
        self.nElem = self.Nx*self.Ny
        self.IDmin = 0
        self.IDmax = self.nElem - 1
        self.IDs = np.arange(self.IDmin,self.IDmax+1)  
        
        self.IDl = np.delete(self.IDs, np.arange(self.Nx-1,self.Nx*self.Ny, self.Nx))
        self.IDr = np.delete(self.IDs, np.arange(0,self.Nx*self.Ny, self.Nx))
        
        self.dx = []
        self.dy = []
        self.X  = []
        self.Y  = []
        for IDe in range(self.nElem):  
            Pld = IDe + IDe//self.Nx  # Point left down of the element 
            Prd = Pld +1
            Plu = Pld + dom.Nx
            Pru = Prd + dom.Nx
             
            self.dx += [(dom.X[Prd]-dom.X[Pld])]    # Nx - 1: Elements number
            self.dy += [(dom.Y[Plu]-dom.Y[Pld])]    # Ny - 1: Elements number
            self.X  += [ dom.X[Pld] + self.dx[-1]/2]
            self.Y  += [ dom.Y[Pld] + self.dy[-1]/2]
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.dx = np.array(self.dx)
        self.dy = np.array(self.dy)
        
        #Matrix form
        self.X_m  = self.X.reshape((self.Nx,self.Ny) , order='F')
        self.Y_m  = self.Y.reshape((self.Nx,self.Ny) , order='F')
        self.dx_m = self.dx.reshape((self.Nx,self.Ny), order='F')
        self.dy_m = self.dy.reshape((self.Nx,self.Ny), order='F')
        
        
        self.kk1  = np.arange(0           ,   self.nElem)
        self.kk2  = np.arange(1*self.nElem, 2*self.nElem)
        self.kk3  = np.arange(2*self.nElem, 3*self.nElem)
        self.kk4  = np.arange(3*self.nElem, 4*self.nElem)
        
        #Needed for computation of GreenGaussGradients
        self.alfa_x = torch.DoubleTensor(0.5*self.dx_m[1:,:]/(self.X_m[1:,:] - self.X_m[:-1,:]))
        self.alfa_y = torch.DoubleTensor(0.5*self.dy_m[:,1:]/(self.Y_m[:,1:] - self.Y_m[:,:-1]))
        
        self.alfa_x = self.alfa_x[:   ,1:-1] #Dominio interno
        self.alfa_y = self.alfa_y[1:-1, :  ]
        # self.alfa_x = F.pad(self.alfa_x,pad=(0, 0, 1, 1),value=0.5)
        # self.alfa_y = F.pad(self.alfa_y,pad=(1, 1, 0, 0),value=0.5)

        self.getInternalIDs()
        self.getLeftIDs()
        self.getRightIDs()
        self.getUpIDs()
        self.getDownIDs()
        
    def getCentroid(self, IDe):
        Pld = IDe + IDe//self.Nx
        return self.dom.X[Pld] + self.dx[IDe]/2, self.dom.Y[Pld] +self.dy[IDe]/2
    
    def getAdiacent(self, ID):
        IDup = ID + self.Nx
        IDdown = ID - self.Nx
        IDright = ID + 1
        IDleft = ID - 1
        
        IDs = [IDup, IDdown, IDright, IDleft]
        nodeConnec = []
        
        if ID % self.Nx == 0:
            IDs.pop(-1)
    
        if (ID + 1) % self.Nx == 0:
            IDs.pop(-2)
            
        for i in range(len(IDs)):
    
            if (0 <= IDs[i] < self.Nx*self.Ny):
                nodeConnec += [IDs[i]]
                
        return nodeConnec
    
    def getBCMarker(self,ID):
        #Left        
        if ID % self.Nx == 0:
            return 'INLET'
        
        #Down left
        elif 0<ID<1:
            return 'SYM_down'

        #Up
        elif (self.Nx*self.Ny - self.Nx) <= ID < (self.Nx*self.Ny):
            return 'SYM_up'
        
        #Down right
        elif 0<=ID<self.Nx:
            return 'WALL'        

        #Right
        elif (ID + 1) % self.Nx == 0:
            return 'OUTLET'
        
        #Domain
        else:
            return 'FLOW'
    
    def getInternalIDs(self):
        self.ID_int = []        
        for ID in range(self.nElem):
            if self.getBCMarker(ID)=='FLOW':
                self.ID_int += [ID] 
        self.ID_int = np.array(self.ID_int)
    
    def getLeftIDs(self):
        self.ID_left = []        
        for ID in range(self.nElem):
            if self.getBCMarker(ID)=='INLET' or ID==(self.Nx*(self.Ny -1)) or ID==0:
                self.ID_left += [ID] 
        self.ID_left = np.array(self.ID_left)
    
    def getRightIDs(self):
        self.ID_right = []        
        for ID in range(self.nElem):
            if self.getBCMarker(ID)=='OUTLET' or ID==(self.Nx-1) or ID== (self.Nx*self.Ny -1):
                self.ID_right += [ID] 
        self.ID_right = np.array(self.ID_right)

    def getUpIDs(self):
        self.ID_up = []        
        for ID in range(self.nElem):
            if self.getBCMarker(ID)=='SYM_up' or ID==(self.Nx*(self.Ny -1)) or ID== (self.Nx*self.Ny -1):
                self.ID_up += [ID] 
        self.ID_up = np.array(self.ID_up)        
    
    def getDownIDs(self):
        self.ID_down = []        
        for ID in range(self.nElem):
            if self.getBCMarker(ID)=='SYM_down' or self.getBCMarker(ID)=='WALL' or ID==0 or ID== (self.Nx-1):
                self.ID_down += [ID] 
        self.ID_down = np.array(self.ID_down)        
        
        
    def plotMesh(self):
        #plt.close(1200)
        fig = self.dom.plotMesh()
        #fig = plt.figure(1200)
        for i in range(self.nElem):
            for k in range(len(self.getAdiacent(i))):
                x0,y0 = self.getCentroid(i) 
                x1,y1 = self.getCentroid(self.getAdiacent(i)[k])

                plt.scatter([x0,x1], [y0,y1], color='black' )#, markerfacecolor='red')
        plt.show()

class solution():        
    def __init__(self, q, domain, elem, flowProp):
        self.q=         q
        self.domain=    domain
        self.elem =     elem
        self.flowProp=  flowProp
        
    def getConservativeVar(self):
        rho  = self.q[self.elem.kk1]
        rhoU = self.q[self.elem.kk2]
        rhoV = self.q[self.elem.kk3]
        rhoE = self.q[self.elem.kk4]
        
        return rho, rhoU, rhoV, rhoE
        
    def getPrimitiveVar(self):    
        rho, rhoU, rhoV, rhoE = self.getConservativeVar()
        
        u = rhoU/rho
        v = rhoV/rho
        p = (self.flowProp.gamma - 1)*(rhoE - 0.5*(rhoU*rhoU + rhoV*rhoV)/rho)
        T = p/self.flowProp.R/rho
        print("Due parole su Padova:\nDI I O   M E R D A !!")
        
        return rho, u, v, p, T
        
    def convert2torch(self,string=''):          #If I use this the location in memory from q and self.q changes
        All = False
        if string=='':
            All = True
        if string=='q' or All:
            self.q = torch.DoubleTensor(self.q)

    def reshape2matrix(self, var = 'cons'):
        if var == 'cons':
            rho, rhoU, rhoV, rhoE = self.getConservativeVar()
            
            rhoM  = reshape_fortran(rho, (self.elem.Nx,self.elem.Ny))
            rhoUM = reshape_fortran(rhoU,(self.elem.Nx,self.elem.Ny))
            rhoVM = reshape_fortran(rhoV,(self.elem.Nx,self.elem.Ny))
            rhoEM = reshape_fortran(rhoE,(self.elem.Nx,self.elem.Ny))
            return rhoM, rhoUM, rhoVM, rhoEM
        
        if var == 'primitive':
            rho, U, V, p, T = self.getPrimitiveVar()
            
            rhoM  = reshape_fortran(rho,(self.elem.Nx,self.elem.Ny))
            UM    = reshape_fortran(U,  (self.elem.Nx,self.elem.Ny))
            VM    = reshape_fortran(V,  (self.elem.Nx,self.elem.Ny))
            pM    = reshape_fortran(p,  (self.elem.Nx,self.elem.Ny))
            TM    = reshape_fortran(T,  (self.elem.Nx,self.elem.Ny))
            return rhoM, UM, VM, pM, TM

        
    def imposeBC(self, qPrimitive):
        u, v, p, T = qPrimitive
        
        
        
# %% SYS --------------------------------------------------------------------------------------------------------------#

# Mach_trained= 0.5 #int(sys.argv[2])
# print('Mach_trained: ', Mach_trained,'###')

# CaseName        = 'Blasius'
# OutputDir       = "../Output"+ CaseName
# Mach            = 2. #int(sys.argv[1]) #9
# RunDir          = '.'

# Restart         = False
# RestartDir      = OutputDir
# RestartFile     = 'Restart_NS_256_Argon/BlasiusRestart256_DL_N0_H0_M{}.p'.format(Mach)
# #RestartFile     = './Blaisus_DL_N18_H1200_M{}.p'.format(Mach)

# ModelDir        = OutputDir + '/' + 'NN_Model_M{}'.format(Mach_trained)     #'NN_Model'
# TargetDir       = '../Train_Data'
# LR              = 0.001 #2.5e-6
# Train = False
# UseModel = False


# myCase = Case(CaseName, 
#               Mach,
              
#               RunDir, 

#               Restart,
#               RestartDir,
#               RestartFile, 

#               ModelDir, 

#               Train=Train, 
                  
#               targetDir=TargetDir, 
#               LR=LR,

#               outputDir=OutputDir,
#               useModel=UseModel)
# print(myCase)

# num_inputs,H = 0,0


# # %% PROBLEM PARAMETERS -------------------------------------------------------
# Argon = flowModel('Argon')

# M1      = myCase.Mach   # External/midpoint Mach number
# T1      = 300.0         # External/midpoint temperature [K]
# p1      = 6.667         # External/midpoint pressure [Pa]
# Tw      = 370.0         # Wall temperature [K]



# # %% ISTANCES -----------------------------------------------------------------

# flow = flowProperties(Argon,  M1,T1,p1,Tw)

# xS = 0.0
# Lx = 0.5
# Ly = 0.1

# Nx = 6
# Ny = 3


# nIter = 1
# dt=0.7e-8


# passox = 1.5
# passoy = 1.2

# print('xRight:')
# xRight = geomGrid(0., Lx,   Nx,              passox)

# print('xLeft')
# xLeft  =-np.flip(geomGrid(0., 0.02, int(np.ceil(0.2*Nx))+1, passox))[:-1]

# x = np.concatenate([xLeft,xRight])

# print('y')
# y = geomGrid(0., Ly, Ny, passoy) 

