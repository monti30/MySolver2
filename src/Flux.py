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

def AUSMparam(UL,UR, rhoL,rhoR, pL,pR, TL,TR, cstar,flow, direction='x'):
    #Parameters 
    beta = 1/8      
    alfa = 3./16.    
    
    if direction=='x':
            
        cL          = cstar[:-1,1:-1] **2 / ( cstar[:-1,1:-1]*(cstar[:-1,1:-1]>=abs(UL)) + abs(UL)*(cstar[:-1,1:-1]<abs(UL)))
        cR          = cstar[1: ,1:-1] **2 / ( cstar[1: ,1:-1]*(cstar[1: ,1:-1]>=abs(UR)) + abs(UR)*(cstar[1: ,1:-1]<abs(UR)))
    elif direction=='y':
        cL          = cstar[1:-1,:-1] **2 / ( cstar[1:-1,:-1]*(cstar[1:-1,:-1]>=abs(UL)) + abs(UL)*(cstar[1:-1,:-1]<abs(UL)))
        cR          = cstar[1:-1 ,1:] **2 / ( cstar[1:-1 ,1:]*(cstar[1:-1 ,1:]>=abs(UR)) + abs(UR)*(cstar[1:-1 ,1:]<abs(UR)))
    
    c05         = cL*(cL<cR) + cR*(cL>=cR)
    
    # Numerical interface mach
    ML   =  UL/c05
    MR   =  UR/c05
    
    MLplus      = ( 0.25*(ML + 1)**2 + beta*(ML**2 -1)**2)*(abs(ML) < 1)      +    0.5*(ML + abs(ML))*(abs(ML) >=1)
    MRminus     = (-0.25*(MR - 1)**2 - beta*(MR**2 -1)**2)*(abs(MR) < 1)      +    0.5*(MR - abs(MR))*(abs(MR) >=1)
    
    M05         = MLplus + MRminus 
    
    # Numerical interface momentum
    mdot        = M05*c05*( rhoL*(M05>=0) + rhoR*(M05<0))
      
    
    # Numerical pressure
    psiLplus    = ( 0.25*(2 - ML)*(ML + 1)**2  + alfa*ML*(ML**2 -1)**2)*(abs(ML) < 1)      +    0.5*(1 + torch.sign(ML))*(abs(ML) >=1)
    psiRminus   = ( 0.25*(2 + MR)*(MR - 1)**2  - alfa*MR*(MR**2 -1)**2)*(abs(MR) < 1)      +    0.5*(1 - torch.sign(MR))*(abs(MR) >=1)
    
    ps          = psiLplus*pL  +  psiRminus*pR
    
    #print('\n\n',direction,'\nc05:',c05,'\nM05',M05, '\nmdot:',mdot,'\nps:', ps)
    return mdot, ps

def AUSMplusUPparam(UL,UR, rhoL,rhoR, pL,pR, TL,TR, cstar,flow, direction='x'):
    #Parameters
    sigma = 1.0
    Kp    = 0.25 
    Ku    = 0.75
    beta = 1/8      
    #alfa = 3./16.    
    
    if direction=='x':
            
        cL          = cstar[:-1,1:-1] **2 / ( cstar[:-1,1:-1]*(cstar[:-1,1:-1]>= UL) + UL*(cstar[:-1,1:-1]< UL))
        cR          = cstar[1: ,1:-1] **2 / ( cstar[1: ,1:-1]*(cstar[1: ,1:-1]>=-UR) - UR*(cstar[1: ,1:-1]<-UR))
    elif direction=='y':
        cL          = cstar[1:-1,:-1] **2 / ( cstar[1:-1,:-1]*(cstar[1:-1,:-1]>= UL) + UL*(cstar[1:-1,:-1]< UL))
        cR          = cstar[1:-1 ,1:] **2 / ( cstar[1:-1 ,1:]*(cstar[1:-1 ,1:]>=-UR) - UR*(cstar[1:-1 ,1:]<-UR))
    
    c05         = 0.5*(cL + cR) #cL*(cL<cR) + cR*(cL>=cR)
    

    # Numerical interface mach
    ML   =  UL/c05
    MR   =  UR/c05
    
    MLplus      = ( 0.25*(ML + 1)**2 + beta*(ML**2 -1)**2)*(abs(ML) < 1)      +    0.5*(ML + abs(ML))*(abs(ML) >=1)
    MRminus     = (-0.25*(MR - 1)**2 - beta*(MR**2 -1)**2)*(abs(MR) < 1)      +    0.5*(MR - abs(MR))*(abs(MR) >=1)
    
    #-----------Mp----------#
    Mseg  = torch.sqrt((UL**2 + UR**2)/2/(c05**2))
    
    maX   = flow.M1**2 + (Mseg**2 - flow.M1**2)*(Mseg**2 > flow.M1**2)
    M0    = torch.sqrt(1 + (maX - 1)*(maX<1))
    
    fa    = M0*(2 - M0)

    Mp = -Kp/fa * ( 1 - sigma*Mseg**2 )*((1 - sigma*Mseg**2)>0)*(pR -pR)/(0.5*(rhoL + rhoR) * c05**2)
        
    
    # M at interface
    M05         = MLplus + MRminus + Mp
    
    # Numerical interface momentum
    mdot        = M05*c05*( rhoL*(M05>=0) + rhoR*(M05<0))
      
    
    # Numerical pressure
    alfa = 3/16*(-4 + 5*fa**2)
    
    psiLplus    = ( 0.25*(2 - ML)*(ML + 1)**2  + alfa*ML*(ML**2 -1)**2)*(abs(ML) < 1)      +    0.5*(1 + torch.sign(ML))*(abs(ML) >=1)
    psiRminus   = ( 0.25*(2 + MR)*(MR - 1)**2  - alfa*MR*(MR**2 -1)**2)*(abs(MR) < 1)      +    0.5*(1 - torch.sign(MR))*(abs(MR) >=1)
    
    #-----------pU----------#
    pU = -Ku*psiLplus*psiRminus*(rhoL + rhoR)*fa*c05*(UR - UL)
    
    
    
    ps          = psiLplus*pL  +  psiRminus*pR  +  pU
    
    #print('\n\n',direction,'\nc05:',c05,'\nM05',M05, '\nmdot:',mdot,'\nps:', ps)
    return mdot, ps 

def AUSM(u,v,rho,p,T,flow,elem):
    UL =    u  [:-1,1:-1]
    UR =    u  [1: ,1:-1]
    rhoL=   rho[:-1,1:-1]
    rhoR=   rho[1: ,1:-1]
    pL=     p  [:-1,1:-1]
    pR=     p  [1: ,1:-1]
    TL=     T  [:-1,1:-1]
    TR=     T  [1: ,1:-1]
    
    # Entalpy
    h           = flow.gamma*flow.R*T/(flow.gamma-1) + 0.5*(u**2 + v**2) #flow.gamma*(flow.R*T/(flow.gamma-1)   +  (u**2 + v**2)*(flow.gamma-1)/(flow.gamma+1) )
    
    
    
    cstar       = torch.sqrt(( 2.*h)*(flow.gamma-1)/(flow.gamma+1) )
    
    mdotX, psX = AUSMplusUPparam(UL,UR, rhoL,rhoR, pL,pR, TL,TR, cstar, flow, direction='x')
    
    # Fluxes in X
    df1rho      = 0.5*(mdotX + abs(mdotX))               +   0.5*(mdotX - abs(mdotX))                 +  0.
    df1rhoU     = 0.5*(mdotX + abs(mdotX))*u[:-1,1:-1]   +   0.5*(mdotX - abs(mdotX))*u[1:,1:-1]      +  psX
    df1rhoV     = 0.5*(mdotX + abs(mdotX))*v[:-1,1:-1]   +   0.5*(mdotX - abs(mdotX))*v[1:,1:-1]      +  0.
    df1rhoE     = 0.5*(mdotX + abs(mdotX))*h[:-1,1:-1]   +   0.5*(mdotX - abs(mdotX))*h[1:,1:-1]      +  0.
    
    
    
    UL =    v  [1:-1 ,:-1]
    UR =    v  [1:-1 ,1: ]
    rhoL=   rho[1:-1 ,:-1]
    rhoR=   rho[1:-1 ,1: ]
    pL=     p  [1:-1 ,:-1]
    pR=     p  [1:-1 ,1: ]
    TL=     T  [1:-1 ,:-1]
    TR=     T  [1:-1 ,1: ]
    
    mdotY, psY = AUSMplusUPparam(UL,UR, rhoL,rhoR, pL,pR, TL,TR, cstar, flow, direction='y')
    
    # Fluxes in Y
    df2rho      = 0.5*(mdotY + abs(mdotY))               +   0.5*(mdotY - abs(mdotY))                  +  0.
    df2rhoU     = 0.5*(mdotY + abs(mdotY))*u[1:-1,:-1]   +   0.5*(mdotY - abs(mdotY))*u[1:-1 ,1:]      +  0.
    df2rhoV     = 0.5*(mdotY + abs(mdotY))*v[1:-1,:-1]   +   0.5*(mdotY - abs(mdotY))*v[1:-1 ,1:]      +  psY
    df2rhoE     = 0.5*(mdotY + abs(mdotY))*h[1:-1,:-1]   +   0.5*(mdotY - abs(mdotY))*h[1:-1 ,1:]      +  0.
    
    return df1rho, df1rhoU, df1rhoV, df1rhoE,  df2rho, df2rhoU, df2rhoV, df2rhoE             


def UPWIND(u,v,rho,p,T,flow,elem): #NOT WORKIMG
    # Total entalpy
    h  = flow.gamma*flow.R*T/(flow.gamma-1) + 0.5*(u**2 + v**2)
    
    # Fluxes in X
    uL =    u  [:-1,1:-1]
    uR =    u  [1: ,1:-1]
    vL =    v  [:-1,1:-1]
    vR =    v  [1: ,1:-1]
    rhoL=   rho[:-1,1:-1]
    rhoR=   rho[1: ,1:-1]
    pL=     p  [:-1,1:-1]
    pR=     p  [1: ,1:-1]
    hL=     h  [:-1,1:-1]
    hR=     h  [1: ,1:-1]
    
    
    
    # df1rho      = (rhoR*uR)            -     (rhoL*uL)              
    # df1rhoU     = (rhoR*uR**2 + pR)    -     (rhoL*uL**2 + pL)
    # df1rhoV     = (rhoR*uR*vR)         -     (rhoL*uL*vL)
    # df1rhoE     = (rhoR*uR*hR)         -     (rhoL*uL*hL)
    df1rho      = (rhoL*uL)          
    df1rhoU     = (rhoL*uL**2 + pL)  
    df1rhoV     = (rhoL*uL*vL)         
    df1rhoE     = (rhoL*uL*hL)       
    
    
    # Fluxes in Y
    uL =    u  [1:-1 ,:-1]
    uR =    u  [1:-1 ,1: ]
    vL =    v  [1:-1 ,:-1]
    vR =    v  [1:-1 ,1: ]
    rhoL=   rho[1:-1 ,:-1]
    rhoR=   rho[1:-1 ,1: ]
    pL=     p  [1:-1 ,:-1]
    pR=     p  [1:-1 ,1: ]
    hL=     h  [1:-1 ,:-1]
    hR=     h  [1:-1 ,1: ]
    
    
    # df2rho      = (rhoR*uR)            -     (rhoL*uL) 
    # df2rhoU     = (rhoR*uR*vR)         -     (rhoL*uL*vL)
    # df2rhoV     = (rhoR*vR**2 + pR)    -     (rhoL*vL**2 + pL)
    # df2rhoE     = (rhoR*uR*hR)         -     (rhoL*uL*hL)
    df2rho      = (rhoL*uL)            
    df2rhoU     = (rhoL*uL*vL)        
    df2rhoV     = (rhoL*vL**2 + pL)  
    df2rhoE     = (rhoL*uL*hL)       
    
    return df1rho, df1rhoU, df1rhoV, df1rhoE,  df2rho, df2rhoU, df2rhoV, df2rhoE   


def HLLCparam(UL,UR,VL,VR,rhoL,rhoR,pL,pR,TL,TR,cL,cR,hL,hR,EL,ER,flow,  direction ='x'):
    uRoe=   (torch.sqrt(rhoL)*UL + torch.sqrt(rhoR)*UR)/(torch.sqrt(rhoL) + torch.sqrt(rhoR))
    hRoe=   (torch.sqrt(rhoL)*hL + torch.sqrt(rhoR)*hR)/(torch.sqrt(rhoL) + torch.sqrt(rhoR))
    cRoe=    torch.sqrt( (flow.gamma-1)*(hRoe - 0.5*uRoe**2))
    
    SL = torch.min( UL - cL , uRoe - cRoe)
    SR = torch.max( UR + cR , uRoe + cRoe)
    
    SM = ((pR - pL) + rhoL*UL*(SL - UL) - rhoR*UR*(SR - UR))/(rhoL*(SL - UL) -rhoR*(SR - UR))
    
    if direction=='y':
        # if (SL>=0):
        # Fluxes in Y
        GrhoUP      = (rhoL*UL)           *(SL >=0)  
        GrhoUUP     =-(rhoL*UL*VL)        *(SL >=0)
        GrhoVUP     = (rhoL*UL**2 + pL)   *(SL >=0)
        GrhoEUP     = (UL*(rhoL*EL + pL)) *(SL >=0)
        #print(GrhoUP)
        
        # if (SL<0 and SM>=0):
        # Fluxes in Y
        GrhoUP     += (rhoL*UL               + SL*(rhoL*(SL - UL)/(SL - SM)     - rhoL))      *(SL <0 )*( SM>=0)
        GrhoUUP    +=-(rhoL*UL*VL            + SL*(rhoL*(SL - VL)/(SL - SM)*VL  - rhoL*VL))   *(SL <0 )*( SM>=0)
        GrhoVUP    += (rhoL*UL**2 + pL       + SL*(rhoL*(SL - UL)/(SL - SM)*SM  - rhoL*UL))   *(SL <0 )*( SM>=0)
        GrhoEUP    += (UL*(rhoL*EL + pL)     + SL*(rhoL*(SL - UL)/(SL - SM)*(EL + (SM - UL)   *(SM + pL/(rhoL*(SL-UL))))  -rhoL*EL) )*(SL <0 )*( SM>=0)
        #print(GrhoUP)    
            
        # if (SR <=0):
        # Fluxes in Y
        GrhoUP     += (rhoR*UR)*          (SR <=0)
        GrhoUUP    +=-(rhoR*UR*VR)*       (SR <=0)
        GrhoVUP    += (rhoR*UR**2 + pR)*  (SR <=0)
        GrhoEUP    += (UR*(rhoR*ER + pR))*(SR <=0)
        #print(GrhoUP)
        
        # if (SR >0 and SM<0):
        #     # Fluxes in Y
        GrhoUP     += (rhoR*UR                + SR*(rhoR*(SR - UR)/(SR - SM)     - rhoR))                                           *(SR >0)* (SM<0)
        GrhoUUP    +=-(rhoR*UR*VR             + SR*(rhoR*(SR - VR)/(SR - SM)*VR  - rhoR*VR))                                       *(SR >0)* (SM<0)
        GrhoVUP    += (rhoR*UR**2 + pR        + SR*(rhoR*(SR - UR)/(SR - SM)*SM  - rhoR*UR))                                       *(SR >0)* (SM<0)
        GrhoEUP    += (UR*(rhoR*ER + pR)      + SR*(rhoR*(SR - UR)/(SR - SM)*(ER + (SM - UR)*(SM + pR/(rhoR*(SR-UR))))  -rhoR*ER)) *(SR >0)* (SM<0)
        #print(GrhoUP)
    
    elif direction=='x':
        # if (SL>=0):
        # Fluxes in Y
        GrhoUP      = (rhoL*UL)           *(SL >=0)  
        GrhoVUP     = (rhoL*UL*VL)        *(SL >=0)
        GrhoUUP     = (rhoL*UL**2 + pL)   *(SL >=0)
        GrhoEUP     = (UL*(rhoL*EL + pL)) *(SL >=0)
        #print(GrhoUP)
        
        # if (SL<0 and SM>=0):
        # Fluxes in Y
        GrhoUP     += (rhoL*UL               + SL*(rhoL*(SL - UL)/(SL - SM)     - rhoL))      *(SL <0 )*( SM>=0)
        GrhoVUP    += (rhoL*UL*VL            + SL*(rhoL*(SL - VL)/(SL - SM)*VL  - rhoL*VL))   *(SL <0 )*( SM>=0)
        GrhoUUP    += (rhoL*UL**2 + pL       + SL*(rhoL*(SL - UL)/(SL - SM)*SM  - rhoL*UL))   *(SL <0 )*( SM>=0)
        GrhoEUP    += (UL*(rhoL*EL + pL)     + SL*(rhoL*(SL - UL)/(SL - SM)*(EL + (SM - UL)   *(SM + pL/(rhoL*(SL-UL))))  -rhoL*EL) )*(SL <0 )*( SM>=0)
        #print(GrhoUP)    
            
        # if (SR <=0):
        # Fluxes in Y
        GrhoUP     += (rhoR*UR)*          (SR <=0)
        GrhoVUP    += (rhoR*UR*VR)*       (SR <=0)
        GrhoUUP    += (rhoR*UR**2 + pR)*  (SR <=0)
        GrhoEUP    += (UR*(rhoR*ER + pR))*(SR <=0)
        #print(GrhoUP)
        
        # if (SR >0 and SM<0):
        #     # Fluxes in Y
        GrhoUP     += (rhoR*UR                + SR*(rhoR*(SR - UR)/(SR - SM)     - rhoR))                                           *(SR >0)* (SM<0)
        GrhoVUP    += (rhoR*UR*VR             + SR*(rhoR*(SR - VR)/(SR - SM)*VR  - rhoR*VR))                                       *(SR >0)* (SM<0)
        GrhoUUP    += (rhoR*UR**2 + pR        + SR*(rhoR*(SR - UR)/(SR - SM)*SM  - rhoR*UR))                                       *(SR >0)* (SM<0)
        GrhoEUP    += (UR*(rhoR*ER + pR)      + SR*(rhoR*(SR - UR)/(SR - SM)*(ER + (SM - UR)*(SM + pR/(rhoR*(SR-UR))))  -rhoR*ER)) *(SR >0)* (SM<0)
        #print(GrhoUP)
    
    # Fluxes in Y -------------------------------------------------------------------------------------
    f2rhoUP      = +   GrhoUP
    f2rhoUUP     = +   GrhoUUP
    f2rhoVUP     = +   GrhoVUP
    f2rhoEUP     = +   GrhoEUP

    return f2rhoUP, f2rhoUUP, f2rhoVUP, f2rhoEUP

def HLLC(u,v,rho,p,T,flow,elem):  #NOT WORKING
    
    c = torch.sqrt( flow.gamma*flow.R*T )
    h  = flow.gamma*flow.R*T/(flow.gamma-1) + 0.5*(u**2 + v**2) #flow.gamma*(flow.R*T/(flow.gamma-1)   +  (u**2 + v**2)*(flow.gamma-1)/(flow.gamma+1) )
    E  = (flow.Cv*T + 0.5*(u**2 + v**2))

    # Y -------------------------------------------------------------------------------- 

    
    UL =    v  [1:-1 , :-1]
    UR =    v  [1:-1 ,1:  ]
    VL =   -u  [1:-1 , :-1]
    VR =   -u  [1:-1 ,1:  ]
    rhoL=   rho[1:-1 , :-1]
    rhoR=   rho[1:-1 ,1:  ]
    pL=     p  [1:-1 , :-1]
    pR=     p  [1:-1 ,1:  ]
    TL=     T  [1:-1 , :-1]
    TR=     T  [1:-1 ,1:  ]
    cL=     c  [1:-1 , :-1]
    cR=     c  [1:-1 ,1:  ]
    hL=     h  [1:-1 , :-1] 
    hR=     h  [1:-1 ,1:  ]
    EL=     E  [1:-1 , :-1] 
    ER=     E  [1:-1 ,1:  ]
    
    
    
    Grho, GrhoU, GrhoV, GrhoE = HLLCparam(UL,UR,VL,VR,rhoL,rhoR,pL,pR,TL,TR,cL,cR,hL,hR,EL,ER,flow, direction='y')
    

    
    # X --------------------------------------------------------------------------------------------------    
    UL =    u  [ :-1 ,1:-1]
    UR =    u  [1:   ,1:-1]
    VL =    v  [ :-1 ,1:-1]
    VR =    v  [1:   ,1:-1]
    rhoL=   rho[ :-1 ,1:-1]
    rhoR=   rho[1:   ,1:-1]
    pL=     p  [ :-1 ,1:-1]
    pR=     p  [1:   ,1:-1]
    TL=     T  [ :-1 ,1:-1]
    TR=     T  [1:   ,1:-1]
    cL=     c  [ :-1 ,1:-1]
    cR=     c  [1:   ,1:-1]
    hL=     h  [ :-1 ,1:-1] 
    hR=     h  [1:   ,1:-1]
    EL=     E  [ :-1 ,1:-1]
    ER=     E  [1:   ,1:-1]
    
    
    Frho, FrhoU, FrhoV, FrhoE  = HLLCparam(UL,UR,VL,VR,rhoL,rhoR,pL,pR,TL,TR,cL,cR,hL,hR,EL,ER,flow, direction='x')

    
    return Frho, FrhoU, FrhoV, FrhoE,  Grho, GrhoU, GrhoV, GrhoE            

def HLLparam(UL,UR,VL,VR,rhoL,rhoR,pL,pR,TL,TR,cL,cR,hL,hR,EL,ER,flow,  direction ='x'):
    uRoe=   (torch.sqrt(rhoL)*UL + torch.sqrt(rhoR)*UR)/(torch.sqrt(rhoL) + torch.sqrt(rhoR))
    hRoe=   (torch.sqrt(rhoL)*hL + torch.sqrt(rhoR)*hR)/(torch.sqrt(rhoL) + torch.sqrt(rhoR))
    cRoe=    torch.sqrt( (flow.gamma-1)*(hRoe - 0.5*uRoe**2))
    
    SL = torch.min( UL - cL , uRoe - cRoe)
    SR = torch.max( UR + cR , uRoe + cRoe)
    
    SM = ((pR - pL) + rhoL*UL*(SL - UL) - rhoR*UR*(SR - UR))/(rhoL*(SL - UL) -rhoR*(SR - UR))
    
    if direction=='y':
        # if (SL>=0):
        # Fluxes in Y
        GrhoUP      = (rhoL*UL)           *(SL >=0)  
        GrhoUUP     =-(rhoL*UL*VL)        *(SL >=0)
        GrhoVUP     = (rhoL*UL**2 + pL)   *(SL >=0)
        GrhoEUP     = (UL*(rhoL*EL + pL)) *(SL >=0)
        #print(GrhoUP)
            
        # if (SR <=0):
        # Fluxes in Y
        GrhoUP     += (rhoR*UR)*          (SR <=0)
        GrhoUUP    +=-(rhoR*UR*VR)*       (SR <=0)
        GrhoVUP    += (rhoR*UR**2 + pR)*  (SR <=0)
        GrhoEUP    += (UR*(rhoR*ER + pR))*(SR <=0)
        #print(GrhoUP)
        
        # if (SR >0 and SL<0):
        #     # Fluxes in Y
        GrhoUP     += (SR*rhoL*UL           - SL*rhoR*UR           + SR*SL*(rhoR-rhoL) )      /(SR-SL)                  *(SR >0)* (SL<0)
        GrhoUUP    +=-(SR*(rhoL*UL*VL)      - SL*(rhoR*UR*VR)      + SR*SL*(rhoR*VR-rhoL*VL) )/(SR-SL)                  *(SR >0)* (SL<0)
        GrhoVUP    += (SR*(rhoL*UL**2 + pL) - SL*(rhoR*UR**2 + pR) + SR*SL*(rhoR*UR-rhoL*UL) )/(SR-SL)                  *(SR >0)* (SL<0)
        GrhoEUP    += (SR*UL*(rhoL*EL + pL) - SL*UR*(rhoR*ER  + pR)+ SR*SL*(rhoR*ER-rhoL*EL) )/(SR-SL)                  *(SR >0)* (SL<0)
    
    elif direction=='x':
        # if (SL>=0):
        # Fluxes in X
        GrhoUP      = (rhoL*UL)           *(SL >=0)  
        GrhoVUP     = (rhoL*UL*VL)        *(SL >=0)
        GrhoUUP     = (rhoL*UL**2 + pL)   *(SL >=0)
        GrhoEUP     = (UL*(rhoL*EL + pL)) *(SL >=0)
        #print(GrhoUP)
            
        # if (SR <=0):
        # Fluxes in Y
        GrhoUP     += (rhoR*UR)*          (SR <=0)
        GrhoVUP    += (rhoR*UR*VR)*       (SR <=0)
        GrhoUUP    += (rhoR*UR**2 + pR)*  (SR <=0)
        GrhoEUP    += (UR*(rhoR*ER + pR))*(SR <=0)
        #print(GrhoUP)
        
        # if (SR >0 and SL<0):
        #     # Fluxes in Y
        GrhoUP     += (SR*rhoL*UL           - SL*rhoR*UR           + SR*SL*(rhoR-rhoL) )      /(SR-SL)                  *(SR >0)* (SL<0)
        GrhoVUP    += (SR*(rhoL*UL*VL)      - SL*(rhoR*UR*VR)      + SR*SL*(rhoR*VR-rhoL*VL) )/(SR-SL)                  *(SR >0)* (SL<0)
        GrhoUUP    += (SR*(rhoL*UL**2 + pL) - SL*(rhoR*UR**2 + pR) + SR*SL*(rhoR*UR-rhoL*UL) )/(SR-SL)                  *(SR >0)* (SL<0)
        GrhoEUP    += (SR*UL*(rhoL*EL + pL) - SL*UR*(rhoR*ER  + pR)+ SR*SL*(rhoR*ER-rhoL*EL) )/(SR-SL)                  *(SR >0)* (SL<0)
        #print(GrhoUP)
    
    # Fluxes in Y -------------------------------------------------------------------------------------
    f2rhoUP      = +   GrhoUP
    f2rhoUUP     = +   GrhoUUP
    f2rhoVUP     = +   GrhoVUP
    f2rhoEUP     = +   GrhoEUP

    return f2rhoUP, f2rhoUUP, f2rhoVUP, f2rhoEUP

def HLL(u,v,rho,p,T,flow,elem):
    
    c =  torch.sqrt( flow.gamma*flow.R*T )
    h  = (flow.Cv + flow.R)*T + 0.5*(u**2 + v**2) #flow.gamma*(flow.R*T/(flow.gamma-1)   +  (u**2 + v**2)*(flow.gamma-1)/(flow.gamma+1) )
    E  = (flow.Cv)         *T + 0.5*(u**2 + v**2)

    # Y -------------------------------------------------------------------------------- 

    
    UL =    v  [1:-1 , :-1]
    UR =    v  [1:-1 ,1:  ]
    VL =   -u  [1:-1 , :-1]
    VR =   -u  [1:-1 ,1:  ]
    rhoL=   rho[1:-1 , :-1]
    rhoR=   rho[1:-1 ,1:  ]
    pL=     p  [1:-1 , :-1]
    pR=     p  [1:-1 ,1:  ]
    TL=     T  [1:-1 , :-1]
    TR=     T  [1:-1 ,1:  ]
    cL=     c  [1:-1 , :-1]
    cR=     c  [1:-1 ,1:  ]
    hL=     h  [1:-1 , :-1] 
    hR=     h  [1:-1 ,1:  ]
    EL=     E  [1:-1 , :-1] 
    ER=     E  [1:-1 ,1:  ]
    
    
    
    Grho, GrhoU, GrhoV, GrhoE = HLLparam(UL,UR,VL,VR,rhoL,rhoR,pL,pR,TL,TR,cL,cR,hL,hR,EL,ER,flow, direction='y')
    

    
    # X --------------------------------------------------------------------------------------------------    
    UL =    u  [ :-1 ,1:-1]
    UR =    u  [1:   ,1:-1]
    VL =    v  [ :-1 ,1:-1]
    VR =    v  [1:   ,1:-1]
    rhoL=   rho[ :-1 ,1:-1]
    rhoR=   rho[1:   ,1:-1]
    pL=     p  [ :-1 ,1:-1]
    pR=     p  [1:   ,1:-1]
    TL=     T  [ :-1 ,1:-1]
    TR=     T  [1:   ,1:-1]
    cL=     c  [ :-1 ,1:-1]
    cR=     c  [1:   ,1:-1]
    hL=     h  [ :-1 ,1:-1] 
    hR=     h  [1:   ,1:-1]
    EL=     E  [ :-1 ,1:-1]
    ER=     E  [1:   ,1:-1]
    
    
    Frho, FrhoU, FrhoV, FrhoE  = HLLCparam(UL,UR,VL,VR,rhoL,rhoR,pL,pR,TL,TR,cL,cR,hL,hR,EL,ER,flow, direction='x')

    
    return Frho, FrhoU, FrhoV, FrhoE,  Grho, GrhoU, GrhoV, GrhoE            