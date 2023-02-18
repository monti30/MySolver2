"""
------------------------------------------------------------------------
PyShock: A Python-based, compressible 1D Navier-Stokes solver for
data assimilation
------------------------------------------------------------------------

@file GlobalSettings.py

"""

__copyright__ = """
Copyright (c) 2022 Jonathan F. MacArt
"""

__license__ = """
 Permission is hereby granted, free of charge, to any person 
 obtaining a copy of this software and associated documentation 
 files (the "Software"), to deal in the Software without 
 restriction, including without limitation the rights to use, 
 copy, modify, merge, publish, distribute, sublicense, and/or 
 sell copies of the Software, and to permit persons to whom the 
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be 
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
 WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
 OTHER DEALINGS IN THE SOFTWARE.
"""


import torch
import torch.nn as nn
from scipy import linalg

# Offloading settings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ----------------------------------------------
# NN model structure
# ----------------------------------------------
class NeuralNetworkModel(nn.Module):
    def __init__(self, num_inputs, H):
        super(NeuralNetworkModel, self).__init__()
        
        self.fc1 = nn.Linear(num_inputs, H).type(torch.FloatTensor)
        self.fcG = nn.Linear(num_inputs, H).type(torch.FloatTensor)
        
        self.fc2 = nn.Linear(H, H).type(torch.FloatTensor)
        self.fc4 = nn.Linear(H, 2).type(torch.FloatTensor)

        
    def forward(self, x):
        
        L1 =  self.fc1( x ) 
        
        H1 = torch.relu( L1 )
        
        L2 = self.fc2( H1 ) 
        
        H2 = torch.relu(L2 )
        
        L3 =  self.fcG( x ) 
        G = torch.sigmoid(L3)
        
        H2_G = G*H2

        C_out = 1.0 #1e-2
        f_out = C_out*self.fc4( H2_G )
        
        return f_out
    


# ----------------------------------------------
# NN model - ELU
# ----------------------------------------------
#ELU function
#alpha = 1.0
#if x >= 0, sigma(x) = x.
#if x <= 0, sigma(x) = alpha*(exp(x) - 1)

#relu implementation
#torch.relu(x) - torch.relu(- alpha*( exp(x) - 1) )
class NeuralNetworkModel_ELU(nn.Module):
    def __init__(self,num_inputs,H):
        super(NeuralNetworkModel_ELU, self).__init__()
        
        self.fc1 = nn.Linear(num_inputs, H).type(torch.FloatTensor)
        self.fcG = nn.Linear(num_inputs, H).type(torch.FloatTensor)
        
        self.fc2 = nn.Linear(H, H).type(torch.FloatTensor)
        self.fc4 = nn.Linear(H, 2).type(torch.FloatTensor)

        self.alpha = 1.0
        self.C_out = 1.0
        
    def forward(self, x):
        
        L1 =  self.fc1( x ) 
        
        H1 = torch.relu(L1) - torch.relu(- self.alpha*( torch.exp(L1) - 1) )
        
        L2 = self.fc2( H1 ) 
        
        H2 = torch.relu(L2) - torch.relu(- self.alpha*( torch.exp(L2) - 1) )
        
        L3 =  self.fcG( x ) 
        G = torch.sigmoid(L3)
        
        H2_G = G*H2
        
        f_out = self.C_out*self.fc4( H2_G )
        
        return f_out    

class Theta(nn.Module):
    def __init__(self, H):
        super(Theta, self).__init__()
        
        self.fc2 = nn.Linear(H, H).type(torch.FloatTensor)
        self.C_out = 1.0
        
        
    def forward(self, x):
        
        f_out = self.fc2(x)
        
        return f_out    


# ----------------------------------------------
# Homebrew Thomas Algorithm solver
# ----------------------------------------------
def Thomas_Solve(Jac,d,x_out):
    ndof = Jac.shape[0]
    a = Jac[1: ,0]
    b = Jac[:  ,1]
    c = Jac[:-1,2]
    c_prime = torch.zeros((ndof),dtype=torch.float64).to(device)
    d_prime = torch.zeros((ndof),dtype=torch.float64).to(device)

    # Forward
    c_prime[0] = c[0]/b[0]
    d_prime[0] = d[0]/b[0]

    for i in range(1,ndof-1):
        c_prime[i] = c[i]/(b[i] - a[i-1]*c_prime[i-1])

    for i in range(1,ndof):
        d_prime[i] = (d[i] - a[i-1]*d_prime[i-1])/(b[i] - a[i-1]*c_prime[i-1])

    # Backward
    x_out[-1] = d_prime[-1]

    for i in range(1,ndof):
        x_out[-1-i] = d_prime[-1-i] - c_prime[-1-i]*x_out[-i]

    # Clean up
    del c_prime,d_prime

    return


# ----------------------------------------------
# Wrapper to LAPACK's Thomas Algorithm solver -- USE THIS
# ----------------------------------------------
def Thomas_Solve_L(Jac,d,x_out):
    ndof = Jac.shape[0]
    a = Jac[1: ,0]
    b = Jac[:  ,1]
    c = Jac[:-1,2]

    linalg.lapack.dgtsv(a, b, c, d)
    x_out = d
    
    return      
