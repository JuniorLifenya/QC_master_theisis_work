import numpy as np 
import qutip as qt
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate 



#---------- Field Framework ---------------------------------------#
class Field:
    """ Base field class for scalar fields, spinor, vector fields etc."""
    def __init__(self, name, dimensions, mass=0 ):
        self.name = name
        self.mass = mass  # Mass of the field (0 for massless fields)
        self.dim =  dimensions  # Number of spatial dimensions
        self.hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
        self.c = 3e8  # Speed of light (m/s)
    #------------------ Objects in field theory ------------------#

    def lagrangian_density(self,Kinetic_Term,Potential_Term,ø, dø, t,): # This defines L = L (ø, dø, t) function for scalar field ø
        """Args: ø (float): Field value , dø (float): Time derivative of the field , t (float): Time"""
        return (Kinetic_Term(dø) - Potential_Term(ø)) # L = T - V
    def Hamiltonian_density(self,Kinetic_Term,Potential_Term,ø, dø, t,): # This defines H = H (ø, dø, t) function for scalar field ø
        """Args: ø (float): Field value, dø (float): Time derivative of the field, t (float): Time"""
        return (Kinetic_Term(dø) + Potential_Term(ø)) # H = T + V
    def action(self,Lagrangian_Density,ø, dø, t, t1, t2): # This defines S = ∫ L dt function for scalar field ø
        """Args: Lagrangian_Density (function): Function to compute the Lagrangian density
            ø (float): Field value
            dø (float): Time derivative of the field
            t (float): Time
            t1 (float): Start time
            t2 (float): End time
        """
        S, _ = integrate.quad(lambda t: Lagrangian_Density(ø, dø, t), t1, t2)
        return S # S = ∫ L dt
    

    def energy_momentum_tensor(self,Lagrangian_Density,ø, dø, t): # This defines Tμν = Tμν(ø, dø, t) function for scalar field ø
        """Args: 
            Lagrangian_Density (function): Function to compute the Lagrangian density
            ø (float): Field value
            dø (float): Time derivative of the field
            t (float): Time
        """
        # Placeholder implementation - actual tensor calculation depends on field type
        T00 = self.Hamiltonian_density(Lagrangian_Density,ø, dø, t)  # Energy density component
        T11 = T22 = T33 = 0  # Spatial components (simplified)
        return np.array([[T00, 0, 0, 0],
                         [0, T11, 0, 0],
                         [0, 0, T22, 0],
                         [0, 0, 0, T33]])
    
    #------------------ Objects in field theory ------------------#

class ScalarField(Field):
    """ Class for scalar fields (spin-0) """
    def __init__(self, name = "scalar", mass=0.0):
        super().__init__(name, dimensions=4, mass=mass)  # 4D ø = ø(x,y,z,t)
        self.mass = mass # Mass of the scalar field (0 for massless fields)

    def lagrangian_density(self,ø,dø,g_munu): # This defines L = L (ø, dø, t) function for scalar field ø
        """Args: ø (float): Field value , dø (float): Time derivative of the"""
        pass # Implement specific lagrangian for scalar field

    def Hamiltonian_density(self,ø,dø,g_munu): # This defines H = H (ø, dø, t) function for scalar field ø
        """Args: ø (float): Field value, dø (float): Time derivative of the field"""
        pass # Implement specific hamiltonian for scalar field

#---------- Field Framework ---------------------------------------#

#---------- Exercise-----------------------------------------------#

# Implement scalar field Lagrangian in both Python and C++
# Create metric tensor class that can handle flat and curved spacetime
# Code variational derivative function using FDM

#---------- Exercise-----------------------------------------------#

#---------- Example usage ------------------------------------------#

# Define a simple scalar field with a quadratic potential V(ø) = (1/2)m(v_ø)^2
def kinetic_term(y, dy, m=1.0):
    t = smp.symbols('t')
    m,k = smp.symbols('m k')
    y = smp.symbols('y', cls = smp.Function) # smp.Function states that the position is a function
    y = y(t)
    dy = dy_dt = smp.diff(y,t)
    ddy_dt = smp.diff(dy,t)
    return 0.5 *(m)* (dy_dt**2)
def potential_term(ø, m=1.0):
    return 0.5 * m**2 * ø**2

#---------- Example usage ------------------------------------------#