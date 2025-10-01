import numpy as np 
import qutip as qt
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate 



#---------- Field Framework ---------------------------------------#
class FieldFramework:
    def lagrangian_density(self,Kinetic_Term,Potential_Term,ø, dø, t,): # This defines L = L (ø, dø, t) function
        """
        Compute the Lagrangian density for a scalar field.
        Args:
            ø (float): Field value
            dø (float): Time derivative of the field
            t (float): Time
        """
        
        return (Kinetic_Term(dø) - Potential_Term(ø)) # L = T - V
    def Hamiltonian_density(self,Kinetic_Term,Potential_Term,ø, dø, t,): # This defines H = H (ø, dø, t) function
        """
        Compute the Hamiltonian density for a scalar field.
        Args:
            ø (float): Field value
            dø (float): Time derivative of the field
            t (float): Time
        """
        return (Kinetic_Term(dø) + Potential_Term(ø)) # H = T + V
    def action(self,Lagrangian_Density,ø, dø, t, t1, t2): # This defines S = ∫ L dt function
        """
        Compute the action for a scalar field over a time interval.
        Args:
            Lagrangian_Density (function): Function to compute the Lagrangian density
            ø (float): Field value
            dø (float): Time derivative of the field
            t (float): Time
            t1 (float): Start time
            t2 (float): End time
        """
        S, _ = integrate.quad(lambda t: Lagrangian_Density(ø, dø, t), t1, t2)
        return S # S = ∫ L dt
    