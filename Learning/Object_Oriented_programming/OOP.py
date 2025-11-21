import qutip as qt
import numpy as np 

# --------------------------------------------------------------------
# Small tutorial on OOP in Python
# --------------------------------------------------------------------

class NVcenter_demo:
    """" Init runs autumatically when we create an object"""
    """ Self refers to THIS specific NV center instance not all NV centers"""

    def __init__(self, D=2.87e9, Bz=0.0):
        self.D = D   # Zero-field splitting in Hz
        self.Bz = Bz  # Magnetic field along the NV axis in Tesla
        self.setup_operators() 

    def setup_operators(self):
        self.Sx = qt.jmat(1, 'x')  # type: ignore
        self.Sy = qt.jmat(1, 'y')  # type: ignore
        self.Sz = qt.jmat(1, 'z')  # type: ignore
        self.Sx2 = self.Sx * self.Sx  # type: ignore
        self.Sy2 = self.Sy * self.Sy  # type: ignore
        self.Sz2 = self.Sz * self.Sz  # type: ignore

    def some_method(self):
        # All methods need "self" to access the object'sdata
        return self.D * self .Sz2  # type: ignore
# usage example
nv1 = NVcenter_demo(D=2.87e9, Bz=0.01)
nv2 = NVcenter_demo(D=2.87e9, Bz=0.02)
H1 = nv1.some_method()
H2 = nv2.some_method()
H_total = H1 + H2

# --------------------------------------------------------------------
# Mini Refraction code 
# --------------------------------------------------------------------
# Minimal test - add this at the end of your file
def minimal_test():
    """Absolute minimal test to verify QuTiP works"""
    print("Running minimal test...")
    
    # Simple spin-1 system
    Sz = qt.jmat(1, 'z')
    psi0 = qt.basis(3, 1)  # |0>
    
    # Simple static Hamiltonian
    H = 2.87e9 * Sz**2
    
    # Short time evolution
    tlist = np.linspace(0, 1e-9, 10)
    result = qt.sesolve(H, psi0, tlist, [Sz])
    
    print("Minimal test passed!")
    return result
minimal_test()

# --------------------------------------------------------------------
