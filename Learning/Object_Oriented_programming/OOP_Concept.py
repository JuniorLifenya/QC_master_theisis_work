import qutip as qt
import numpy as np 


# --------------------------------------------------------------------
# Small tutorial on OOP in Python
# --------------------------------------------------------------------
# Data class = A special kind of class thats designed mostly for holding data
# Without writing a lot of the boilerplate code for regular classes.
# They automatically generate: __init__, __repr__, __eq__

class Person: #The class is just holding daata
    def __init__(self,name,age):
        self.name = name 
        self.age = age
        self.is_alive= True 
    
    def __repr__(self):
        return f"(name={self.name}, age= {self.age}, is_alive ={self.is_alive})"
    
    def __eq__(self,other):
        if not isinstance(other,Person):
            return NotImplemented
        return self.name == other.name and self.age == other.age
person1 = Person("Dude",30)
person2 = Person("Bro",30)
print(person1)
print(person2)

# --------------------------------------------------------------------

# Same as above but we dont need dunder init for example and so on 
from dataclasses import dataclass, field

@dataclass
class Person:
    # Only need to list attributes and data types 
    name: str
    age :int
    is_alive: bool = True 
    password : str = field(repr=False) # The field is a helper function to costumize individual behavior of attributes
    # With false we say that its display will be false( meaning you cannot see it)

    def __post_init__(self): # We can write some logic here now
        if self.age < 0:
            raise ValueError("Age cannot be negative")

# Now create person objects 
person3 = (Person("Dude",30))
person4 = (Person("Patrick",35, "Password"))
print(person3)
print(person4)
print( person3 == person2)

person_negative_age = (Person("Neg-Dude",-1))
print(person_negative_age)
# --------------------------------------------------------------------



# --------------------------------------------------------------------
# Small tutorial on OOP in Python
# --------------------------------------------------------------------

class NVcenter_demo:
    """ Init runs automatically when we create an object
    Self refers to THIS specific NV center instance not all NV centers
    
    The self variable allows us to add attributes to our object.
    It also prevents name conflicts, since name and self.name are different variables.
    """

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
