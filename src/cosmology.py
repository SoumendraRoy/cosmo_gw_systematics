import numpy as np
import aesara.tensor as at
from .utils import cumtrapz

class FlatLCDM:
    """FLRW cosmology with no spatial curvature and cosmological constant
    
    See [Hogg (1999)](https://arxiv.org/abs/astro-ph/9905116). 
    
    Parameters
    ----------
    zs : real array
            The redshifts at which the comoving distance should be calculated. These
            must be sufficiently dense that a trapezoidal approximation to the
            integral is sufficiently accurate.  A common choice is to choose them
            uniformly in `log(1+z)` via `zs = np.expm1(np.linspace(np.log(1),
            np.log(1+zmax), Nz))`
    Om : real 
        Present day matter density relative to the critical density `0 <= Om <= 1`.
    """
    
    def __init__(self, Om):
        self.Om = Om
        
        
    def Ez(self, zs):
        """Pytensor definition of the cosmological integrand for a flat LCDM cosmology. 
        """
        opz = 1 + zs
        return at.sqrt(self.Om*opz*opz*opz + (1-self.Om))

    def dCs(self, zs):
        """Evaluate the ratio of comoving distance integral and Hubble constant on the grid `zs`.

         """
        fz = 1/self.Ez(zs)
        return cumtrapz(fz, zs)

    def dLs(self, zs):
        """Ratio of Luminosity distance and Hubble constant on the grid `zs` with previously evaluated comoving distances.

        Parameters
        ----------
        dCs : real array
            Comoving distances on the grid of redshifts.
        """
        return self.dCs(zs)*(1+zs)

    def dVdz(self, zs):
        """Evaluate the (unitless) differential comoving volume on the grid `zs`.

        Parameters
        ----------
        dCs : real array
            The (unitless) comoving distances at the redshift grid.
       """
        return 4*np.pi*self.dCs(zs)*self.dCs(zs)/self.Ez(zs)
    
class Flatw0CDM:
    """FLRW cosmology with no spatial curvature and constant dark energy equation of state
    
    See [Hogg (1999)](https://arxiv.org/abs/astro-ph/9905116). 
    
    Parameters
    ----------
    zs : real array
            The redshifts at which the comoving distance should be calculated. These
            must be sufficiently dense that a trapezoidal approximation to the
            integral is sufficiently accurate.  A common choice is to choose them
            uniformly in `log(1+z)` via `zs = np.expm1(np.linspace(np.log(1),
            np.log(1+zmax), Nz))`
    Om : real 
        Present day matter density relative to the critical density `0 <= Om <= 1`.
    w0 : real 
        Dark energy equation of state parameter.
    """
    
    def __init__(self, Om, w0):
        self.Om = Om
        self.w0 = w0
        
    def Ez(self, zs):
        """Pytensor definition of the cosmological integrand for a flat LCDM cosmology. 
        """
        opz = 1 + zs
        return at.sqrt(self.Om*opz*opz*opz + (1-self.Om)*opz**(3*(1+self.w0)))

    def dCs(self, zs):
        """Evaluate the ratio of comoving distance integral and Hubble constant on the grid `zs`.

         """
        fz = 1/self.Ez(zs)
        return cumtrapz(fz, zs)

    def dLs(self, zs):
        """Ratio of Luminosity distance and Hubble constant on the grid `zs` with previously evaluated comoving distances.

        Parameters
        ----------
        dCs : real array
            Comoving distances on the grid of redshifts.
        """
        return self.dCs(zs)*(1+zs)

    def dVdz(self, zs):
        """Evaluate the (unitless) differential comoving volume on the grid `zs`.

        Parameters
        ----------
        dCs : real array
            The (unitless) comoving distances at the redshift grid.
       """
        return 4*np.pi*self.dCs(zs)*self.dCs(zs)/self.Ez(zs)
    
class FlatCPLCDM:
    """FLRW cosmology with no spatial curvature and CPL parametrized dark energy equation of state
    
    See [Hogg (1999)](https://arxiv.org/abs/astro-ph/9905116). 
    
    Parameters
    ----------
    zs : real array
            The redshifts at which the comoving distance should be calculated. These
            must be sufficiently dense that a trapezoidal approximation to the
            integral is sufficiently accurate.  A common choice is to choose them
            uniformly in `log(1+z)` via `zs = np.expm1(np.linspace(np.log(1),
            np.log(1+zmax), Nz))`
    Om : real 
        Present day matter density relative to the critical density `0 <= Om <= 1`.
    w0 : real 
        First CPL dark energy equation of state parameter.
    wa : real 
        Second CPL dark energy equation of state parameter.
    """
    
    def __init__(self, Om, w0, wa):
        self.Om = Om
        self.w0 = w0
        self.wa = wa
        
    def Ez(self, zs):
        """Pytensor definition of the cosmological integrand for a flat LCDM cosmology. 
        """
        opz = 1 + zs
        w = self.w0 + self.wa*(zs/opz)
        return at.sqrt(self.Om*opz*opz*opz + (1-self.Om)*opz**(3*(1+w)))

    def dCs(self, zs):
        """Evaluate the ratio of comoving distance integral and Hubble constant on the grid `zs`.
        """
        fz = 1/self.Ez(zs)
        return cumtrapz(fz, zs)

    def dLs(self, zs):
        """Ratio of Luminosity distance and Hubble constant on the grid `zs` with previously evaluated comoving distances.

        Parameters
        ----------
        dCs : real array
            Comoving distances on the grid of redshifts.
        """
        return self.dCs(zs)*(1+zs)

    def dVdz(self, zs):
        """Evaluate the (unitless) differential comoving volume on the grid `zs`.

        Parameters
        ----------
        dCs : real array
            The (unitless) comoving distances at the redshift grid.
       """
        return 4*np.pi*self.dCs(zs)*self.dCs(zs)/self.Ez(zs)