#!/usr/bin/env python
from pyscf import lib
from pyscf.x2c import x2c, _response_functions
from . import ghf


class Polarizability(ghf.Polarizability):
    def get_h1(self, picture_change=True):
        '''
        Generate the dipole matrix in AO basis.
        '''
        mf = self.mf
        mol = mf.mol
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        charge_center = lib.einsum('i,ix->x', charges, coords) / charges.sum()
        with mol.with_common_orig(charge_center):
            if picture_change:
                ao_dip = mf.with_x2c.picture_change(('int1e_r_spinor',
                                                     'int1e_sprsp_spinor'))
            else:
                ao_dip = mol.intor_symmetric('int1e_r_spinor')
        return -ao_dip


x2c.SCF.Polarizability = lib.class_as_method(Polarizability)
    

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    from pyscf import gto
    mol = gto.M(atom ='''O     0.000000     0.000000     0.123323
                         H     0.000000     0.757497    -0.493291
                         H     0.000000    -0.757497    -0.493291''',
                #charge = 1,
                #spin = 1,
                basis = 'ccpvdz')
    
    xc = 'TPSS'
    '''
    mf1 = mol.GHF()
    mf1.conv_tol = 1e-11
    mf1.kernel()
    polar1 = Polarizability(mf1)
    #polar1.conv_tol = 1e-11
    print(polar1.polarizability())
    print(polar1.hyper_polarizability())
    '''
    mfg = mol.GKS()
    mfg.xc = xc
    mfg.conv_tol = 1e-11
    mfg.kernel()
    plg = Polarizability(mfg)
    print(plg.polarizability())

    mfg.collinear = 'mcol'
    mfg.kernel()
    plgm = Polarizability(mfg)
    print(plgm.polarizability())

    mfx = mol.GKS().x2c()
    mfx.xc = xc
    mfx.conv_tol = 1e-11
    mfx.kernel()
    plx = Polarizability(mfx)
    print(plx.polarizability())

    mfx.collinear = 'mcol'
    mfx.kernel()
    plxm = Polarizability(mfx)
    print(plxm.polarizability())