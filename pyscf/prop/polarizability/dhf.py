#!/usr/bin/python3
import numpy
from pyscf import lib
from pyscf.lib import logger
from . import ghf


def get_dm1(polobj, mo1):
    '''
    Generate the 1st-order density matrix in AO basis.
    '''
    mf = polobj.mf
    n2c = mf.mol.nao_2c()
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = n2c + numpy.where(mo_occ[n2c:]==1)[0]
    viridx = n2c + numpy.where(mo_occ[n2c:]==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    if len(mo1) == 2:
        dm1  = lib.einsum('pj,xji,qi->xpq', orbv, mo1[0]       , orbo.conj())
        dm1 += lib.einsum('pi,xji,qj->xpq', orbo, mo1[1].conj(), orbv.conj())
    elif len(mo1) == 3:
        dm1  = lib.einsum('pj,xji,qi->xpq', orbv, mo1, orbo.conj())
        dm1 += dm1.transpose(0,2,1).conj()
    else:
        raise ValueError('mo1 does not have the correct shape (3,nvir,nocc)'
                         ' for freq = 0 or (2,3,nvir,nocc) for freq != 0.')
    return dm1

def get_h1(polobj, picture_change=True):
    mf = polobj.mf
    mol = mf.mol
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = lib.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig((charge_center)):
        c = lib.param.LIGHT_SPEED
        ll_dip = mol.intor_symmetric('int1e_r_spinor', comp=3)
        ss_dip = mol.intor_symmetric('int1e_sprsp_spinor', comp=3) * (.5/c)**2
        dip = _block_diag(ll_dip, ss_dip)
    return -dip

def get_e_vo(mf):
    '''
    Generate $e_vo = e_v - e_o$.
    '''
    n2c = mf.mol.nao_2c()
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    occidx = n2c + numpy.where(mo_occ[n2c:]==1)[0]
    viridx = n2c + numpy.where(mo_occ[n2c:]==0)[0]
    e_v = mo_energy[viridx]
    e_o = mo_energy[occidx]
    e_vo = e_v[:,None] - e_o
    return e_vo

def _to_vo(mf, ao):
    '''
    Convert some quantity in AO basis to that in vir-occ MO basis.
    '''
    n2c = mf.mol.nao_2c()
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = n2c + numpy.where(mo_occ[n2c:]==1)[0]
    viridx = n2c + numpy.where(mo_occ[n2c:]==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    vo = lib.einsum('pj,xpq,qi->xji', orbv.conj(), ao, orbo)
    return vo

def _to_vv(mf, ao):
    '''
    Convert some quantity in AO basis to that in vir-vir MO basis.
    '''
    n2c = mf.mol.nao_2c()
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    viridx = n2c + numpy.where(mo_occ[n2c:] == 0)[0]
    orbv = mo_coeff[:,viridx]
    vv = lib.einsum('pj,xpq,qi->xji', orbv.conj(), ao, orbv)
    return vv

def _to_oo(mf, ao):
    '''
    Convert some quantity in AO basis to that in occ-occ MO basis.
    '''
    n2c = mf.mol.nao_2c()
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = n2c + numpy.where(mo_occ[n2c:] == 1)[0]
    orbo = mo_coeff[:,occidx]
    oo = lib.einsum('pj,xpq,qi->xji', orbo.conj(), ao, orbo)
    return oo

def _block_diag(a, b):
    '''
    3-D block diagonalizer.
    [[A 0],
     [0 B]]
    '''
    assert len(a) == len(b) and a.ndim == b.ndim == 3
    diag = numpy.zeros((len(a), a.shape[1]+b.shape[1],
                                a.shape[2]+b.shape[2]), dtype='complex128')
    diag[...,:a.shape[1],:a.shape[2]] = a
    diag[...,a.shape[1]:,a.shape[2]:] = b
    return diag


class Polarizability(ghf.Polarizability):
    @lib.with_doc(get_dm1.__doc__)
    def get_dm1(self, freq=None, mo1=None, picture_change=True, solver='krylov'):
        if freq is None: freq = 0
        if mo1 is None: mo1 = self.get_mo1(freq, picture_change, solver)
        return get_dm1(self, mo1)
    
    get_h1 = get_h1

    @lib.with_doc(get_e_vo.__doc__)
    def _get_e_vo(self) : return get_e_vo(self.mf)
    
    def _to_vo(self, ao): return _to_vo(self.mf, ao)
    def _to_vv(self, ao): return _to_vv(self.mf, ao)
    def _to_oo(self, ao): return _to_oo(self.mf, ao)
    
from pyscf.scf import dhf
from pyscf.dft import dks
dhf.DHF.Polarizability = dks.DKS.Polarizability = lib.class_as_method(Polarizability)


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom = '''O     0.000000     0.000000     0.123323
                          H     0.000000     0.757497    -0.493291
                          H     0.000000    -0.757497    -0.493291''',
                charge = 1,
                spin = 1,
                basis = 'ccpvdz')
    
    mf1 = mol.DHF()
    mf1.conv_tol = 1e-11
    mf1.kernel()
    polar1 = Polarizability(mf1)
    #polar1.conv_tol = 1e-11
    print(polar1.polarizability())
    print(polar1.polarizability_with_freq())
    #print(polar1.hyper_polarizability())

    #mf2 = mol.DKS().x2c()
    #mf2.xc = 'b3lyp'
    #mf2.collinear = 'mcol'
    #mf2.conv_tol = 1e-11
    #mf2.kernel()
    #polar2 = Polarizability(mf2)
    #polar2.conv_tol = 1e-11
    #print(polar2.polarizability())
    #print(polar2.hyper_polarizability())