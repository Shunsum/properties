#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic magnetizability tensor for RHF
(In testing)

Refs:
[1] R. Cammi, J. Chem. Phys., 109, 3185 (1998)
[2] Todd A. Keith, Chem. Phys., 213, 123 (1996)
[3] S. Sauer et al., Mol. Phys., 76, 445 (1991)
'''


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import jk, _response_functions  # noqa
from pyscf.prop.nmr import rhf as rhf_nmr
from pyscf.prop.cphf import CPHFBase
from pyscf.data import nist

def el_mag_moment(magobj, gauge_orig=None, with_imag=False):
    '''Electronic magnetic moment'''
    mol = magobj.mol
    mf = magobj.mf
    dm0 = mf.make_rdm1()
    if with_imag:
        h1 = get_h1(mol, gauge_orig)
        if gauge_orig is None:
            jk1 = get_jk1(mol, dm0) * .5
            mu = -lib.einsum('vu,xuv->x', dm0, h1+jk1)
            s1 = magobj._to_oo(get_s1(mol))
            mu += lib.einsum('xii,i->x', s1, mf.mo_energy[mf.mo_occ>0]) * 2
        else:
            mu = -lib.einsum('vu,xuv->x', dm0, h1)
        return mu
    else:
        return numpy.zeros(3)

def diamag(magobj, gauge_orig=None):
    '''Diamagnetic magnetizability'''
    mol = magobj.mol
    mf = magobj.mf
    dm0 = mf.make_rdm1()
    
    h2 = get_h2(mol, gauge_orig)
    if gauge_orig is None:
        jk2 = get_jk2(mol, dm0) * .5
        xi = -lib.einsum('vu,xyuv->xy', dm0, h2+jk2)
        s2 = magobj._to_oo(get_s2(mol))
        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy[mo_occ>0]
        mo_occ = mo_occ[mo_occ>0]
        xi += lib.einsum('xyii,i,i->xy', s2, mo_occ, mo_energy)
    else:
        xi = -lib.einsum('vu,xyuv->xy', dm0, h2)
    
    return xi

def paramag(magobj, gauge_orig=None, freq=0):
    '''Paramagnetic magnetizability'''
    log = logger.Logger(magobj.stdout, magobj.verbose)
    cput1 = (logger.process_clock(), logger.perf_counter())

    mol = magobj.mol
    mf = magobj.mf
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:,occidx]

    mo1 = magobj.solve_mo1(freq=freq)
    e0to = magobj.get_e0to()
    s1 = magobj._to_to(magobj.get_s1())
    f1 = magobj._to_to(magobj.get_f1(mo1, freq))
    v1 = magobj._to_to(magobj.get_vind(mo1, freq))
    
    mag_para = numpy.einsum('xji,yji,i->xy', mo1.conj(), s1, mo_energy[occidx]) * 2
    mag_para-= numpy.einsum('xji,yji->xy', mo1, f1.conj()) * 2
    mag_para+= mag_para.T
    mag_para-= numpy.einsum('xji,yji,ji->xy', mo1, mo1.conj(), e0to) * 2
    mag_para+= numpy.einsum('xji,yji->xy', v1, mo1.conj()) * 2
    # + c.c.
    mag_para+= mag_para.conj()

    return mag_para.real

def get_h1(mol, gauge_orig=None, picture_change=True):
    r'''The first-order core Hamiltonian wrt the external magnetic field.
        Without gauge origin (using GIAO):
            h_{\mu\nu}^(1) = (\mu|g kin|\nu) - (\mu|Rxp|\nu) * .5
                           + (\mu|g nuc|\nu) + (\mu|rxp|\nu) * .5
        With a gauge origin:
            h_{\mu\nu}^(1) = (\mu|rxp|\nu) * .5'''
    if gauge_orig is None:
        h1  = mol.intor('int1e_igkin', comp=3)
        h1 += mol.intor_asymmetric('int1e_ignuc', comp=3)
        h1 += mol.intor('int1e_giao_irjxp', comp=3) * .5
    else:
        mol.set_common_origin(gauge_orig)
        h1 = mol.intor_asymmetric('int1e_cg_irxp', comp=3) * .5
    return h1 * -1j

def get_jk1(mol, dm0):
    r'''The first-order JK wrt the external magnetic field.
        J_{\mu\nu}^(1) = D_{\lambda\kappa}^0 (g\mu\nu|\kappa\lambda) * 2
        K_{\mu\nu}^(1) = D_{\lambda\kappa}^0 (g\mu\lambda|\kappa\nu) * 2
        JK^(1) = J^(1) - .5*K^(1)'''
    j1 = jk.get_jk(mol, dm0, 'ijkl,lk->a2ij', 'int2e_ig1', 'a4ij', comp=3, hermi=2)
    k1 = jk.get_jk(mol, dm0, 'ijkl,jk->s1il', 'int2e_ig1', 'a4ij', comp=3, hermi=0)
    return (j1 * 2 - k1) * -1j

def get_s1(mol):
    r'''The first-order overlap matrix wrt the external magnetic field.
        S_{\mu\nu}^(1) = (\mu|g|\nu)'''
    return mol.intor_asymmetric('int1e_igovlp', comp=3) * -1j

def get_h2(mol, gauge_orig=None, picture_change=True):
    r'''The second-order core Hamiltonian wrt the external magnetic field.
        Without gauge origin (using GIAO):
            h_{\mu\nu}^(2) = (\mu|gg kin|\nu) + (\mu|gg nuc|\nu)
                + .5 (\mu|g rxp|\nu) - .5 (\mu|g Rxp|\nu) + transpose(x,y)
                + .25 (\mu|r·r-rr|\nu) - .5 (\mu|R·r-Rr|\nu) + .25 (\mu|R·R-RR|\nu)
        With a gauge origin:
            h_{\mu\nu}^(2) = .25 (\mu|r·r-rr|\nu)'''
    nao = mol.nao
    if gauge_orig is None:
        h2gg  = mol.intor('int1e_ggkin', comp=9)
        h2gg += mol.intor_symmetric('int1e_ggnuc', comp=9)

        h2g  = mol.intor('int1e_grjxp', comp=9).reshape(3,3,nao,nao) * .5
        h2g += h2g.transpose(1,0,2,3)

        h2 = mol.intor('int1e_rr_origj', comp=9).reshape(3,3,nao,nao) * .25
        h2 = lib.einsum('xy,zzuv->xyuv', numpy.eye(3), h2) - h2

        h2 = h2gg.reshape(3,3,nao,nao) + h2g + h2
    else:
        mol.set_common_origin(gauge_orig)
        h2 = mol.intor_symmetric('int1e_rr', comp=9).reshape(3,3,nao,nao) * .25
        h2 = lib.einsum('xy,zzuv->xyuv', numpy.eye(3), h2) - h2
    return h2

def get_jk2(mol, dm0):
    r'''The second-order JK wrt the external magnetic field.
        J_{\mu\nu}^(2) = D_{\lambda\kappa}^0 (gg\mu\nu|\kappa\lambda) * 2
                       + D_{\lambda\kappa}^0 (g\mu\nu|g\kappa\lambda) * 2
        K_{\mu\nu}^(2) = D_{\lambda\kappa}^0 (gg\mu\lambda|\kappa\nu) * 2
                       + D_{\lambda\kappa}^0 (g\mu\lambda|g\kappa\nu) * 2
        JK^(2) = J^(2) - .5*K^(2)'''
    j2  = jk.get_jk(mol, dm0, 'ijkl,lk->s2ij', 'int2e_gg1' , 's4' , comp=9, hermi=1)
    j2 += jk.get_jk(mol, dm0, 'ijkl,lk->s2ij', 'int2e_g1g2', 'aa4', comp=9, hermi=1)
    k2  = jk.get_jk(mol, dm0, 'ijkl,jk->s1il', 'int2e_gg1' , 's4' , comp=9, hermi=0)
    k2 += jk.get_jk(mol, dm0, 'ijkl,jk->s1il', 'int2e_g1g2', 'aa4', comp=9, hermi=0)
    jk2 = j2 * 2 - k2
    return jk2.reshape(3,3,*jk2.shape[-2:])

def get_s2(mol):
    r'''The second-order overlap matrix wrt the external magnetic field.
        S_{\mu\nu}^(2) = (\mu|gg|\nu)'''
    nao = mol.nao
    return mol.intor_symmetric('int1e_ggovlp', comp=9).reshape(3,3,nao,nao)

def _get_ao_coords(mol):
    atom_coords = mol.atom_coords()
    nao = mol.nao_nr()
    ao_coords = numpy.empty((nao, 3))
    aoslices = mol.aoslice_by_atom()
    for atm_id, (ish0, ish1, i0, i1) in enumerate(aoslices):
        ao_coords[i0:i1] = atom_coords[atm_id]
    return ao_coords


class Magnetizability(CPHFBase):
    def __init__(self, mf):
        super().__init__(mf)
        self._gauge_orig = None
        self.with_s1 = True

    @property
    def gauge_orig(self):
        return self._gauge_orig

    @gauge_orig.setter
    def gauge_orig(self, orig):
        self._gauge_orig = orig
        self.with_s1 = (self._gauge_orig is None)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s ********',
                 self.__class__, self.mf.__class__)
        if self.gauge_orig is None:
            log.info('gauge = GIAO')
        else:
            log.info('Common gauge = %s', str(self.gauge_orig))
        if self.cphf:
            log.info('Solving MO10 eq with CPHF.')
            log.info('CPHF conv_tol = %g', self.conv_tol)
            log.info('CPHF max_cycle_cphf = %d', self.max_cycle_cphf)
        if not self.mf.converged:
            log.warn('Ground state SCF is not converged')
        return self

    def kernel(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()

        mag_dia = self.dia(self.gauge_orig)
        mag_para = self.para(self.gauge_orig)
        ksi = mag_para + mag_dia

        logger.timer(self, 'Magnetizability', *cput0)
        if self.verbose >= logger.NOTE:
            _write = rhf_nmr._write
            _write(self.stdout, ksi, '\nMagnetizability (au)')
            _write(self.stdout, mag_dia, 'dia-magnetic contribution (au)')
            _write(self.stdout, mag_para, 'para-magnetic contribution (au)')
            #if self.verbose >= logger.INFO:
            #    _write(self.stdout, para_occ, 'occ part of para-magnetic term')
            #    _write(self.stdout, para_vir, 'vir part of para-magnetic term')

            unit = nist.HARTREE2J / nist.AU2TESLA**2 * 1e30
            _write(self.stdout, ksi*unit, '\nMagnetizability (10^{-30} J/T^2)')
        return ksi

    def get_e0to(self):
        '''e0to = e0t - e0o.'''
        mf = self.mf
        mo_energy = mf.mo_energy
        e0o = mo_energy[mf.mo_occ>0]
        e0to = mo_energy[:,None] - e0o
        return e0to

    el_mag_moment = el_mag_moment
    dia = diamag = diamag
    para = paramag = paramag
    get_fock = rhf_nmr.get_fock

    def get_h1(self, **kwargs):
        mol = self.mol
        gauge_orig = self.gauge_orig
        return get_h1(mol, gauge_orig, **kwargs)
    
    def get_jk1(self, dm0=None):
        mol = self.mol
        if dm0 is None: dm0 = self.mf.make_rdm1()
        return get_jk1(mol, dm0)
    
    def get_s1(self):
        mol = self.mol
        return get_s1(mol)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = '''h  ,  0.   0.   0.
                  F  ,  0.   0.   .917'''
    mol.basis = '631g'
    mol.build()

    mf = scf.RHF(mol).run()
    mag = Magnetizability(mf)
    mag.cphf = True
    m = mag.kernel()
    print(lib.finger(m) - -0.43596639996758657)

    mag.gauge_orig = (0,0,1)
    m = mag.kernel()
    print(lib.finger(m) - -0.76996086788058238)

    mag.gauge_orig = (0,0,1)
    mag.cphf = False
    m = mag.kernel()
    print(lib.finger(m) - -0.7973915717274408)


    mol = gto.M(atom='''O      0.   0.       0.
                        H      0.  -0.757    0.587
                        H      0.   0.757    0.587''',
                basis='ccpvdz')
    mf = scf.RHF(mol).run()
    mag = Magnetizability(mf)
    mag.cphf = True
    m = mag.kernel()
    print(lib.finger(m) - -0.62173669377370366)
