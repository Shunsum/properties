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
Non-relativistic static and dynamic polarizability and hyper-polarizability tensor
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, _response_functions
from pyscf.x2c import sfx2c1e, x2c


def get_dm1(polobj, mo1, freq=0):
    '''
    Generate the 1st-order density matrix in AO basis.
    '''
    mf = polobj.mf
    mo_coeff = mf.mo_coeff
    occidx = mf.mo_occ > 0
    orbv = mo_coeff[:,~occidx]
    orbo = mo_coeff[:, occidx]
    if freq == 0 and len(mo1) != 2:
        dm1  = lib.einsum('pj,xji,qi->xpq', orbv, mo1, orbo.conj()) * 2
        dm1 += dm1.transpose(0,2,1).conj()
    else:
        dm1  = lib.einsum('pj,xji,qi->xpq', orbv, mo1[0]       , orbo.conj()) * 2
        dm1 += lib.einsum('pi,xji,qj->xpq', orbo, mo1[1].conj(), orbv.conj()) * 2
    return dm1

def get_h1(polobj, picture_change=True):
    '''
    Generate the dipole matrix in AO basis.
    '''
    mf = polobj.mf
    mol = mf.mol
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = lib.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center):
        if isinstance(mf, sfx2c1e.SFX2C1E_SCF) and picture_change:
            xmol = mf.with_x2c.get_xmol()[0]
            nao = xmol.nao
            prp = xmol.intor_symmetric('int1e_sprsp').reshape(3,4,nao,nao)[:,3]
            c1 = 0.5/lib.param.LIGHT_SPEED
            ao_dip = mf.with_x2c.picture_change(('int1e_r', prp*c1**2))
        else:
            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    return -ao_dip

def get_v1(polobj, mo1, freq=0):
    '''
    Generate the 1st-order effective potential matrix in AO basis.
    '''
    mf = polobj.mf
    dm1 = polobj.get_dm1(mo1, freq)

    if isinstance(mo1, tuple): # UHF/UKS
        if freq == 0 and len(mo1) != 4:
            v1 = mf.gen_response(hermi=1)
        else:
            v1 = mf.gen_response(hermi=0)
    
    else:
        if freq == 0 and len(mo1) != 2:
            v1 = mf.gen_response(hermi=1)
        else:
            v1 = mf.gen_response(hermi=0)
            if isinstance(mf, x2c.X2C1E_GSCF) and \
            hasattr(mf, 'collinear') and mf.collinear == 'col':
                dm1r = dm1.real
                dm1i = dm1.imag
                return v1(dm1r) + v1(dm1i)*1j
    
    return v1(dm1)

def get_f1(polobj, mo1, freq=0, picture_change=True):
    '''
    Generate the 1st-order Fock matrix in AO basis.
    '''
    h1 = polobj.get_h1(picture_change)
    v1 = get_v1(polobj, mo1, freq)
    return h1 + v1

def get_h1vo(polobj, picture_change=True):
    '''
    Generate the dipole matrix $-<vir.|r|occ.>$ in MO basis.
    '''
    h1 = polobj.get_h1(picture_change)
    h1 = polobj._to_vo(h1)
    return h1

def get_v1vo(polobj, mo1, freq=0):
    '''
    Generate the 1st-order effective potential matrix $<vir.|\hat{v}|occ.>$ in MO basis.
    '''
    v1 = get_v1(polobj, mo1, freq)
    v1 = polobj._to_vo(v1)
    return v1

def get_f1vo(polobj, mo1, freq=0, picture_change=True):
    '''
    Generate the 1st-order Fock matrix $<vir.|\hat{f}|occ.>$ in MO basis.
    '''
    f1 = get_f1(polobj, mo1, freq, picture_change)
    f1 = polobj._to_vo(f1)
    return f1

def get_f1vv(polobj, mo1, freq=0, picture_change=True):
    '''
    Generate the 1st-order Fock matrix $<vir.|\hat{f}|vir.>$ in MO basis.
    '''
    f1 = get_f1(polobj, mo1, freq, picture_change)
    f1 = polobj._to_vv(f1)
    return f1

def get_f1oo(polobj, mo1, freq=0, picture_change=True):
    '''
    Generate the 1st-order Fock matrix $<occ.|\hat{f}|occ.>$ in MO basis,
    which is equivalently the 1st-order occ-occ orbital energy matrix.
    '''
    f1 = get_f1(polobj, mo1, freq, picture_change)
    f1 = polobj._to_oo(f1)
    return f1

def get_e_vo(mf):
    '''
    Generate $e_vo = e_v - e_o$.
    '''
    occidx = mf.mo_occ > 0
    mo_energy = mf.mo_energy
    e_v = mo_energy[~occidx]
    e_o = mo_energy[ occidx]
    e_vo = e_v[:,None] - e_o
    return e_vo

def _to_vo(mf, ao):
    '''
    Convert some quantity in AO basis to that in vir-occ MO basis.
    '''
    mo_coeff = mf.mo_coeff
    occidx = mf.mo_occ > 0
    orbv = mo_coeff[:,~occidx]
    orbo = mo_coeff[:, occidx]
    vo = lib.einsum('pj,xpq,qi->xji', orbv.conj(), ao, orbo)
    return vo

def _to_vv(mf, ao):
    '''
    Convert some quantity in AO basis to that in vir-occ MO basis.
    '''
    mo_coeff = mf.mo_coeff
    viridx = mf.mo_occ == 0
    orbv = mo_coeff[:, viridx]
    vv = lib.einsum('pj,xpq,qi->xji', orbv.conj(), ao, orbv)
    return vv

def _to_oo(mf, ao):
    '''
    Convert some quantity in AO basis to that in vir-occ MO basis.
    '''
    mo_coeff = mf.mo_coeff
    occidx = mf.mo_occ > 0
    orbo = mo_coeff[:, occidx]
    oo = lib.einsum('pj,xpq,qi->xji', orbo.conj(), ao, orbo)
    return oo


def el_dip_moment(polobj, picture_change=True):
    '''
    Electronic dipole moment (with picture change correction if in SFX2C).
    '''
    log = logger.new_logger(polobj)
    mf = polobj.mf
    h1 = polobj.get_h1(picture_change)
    dm = mf.make_rdm1()

    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        # UHF density matrices
        dm = dm[0] + dm[1]
    
    mu = -lib.einsum('xpq,qp->x', h1, dm)

    if mf.verbose >= logger.INFO:
        log.note('Electronic dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mu)

    return mu

# Note: polarizability and relevant properties are demanding on basis sets.
# ORCA recommends to use Sadlej basis for these properties.
def polarizability(polobj, freq=0, picture_change=True, solver='krylov'):
    '''
    Polarizability (with picture change correction if in SFX2C).
    '''
    log = logger.new_logger(polobj)
    mf = polobj.mf
    h1 = polobj.get_h1vo(picture_change)
    mo1 = polobj.get_mo1(freq, picture_change, solver)

    if freq == 0:
        # alpha(0;0)
        alpha  = -lib.einsum('xji,yji->xy', h1.conj(), mo1) * 2
        alpha += alpha.conj()

    else:
        # alpha(-omega;omega)
        alpha  = -lib.einsum('xji,yji->xy', h1.conj(), mo1[0]) * 2
        alpha += -lib.einsum('xji,yji->xy', h1, mo1[1].conj()) * 2

    if mf.verbose >= logger.INFO:
        xx, yy, zz = alpha.diagonal()
        log.note('Isotropic polarizability %.12g', (xx+yy+zz)/3)
        log.note('Polarizability anisotropy %.12g',
                 (.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2))**.5)
        log.debug(f'Polarizability tensor alpha({-freq};{freq})')
        log.debug(f'{alpha}')

    return alpha

def hyperpolarizability(polobj, freq=0, type='SHG', picture_change=True, solver='krylov'):
    '''
    Hyperpolarizability (with picture change correction if in SFX2C).
    '''
    log = logger.new_logger(polobj)
    mf = polobj.mf

    if freq == 0:
        mo1 = polobj.get_mo1(freq, picture_change, solver)
        f1 = polobj.get_f1vv(mo1, freq, picture_change)
        e1 = polobj.get_e1(mo1, freq, picture_change)

        # beta(0;0,0)
        beta = -lib.einsum('xjl,yji,zli->xyz', f1, mo1.conj(), mo1) * 2
        beta += lib.einsum('xik,yjk,zji->xyz', e1, mo1.conj(), mo1) * 2
        beta += beta.transpose(0,2,1) + beta.transpose(1,0,2) + \
                beta.transpose(1,2,0) + beta.transpose(2,0,1) + beta.transpose(2,1,0)
        
        if isinstance(mf, hf.KohnShamDFT):
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm = mf.make_rdm1()
            dm1 = polobj.get_dm1(mo1, freq)

            if xctype == 'LDA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype)
                rho1 = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                    for dmx in dm1]).reshape(3,1,-1) # reshape to match kxc

            elif xctype == 'GGA' or 'MGGA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype, with_lapl=False)
                rho1 = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                with_lapl=False) for dmx in dm1])

            else:
                raise RuntimeError(f"Unknown xctype '{xctype}'")
            
            kxc = ni.eval_xc_eff(mf.xc, rho, deriv=3)[3] * mf.grids.weights
            beta -= lib.einsum('xig,yjg,zkg,ijkg->xyz', rho1, rho1, rho1, kxc)
        
        if mf.verbose >= logger.INFO:
            log.debug(f'Static hyperpolarizability tensor beta({freq};{freq},{freq})')
            log.debug(f'{beta}')
    
    elif type.upper() == 'SHG':
        mo1_2o = polobj.get_mo1(freq*2, picture_change, solver)
        f1_m2o = polobj.get_f1vv(mo1_2o, freq*2, picture_change).transpose(0,2,1).conj()
        e1_m2o = polobj.get_e1(mo1_2o, freq*2, picture_change).transpose(0,2,1).conj()

        mo1_1o = polobj.get_mo1(freq, picture_change, solver)
        f1_p1o = polobj.get_f1vv(mo1_1o, freq, picture_change)
        e1_p1o = polobj.get_e1(mo1_1o, freq, picture_change)

        # beta(-2omega;omega,omega)
        beta = -lib.einsum('xjl,yji,zli->xyz', f1_m2o, mo1_1o[1].conj(), mo1_1o[0]) * 2
        beta +=-lib.einsum('yjl,xji,zli->xyz', f1_p1o, mo1_2o[0].conj(), mo1_1o[0]) * 2
        beta +=-lib.einsum('zjl,yji,xli->xyz', f1_p1o, mo1_1o[1].conj(), mo1_2o[1]) * 2
        beta += lib.einsum('xik,yjk,zji->xyz', e1_m2o, mo1_1o[1].conj(), mo1_1o[0]) * 2
        beta += lib.einsum('yik,xjk,zji->xyz', e1_p1o, mo1_2o[0].conj(), mo1_1o[0]) * 2
        beta += lib.einsum('zik,yjk,xji->xyz', e1_p1o, mo1_1o[1].conj(), mo1_2o[1]) * 2
        beta += beta.transpose(0,2,1)
        
        if isinstance(mf, hf.KohnShamDFT):
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm = mf.make_rdm1()
            dm1_m2o = polobj.get_dm1(mo1_2o, freq*2).transpose(0,2,1).conj()
            dm1_p1o = polobj.get_dm1(mo1_1o, freq)

            if xctype == 'LDA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype)
                rho1_m2o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                        for dmx in dm1_m2o]).reshape(3,1,-1) # reshape to match kxc
                rho1_p1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                        for dmx in dm1_p1o]).reshape(3,1,-1) # reshape to match kxc
                
            elif xctype == 'GGA' or 'MGGA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype, with_lapl=False)
                rho1_m2o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                    with_lapl=False) for dmx in dm1_m2o])
                rho1_p1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                    with_lapl=False) for dmx in dm1_p1o])

            else:
                raise RuntimeError(f"Unknown xctype '{xctype}'")
            
            kxc = ni.eval_xc_eff(mf.xc, rho, deriv=3)[3] * mf.grids.weights
            beta -= lib.einsum('xig,yjg,zkg,ijkg->xyz', rho1_m2o, rho1_p1o, rho1_p1o, kxc)

        if mf.verbose >= logger.INFO:
            log.debug(f'{type} hyperpolarizability tensor beta({-2*freq};{freq},{freq})')
            log.debug(f'{beta}')
        
    elif type.upper() == 'EOPE' or 'OR':    
        mo1_1o = polobj.get_mo1(freq, picture_change, solver)
        f1_p1o = polobj.get_f1vv(mo1_1o, freq, picture_change)
        f1_m1o = f1_p1o.transpose(0,2,1).conj()
        e1_p1o = polobj.get_e1(mo1_1o, freq, picture_change)
        e1_m1o = e1_p1o.transpose(0,2,1).conj()

        mo1 = polobj.get_mo1(0, picture_change, solver)
        f1 = polobj.get_f1vv(mo1, 0, picture_change)
        e1 = polobj.get_e1(mo1, 0, picture_change)

        # beta(-omega;omega,0)
        beta = -lib.einsum('xjl,yji,zli->xyz', f1_m1o, mo1_1o[1].conj(), mo1) * 2
        beta +=-lib.einsum('xjl,zji,yli->xyz', f1_m1o, mo1.conj(), mo1_1o[0]) * 2
        beta +=-lib.einsum('yjl,xji,zli->xyz', f1_p1o, mo1_1o[0].conj(), mo1) * 2
        beta +=-lib.einsum('yjl,zji,xli->xyz', f1_p1o, mo1.conj(), mo1_1o[1]) * 2
        beta +=-lib.einsum('zjl,xji,yli->xyz', f1, mo1_1o[0].conj(), mo1_1o[0]) * 2
        beta +=-lib.einsum('zjl,yji,xli->xyz', f1, mo1_1o[1].conj(), mo1_1o[1]) * 2
        beta += lib.einsum('xik,yjk,zji->xyz', e1_m1o, mo1_1o[1].conj(), mo1) * 2
        beta += lib.einsum('xik,zjk,yji->xyz', e1_m1o, mo1.conj(), mo1_1o[0]) * 2
        beta += lib.einsum('yik,xjk,zji->xyz', e1_p1o, mo1_1o[0].conj(), mo1) * 2
        beta += lib.einsum('yik,zjk,xji->xyz', e1_p1o, mo1.conj(), mo1_1o[1]) * 2
        beta += lib.einsum('zik,xjk,yji->xyz', e1, mo1_1o[0].conj(), mo1_1o[0]) * 2
        beta += lib.einsum('zik,yjk,xji->xyz', e1, mo1_1o[1].conj(), mo1_1o[1]) * 2
        
        if isinstance(mf, hf.KohnShamDFT):
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm = mf.make_rdm1()
            dm1 = polobj.get_dm1(mo1, 0)
            dm1_p1o = polobj.get_dm1(mo1_1o, freq)
            dm1_m1o = dm1_p1o.transpose(0,2,1).conj()

            if xctype == 'LDA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype)
                rho1 = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                    for dmx in dm1]).reshape(3,1,-1) # reshape to match kxc
                rho1_p1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                        for dmx in dm1_p1o]).reshape(3,1,-1) # reshape to match kxc
                rho1_m1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                        for dmx in dm1_m1o]).reshape(3,1,-1) # reshape to match kxc
                
            elif xctype == 'GGA' or 'MGGA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype, with_lapl=False)
                rho1 = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                with_lapl=False) for dmx in dm1])
                rho1_p1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                    with_lapl=False) for dmx in dm1_p1o])
                rho1_m1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                    with_lapl=False) for dmx in dm1_m1o])
                
            else:
                raise RuntimeError(f"Unknown xctype '{xctype}'")
            
            kxc = ni.eval_xc_eff(mf.xc, rho, deriv=3)[3] * mf.grids.weights
            beta -= lib.einsum('xig,yjg,zkg,ijkg->xyz', rho1_m1o, rho1_p1o, rho1, kxc)
        
        if type.upper() == 'EOPE':
            if mf.verbose >= logger.INFO:
                log.debug(f'{type} hyperpolarizability tensor beta({-freq};{freq},0)')
                log.debug(f'{beta}')
        
        if type.upper() == 'OR':
            if mf.verbose >= logger.INFO:
                log.debug(f'{type} hyperpolarizability tensor beta(0;{freq},{-freq})')
                log.debug(f'{beta.transpose(2,1,0)}')
                
            beta = beta.transpose(2,1,0)
    
    else:
        raise NotImplementedError(f'{type}')
    
    return beta

# Solve the frequency-dependent CPHF problem
# [A+wI, B   ] [X] + [h1] = [0]
# [B   , A-wI] [Y]   [h1]   [0]

# TODO: new solver with Arnoldi iteration.
# The krylov solver in this implementation often fails. see
# https://github.com/pyscf/pyscf/issues/507

def krylov_solver(polobj, freq=0, picture_change=True, verbose=logger.WARN):
    '''
    1st-order CPHF/KS iterative solver by `pyscf.lib.krylov`.
    '''
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    e_vo = polobj._get_e_vo()
    nvir, nocc = e_vo.shape
    h1 = polobj.get_h1vo(picture_change)

    if freq == 0:
        def vind(mo1):
            mo1 = mo1.reshape(3,nvir,nocc)
            v1 = polobj.get_v1vo(mo1, freq) / e_vo
            return v1.ravel()
        
        mo1base = -h1 / e_vo
        mo1 = lib.krylov(vind, mo1base.ravel(), max_cycle=polobj.max_cycle_cphf,
                         tol=polobj.conv_tol, verbose=log).reshape(3,nvir,nocc)
    
    else:
        def vind(mo1):
            mo1 = mo1.reshape(2,-1,nvir,nocc) # mo1[0]=U(+omega), mo1[1]=U(-omega)
            v1 = get_v1(polobj, mo1, freq)
            v1_p = polobj._to_vo(v1)
            v1_m = polobj._to_vo(v1.transpose(0,2,1).conj())
            v1_p /= (e_vo + freq)
            v1_m /= (e_vo - freq)
            v1 = numpy.stack((v1_p, v1_m))
            return v1.ravel()
        # calculating each direction separately can give a more exact result
        mo1base = -numpy.stack((h1/(e_vo+freq), h1/(e_vo-freq))) # shape: (2,3,nvir,nocc)
        mo1 = numpy.empty_like(mo1base)
        mo1[:,0] = lib.krylov(vind, mo1base[:,0].ravel(), max_cycle=polobj.max_cycle_cphf,
                              tol=polobj.conv_tol, verbose=log).reshape(2,nvir,nocc)
        mo1[:,1] = lib.krylov(vind, mo1base[:,1].ravel(), max_cycle=polobj.max_cycle_cphf,
                              tol=polobj.conv_tol, verbose=log).reshape(2,nvir,nocc)
        mo1[:,2] = lib.krylov(vind, mo1base[:,2].ravel(), max_cycle=polobj.max_cycle_cphf,
                              tol=polobj.conv_tol, verbose=log).reshape(2,nvir,nocc)
    
    log.timer('Krylov solver in CPHF/KS', *t0)
    return mo1

def newton_solver(polobj, freq=0, picture_change=True, verbose=logger.WARN):
    '''
    1st-order CPHF/KS iterative solver by `scipy.optimize.newton_krylov`.
    '''
    from scipy.optimize import newton_krylov
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    e_vo = polobj._get_e_vo()
    h1 = polobj.get_h1vo(picture_change)

    if freq == 0:
        def vind(mo1):
            v1  = polobj.get_v1vo(mo1, freq)
            v1 += e_vo * mo1
            return v1 + h1

    else:
        h1 = numpy.stack((h1, h1))
        def vind(mo1):
            v1 = get_v1(polobj, mo1, freq)
            v1_p = polobj._to_vo(v1)
            v1_m = polobj._to_vo(v1.transpose(0,2,1).conj())
            v1_p += (e_vo + freq) * mo1[0]
            v1_m += (e_vo - freq) * mo1[1]
            v1 = numpy.stack((v1_p, v1_m))
            return v1 + h1

    mo1 = newton_krylov(vind, -h1, maxiter=polobj.max_cycle_cphf,
                        f_tol=polobj.conv_tol)
    
    log.timer('Newton-Krylov solver in CPHF/KS', *t0)
    return mo1

def direct_solver(polobj, freq=0, picture_change=True, verbose=logger.WARN):
    '''
    1st-order CPHF/KS direct solver by `numpy.linalg.solve`.
    '''
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    e_vo = polobj._get_e_vo()
    nvir, nocc = e_vo.shape
    h1 = polobj.get_h1vo(picture_change)
    h1 = numpy.stack((h1, h1.conj()), axis=1)
    
    def vind(mo1):
        mo1 = mo1.reshape(2,1,nvir,nocc) # mo1[0]=U(+omega), mo1[1]=U*(-omega)
        # mo1[1] = mo1[1].conj() # mo1[1] = U(-omega) <unnecessary with the identity passed in>
        v1 = get_v1(polobj, mo1, freq)
        v1_p = polobj._to_vo(v1)
        v1_m = polobj._to_vo(v1.transpose(0,2,1).conj())
        v1_p += (e_vo + freq) * mo1[0]
        v1_m += (e_vo - freq) * mo1[1]
        v1 = numpy.stack((v1_p, v1_m.conj())) # shape: (2,1,nvir,nocc)
        return v1.ravel()

    size = h1[0].size
    operator = numpy.empty((size, size))
    iden = numpy.eye(size)
    for i, row in enumerate(iden):
        operator[:,i] = vind(row)
    
    mo1 = numpy.linalg.solve(operator, -h1.reshape(3,-1).T).T
    mo1 = mo1.reshape(h1.shape).transpose(1,0,2,3)
    #import pdb; pdb.set_trace()
    if freq == 0: return mo1[0]
    else: mo1[1] = mo1[1].conj()
    
    log.timer('Direct solver in CPHF/KS', *t0)
    return mo1


class Polarizability(lib.StreamObject):
    def __init__(self, mf):
        self.mf = mf
        self.stdout = mf.stdout
        self.verbose = mf.verbose

        self.max_cycle_cphf = 50
        self.conv_tol = 1e-9

        self._keys = set(self.__dict__.keys())
    
    def get_mo1(self, freq=0, picture_change=True, solver='krylov'):
        '''
        Generate the transformation matrix $U(0)$ or ($U(+\omega)$, $U(-\omega)$),
        which is the coefficient matrix of the 1st-order derivative of MO.
        '''
        solver = solver.lower()
        if   'direct' in solver: # exact solver
            mo1 = direct_solver(self, freq, picture_change)
        elif 'newton' in solver: # only newton-krylov recommended
            mo1 = newton_solver(self, freq, picture_change)
        elif 'krylov' in solver: # very recommended
            mo1 = krylov_solver(self, freq, picture_change)
        else:
            raise NotImplementedError(solver)
        return mo1
    
    @lib.with_doc(get_e_vo.__doc__)
    def _get_e_vo(self) : return get_e_vo(self.mf)
    
    def _to_vo(self, ao): return _to_vo(self.mf, ao)
    def _to_vv(self, ao): return _to_vv(self.mf, ao)
    def _to_oo(self, ao): return _to_oo(self.mf, ao)

    get_dm1 = get_dm1
    get_h1 = get_h1

    get_h1vo = get_h1vo
    get_v1vo = get_v1vo
    get_f1vv = get_f1vv
    get_e1 = get_f1oo = get_f1oo

    el_dip_moment = el_dip_moment
    polarizability = polarizability
    hyperpolarizability = hyperpolarizability


hf.RHF.Polarizability = sfx2c1e.SFX2C1E_SCF.Polarizability = lib.class_as_method(Polarizability)


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.Mole()
    mol.atom = '''h  ,  0.   0.   0.
                  F  ,  0.   0.   .917'''
    mol.basis = '631g'
    mol.build()

    mf = scf.RHF(mol).run(conv_tol=1e-14)
    polobj = mf.Polarizability().polarizability()
    hpol = mf.Polarizability().hyperpolarizability()
    print(polobj)

    mf.verbose = 0
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    h1 = mf.get_hcore()
    def apply_E(E):
        mf.get_hcore = lambda *args, **kwargs: h1 + numpy.einsum('x,xij->ij', E, ao_dip)
        mf.run(conv_tol=1e-14)
        return mf.dip_moment(mol, mf.make_rdm1(), unit='AU', verbose=0)
    e1 = apply_E([ 0.0001, 0, 0])
    e2 = apply_E([-0.0001, 0, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([0, 0.0001, 0])
    e2 = apply_E([0,-0.0001, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([0, 0, 0.0001])
    e2 = apply_E([0, 0,-0.0001])
    print((e1 - e2) / 0.0002)

    print(hpol)
    def apply_E(E):
        mf.get_hcore = lambda *args, **kwargs: h1 + numpy.einsum('x,xij->ij', E, ao_dip)
        mf.run(conv_tol=1e-14)
        return Polarizability(mf).polarizability()
    e1 = apply_E([ 0.0001, 0, 0])
    e2 = apply_E([-0.0001, 0, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([0, 0.0001, 0])
    e2 = apply_E([0,-0.0001, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([0, 0, 0.0001])
    e2 = apply_E([0, 0,-0.0001])
    print((e1 - e2) / 0.0002)

    mol = gto.M(atom='''O      0.   0.       0.
                        H      0.  -0.757    0.587
                        H      0.   0.757    0.587''',
                basis='6-31g')
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    print(Polarizability(mf).polarizability())
    print(Polarizability(mf).polarizability_with_freq(freq= 0.))

    print(Polarizability(mf).polarizability_with_freq(freq= 0.1))
    print(Polarizability(mf).polarizability_with_freq(freq=-0.1))

