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
from pyscf.x2c.sfx2c1e import SFX2C1E_SCF
from pyscf.prop.cphf import CPHFBase


def el_dip_moment(polobj, **kwargs):
    '''Electronic dipole moment (with picture change correction if in SFX2C).
    
    Kwargs:
        picture_change : bool
            Whether to include the picture change correction in SFX2C.'''
    log = logger.new_logger(polobj)
    mf = polobj.mf
    h1 = polobj.get_h1(**kwargs)
    dm = mf.make_rdm1()

    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        # U-HF/KS density matrices
        dm = dm[0] + dm[1]
    
    mu = -lib.einsum('xpq,qp->x', h1, dm)

    if mf.verbose >= logger.INFO:
        log.note('Electronic dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mu)

    return mu

def polarizability(polobj, freq=(0,0), **kwargs):
    '''Polarizability (with picture change correction if in SFX2C).
    
    Kwargs:
        picture_change : bool
            Whether to include the picture change correction in SFX2C.
            Default is True.
        solver : str
            The solver to use for the CP-HF/KS equations. Only
            'direct': Direct method to solve the linear equations;
            'newton': Newton iterative method with the inverse projected into
                      the Krylov subspace; and
            'krylov': Krylov subspace method to project the solution into the
                      Krylov subspace;
            are supported. Default is 'krylov'.
    '''
    assert isinstance(freq, tuple) and len(freq) == 2
    assert abs(freq[0]) == abs(freq[1]) or 0 in freq, '''
    For a given frequency w, only 9 choices of (a,b) are allowed,
    where a, b are in {0, w, -w}.'''
    log = logger.new_logger(polobj)
    if polobj.with_s1:
        h1 = polobj._to_to(polobj.get_h1(**kwargs))
    else:
        h1 = polobj._to_vo(polobj.get_h1(**kwargs))
    try : mo12 = polobj.mo1[freq[1]]
    except KeyError: mo12 = polobj.solve_mo1(freq[1], **kwargs)

    if freq[1] == 0:
        # a(w1,w2=0)
        alpha = -lib.einsum('xji,yji->xy', h1.conj(), mo12)
        alpha += alpha.conj()

    else:
        # a(w1,w2)
        h1 = numpy.stack((h1.conj(), h1))
        alpha = -lib.einsum('sxji,syji->xy', h1, mo12)

        try: mo11 = polobj.mo1[freq[0]]
        except KeyError: mo11 = polobj.solve_mo1(freq[0], **kwargs)
        if freq[0] == 0:
            mo11 = numpy.stack((mo11.conj(), -mo11))
        else:
            mo11 = mo11[[1,0]]; mo11[1] = -mo11[1]
        alpha += lib.einsum('sxji,syji->xy', mo11, mo12) * freq[1]

    if polobj.verbose >= logger.INFO:
        xx, yy, zz = alpha.diagonal()
        log.note('Isotropic polarizability %.12g', (xx+yy+zz)/3)
        log.note('Polarizability anisotropy %.12g',
                 (.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2))**.5)
        log.debug(f'Polarizability tensor a{freq}')
        log.debug(f'{alpha}')

    return alpha*2 if isinstance(polobj.mf, hf.RHF) else alpha.real

def hyperpolarizability(polobj, freq=(0,0,0), **kwargs):
    '''(First) Hyperpolarizability (with picture change correction if in SFX2C).
    
    Kwargs:
        picture_change : bool
            Whether to include the picture change correction in SFX2C.
            Default is True.
        solver : str
            The solver to use for the CP-HF/KS equations. Only
            'direct': Direct method to solve the linear equations;
            'newton': Newton iterative method with the inverse projected into
                      the Krylov subspace; and
            'krylov': Krylov subspace method to project the solution into the
                      Krylov subspace;
            are supported. Default is 'krylov'.
    '''
    assert isinstance(freq, tuple) and len(freq) == 3
    assert len({abs(f) for f in freq if f != 0}) in (0, 1), '''
    For a given frequency w, only 27 choices of (a,b,c) are allowed,
    where a, b, c are in {0, w, -w}.'''
    log = logger.new_logger(polobj)
    mf = polobj.mf

    if freq == 0:
        mo1 = polobj.get_mo1(freq, picture_change, solver)
        f1 = polobj.get_f1vv(mo1, picture_change)
        e1 = polobj.get_e1oo(mo1, picture_change)

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
            dm1 = polobj.get_dm1(mo1=mo1)

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
        f1_m2o = polobj.get_f1vv(mo1_2o, picture_change).transpose(0,2,1).conj()
        e1_m2o = polobj.get_e1oo(mo1_2o, picture_change).transpose(0,2,1).conj()

        mo1_1o = polobj.get_mo1(freq, picture_change, solver)
        f1_p1o = polobj.get_f1vv(mo1_1o, picture_change)
        e1_p1o = polobj.get_e1oo(mo1_1o, picture_change)

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
            dm1_m2o = polobj.get_dm1(mo1=mo1_2o).transpose(0,2,1).conj()
            dm1_p1o = polobj.get_dm1(mo1=mo1_1o)

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
        f1_p1o = polobj.get_f1vv(mo1_1o, picture_change)
        f1_m1o = f1_p1o.transpose(0,2,1).conj()
        e1_p1o = polobj.get_e1oo(mo1_1o, picture_change)
        e1_m1o = e1_p1o.transpose(0,2,1).conj()

        mo1 = polobj.get_mo1(0, picture_change, solver)
        f1 = polobj.get_f1vv(mo1, picture_change)
        e1 = polobj.get_e1oo(mo1, picture_change)

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
            dm1 = polobj.get_dm1(mo1=mo1)
            dm1_p1o = polobj.get_dm1(mo1=mo1_1o)
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

class Polarizability(CPHFBase):
    def get_h1(polobj, picture_change=True, **kwargs):
        '''The dipole matrix in AO basis.'''
        mf = polobj.mf
        mol = mf.mol
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        charge_center = lib.einsum('i,ix->x', charges, coords) / charges.sum()
        with mol.with_common_orig(charge_center):
            if isinstance(mf, SFX2C1E_SCF) and picture_change:
                xmol = mf.with_x2c.get_xmol()[0]
                nao = xmol.nao
                prp = xmol.intor_symmetric('int1e_sprsp').reshape(3,4,nao,nao)[:,3]
                c1 = 0.5/lib.param.LIGHT_SPEED
                ao_dip = mf.with_x2c.picture_change(('int1e_r', prp*c1**2))
            else:
                ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        return -ao_dip

    dipole = el_dip_moment = el_dip_moment
    polar = polarizability = polarizability
    hyperpolar = hyperpolarizability = hyperpolarizability

hf.RHF.Polarizability = SFX2C1E_SCF.Polarizability = lib.class_as_method(Polarizability)


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
        mf.get_hcore = lambda *args, **kwargss: h1 + numpy.einsum('x,xij->ij', E, ao_dip)
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
        mf.get_hcore = lambda *args, **kwargss: h1 + numpy.einsum('x,xij->ij', E, ao_dip)
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

