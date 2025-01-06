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


def el_dip_moment(pl:'Polarizability', **kwargs):
    '''Electronic dipole moment (with picture change correction if in SFX2C).
    
    Kwargs:
        picture_change : bool
            Whether to include the picture change correction in SFX2C.'''
    log = logger.new_logger(pl)
    mf = pl.mf
    h1 = pl.get_h1(**kwargs)
    dm = mf.make_rdm1()

    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        # U-HF/KS density matrices
        dm = dm[0] + dm[1]
    
    mu = -lib.einsum('xpq,qp->x', h1, dm)

    if mf.verbose >= logger.INFO:
        log.note('Electronic dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mu)

    return mu

def polarizability(pl:'Polarizability', freq=(0,0), **kwargs):
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
    log = logger.new_logger(pl)
    if pl.with_s1:
        h1 = pl._to_to(pl.get_h1(**kwargs))
    else:
        h1 = pl._to_vo(pl.get_h1(**kwargs))

    if 0 in freq: # a(0,0) / a(0,w) / a(w,0)
        try: mo1 = pl.mo1[0]
        except KeyError: mo1 = pl.solve_mo1(0, **kwargs)
    
        alpha = -lib.einsum('xji,yji->xy', h1.conj(), mo1) * .5
        alpha += alpha.conj()
        alpha += alpha.T

    elif abs(freq[0]) == abs(freq[1]): # a(w,w) / a(-w,w) / a(w,-w)
        try: mo1 = pl.mo1[freq[1]]
        except KeyError: mo1 = pl.solve_mo1(freq[1], **kwargs)

        alpha = -lib.einsum('sxji,syji->xy', (h1.conj(),h1), mo1) * .5
        if freq[0] == freq[1]:
            alpha += alpha.T
        else:
            alpha += alpha.T.conj()
            alpha += lib.einsum('sxji,syji,s->xy', mo1.conj(), mo1, freq)
    
    else: # a(w1,w2)
        try: mo10 = pl.mo1[freq[0]]
        except KeyError: mo10 = pl.solve_mo1(freq[0], **kwargs)
        try: mo11 = pl.mo1[freq[1]]
        except KeyError: mo11 = pl.solve_mo1(freq[1], **kwargs)

        h1 = (h1.conj(), h1)
        alpha = -lib.einsum('syji,sxji->xy', h1, mo10)
        alpha -= lib.einsum('sxji,syji->xy', h1, mo11)
        alpha += lib.einsum('xji,yji->xy', mo10[0], mo11[1]) * (freq[0]-freq[1])
        alpha += lib.einsum('xji,yji->xy', mo10[1], mo11[0]) * (freq[1]-freq[0])

    if pl.verbose >= logger.INFO:
        xx, yy, zz = alpha.diagonal()
        log.note('Isotropic polarizability %.12g', (xx+yy+zz)/3)
        log.note('Polarizability anisotropy %.12g',
                 (.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2))**.5)
        log.debug(f'Polarizability tensor a{freq}')
        log.debug(f'{alpha}')

    return alpha*2 if isinstance(pl.mf, hf.RHF) else alpha.real

def hyperpolarizability(pl:'Polarizability', freq=(0,0,0), **kwargs):
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
    #assert len({abs(f) for f in freq if f != 0}) in (0, 1), '''
    #For a given frequency w, only 27 freq input choices of (a,b,c) are allowed,
    #where a, b, c are in {0, w, -w}.'''
    log = logger.new_logger(pl)
    mf = pl.mf

    if freq[0] == freq[1] == freq[2]:
        try: mo1 = pl.mo1[freq[0]]
        except KeyError: mo1 = pl.solve_mo1(freq[0], **kwargs)
        f1vv = pl._to_vv(pl.get_f1(mo1, freq[0], **kwargs))
        f1oo = pl._to_oo(pl.get_f1(mo1, freq[0], **kwargs))
        
        # b(0,0,0) / b(w,w,w)
        if freq[0] == 0:
            beta = -lib.einsum('xjl,yli,zji->xyz', f1vv, mo1, mo1.conj())
            beta += lib.einsum('xik,yji,zjk->xyz', f1oo, mo1, mo1.conj())
        else:
            beta = -lib.einsum('xjl,yli,zji->xyz', f1vv, mo1[0], mo1[1])
            beta += lib.einsum('xik,yji,zjk->xyz', f1oo, mo1[0], mo1[1])

            try: mo2 = pl.mo2[freq[:-1]]
            except KeyError: mo2 = pl.solve_mo2(freq[:-1], **kwargs)
            mo2 = mo2[...,mf.mo_occ==0,:]; mo2[0] *= -1
            beta += lib.einsum('syzji,sxji->xyz', mo2, mo1[[1,0]])*freq[0]*.5
        
        beta += beta.transpose(0,2,1) + beta.transpose(1,0,2) + \
                beta.transpose(1,2,0) + beta.transpose(2,0,1) + beta.transpose(2,1,0)
        beta *= 2
        
        if isinstance(mf, hf.KohnShamDFT):
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm = mf.make_rdm1()
            dm1 = pl.get_dm1(mo1, freq[0])

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

    else: # for testing purpose
        try: mo10 = pl.mo1[freq[0]]
        except KeyError: mo10 = pl.solve_mo1(freq[0], **kwargs)
        try: mo11 = pl.mo1[freq[1]]
        except KeyError: mo11 = pl.solve_mo1(freq[1], **kwargs)
        try: mo12 = pl.mo1[freq[2]]
        except KeyError: mo12 = pl.solve_mo1(freq[2], **kwargs)
        f1vv0 = pl._to_vv(pl.get_f1(mo10, freq[0], **kwargs))
        f1oo0 = pl._to_oo(pl.get_f1(mo10, freq[0], **kwargs))
        f1vv1 = pl._to_vv(pl.get_f1(mo11, freq[1], **kwargs))
        f1oo1 = pl._to_oo(pl.get_f1(mo11, freq[1], **kwargs))
        f1vv2 = pl._to_vv(pl.get_f1(mo12, freq[2], **kwargs))
        f1oo2 = pl._to_oo(pl.get_f1(mo12, freq[2], **kwargs))

        if freq[0] == 0: mo10 = (mo10, mo10.conj())
        if freq[1] == 0: mo11 = (mo11, mo11.conj())
        if freq[2] == 0: mo12 = (mo12, mo12.conj())
        # b(w1,w2,w3)
        beta = -lib.einsum('xjl,yli,zji->xyz', f1vv0, mo11[0], mo12[1])
        beta -= lib.einsum('xjl,zli,yji->xyz', f1vv0, mo12[0], mo11[1])
        beta -= lib.einsum('yjl,xli,zji->xyz', f1vv1, mo10[0], mo12[1])
        beta -= lib.einsum('yjl,zli,xji->xyz', f1vv1, mo12[0], mo10[1])
        beta -= lib.einsum('zjl,xli,yji->xyz', f1vv2, mo10[0], mo11[1])
        beta -= lib.einsum('zjl,yli,xji->xyz', f1vv2, mo11[0], mo10[1])
        beta += lib.einsum('xik,yji,zjk->xyz', f1oo0, mo11[0], mo12[1])
        beta += lib.einsum('xik,zji,yjk->xyz', f1oo0, mo12[0], mo11[1])
        beta += lib.einsum('yik,xji,zjk->xyz', f1oo1, mo10[0], mo12[1])
        beta += lib.einsum('yik,zji,xjk->xyz', f1oo1, mo12[0], mo10[1])
        beta += lib.einsum('zik,xji,yjk->xyz', f1oo2, mo10[0], mo11[1])
        beta += lib.einsum('zik,yji,xjk->xyz', f1oo2, mo11[0], mo10[1])
        
        if freq[0] == 0:
            mo10 = mo10[0]
        else:
            try: mo2 = pl.mo2[freq[1:]]
            except KeyError: mo2 = pl.solve_mo2(freq[1:], **kwargs)
            if freq[1:] == (0,0): mo2 = numpy.stack((mo2, mo2.conj()))
            mo2 = mo2[...,mf.mo_occ==0,:]; mo2[0] *= -1
            beta += lib.einsum('syzji,sxji->xyz', mo2, mo10[[1,0]]) * freq[0]
        if freq[1] == 0:
            mo11 = mo11[0]
        else:
            try: mo2 = pl.mo2[(freq[0],freq[2])]
            except KeyError: mo2 = pl.solve_mo2((freq[0],freq[2]), **kwargs)
            if freq[0] == freq[2] == 0: mo2 = numpy.stack((mo2, mo2.conj()))
            mo2 = mo2[...,mf.mo_occ==0,:]; mo2[0] *= -1
            beta += lib.einsum('sxzji,syji->xyz', mo2, mo11[[1,0]]) * freq[1]
        if freq[2] == 0:
            mo12 = mo12[0]
        else:
            try: mo2 = pl.mo2[freq[:-1]]
            except KeyError: mo2 = pl.solve_mo2(freq[:-1], **kwargs)
            if freq[:-1] == (0,0): mo2 = numpy.stack((mo2, mo2.conj()))
            mo2 = mo2[...,mf.mo_occ==0,:]; mo2[0] *= -1
            beta += lib.einsum('sxyji,szji->xyz', mo2, mo12[[1,0]]) * freq[2]
        beta *= 2
        
        if isinstance(mf, hf.KohnShamDFT):
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm = mf.make_rdm1()
            dm10 = pl.get_dm1(mo10, freq[0])
            dm11 = pl.get_dm1(mo11, freq[1])
            dm12 = pl.get_dm1(mo12, freq[2])

            if xctype == 'LDA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype)
                rho10 = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                     for dmx in dm10]).reshape(3,1,-1) # reshape to match kxc
                rho11 = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                     for dmx in dm11]).reshape(3,1,-1) # reshape to match kxc
                rho12 = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                     for dmx in dm12]).reshape(3,1,-1) # reshape to match kxc
                
            elif xctype == 'GGA' or 'MGGA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype, with_lapl=False)
                rho10 = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                     with_lapl=False) for dmx in dm10])
                rho11 = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                     with_lapl=False) for dmx in dm11])
                rho12 = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                     with_lapl=False) for dmx in dm12])
                
            else:
                raise RuntimeError(f"Unknown xctype '{xctype}'")
            
            kxc = ni.eval_xc_eff(mf.xc, rho, deriv=3)[3] * mf.grids.weights
            beta -= lib.einsum('xig,yjg,zkg,ijkg->xyz', rho10, rho11, rho12, kxc)

    if mf.verbose >= logger.INFO:
        log.debug(f'Hyperpolarizability tensor b{freq}')
        log.debug(f'{beta}')
    
    return beta

class Polarizability(CPHFBase):
    def get_h1(pl, picture_change=True, **kwargs):
        '''The dipole matrix in AO basis.'''
        mf = pl.mf
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
    pl = mf.Polarizability().polarizability()
    hpol = mf.Polarizability().hyperpolarizability()
    print(pl)

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

