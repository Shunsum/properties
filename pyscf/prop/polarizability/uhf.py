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
from pyscf.scf import hf, uhf, _response_functions
from pyscf.prop.cphf import UCPHFBase
from .rhf import RHFPolar
# Note: polarizability and relevant properties are demanding on basis sets.
# ORCA recommends to use Sadlej basis for these properties.
def polarizability(pl:'UHFPolar', freq=(0,0), **kwargs):
    '''The second energy response tensor (with picture change correction if in SFX2C).
    
    Kwargs:
        freq : tuple
            The frequency tuple (w1, w2) in a.u. Default is (0, 0).
        picture_change : bool
            Whether to include the picture change correction in SFX2C.
            Default is True.
        solver : str
            The solver to use for the CP-HF/KS equations. Only
            `direct`: Direct method to solve the linear equations;
            `newton`: Newton iterative method with the inverse projected into
                      the Krylov subspace; and
            `krylov`: Krylov subspace method to project the solution into the
                      Krylov subspace;
            are supported. Default is `krylov`.
    '''
    assert isinstance(freq, tuple) and len(freq) == 2
    log = logger.new_logger(pl)
    if pl.with_s1:
        h1a, h1b = pl._to_to(pl.get_h1(**kwargs))
    else:
        h1a, h1b = pl._to_vo(pl.get_h1(**kwargs))

    if 0 in freq: # a(0,0) / a(0,w) / a(w,0)
        try: mo1a, mo1b = pl.mo1[0]
        except KeyError: mo1a, mo1b = pl.solve_mo1(0, **kwargs)
        
        alpha = -lib.einsum('xji,yji->xy', h1a.conj(), mo1a) * .5
        alpha -= lib.einsum('xji,yji->xy', h1b.conj(), mo1b) * .5
        alpha += alpha.conj()
        alpha += alpha.T

    elif abs(freq[0]) == abs(freq[1]): # a(w,w) / a(-w,w) / a(w,-w)
        try: mo1a, mo1b = pl.mo1[freq[1]]
        except KeyError: mo1a, mo1b = pl.solve_mo1(freq[1], **kwargs)

        alpha = -lib.einsum('sxji,syji->xy', (h1a.conj(),h1a), mo1a) * .5
        alpha -= lib.einsum('sxji,syji->xy', (h1b.conj(),h1b), mo1b) * .5
        if freq[0] == freq[1]:
            alpha += alpha.T
        else:
            alpha += alpha.T.conj()
            alpha += lib.einsum('sxji,syji,s->xy', mo1a.conj(), mo1a, freq)
            alpha += lib.einsum('sxji,syji,s->xy', mo1b.conj(), mo1b, freq)
    
    else: # a(w1,w2)
        try: mo10a, mo10b = pl.mo1[freq[0]]
        except KeyError: mo10a, mo10b = pl.solve_mo1(freq[0], **kwargs)
        try: mo11a, mo11b = pl.mo1[freq[1]]
        except KeyError: mo11a, mo11b = pl.solve_mo1(freq[1], **kwargs)

        h1a = (h1a.conj(), h1a)
        h1b = (h1b.conj(), h1b)
        alpha = -lib.einsum('syji,sxji->xy', h1a, mo10a)
        alpha -= lib.einsum('syji,sxji->xy', h1b, mo10b)
        alpha -= lib.einsum('sxji,syji->xy', h1a, mo11a)
        alpha -= lib.einsum('sxji,syji->xy', h1b, mo11b)
        alpha += lib.einsum('xji,yji->xy', mo10a[0], mo11a[1]) * (freq[0]-freq[1])
        alpha += lib.einsum('xji,yji->xy', mo10b[0], mo11b[1]) * (freq[0]-freq[1])
        alpha += lib.einsum('xji,yji->xy', mo10a[1], mo11a[0]) * (freq[1]-freq[0])
        alpha += lib.einsum('xji,yji->xy', mo10b[1], mo11b[0]) * (freq[1]-freq[0])

    if pl.verbose >= logger.INFO:
        xx, yy, zz = alpha.diagonal()
        log.note('Isotropy %.12g', (xx+yy+zz)/3)
        log.note('Anisotropy %.12g', (((xx-yy)**2 + (yy-zz)**2
                                       + (zz-xx)**2)*.5)**.5)
        log.debug(f'The second energy response tensor a{freq}')
        log.debug(f'{alpha}')

    return alpha

def hyperpolarizability(pl:'UHFPolar', freq=(0,0,0), **kwargs):
    '''The third energy response tensor (with picture change correction if in SFX2C).

    Kwargs:
        freq : tuple
            The frequency tuple (w1, w2, w3) in a.u. Default is (0, 0, 0).
        picture_change : bool
            Whether to include the picture change correction in SFX2C.
            Default is True.
        solver : str
            The solver to use for the CP-HF/KS equations. Only
            `direct`: Direct method to solve the linear equations;
            `newton`: Newton iterative method with the inverse projected into
                      the Krylov subspace; and
            `krylov`: Krylov subspace method to project the solution into the
                      Krylov subspace;
            are supported. Default is `krylov`.
    '''
    assert isinstance(freq, tuple) and len(freq) == 3
    log = logger.new_logger(pl)
    mf = pl.mf

    if freq[0] == freq[1] == freq[2]: # b(0,0,0) / b(w,w,w)
        try: mo1a, mo1b = pl.mo1[freq[0]]
        except KeyError: mo1a, mo1b = pl.solve_mo1(freq[0], **kwargs)
        f1 = pl.get_f1((mo1a, mo1b), freq[0], **kwargs)
        f1vva, f1vvb = pl._to_vv(f1)
        f1ooa, f1oob = pl._to_oo(f1)
        
        if freq[0] == 0:
            beta = -lib.einsum('xjl,yli,zji->xyz', f1vva, mo1a, mo1a.conj())
            beta -= lib.einsum('xjl,yli,zji->xyz', f1vvb, mo1b, mo1b.conj())
            beta += lib.einsum('xik,yji,zjk->xyz', f1ooa, mo1a, mo1a.conj())
            beta += lib.einsum('xik,yji,zjk->xyz', f1oob, mo1b, mo1b.conj())
        else:
            beta = -lib.einsum('xjl,yli,zji->xyz', f1vva, mo1a[0], mo1a[1])
            beta -= lib.einsum('xjl,yli,zji->xyz', f1vvb, mo1b[0], mo1b[1])
            beta += lib.einsum('xik,yji,zjk->xyz', f1ooa, mo1a[0], mo1a[1])
            beta += lib.einsum('xik,yji,zjk->xyz', f1oob, mo1b[0], mo1b[1])

            try: mo2a, mo2b = pl.mo2[freq[1:]]
            except KeyError: mo2a, mo2b = pl.solve_mo2(freq[1:], **kwargs)
            mo2a = mo2a[...,mf.mo_occ[0]==0,:]
            mo2b = mo2b[...,mf.mo_occ[1]==0,:]
            mo2a = mo2a.reshape(2,3,3,*mo2a.shape[-2:])
            mo2b = mo2b.reshape(2,3,3,*mo2b.shape[-2:])
            mo2a[0] *= -1
            mo2b[0] *= -1
            beta += lib.einsum('syzji,sxji->xyz', mo2a, mo1a[[1,0]]) * freq[0] * .5
            beta += lib.einsum('syzji,sxji->xyz', mo2b, mo1b[[1,0]]) * freq[0] * .5
        
        beta += beta.transpose(0,2,1) + beta.transpose(1,0,2) + \
                beta.transpose(1,2,0) + beta.transpose(2,0,1) + beta.transpose(2,1,0)
        
        if isinstance(mf, hf.KohnShamDFT):
            mol = mf.mol
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm0 = mf.make_rdm1()
            dm1 = pl.get_dm1((mo1a, mo1b), freq[0])
            nao = dm0.shape[-1]
            cur_mem = lib.current_memory()[0]
            max_mem = max(2000, mf.max_memory*.8 - cur_mem)
            
            if xctype == 'LDA':
                for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, 0, max_mem):
                    rho0 = numpy.array([ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=1) for dm in dm0])
                    rho1 = numpy.array([[ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=freq[0]==0) for dm in dms] for dms in dm1])
                    kxc = ni.eval_xc_eff(mf.xc, rho0, deriv=3, xctype=xctype)[3]
                    kxc = kxc[:,0,:,0,:,0] * weight
                    beta -= lib.einsum('axg,byg,czg,abcg->xyz', rho1, rho1, rho1, kxc)

            elif xctype in ('GGA', 'MGGA'):
                for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, 1, max_mem):
                    rho0 = numpy.array([ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=1, with_lapl=False) for dm in dm0])
                    rho1 = numpy.array([[ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=freq[0]==0, with_lapl=False)
                                        for dm in dms] for dms in dm1])
                    kxc = ni.eval_xc_eff(mf.xc, rho0, deriv=3, xctype=xctype)[3] * weight
                    beta -= lib.einsum('axig,byjg,czkg,aibjckg->xyz',
                                       rho1, rho1, rho1, kxc)

            elif xctype == 'HF':
                pass

            else:
                raise NotImplementedError(xctype)

    elif len(set(freq)) == 2: # b(we,we,wi) -> b(we,wi,we) / b(wi,we,we)
        we = next(f for f in set(freq) if freq.count(f) == 2)
        wi = next(f for f in set(freq) if f != we)
        try: mo1ea, mo1eb = pl.mo1[we]
        except KeyError: mo1ea, mo1eb = pl.solve_mo1(we, **kwargs)
        try: mo1ia, mo1ib = pl.mo1[wi]
        except KeyError: mo1ia, mo1ib = pl.solve_mo1(wi, **kwargs)
        f1e = pl.get_f1((mo1ea, mo1eb), we, **kwargs)
        f1i = pl.get_f1((mo1ia, mo1ib), wi, **kwargs)
        f1vvea, f1vveb = pl._to_vv(f1e)
        f1ooea, f1ooeb = pl._to_oo(f1e)
        f1vvia, f1vvib = pl._to_vv(f1i)
        f1ooia, f1ooib = pl._to_oo(f1i)

        if we == 0:
            mo1ea = (mo1ea, mo1ea.conj())
            mo1eb = (mo1eb, mo1eb.conj())
        if wi == 0:
            mo1ia = (mo1ia, mo1ia.conj())
            mo1ib = (mo1ib, mo1ib.conj())
        
        beta = -lib.einsum('xjl,yli,zji->xyz', f1vvea, mo1ea[0], mo1ia[1])
        beta -= lib.einsum('xjl,yli,zji->xyz', f1vveb, mo1eb[0], mo1ib[1])
        beta -= lib.einsum('yjl,zli,xji->xyz', f1vvea, mo1ia[0], mo1ea[1])
        beta -= lib.einsum('yjl,zli,xji->xyz', f1vveb, mo1ib[0], mo1eb[1])
        beta -= lib.einsum('zjl,xli,yji->xyz', f1vvia, mo1ea[0], mo1ea[1])
        beta -= lib.einsum('zjl,xli,yji->xyz', f1vvib, mo1eb[0], mo1eb[1])
        beta += lib.einsum('xik,yji,zjk->xyz', f1ooea, mo1ea[0], mo1ia[1])
        beta += lib.einsum('xik,yji,zjk->xyz', f1ooeb, mo1eb[0], mo1ib[1])
        beta += lib.einsum('yik,zji,xjk->xyz', f1ooea, mo1ia[0], mo1ea[1])
        beta += lib.einsum('yik,zji,xjk->xyz', f1ooeb, mo1ib[0], mo1eb[1])
        beta += lib.einsum('zik,xji,yjk->xyz', f1ooia, mo1ea[0], mo1ea[1])
        beta += lib.einsum('zik,xji,yjk->xyz', f1ooib, mo1eb[0], mo1eb[1])

        if we == 0:
            mo1ea = mo1ea[0]
            mo1eb = mo1eb[0]
        else:
            try: mo2a, mo2b = pl.mo2[(we,wi)]
            except KeyError: mo2a, mo2b = pl.solve_mo2((we,wi), **kwargs)
            mo2a = mo2a[...,mf.mo_occ[0]==0,:]
            mo2b = mo2b[...,mf.mo_occ[1]==0,:]
            mo2a = mo2a.reshape(2,3,3,*mo2a.shape[-2:])
            mo2b = mo2b.reshape(2,3,3,*mo2b.shape[-2:])
            mo2a[0] *= -1
            mo2b[0] *= -1
            beta += lib.einsum('syzji,sxji->xyz', mo2a, mo1ea[[1,0]]) * we
            beta += lib.einsum('syzji,sxji->xyz', mo2b, mo1eb[[1,0]]) * we
        if wi == 0:
            mo1ia = mo1ia[0]
            mo1ib = mo1ib[0]
        else:
            try: mo2a, mo2b = pl.mo2[(we,we)]
            except KeyError: mo2a, mo2b = pl.solve_mo2((we,we), **kwargs)
            mo2a = mo2a[...,mf.mo_occ[0]==0,:]
            mo2b = mo2b[...,mf.mo_occ[1]==0,:]
            if we == 0:
                mo2a = numpy.array((mo2a, mo2a.conj()))
                mo2b = numpy.array((mo2b, mo2b.conj()))
            mo2a = mo2a.reshape(2,3,3,*mo2a.shape[-2:])
            mo2b = mo2b.reshape(2,3,3,*mo2b.shape[-2:])
            mo2a[0] *= -1
            mo2b[0] *= -1
            beta += lib.einsum('sxyji,szji->xyz', mo2a, mo1ia[[1,0]]) * wi * .5
            beta += lib.einsum('sxyji,szji->xyz', mo2b, mo1ib[[1,0]]) * wi * .5

        beta += beta.transpose(1,0,2)

        if isinstance(mf, hf.KohnShamDFT):
            mol = mf.mol
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm0 = mf.make_rdm1()
            dm1e = pl.get_dm1((mo1ea, mo1eb), we)
            dm1i = pl.get_dm1((mo1ia, mo1ib), wi)
            nao = dm0.shape[-1]
            cur_mem = lib.current_memory()[0]
            max_mem = max(2000, mf.max_memory*.8 - cur_mem)

            if xctype == 'LDA':
                for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, 0, max_mem):
                    rho0 = numpy.array([ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=1) for dm in dm0])
                    rho1e = numpy.array([[ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=we==0) for dm in dms] for dms in dm1e])
                    rho1i = numpy.array([[ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=wi==0) for dm in dms] for dms in dm1i])
                    kxc = ni.eval_xc_eff(mf.xc, rho0, deriv=3, xctype=xctype)[3]
                    kxc = kxc[:,0,:,0,:,0] * weight
                    beta -= lib.einsum('axg,byg,czg,abcg->xyz', rho1e, rho1e, rho1i, kxc)
            
            elif xctype in ('GGA', 'MGGA'):
                for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, 1, max_mem):
                    rho0 = numpy.array([ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=1, with_lapl=False) for dm in dm0])
                    rho1e = numpy.array([[ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=we==0, with_lapl=False)
                                        for dm in dms] for dms in dm1e])
                    rho1i = numpy.array([[ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=wi==0, with_lapl=False)
                                        for dm in dms] for dms in dm1i])
                    kxc = ni.eval_xc_eff(mf.xc, rho0, deriv=3, xctype=xctype)[3] * weight
                    beta -= lib.einsum('axig,byjg,czkg,aibjckg->xyz',
                                       rho1e, rho1e, rho1i, kxc)
            
            elif xctype == 'HF':
                pass

            else:
                raise NotImplementedError(xctype)

        if   freq.index(wi) == 1: beta = beta.transpose(0,2,1)
        elif freq.index(wi) == 0: beta = beta.transpose(2,1,0)

    else: # b(w1,w2,w3)
        try: mo10a, mo10b = pl.mo1[freq[0]]
        except KeyError: mo10a, mo10b = pl.solve_mo1(freq[0], **kwargs)
        try: mo11a, mo11b = pl.mo1[freq[1]]
        except KeyError: mo11a, mo11b = pl.solve_mo1(freq[1], **kwargs)
        try: mo12a, mo12b = pl.mo1[freq[2]]
        except KeyError: mo12a, mo12b = pl.solve_mo1(freq[2], **kwargs)
        f10 = pl.get_f1((mo10a ,mo10b), freq[0], **kwargs)
        f11 = pl.get_f1((mo11a, mo11b), freq[1], **kwargs)
        f12 = pl.get_f1((mo12a, mo12b), freq[2], **kwargs)
        f1vv0a, f1vv0b = pl._to_vv(f10)
        f1oo0a, f1oo0b = pl._to_oo(f10)
        f1vv1a, f1vv1b = pl._to_vv(f11)
        f1oo1a, f1oo1b = pl._to_oo(f11)
        f1vv2a, f1vv2b = pl._to_vv(f12)
        f1oo2a, f1oo2b = pl._to_oo(f12)

        if freq[0] == 0:
            mo10a = (mo10a, mo10a.conj())
            mo10b = (mo10b, mo10b.conj())
        if freq[1] == 0:
            mo11a = (mo11a, mo11a.conj())
            mo11b = (mo11b, mo11b.conj())
        if freq[2] == 0:
            mo12a = (mo12a, mo12a.conj())
            mo12b = (mo12b, mo12b.conj())
        
        beta = -lib.einsum('xjl,yli,zji->xyz', f1vv0a, mo11a[0], mo12a[1])
        beta -= lib.einsum('xjl,yli,zji->xyz', f1vv0b, mo11b[0], mo12b[1])
        beta -= lib.einsum('xjl,zli,yji->xyz', f1vv0a, mo12a[0], mo11a[1])
        beta -= lib.einsum('xjl,zli,yji->xyz', f1vv0b, mo12b[0], mo11b[1])
        beta -= lib.einsum('yjl,xli,zji->xyz', f1vv1a, mo10a[0], mo12a[1])
        beta -= lib.einsum('yjl,xli,zji->xyz', f1vv1b, mo10b[0], mo12b[1])
        beta -= lib.einsum('yjl,zli,xji->xyz', f1vv1a, mo12a[0], mo10a[1])
        beta -= lib.einsum('yjl,zli,xji->xyz', f1vv1b, mo12b[0], mo10b[1])
        beta -= lib.einsum('zjl,xli,yji->xyz', f1vv2a, mo10a[0], mo11a[1])
        beta -= lib.einsum('zjl,xli,yji->xyz', f1vv2b, mo10b[0], mo11b[1])
        beta -= lib.einsum('zjl,yli,xji->xyz', f1vv2a, mo11a[0], mo10a[1])
        beta -= lib.einsum('zjl,yli,xji->xyz', f1vv2b, mo11b[0], mo10b[1])
        beta += lib.einsum('xik,yji,zjk->xyz', f1oo0a, mo11a[0], mo12a[1])
        beta += lib.einsum('xik,yji,zjk->xyz', f1oo0b, mo11b[0], mo12b[1])
        beta += lib.einsum('xik,zji,yjk->xyz', f1oo0a, mo12a[0], mo11a[1])
        beta += lib.einsum('xik,zji,yjk->xyz', f1oo0b, mo12b[0], mo11b[1])
        beta += lib.einsum('yik,xji,zjk->xyz', f1oo1a, mo10a[0], mo12a[1])
        beta += lib.einsum('yik,xji,zjk->xyz', f1oo1b, mo10b[0], mo12b[1])
        beta += lib.einsum('yik,zji,xjk->xyz', f1oo1a, mo12a[0], mo10a[1])
        beta += lib.einsum('yik,zji,xjk->xyz', f1oo1b, mo12b[0], mo10b[1])
        beta += lib.einsum('zik,xji,yjk->xyz', f1oo2a, mo10a[0], mo11a[1])
        beta += lib.einsum('zik,xji,yjk->xyz', f1oo2b, mo10b[0], mo11b[1])
        beta += lib.einsum('zik,yji,xjk->xyz', f1oo2a, mo11a[0], mo10a[1])
        beta += lib.einsum('zik,yji,xjk->xyz', f1oo2b, mo11b[0], mo10b[1])
        
        if freq[0] == 0:
            mo10a = mo10a[0]
            mo10b = mo10b[0]
        else:
            try: mo2a, mo2b = pl.mo2[freq[1:]]
            except KeyError: mo2a, mo2b = pl.solve_mo2(freq[1:], **kwargs)
            mo2a = mo2a[...,mf.mo_occ[0]==0,:]
            mo2b = mo2b[...,mf.mo_occ[1]==0,:]
            mo2a = mo2a.reshape(2,3,3,*mo2a.shape[-2:])
            mo2b = mo2b.reshape(2,3,3,*mo2b.shape[-2:])
            mo2a[0] *= -1
            mo2b[0] *= -1
            beta += lib.einsum('syzji,sxji->xyz', mo2a, mo10a[[1,0]]) * freq[0]
            beta += lib.einsum('syzji,sxji->xyz', mo2b, mo10b[[1,0]]) * freq[0]
        if freq[1] == 0:
            mo11a = mo11a[0]
            mo11b = mo11b[0]
        else:
            try: mo2a, mo2b = pl.mo2[(freq[0],freq[2])]
            except KeyError: mo2a, mo2b = pl.solve_mo2((freq[0],freq[2]), **kwargs)
            mo2a = mo2a[...,mf.mo_occ[0]==0,:]
            mo2b = mo2b[...,mf.mo_occ[1]==0,:]
            mo2a = mo2a.reshape(2,3,3,*mo2a.shape[-2:])
            mo2b = mo2b.reshape(2,3,3,*mo2b.shape[-2:])
            mo2a[0] *= -1
            mo2b[0] *= -1
            beta += lib.einsum('sxzji,syji->xyz', mo2a, mo11a[[1,0]]) * freq[1]
            beta += lib.einsum('sxzji,syji->xyz', mo2b, mo11b[[1,0]]) * freq[1]
        if freq[2] == 0:
            mo12a = mo12a[0]
            mo12b = mo12b[0]
        else:
            try: mo2a, mo2b = pl.mo2[freq[:-1]]
            except KeyError: mo2a, mo2b = pl.solve_mo2(freq[:-1], **kwargs)
            mo2a = mo2a[...,mf.mo_occ[0]==0,:]
            mo2b = mo2b[...,mf.mo_occ[1]==0,:]
            mo2a = mo2a.reshape(2,3,3,*mo2a.shape[-2:])
            mo2b = mo2b.reshape(2,3,3,*mo2b.shape[-2:])
            mo2a[0] *= -1
            mo2b[0] *= -1
            beta += lib.einsum('sxyji,szji->xyz', mo2a, mo12a[[1,0]]) * freq[2]
            beta += lib.einsum('sxyji,szji->xyz', mo2b, mo12b[[1,0]]) * freq[2]
        
        if isinstance(mf, hf.KohnShamDFT):
            mol = mf.mol
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm0 = mf.make_rdm1()
            dm10 = pl.get_dm1((mo10a, mo10b), freq[0])
            dm11 = pl.get_dm1((mo11a, mo11b), freq[1])
            dm12 = pl.get_dm1((mo12a, mo12b), freq[2])
            nao = dm0.shape[-1]
            cur_mem = lib.current_memory()[0]
            max_mem = max(2000, mf.max_memory*.8 - cur_mem)

            if xctype == 'LDA':
                for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, 0, max_mem):
                    rho0 = numpy.array([ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=1) for dm in dm0])
                    rho10 = numpy.array([[ni.eval_rho(mol, ao, dm, mask, xctype,
                                         hermi=freq[0]==0)
                                         for dm in dms] for dms in dm10])
                    rho11 = numpy.array([[ni.eval_rho(mol, ao, dm, mask, xctype,
                                         hermi=freq[1]==0)
                                         for dm in dms] for dms in dm11])
                    rho12 = numpy.array([[ni.eval_rho(mol, ao, dm, mask, xctype,
                                         hermi=freq[2]==0)
                                         for dm in dms] for dms in dm12])
                    kxc = ni.eval_xc_eff(mf.xc, rho0, deriv=3, xctype=xctype)[3]
                    kxc = kxc[:,0,:,0,:,0] * weight
                    beta -= lib.einsum('axg,byg,czg,abcg->xyz', rho10, rho11, rho12, kxc)
            
            elif xctype in ('GGA', 'MGGA'):
                for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, 1, max_mem):
                    rho0 = numpy.array([ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=1, with_lapl=False) for dm in dm0])
                    rho10 = numpy.array([[ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=freq[0]==0, with_lapl=False)
                                        for dm in dms] for dms in dm10])
                    rho11 = numpy.array([[ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=freq[1]==0, with_lapl=False)
                                        for dm in dms] for dms in dm11])
                    rho12 = numpy.array([[ni.eval_rho(mol, ao, dm, mask, xctype,
                                        hermi=freq[2]==0, with_lapl=False)
                                        for dm in dms] for dms in dm12])
                    kxc = ni.eval_xc_eff(mf.xc, rho0, deriv=3, xctype=xctype)[3] * weight
                    beta -= lib.einsum('axig,byjg,czkg,aibjckg->xyz',
                                       rho10, rho11, rho12, kxc)
            
            elif xctype == 'HF':
                pass

            else:
                raise NotImplementedError(xctype)

    if mf.verbose >= logger.INFO:
        log.debug(f'The third energy response tensor b{freq}')
        log.debug(f'{beta}')
    
    return beta


class UHFPolar(UCPHFBase, RHFPolar):
    polar = polarizability = polarizability
    hyperpolar = hyperpolarizability = hyperpolarizability

uhf.UHF.Polarizability = lib.class_as_method(UHFPolar)


if __name__ == '__main__':
    # static polarizabilities computed via analytical gradient vs. finite field
    from pyscf import gto
    mol = gto.M(atom = '''O    0.    0.       0.
                          H    0.   -0.757    0.587
                          H    0.    0.757    0.587''',
                charge = 1,
                spin = 1)

    mf = mol.UHF().run(conv_tol=1e-14)
    hcore = mf.get_hcore()
    pl = UHFPolar(mf)
    h1 = pl.get_h1()
    polar = pl.polar()
    hyperpolar = pl.hyperpolar()
    
    def apply_E(E):
        mf.get_hcore = lambda *args, **kwargs: hcore + lib.einsum('x,xuv->uv', E, h1)
        mf.run(conv_tol=1e-14)
        return mf.dip_moment(mol, mf.make_rdm1(), unit='AU', verbose=0)
    print(polar)
    e1 = apply_E([ 0.0001, 0, 0])
    e2 = apply_E([-0.0001, 0, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([ 0, 0.0001, 0])
    e2 = apply_E([ 0,-0.0001, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([ 0, 0, 0.0001])
    e2 = apply_E([ 0, 0,-0.0001])
    print((e1 - e2) / 0.0002)
    
    def apply_E(E):
        mf.get_hcore = lambda *args, **kwargs: hcore + lib.einsum('x,xuv->uv', E, h1)
        mf.run(conv_tol=1e-14)
        return UHFPolar(mf).polarizability()
    print(hyperpolar)
    e1 = apply_E([ 0.0001, 0, 0])
    e2 = apply_E([-0.0001, 0, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([ 0, 0.0001, 0])
    e2 = apply_E([ 0,-0.0001, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([ 0, 0, 0.0001])
    e2 = apply_E([ 0, 0,-0.0001])
    print((e1 - e2) / 0.0002)
