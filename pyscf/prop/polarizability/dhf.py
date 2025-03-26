#!/usr/bin/env python

import numpy
from scipy.linalg import block_diag
from pyscf import lib
from pyscf.scf import dhf
from .ghf import GHFPolar


def _block_diag(a, b):
    '''
    3-D block diagonalizer using SciPy's block_diag.
    '''
    assert len(a) == len(b) and a.ndim == b.ndim == 3

    return numpy.array([block_diag(a[i], b[i]) for i in range(len(a))], dtype=a.dtype)


class DHFPolar(GHFPolar):
    def get_dm1(self, mo1=None, freq=0, **kwargs):
        '''The first-order density matrix in AO basis.'''
        if mo1 is None:
            try: mo1 = self.mo1[freq]
            except KeyError: mo1 = self.solve_mo1(freq=freq, **kwargs)
        mf = self.mf
        n2c = mf.mol.nao_2c()
        mo_coeff = mf.mo_coeff[:,n2c:]
        occidx = mf.mo_occ[n2c:] > 0
        orbv = mo_coeff[:,~occidx]
        orbo = mo_coeff[:, occidx]
        
        if mo1.ndim == 3:
            if mo1.shape[-2] == orbv.shape[-1]:
                dm1 = lib.einsum('pj,xji,qi->xpq', orbv, mo1, orbo.conj())
            elif mo1.shape[-2] == orbo.shape[-1]: # mo1 = s1oo * -.5
                dm1 = lib.einsum('pj,xji,qi->xpq', orbo, mo1, orbo.conj())
            else: # mo1.shape[-2] == mo_coeff.shape[-1]
                dm1 = lib.einsum('pj,xji,qi->xpq', mo_coeff, mo1, orbo.conj())
            dm1 += dm1.transpose(0,2,1).conj()
        
        else: # mo1[0] = U(w), mo1[1] = U*(-w)
            if mo1.shape[-2] == orbv.shape[-1]:
                dm1 = lib.einsum('pj,xji,qi->xpq', orbv, mo1[0], orbo.conj())
                dm1+= lib.einsum('pi,xji,qj->xpq', orbo, mo1[1], orbv.conj())
            elif mo1.shape[-2] == orbo.shape[-1]: # mo1 = s1oo * -.5
                dm1 = lib.einsum('pj,xji,qi->xpq', orbo, mo1, orbo.conj())
                dm1+= dm1.transpose(0,2,1).conj()
            else: # mo1.shape[-2] == mo_coeff.shape[-1]
                dm1 = lib.einsum('pj,xji,qi->xpq', mo_coeff, mo1[0], orbo.conj())
                dm1+= lib.einsum('pi,xji,qj->xpq', orbo, mo1[1], mo_coeff.conj())
        
        return dm1

    def get_dm2(self, mo2=None, freq=(0,0), with_mo1=True, with_mo2=True, **kwargs):
        '''The second-order density matrix in AO basis.'''
        if mo2 is None and with_mo2:
            try: mo2 = self.mo2[freq]
            except KeyError: mo2 = self.solve_mo2(freq=freq, **kwargs)
        mf = self.mf
        n2c = mf.mol.nao_2c()
        mo_coeff = mf.mo_coeff[:,n2c:]
        occidx = mf.mo_occ[n2c:] > 0
        orbv = mo_coeff[:,~occidx]
        orbo = mo_coeff[:, occidx]
        dm2 = 0

        if freq == (0,0) and (mo2 is None or mo2.ndim == 3): # D(0,0)
            if with_mo2:
                if mo2.shape[-2] == orbv.shape[-1]:
                    dm22 = lib.einsum('pj,xji,qi->xpq', orbv, mo2, orbo.conj())
                elif mo2.shape[-2] == orbo.shape[-1]:
                    dm22 = lib.einsum('pj,xji,qi->xpq', orbo, mo2, orbo.conj())
                else:
                    dm22 = lib.einsum('pj,xji,qi->xpq', mo_coeff, mo2, orbo.conj())
                dm2 += dm22
            
            if with_mo1:
                try: mo1 = self.mo1[0]
                except KeyError: mo1 = self.solve_mo1(freq=0, **kwargs)
                if self.with_s1: # mo1.shape = (3,ntot,nocc)
                    dm21 = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff,
                                      mo1, mo1.conj(), mo_coeff.conj())
                else:            # mo1.shape = (3,nvir,nocc)
                    dm21 = lib.einsum('pj,xji,yki,qk->xypq', orbv,
                                      mo1, mo1.conj(), orbv.conj())
                dm2 += dm21.reshape(9,*dm21.shape[-2:])
            
            try: dm2 += dm2.transpose(0,2,1).conj()
            except SyntaxError: pass
        
        else: # mo2[0] = U(w1,w2), mo2[1] = U*(-w1,-w2)
            if with_mo2:
                if mo2.shape[-2] == orbv.shape[-1]:
                    dm22 = lib.einsum('pj,xji,qi->xpq', orbv, mo2[0], orbo.conj())
                    dm22+= lib.einsum('pi,xji,qj->xpq', orbo, mo2[1], orbv.conj())
                elif mo2.shape[-2] == orbo.shape[-1]:
                    dm22 = lib.einsum('pj,xji,qi->xpq', orbo, mo2[0], orbo.conj())
                    dm22+= lib.einsum('pi,xji,qj->xpq', orbo, mo2[1], orbo.conj())
                else:
                    dm22 = lib.einsum('pj,xji,qi->xpq', mo_coeff, mo2[0], orbo.conj())
                    dm22+= lib.einsum('pi,xji,qj->xpq', orbo, mo2[1], mo_coeff.conj())
                dm2 += dm22
            
            if with_mo1:
                if freq[0] == freq[1]: # D(w,w)
                    try: mo1 = self.mo1[freq[0]]
                    except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
                    if self.with_s1:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff,
                                          mo1[0], mo1[1], mo_coeff.conj())
                    else:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq', orbv,
                                          mo1[0], mo1[1], orbv.conj())
                    dm21 += dm21.transpose(1,0,2,3)
                
                elif freq[0] == -freq[1]: # D(w,-w)
                    try: mo1 = self.mo1[freq[0]]
                    except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
                    if self.with_s1:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff,
                                          mo1[0], mo1[0].conj(), mo_coeff.conj())
                        dm21+= lib.einsum('pj,xki,yji,qk->xypq', mo_coeff,
                                          mo1[1], mo1[1].conj(), mo_coeff.conj())
                    else:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq', orbv,
                                          mo1[0], mo1[0].conj(), orbv.conj())
                        dm21+= lib.einsum('pj,xki,yji,qk->xypq', orbv,
                                          mo1[1], mo1[1].conj(), orbv.conj())
                
                elif 0 in freq: # D(0,w) / D(w,0)
                    w = freq[0] if freq[0] != 0 else freq[1]
                    try: mo10 = self.mo1[0]
                    except KeyError: mo10 = self.solve_mo1(freq=0, **kwargs)
                    try: mo11 = self.mo1[w]
                    except KeyError: mo11 = self.solve_mo1(freq=w, **kwargs)
                    if self.with_s1:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff,
                                          mo10, mo11[1], mo_coeff.conj())
                        dm21+= lib.einsum('pk,xji,yki,qj->xypq', mo_coeff,
                                          mo10.conj(), mo11[0], mo_coeff.conj())
                    else:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq', orbv,
                                          mo10, mo11[1], orbv.conj())
                        dm21+= lib.einsum('pk,xji,yki,qj->xypq', orbv,
                                          mo10.conj(), mo11[0], orbv.conj())
                    if freq.index(0) == 1: dm21 = dm21.transpose(1,0,2,3)
                
                else: # D(w1,w2)
                    w0 = freq[0]
                    w1 = freq[1]
                    try: mo10 = self.mo1[w0]
                    except KeyError: mo10 = self.solve_mo1(freq=w0, **kwargs)
                    try: mo11 = self.mo1[w1]
                    except KeyError: mo11 = self.solve_mo1(freq=w1, **kwargs)
                    if self.with_s1:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff,
                                          mo10[0], mo11[1], mo_coeff.conj())
                        dm21+= lib.einsum('pk,xji,yki,qj->xypq', mo_coeff,
                                          mo10[1], mo11[0], mo_coeff.conj())
                    else:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq', orbv,
                                          mo10[0], mo11[1], orbv.conj())
                        dm21+= lib.einsum('pk,xji,yki,qj->xypq', orbv,
                                          mo10[1], mo11[0], orbv.conj())
                
                dm2 += dm21.reshape(9,*dm21.shape[-2:])
        
        return dm2

    def get_h1(self, **kwargs):
        mf = self.mf
        mol = mf.mol
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        charge_center = lib.einsum('i,ix->x', charges, coords) / charges.sum()
        with mol.with_common_orig((charge_center)):
            c = lib.param.LIGHT_SPEED
            ll_dip = mol.intor_symmetric('int1e_r_spinor', comp=3)
            ss_dip = mol.intor_symmetric('int1e_sprsp_spinor', comp=3) * (.5/c)**2
            dip = _block_diag(ll_dip, ss_dip)
        return dip

    def get_e0vo(self):
        '''e0vo = e0v - e0o.'''
        mf = self.mf
        n2c = mf.mol.nao_2c()
        occidx = mf.mo_occ[n2c:] > 0
        e0 = mf.mo_energy[n2c:]
        e0v = e0[~occidx]
        e0o = e0[ occidx]
        e0vo = e0v[:,None] - e0o
        return e0vo

    def get_e0to(self):
        '''e0to = e0t - e0o.'''
        mf = self.mf
        n2c = mf.mol.nao_2c()
        e0 = mf.mo_energy[n2c:]
        e0o = e0[mf.mo_occ[n2c:]>0]
        e0to = e0[:,None] - e0o
        return e0to

    def _to_vo(self, ao):
        '''Convert some quantity in AO basis to that in vir.-occ. MO basis.'''
        mf = self.mf
        n2c = mf.mol.nao_2c()
        mo_coeff = mf.mo_coeff[:,n2c:]
        occidx = mf.mo_occ[n2c:] > 0
        orbv = mo_coeff[:,~occidx]
        orbo = mo_coeff[:, occidx]

        if ao.ndim == 2:
            vo = orbv.T.conj() @ ao @ orbo
        elif ao.ndim == 3:
            vo = lib.einsum('pj,xpq,qi->xji', orbv.conj(), ao, orbo)
        elif ao.ndim == 4:
            vo = lib.einsum('pj,xypq,qi->xyji', orbv.conj(), ao, orbo)
        else:
            raise NotImplementedError
        
        return vo

    def _to_vv(self, ao):
        '''Convert some quantity in AO basis to that in vir.-occ. MO basis.'''
        mf = self.mf
        n2c = mf.mol.nao_2c()
        mo_coeff = mf.mo_coeff[:,n2c:]
        viridx = mf.mo_occ[n2c:] == 0
        orbv = mo_coeff[:, viridx]

        if ao.ndim == 2:
            vv = orbv.T.conj() @ ao @ orbv
        elif ao.ndim == 3:
            vv = lib.einsum('pj,xpq,qi->xji', orbv.conj(), ao, orbv)
        elif ao.ndim == 4:
            vv = lib.einsum('pj,xypq,qi->xyji', orbv.conj(), ao, orbv)
        else:
            raise NotImplementedError
        
        return vv

    def _to_oo(self, ao):
        '''Convert some quantity in AO basis to that in occ.-occ. MO basis.'''
        mf = self.mf
        n2c = mf.mol.nao_2c()
        mo_coeff = mf.mo_coeff[:,n2c:]
        occidx = mf.mo_occ[n2c:] > 0
        orbo = mo_coeff[:, occidx]

        if ao.ndim == 2:
            oo = orbo.T.conj() @ ao @ orbo
        elif ao.ndim == 3:
            oo = lib.einsum('pj,xpq,qi->xji', orbo.conj(), ao, orbo)
        elif ao.ndim == 4:
            oo = lib.einsum('pj,xypq,qi->xyji', orbo.conj(), ao, orbo)
        else:
            raise NotImplementedError
        
        return oo
    
    def _to_to(self, ao):
        '''Convert some quantity in AO basis to that in tot.-occ. MO basis.'''
        mf = self.mf
        n2c = mf.mol.nao_2c()
        mo_coeff = mf.mo_coeff[:,n2c:]
        occidx = mf.mo_occ[n2c:] > 0
        orbo = mo_coeff[:, occidx]

        if ao.ndim == 2:
            to = mo_coeff.T.conj() @ ao @ orbo
        elif ao.ndim == 3:
            to = lib.einsum('pj,xpq,qi->xji', mo_coeff.conj(), ao, orbo)
        elif ao.ndim == 4:
            to = lib.einsum('pj,xypq,qi->xyji', mo_coeff.conj(), ao, orbo)
        else:
            raise NotImplementedError
        
        return to

    def _to_vt(self, ao):
        '''Convert some quantity in AO basis to that in vir.-tot. MO basis.'''
        mf = self.mf
        n2c = mf.mol.nao_2c()
        mo_coeff = mf.mo_coeff[:,n2c:]
        viridx = mf.mo_occ[n2c:] == 0
        orbv = mo_coeff[:, viridx]

        if ao.ndim == 2:
            vt = orbv.T.conj() @ ao @ mo_coeff
        elif ao.ndim == 3:
            vt = lib.einsum('pj,xpq,qi->xji', orbv.conj(), ao, mo_coeff)
        elif ao.ndim == 4:
            vt = lib.einsum('pj,xypq,qi->xyji', orbv.conj(), ao, mo_coeff)
        else:
            raise NotImplementedError
        
        return vt

dhf.DHF.Polarizability = lib.class_as_method(DHFPolar)


if __name__ == '__main__':
    # static polarizabilities computed via analytical gradient vs. finite field
    from pyscf import gto
    mol = gto.M(atom = '''H    0.   0.   0.
                          F    0.   0.   0.917''')

    mf = mol.DHF().run(conv_tol=1e-14)
    hcore = mf.get_hcore()
    pl = DHFPolar(mf)
    h1 = pl.get_h1()
    polar = pl.polar()
    hyperpolar = pl.hyperpolar()
    
    def apply_E(E):
        mf.get_hcore = lambda *args, **kwargs: hcore + lib.einsum('x,xuv->uv', E, h1)
        mf.run(conv_tol=1e-14)
        return mf.dip_moment(mol, mf.make_rdm1(), unit_symbol='AU', verbose=0)
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
        return DHFPolar(mf).polarizability()
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
