import numpy
from scipy.optimize import newton_krylov
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, _response_functions
from . import numint, numint2c, r_numint

class CPHFBase(lib.StreamObject):
    def __init__(self, mf):
        self.mf = mf
        self.mol = mf.mol
        self.stdout = mf.stdout
        self.verbose = mf.verbose

        self.mo1 = {}
        self.mo2 = {}
        self.with_s1 = False
        self.max_cycle = 50
        self.conv_tol = 1e-9

        self._keys = set(self.__dict__.keys())

    def get_dm1(self, mo1=None, freq=0, **kwargs):
        '''The first-order density matrix in AO basis.'''
        if mo1 is None:
            try: mo1 = self.mo1[freq]
            except KeyError: mo1 = self.solve_mo1(freq=freq, **kwargs)
        mf = self.mf
        mo_coeff = mf.mo_coeff
        occidx = mf.mo_occ > 0
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
        
        return dm1*2 if isinstance(mf, hf.RHF) else dm1

    def get_dm2(self, mo2=None, freq=(0,0), with_mo1=True, with_mo2=True, **kwargs):
        '''The second-order density matrix in AO basis.'''
        if mo2 is None and with_mo2:
            try: mo2 = self.mo2[freq]
            except KeyError: mo2 = self.solve_mo2(freq=freq, **kwargs)
        mf = self.mf
        mo_coeff = mf.mo_coeff
        occidx = mf.mo_occ > 0
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
        
        return dm2*2 if isinstance(mf, hf.RHF) else dm2

    def get_vind(self, mo_deriv, freq, **kwargs):
        '''The induced potential matrix in AO basis.'''
        if isinstance(freq, (int, float)):
            dm_deriv = self.get_dm1(mo_deriv, freq, **kwargs)
            hermi = freq == 0
        elif len(freq) == 2:
            dm_deriv = self.get_dm2(mo_deriv, freq, **kwargs)
            hermi = freq[0] == freq[1] == 0
        else:
            raise NotImplementedError(freq)
        
        vind = self.mf.gen_response(hermi=hermi)

        try:
            return vind(dm_deriv)
        except NotImplementedError: # nr_fxc does not support complex density matrices
            dmr_deriv = dm_deriv.real
            dmi_deriv = dm_deriv.imag
            return vind(dmr_deriv) + vind(dmi_deriv)*1j

    def get_f1(self, mo1=None, freq=0, **kwargs):
        '''The first-order Fock matrix in AO basis.'''
        if mo1 is None:
            try: mo1 = self.mo1[freq]
            except KeyError: mo1 = self.solve_mo1(freq=freq, **kwargs)
        f1 = self.get_vind(mo1, freq, **kwargs)
        f1+= self.get_h1(**kwargs) # in-place operation does not support broadcast
        if self.with_s1:
            f1 += self.get_jk1()
        return f1
    
    def get_e1(self, mo1=None, freq=0, **kwargs):
        '''The first-order energy for the occ.-occ. block.'''
        if mo1 is None:
            try: mo1 = self.mo1[freq]
            except KeyError: mo1 = self.solve_mo1(freq=freq, **kwargs)
        e1 = self.get_f1(mo1, freq, **kwargs)
        if self.with_s1:
            if freq != 0:
                e1 -= self.get_t1(freq)
            e1 = self._to_oo(e1)
            mf = self.mf
            e0 = mf.mo_energy[mf.mo_occ>0]
            e0oo = (e0[:,None] - e0 + freq) * .5
            e1 -= self._to_oo(self.get_s1()) * e0oo
            return e1
        else:
            return self._to_oo(e1)
        
    def get_h1(self, **kwargs):
        '''The first-order core-Hamiltonian in AO basis.'''
        mf = self.mf
        nao = mf.mo_coeff.shape[-2]
        return numpy.zeros((3,nao,nao))

    def get_jk1(self, dm0=None):
        '''The first-order JK wrt the external magnetic field.'''
        mf = self.mf
        nao = mf.mo_coeff.shape[-2]
        return numpy.zeros((3,nao,nao))

    def get_s1(self):
        '''The first-order overlap matrix in AO basis.'''
        mf = self.mf
        nao = mf.mo_coeff.shape[-2]
        return numpy.zeros((3,nao,nao))
    
    def get_t1(self, freq):
        r'''( \mu | i\partial_t | \nu )^(1).'''
        mf = self.mf
        nao = mf.mo_coeff.shape[-2]
        return numpy.zeros((3,nao,nao))
    
    def get_f2(self, mo2=None, freq=(0,0), with_mo2=True, **kwargs):
        '''The second-order Fock matrix in AO basis.'''
        if mo2 is None and with_mo2:
            try: mo2 = self.mo2[freq]
            except KeyError: mo2 = self.solve_mo2(freq=freq, **kwargs)
        
        f2 = self.get_vind(mo2, freq, with_mo2=with_mo2, **kwargs)

        if isinstance(self.mf, hf.KohnShamDFT):
            mf = self.mf
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            dm0 = mf.make_rdm1()
            cur_mem = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8 - cur_mem)

            try: mo10 = self.mo1[freq[0]]
            except KeyError: mo10 = self.solve_mo1(freq=freq[0], **kwargs)
            dm10 = self.get_dm1(mo10, freq[0])

            if freq[1] == freq[0]:
                dm11 = dm10
            elif freq[1] == -freq[0]:
                dm11 = dm10.transpose(0,2,1).conj()
            else:
                try: mo11 = self.mo1[freq[1]]
                except KeyError: mo11 = self.solve_mo1(freq=freq[1], **kwargs)
                dm11 = self.get_dm1(mo11, freq[1])
            
            try:
                f2 += ni.get_kxc(mf.mol, mf.grids, mf.xc, dm0, (dm10,dm11),
                                 (freq[0]==0,freq[1]==0), max_memory)
            except NotImplementedError:
                f2 += ni.get_kxc(mf.mol, mf.grids, mf.xc, dm0, (dm10.real,
                                 dm11.real), (freq[0]==0,freq[1]==0), max_memory)
                f2 += ni.get_kxc(mf.mol, mf.grids, mf.xc, dm0, (dm10.imag,
                                 dm11.imag), (freq[0]==0,freq[1]==0), max_memory) * 1j

        if hasattr(self, 'get_h2'):
            f2 += self.get_h2(**kwargs)
        
        if self.with_s1:
            f2 += self.get_jk2()
            f2 += self.get_jk11(freq)
        
        return f2

    def get_jk2(self):
        r'''The second-order JK wrt the external magnetic field.
            D^(0)(\ \ | \ \)^(2)'''
        mf = self.mf
        nao = mf.mo_coeff.shape[-2]
        return numpy.zeros((9,nao,nao))
    
    def get_jk11(self, freq):
        r'''The second-order JK wrt the external magnetic field.
            D^(1)(\ \ | \ \)^(1)'''
        mf = self.mf
        nao = mf.mo_coeff.shape[-2]
        return numpy.zeros((9,nao,nao))
    
    def get_s2(self):
        '''The second-order overlap matrix in AO basis.'''
        mf = self.mf
        nao = mf.mo_coeff.shape[-2]
        return numpy.zeros((9,nao,nao))
    
    def get_t2(self, freq):
        r'''( \mu | i\partial_t | \nu )^(2).'''
        mf = self.mf
        nao = mf.mo_coeff.shape[-2]
        return numpy.zeros((9,nao,nao))

    def get_e0vo(self):
        '''e0vo = e0v - e0o.'''
        mf = self.mf
        occidx = mf.mo_occ > 0
        e0 = mf.mo_energy
        e0v = e0[~occidx]
        e0o = e0[ occidx]
        e0vo = e0v[:,None] - e0o
        return e0vo

    def get_e0to(self):
        '''e0to = e0t - e0o.'''
        mf = self.mf
        e0 = mf.mo_energy
        e0o = e0[mf.mo_occ>0]
        e0to = e0[:,None] - e0o
        return e0to

    def solve_mo1(self, freq=0, solver='krylov', **kwargs):
        '''CP-HF/KS solver for the first-order MO response $U$.'''
        assert isinstance(freq, (int, float))
        solver = solver.lower()
        if   'direct' in solver: # exact solver
            mo1 = self._direct_solver(freq, **kwargs)
        elif 'newton' in solver: # only newton-krylov recommended
            mo1 = self._newton_solver(freq, **kwargs)
        elif 'krylov' in solver: # very recommended
            mo1 = self._krylov_solver(freq, **kwargs)
        else:
            raise NotImplementedError(solver)
        self.mo1[freq] = mo1
        if freq != 0:
            if isinstance(mo1, tuple): # U-HF/KS
                self.mo1[-freq] = (mo1[0][[1,0]].conj(),
                                   mo1[1][[1,0]].conj())
            else:
                self.mo1[-freq] = mo1[[1,0]].conj()
        return mo1
    
    def solve_mo2(self, freq=(0,0), solver='krylov', **kwargs):
        '''CP-HF/KS solver for the second-order MO response $U$.'''
        assert isinstance(freq, tuple) and len(freq) == 2
        solver = solver.lower()
        if   'direct' in solver: # exact solver
            mo2 = self._direct_solver(freq, **kwargs)
        elif 'newton' in solver: # only newton-krylov recommended
            mo2 = self._newton_solver(freq, **kwargs)
        elif 'krylov' in solver: # very recommended
            mo2 = self._krylov_solver(freq, **kwargs)
        else:
            raise NotImplementedError(solver)
        self.mo2[freq] = mo2
        if freq != (0,0):
            if isinstance(mo2, tuple):
                self.mo2[(-freq[0],-freq[1])] = (mo2[0][[1,0]].conj(),
                                                 mo2[1][[1,0]].conj())
            else:
                self.mo2[(-freq[0],-freq[1])] = mo2[[1,0]].conj()
        return mo2

    def _mo2oo(self, freq=(0,0), **kwargs):
        '''The second-order MO response $U$ in the occ.-occ. block.'''
        if freq == (0,0): # U(0,0)
            try: mo1 = self.mo1[0]
            except KeyError: mo1 = self.solve_mo1(freq=0, **kwargs)
            mo2oo = lib.einsum('xji,yjk->xyik', mo1.conj(), mo1)
            if self.with_s1:
                s2 = self._to_oo(self.get_s2())
                s1 = self._to_to(self.get_s1())
                us = lib.einsum('xji,yjk->xyik', mo1.conj(), s1)
                us+= us.transpose(0,1,3,2).conj()
                mo2oo += us + s2*.5
            mo2oo += mo2oo.transpose(1,0,2,3)
            mo2oo *= -.5
            mo2oo = mo2oo.reshape(9,*mo2oo.shape[-2:])
        
        else:
            if freq[0] == freq[1]: # U(w,w)
                try: mo1 = self.mo1[freq[0]]
                except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
                mo2oo = lib.einsum('xji,yjk->xyik', mo1[1], mo1[0])
                if self.with_s1:
                    s2 = self._to_oo(self.get_s2())
                    s1 = self._to_to(self.get_s1())
                    us = lib.einsum('xji,yjk->xyik', mo1[1], s1)
                    us+= lib.einsum('xjk,yji->xyik', mo1[0], s1.conj())
                    mo2oo += us + s2*.5
                mo2oo += mo2oo.transpose(1,0,2,3)
            
            elif freq[0] == -freq[1]: # U(w,-w)
                try: mo1 = self.mo1[freq[0]]
                except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
                mo2oo = lib.einsum('xji,yjk->xyik', mo1[1], mo1[0])
                mo2oo+= lib.einsum('xjk,yji->xyik', mo1[0], mo1[1])
                if self.with_s1:
                    s2 = self._to_oo(self.get_s2())
                    s1 = self._to_to(self.get_s1())
                    us = lib.einsum('xji,yjk->xyik', mo1[1], s1)
                    us+= lib.einsum('xjk,yji->xyik', mo1[0], s1.conj())
                    us+= us.transpose(1,0,3,2).conj()
                    mo2oo += us + s2
            
            elif 0 in freq: # U(0,w) / U(w,0)
                w = freq[0] if freq[0] != 0 else freq[1]
                try: mo10 = self.mo1[0]
                except KeyError: mo10 = self.solve_mo1(freq=0, **kwargs)
                try: mo11 = self.mo1[w]
                except KeyError: mo11 = self.solve_mo1(freq=w, **kwargs)
                mo2oo = lib.einsum('xji,yjk->xyik', mo10.conj(), mo11[0])
                mo2oo+= lib.einsum('xjk,yji->xyik', mo10, mo11[1])
                if self.with_s1:
                    s2 = self._to_oo(self.get_s2())
                    s1 = self._to_to(self.get_s1())
                    us = lib.einsum('xji,yjk->xyik', mo10.conj(), s1)
                    us+= lib.einsum('xjk,yji->xyik', mo10, s1.conj())
                    us+= lib.einsum('xji,yjk->xyik', s1.conj(), mo11[0])
                    us+= lib.einsum('xjk,yji->xyik', s1, mo11[1])
                    mo2oo += us + s2
                if freq.index(0) == 1: mo2oo = mo2oo.transpose(1,0,2,3)
            
            else: # U(w1,w2)
                w0 = freq[0]
                w1 = freq[1]
                try: mo10 = self.mo1[w0]
                except KeyError: mo10 = self.solve_mo1(freq=w0, **kwargs)
                try: mo11 = self.mo1[w1]
                except KeyError: mo11 = self.solve_mo1(freq=w1, **kwargs)
                mo2oo = lib.einsum('xji,yjk->xyik', mo10[1], mo11[0])
                mo2oo+= lib.einsum('xjk,yji->xyik', mo10[0], mo11[1])
                if self.with_s1:
                    s2 = self._to_oo(self.get_s2())
                    s1 = self._to_to(self.get_s1())
                    us = lib.einsum('xji,yjk->xyik', mo10[1], s1)
                    us+= lib.einsum('xjk,yji->xyik', mo10[0], s1.conj())
                    us+= lib.einsum('xji,yjk->xyik', s1.conj(), mo11[0])
                    us+= lib.einsum('xjk,yji->xyik', s1, mo11[1])
                    mo2oo += us + s2
            
            mo2oo = mo2oo.reshape(9,*mo2oo.shape[-2:])
            mo2oo = numpy.array((mo2oo, mo2oo.transpose(0,2,1))) * -.5
        
        return mo2oo

    def _rhs1(self, freq=0, **kwargs):
        '''The right-hand side of the working first-order CP-HF/KS equation.'''
        rhs = -self.get_h1(**kwargs)
        if self.with_s1:
            mf = self.mf
            e0 = mf.mo_energy[mf.mo_occ>0]
            s1 = self.get_s1()
            rhs += self.get_vind(self._to_oo(s1)*.5, freq, **kwargs)
            rhs -= self.get_jk1(mf.make_rdm1())
            if freq != 0:
                rhs += self.get_t1(freq)
            rhs  = self._to_vo(rhs)
            rhs += self._to_vo(s1) * e0
            return rhs
        else:
            return self._to_vo(rhs)

    def _rhs2(self, freq=(0,0), **kwargs):
        '''The right-hand side of the working second-order CP-HF/KS equation.'''
        rhs = -self.get_f2(freq=freq, with_mo2=False, **kwargs)
        rhs -= self.get_vind(self._mo2oo(freq, **kwargs), freq, **kwargs)
        rhs = self._to_vo(rhs)

        if freq[0] == freq[1]:
            try: mo1 = self.mo1[freq[0]]
            except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
            f1 = self.get_f1(mo1, freq[0], **kwargs)
            e1 = self.get_e1(mo1, freq[0], **kwargs)
            if self.with_s1: # mo1.shape = (3,ntot,nocc)
                mf = self.mf
                occidx = mf.mo_occ > 0
                e0 = mf.mo_energy[occidx]
                s1 = self.get_s1()
                s1vt = self._to_vt(s1)

                rhs += self._to_vo(self.get_s2()) * e0
                sym = lib.einsum('xjk,yki->xyji', self._to_vo(s1), e1)
                if freq[0] != 0:
                    mo1 = mo1[0]
                    t1vt = self._to_vt(self.get_t1(freq[0]))
                    sym -= lib.einsum('xjl,yli->xyji', s1vt, mo1) * freq[0]
                    sym += lib.einsum('xjl,yli->xyji', t1vt, mo1)
                    rhs += self._to_vo(self.get_t2(freq))
                sym += lib.einsum('xjl,yli,i->xyji', s1vt, mo1, e0)
                sym -= lib.einsum('xjl,yli->xyji', self._to_vt(f1), mo1)
                sym += lib.einsum('xjk,yki->xyji', mo1[:,~occidx], e1)
                sym += sym.transpose(1,0,2,3)
                rhs += sym.reshape(9,*sym.shape[-2:])
            else:
                if freq[0] != 0:
                    mo1 = mo1[0]
                sym = -lib.einsum('xjl,yli->xyji', self._to_vv(f1), mo1)
                sym += lib.einsum('xjk,yki->xyji', mo1, e1)
                sym += sym.transpose(1,0,2,3)
                rhs += sym.reshape(9,*sym.shape[-2:])

        elif freq[0] == -freq[1]:
            try: mo1 = self.mo1[freq[0]]
            except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
            f1p = self.get_f1(mo1, freq[0], **kwargs)
            f1m = f1p.transpose(0,2,1).conj()
            e1p = self.get_e1(mo1, freq[0], **kwargs)
            e1m = e1p.transpose(0,2,1).conj()
            mo1[1] = mo1[1].conj()
            if self.with_s1: # mo1.shape = (3,ntot,nocc)
                mf = self.mf
                occidx = mf.mo_occ > 0
                e0 = mf.mo_energy[occidx]
                s1 = self.get_s1()
                s1vt = self._to_vt(s1)
                s1vo = self._to_vo(s1)
                t1vt = self._to_vt(self.get_t1(freq[0]))
                
                rhs += self._to_vo(self.get_s2()) * e0
                tmp  = lib.einsum('xjl,yli,i->xyji', s1vt, mo1[1], e0)
                tmp += lib.einsum('yjl,xli,i->xyji', s1vt, mo1[0], e0)
                tmp += lib.einsum('xjk,yki->xyji', s1vo, e1m)
                tmp += lib.einsum('yjk,xki->xyji', s1vo, e1p)

                rhs += self._to_vo(self.get_t2(freq))
                tmp -= lib.einsum('xjl,yli->xyji', s1vt, mo1[1]) * freq[1]
                tmp -= lib.einsum('yjl,xli->xyji', s1vt, mo1[0]) * freq[0]
                tmp += lib.einsum('xjl,yli->xyji', t1vt, mo1[1])
                tmp -= lib.einsum('yjl,xli->xyji', t1vt, mo1[0])

                tmp -= lib.einsum('xjl,yli->xyji', self._to_vt(f1p), mo1[1])
                tmp -= lib.einsum('yjl,xli->xyji', self._to_vt(f1m), mo1[0])
                tmp += lib.einsum('xjk,yki->xyji', mo1[0,:,~occidx], e1m)
                tmp += lib.einsum('yjk,xki->xyji', mo1[1,:,~occidx], e1p)
            else:
                tmp = -lib.einsum('xjl,yli->xyji', self._to_vv(f1p), mo1[1])
                tmp -= lib.einsum('yjl,xli->xyji', self._to_vv(f1m), mo1[0])
                tmp += lib.einsum('xjk,yki->xyji', mo1[0], e1m)
                tmp += lib.einsum('yjk,xki->xyji', mo1[1], e1p)
            rhs += tmp.reshape(9,*tmp.shape[-2:])
                
        elif 0 in freq:
            w = freq[0] if freq[0] != 0 else freq[1]
            try: mo10 = self.mo1[0]
            except KeyError: mo10 = self.solve_mo1(freq=0, **kwargs)
            try: mo11 = self.mo1[w]
            except KeyError: mo11 = self.solve_mo1(freq=w, **kwargs)
            f10 = self.get_f1(mo10, 0, **kwargs)
            f11 = self.get_f1(mo11, w, **kwargs)
            e10 = self.get_e1(mo10, 0, **kwargs)
            e11 = self.get_e1(mo11, w, **kwargs)
            mo11 = mo11[0]
            if self.with_s1: # mo1.shape = (3,ntot,nocc)
                mf = self.mf
                occidx = mf.mo_occ > 0
                e0 = mf.mo_energy[occidx]
                s1 = self.get_s1()
                s1vt = self._to_vt(s1)
                s1vo = self._to_vo(s1)
                t1vt = self._to_vt(self.get_t1(freq=w))
                
                tmp  = lib.einsum('xjl,yli,i->xyji', s1vt, mo11, e0)
                tmp += lib.einsum('yjl,xli,i->xyji', s1vt, mo10, e0)
                tmp += lib.einsum('xjk,yki->xyji', s1vo, e11)
                tmp += lib.einsum('yjk,xki->xyji', s1vo, e10)

                rhs += self._to_vo(self.get_s2()) * e0
                rhs += self._to_vo(self.get_t2(freq))
                tmp -= lib.einsum('xjl,yli->xyji', s1vt, mo11) * w
                tmp += lib.einsum('yjl,xli->xyji', t1vt, mo10)

                tmp -= lib.einsum('xjl,yli->xyji', self._to_vt(f10), mo11)
                tmp -= lib.einsum('yjl,xli->xyji', self._to_vt(f11), mo10)
                tmp += lib.einsum('xjk,yki->xyji', mo10[:,~occidx], e11)
                tmp += lib.einsum('yjk,xki->xyji', mo11[:,~occidx], e10)
            else:
                tmp = -lib.einsum('xjl,yli->xyji', self._to_vv(f10), mo11)
                tmp -= lib.einsum('yjl,xli->xyji', self._to_vv(f11), mo10)
                tmp += lib.einsum('xjk,yki->xyji', mo10, e11)
                tmp += lib.einsum('yjk,xki->xyji', mo11, e10)
            
            if freq.index(0) == 1:
                rhs += tmp.transpose(1,0,2,3).reshape(9,*tmp.shape[-2:])
            else:
                rhs += tmp.reshape(9,*tmp.shape[-2:])

        else:
            w0 = freq[0]
            w1 = freq[1]
            try: mo10 = self.mo1[w0]
            except KeyError: mo10 = self.solve_mo1(freq=w0, **kwargs)
            try: mo11 = self.mo1[w1]
            except KeyError: mo11 = self.solve_mo1(freq=w1, **kwargs)
            f10 = self.get_f1(mo10, w0, **kwargs)
            f11 = self.get_f1(mo11, w1, **kwargs)
            e10 = self.get_e1(mo10, w0, **kwargs)
            e11 = self.get_e1(mo11, w1, **kwargs)
            mo10 = mo10[0]
            mo11 = mo11[0]
            if self.with_s1: # mo1.shape = (3,ntot,nocc)
                mf = self.mf
                occidx = mf.mo_occ > 0
                e0 = mf.mo_energy[occidx]
                s1 = self.get_s1()
                s1vt = self._to_vt(s1)
                s1vo = self._to_vo(s1)
                t10 = self.get_t1(freq=w0)
                t11 = self.get_t1(freq=w1)
                
                rhs += self._to_vo(self.get_s2()) * e0
                tmp  = lib.einsum('xjl,yli,i->xyji', s1vt, mo11, e0)
                tmp += lib.einsum('yjl,xli,i->xyji', s1vt, mo10, e0)
                tmp += lib.einsum('xjk,yki->xyji', s1vo, e11)
                tmp += lib.einsum('yjk,xki->xyji', s1vo, e10)

                rhs += self._to_vo(self.get_t2(freq))
                tmp -= lib.einsum('xjl,yli->xyji', s1vt, mo11) * w1
                tmp -= lib.einsum('yjl,xli->xyji', s1vt, mo10) * w0
                tmp += lib.einsum('xjl,yli->xyji', self._to_vt(t10), mo11)
                tmp += lib.einsum('yjl,xli->xyji', self._to_vt(t11), mo10)

                tmp -= lib.einsum('xjl,yli->xyji', self._to_vt(f10), mo11)
                tmp -= lib.einsum('yjl,xli->xyji', self._to_vt(f11), mo10)
                tmp += lib.einsum('xjk,yki->xyji', mo10[:,~occidx], e11)
                tmp += lib.einsum('yjk,xki->xyji', mo11[:,~occidx], e10)
            else:
                tmp = -lib.einsum('xjl,yli->xyji', self._to_vv(f10), mo11)
                tmp -= lib.einsum('yjl,xli->xyji', self._to_vv(f11), mo10)
                tmp += lib.einsum('xjk,yki->xyji', mo10, e11)
                tmp += lib.einsum('yjk,xki->xyji', mo11, e10)
            rhs += tmp.reshape(9,*tmp.shape[-2:])
        
        return rhs

    def _krylov_solver(self, freq=0, verbose=logger.WARN, **kwargs):
        '''CP-HF/KS subspace projection solver by `pyscf.lib.krylov`.'''
        log = logger.new_logger(verbose=verbose)
        t0 = (logger.process_clock(), logger.perf_counter())
        e0vo = self.get_e0vo()
        nvir, nocc = e0vo.shape
        # first-order solver
        if isinstance(freq, (int, float)):
            if freq == 0:
                def lhs(mo1): # U(0)
                    mo1 = mo1.reshape(3,nvir,nocc)
                    v1 = self.get_vind(mo1, freq, **kwargs)
                    v1 = self._to_vo(v1) / e0vo
                    return v1.ravel()
                
                rhs = self._rhs1(freq, **kwargs) / e0vo
                rhs = rhs.ravel()
                # casting multiple vectors into Krylov solver yields poor and inefficient results
                mo1 = lib.krylov(lhs, rhs, max_cycle=self.max_cycle,
                                 tol=self.conv_tol, verbose=log)
                mo1 = mo1.reshape(3,nvir,nocc)
                if self.with_s1:
                    mo1oo = self._to_oo(self.get_s1()) * -.5
                    mo1 = numpy.concatenate((mo1oo, mo1), axis=-2)
            
            else:
                def lhs(mo1): # mo1[0] = U(w), mo1[1] = U*(-w)
                    mo1 = mo1.reshape(2,3,nvir,nocc)
                    v1 = self.get_vind(mo1, freq, **kwargs)
                    v1p = self._to_vo(v1) # shape: (3,nvir,nocc)
                    v1m = self._to_vo(v1.transpose(0,2,1).conj())
                    v1p /= (e0vo + freq)
                    v1m /= (e0vo - freq)
                    v1 = numpy.array((v1p, v1m.conj()))
                    return v1.ravel()
                
                if not self.with_s1:
                    rhs = self._rhs1(freq, **kwargs)
                    rhs = numpy.array((rhs       /(e0vo+freq),
                                       rhs.conj()/(e0vo-freq)))
                else:
                    rhs = (self._rhs1( freq, **kwargs),
                           self._rhs1(-freq, **kwargs))
                    rhs = numpy.array((rhs[0]       /(e0vo+freq),
                                       rhs[1].conj()/(e0vo-freq)))
                rhs = rhs.ravel()
                # casting multiple vectors into Krylov solver yields poor and inefficient results
                mo1 = lib.krylov(lhs, rhs, max_cycle=self.max_cycle,
                                 tol=self.conv_tol, verbose=log)
                mo1 = mo1.reshape(2,3,nvir,nocc)
                if self.with_s1:
                    mo1oo = self._to_oo(self.get_s1()) * -.5
                    mo1oo = numpy.array((mo1oo, mo1oo.transpose(0,2,1)))
                    # transpose() is more efficient than conj() for a hermitian matrix
                    mo1 = numpy.concatenate((mo1oo, mo1), axis=-2)
            
            log.timer('Krylov solver for the first-order CP-HF/KS', *t0)
            return mo1
        # second-order solver
        elif len(freq) == 2:
            if freq == (0,0):
                def lhs(mo2): # U(0,0)
                    mo2 = mo2.reshape(9,nvir,nocc)
                    v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                    v2 = self._to_vo(v2) / e0vo
                    return v2.ravel()
                
                rhs = self._rhs2(freq, **kwargs) / e0vo
                rhs = rhs.ravel()
                # casting multiple vectors into Krylov solver yields poor and inefficient results
                mo2 = lib.krylov(lhs, rhs, max_cycle=self.max_cycle,
                                 tol=self.conv_tol, verbose=log)
                mo2 = mo2.reshape(9,nvir,nocc)
            
            else:
                def lhs(mo2): # mo2[0] = U(w1,w2), mo2[1] = U*(-w1,-w2)
                    mo2 = mo2.reshape(2,9,nvir,nocc)
                    v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                    v2p = self._to_vo(v2) # shape: (9,nvir,nocc)
                    v2m = self._to_vo(v2.transpose(0,2,1).conj())
                    v2p /= (e0vo + freq[0] + freq[1])
                    v2m /= (e0vo - freq[0] - freq[1])
                    v2 = numpy.array((v2p, v2m.conj()))
                    return v2.ravel()
                
                rhs = (self._rhs2(freq, **kwargs),
                       self._rhs2((-freq[0],-freq[1]), **kwargs))
                rhs = numpy.array((rhs[0]       /(e0vo+freq[0]+freq[1]),
                                   rhs[1].conj()/(e0vo-freq[0]-freq[1])))
                rhs = rhs.ravel()
                # casting multiple vectors into Krylov solver yields poor and inefficient results
                mo2 = lib.krylov(lhs, rhs, max_cycle=self.max_cycle,
                                 tol=self.conv_tol, verbose=log)
                mo2 = mo2.reshape(2,9,nvir,nocc)
            
            mo2 = numpy.concatenate((self._mo2oo(freq, **kwargs), mo2), axis=-2)

            log.timer('Krylov solver for the second-order CPHF/KS', *t0)
            return mo2
        
        else:
            raise NotImplementedError(freq)

    def _newton_solver(self, freq=0, verbose=logger.WARN, **kwargs):
        '''CP-HF/KS iterative solver by `scipy.optimize.newton_krylov`.'''
        log = logger.new_logger(verbose=verbose)
        t0 = (logger.process_clock(), logger.perf_counter())

        e0vo = self.get_e0vo()
        # first-order solver
        if isinstance(freq, (int, float)):
            if freq == 0:
                rhs = self._rhs1(freq, **kwargs)

                def lhs(mo1): # U(0)
                    v1 = self.get_vind(mo1, freq, **kwargs)
                    v1 = self._to_vo(v1)
                    v1+= e0vo * mo1
                    return v1 - rhs

            else:
                if not self.with_s1:
                    rhs = self._rhs1(freq, **kwargs)
                    rhs = numpy.array((rhs, rhs.conj()))
                else:
                    rhs = numpy.array((self._rhs1( freq, **kwargs),
                                       self._rhs1(-freq, **kwargs).conj()))

                def lhs(mo1): # mo1[0] = U(w), mo1[1] = U*(-w)
                    v1 = self.get_vind(mo1, freq, **kwargs)
                    v1p = self._to_vo(v1)
                    v1m = self._to_vo(v1.transpose(0,2,1).conj())
                    v1p += (e0vo + freq) * mo1[0]
                    v1m += (e0vo - freq) * mo1[1].conj()
                    v1 = numpy.array((v1p, v1m.conj()))
                    return v1 - rhs

            mo1 = newton_krylov(lhs, rhs, maxiter=self.max_cycle, f_tol=self.conv_tol)
            if self.with_s1:
                mo1oo = self._to_oo(self.get_s1()) * -.5
                if freq != 0:
                    mo1oo = numpy.array((mo1oo, mo1oo.transpose(0,2,1)))
                mo1 = numpy.concatenate((mo1oo, mo1), axis=-2)
        
            log.timer('Newton-Krylov solver for the first-order CP-HF/KS', *t0)
            return mo1
        # second-order solver
        elif len(freq) == 2:
            if freq == (0,0):
                rhs = self._rhs2(freq, **kwargs)

                def lhs(mo2): # U(0,0)
                    v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                    v2 = self._to_vo(v2)
                    v2+= e0vo * mo2
                    return v2 - rhs
                
            else:
                rhs = (self._rhs2(freq, **kwargs),
                       self._rhs2((-freq[0],-freq[1]), **kwargs))
                rhs = numpy.array((rhs[0], rhs[1].conj()))

                def lhs(mo2): # mo2[0] = U(w1,w2), mo2[1] = U*(-w1,-w2)
                    v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                    v2p = self._to_vo(v2)
                    v2m = self._to_vo(v2.transpose(0,2,1).conj())
                    v2p += (e0vo + freq[0] + freq[1]) * mo2[0]
                    v2m += (e0vo - freq[0] - freq[1]) * mo2[1].conj()
                    v2 = numpy.array((v2p, v2m.conj()))
                    return v2 - rhs

            mo2 = newton_krylov(lhs, rhs, maxiter=self.max_cycle, f_tol=self.conv_tol)
            mo2 = numpy.concatenate((self._mo2oo(freq, **kwargs), mo2), axis=-2)

            log.timer('Newton-Krylov solver for the second-order CP-HF/KS', *t0)
            return mo2
        
        else:
            raise NotImplementedError(freq)

    def _direct_solver(self, freq=0, verbose=logger.WARN, **kwargs):
        '''CP-HF/KS direct solver by `numpy.linalg.solve`.'''
        log = logger.new_logger(verbose=verbose)
        t0 = (logger.process_clock(), logger.perf_counter())

        e0vo = self.get_e0vo()
        nvir, nocc = e0vo.shape
        # first-order solver
        if isinstance(freq, (int, float)):
            def lhs(mo1): # mo1[0] = U(w), mo1[1] = U*(-w)
                mo1 = mo1.reshape(2,1,nvir,nocc)
                v1 = self.get_vind(mo1, freq, **kwargs)
                v1p = self._to_vo(v1)
                v1m = self._to_vo(v1.transpose(0,2,1).conj())
                v1p += (e0vo + freq) * mo1[0]
                v1m += (e0vo - freq) * mo1[1].conj()
                v1 = numpy.array((v1p, v1m.conj())) # shape: (2,1,nvir,nocc)
                return v1.ravel()
            
            if not self.with_s1:
                rhs = self._rhs1(freq, **kwargs)
                rhs = numpy.stack((rhs, rhs.conj()), axis=1)
            else:
                rhs = (self._rhs1( freq, **kwargs),
                       self._rhs1(-freq, **kwargs))
                rhs = numpy.stack((rhs[0], rhs[1].conj()), axis=1)
            size = rhs[0].size
            operator = numpy.empty((size, size))
            iden = numpy.eye(size)
            for i, row in enumerate(iden):
                operator[:,i] = lhs(row)
            
            mo1 = numpy.linalg.solve(operator, rhs.reshape(3,-1).T).T
            mo1 = mo1.reshape(3,2,nvir,nocc).swapaxes(0,1)
            if self.with_s1:
                mo1oo = self._to_oo(self.get_s1()) * -.5
                mo1oo = numpy.array((mo1oo, mo1oo.transpose(0,2,1)))
                # transpose() is more efficient than conj() for a hermitian matrix
                mo1 = numpy.concatenate((mo1oo, mo1), axis=-2)
            
            log.timer('Direct solver for the first-order CP-HF/KS', *t0)
            return mo1[0] if freq == 0 else mo1
        # second-order solver
        elif len(freq) == 2:
            def lhs(mo2): # mo2[0] = U(w1,w2), mo2[1] = U*(-w1,-w2)
                mo2 = mo2.reshape(2,1,nvir,nocc)
                v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                v2p = self._to_vo(v2)
                v2m = self._to_vo(v2.transpose(0,2,1).conj())
                v2p += (e0vo + freq[0] + freq[1]) * mo2[0]
                v2m += (e0vo - freq[0] - freq[1]) * mo2[1].conj()
                v2 = numpy.array((v2p, v2m.conj()))
                return v2.ravel()
            
            rhs = (self._rhs2(freq, **kwargs),
                   self._rhs2((-freq[0],-freq[1]), **kwargs))
            rhs = numpy.stack((rhs[0], rhs[1].conj()), axis=1)
            size = rhs[0].size
            operator = numpy.empty((size, size))
            iden = numpy.eye(size)
            for i, row in enumerate(iden):
                operator[:,i] = lhs(row)
            
            mo2 = numpy.linalg.solve(operator, rhs.reshape(9,-1).T).T
            mo2 = mo2.reshape(9,2,nvir,nocc).swapaxes(0,1)
            if freq == (0,0): mo2 = mo2[0]
            mo2 = numpy.concatenate((self._mo2oo(freq, **kwargs), mo2), axis=-2)

            log.timer('Direct solver for the second-order CP-HF/KS', *t0)
            return mo2
        
        else:
            raise NotImplementedError(freq)

    def _to_vo(self, ao):
        '''Convert some quantity in AO basis to that in vir.-occ. MO basis.'''
        mf = self.mf
        mo_coeff = mf.mo_coeff
        occidx = mf.mo_occ > 0
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
        mo_coeff = mf.mo_coeff
        viridx = mf.mo_occ == 0
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
        mo_coeff = mf.mo_coeff
        occidx = mf.mo_occ > 0
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
        mo_coeff = mf.mo_coeff
        occidx = mf.mo_occ > 0
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
        mo_coeff = mf.mo_coeff
        viridx = mf.mo_occ == 0
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


class UCPHFBase(CPHFBase):
    def get_dm1(self, mo1=None, freq=0, **kwargs):
        '''The first-order density matrix in AO basis.'''
        if mo1 is None:
            try: mo1 = self.mo1[freq]
            except KeyError: mo1 = self.solve_mo1(freq=freq, **kwargs)
        mo1a, mo1b = mo1
        mf = self.mf
        mo_coeff = mf.mo_coeff
        occidxa = mf.mo_occ[0] > 0
        occidxb = mf.mo_occ[1] > 0
        orbva = mo_coeff[0][:,~occidxa]
        orboa = mo_coeff[0][:, occidxa]
        orbvb = mo_coeff[1][:,~occidxb]
        orbob = mo_coeff[1][:, occidxb]

        if mo1a.ndim == 3:
            if mo1a.shape[-2] == orbva.shape[-1]:
                dm1a = lib.einsum('pj,xji,qi->xpq', orbva, mo1a, orboa.conj())
                dm1b = lib.einsum('pj,xji,qi->xpq', orbvb, mo1b, orbob.conj())
            elif mo1a.shape[-2] == orboa.shape[-1]: # mo1s = s1oos * -.5
                dm1a = lib.einsum('pj,xji,qi->xpq', orboa, mo1a, orboa.conj())
                dm1b = lib.einsum('pj,xji,qi->xpq', orbob, mo1b, orbob.conj())
            else: # mo1.shape[-2] == mo_coeff.shape[-1]
                dm1a = lib.einsum('pj,xji,qi->xpq', mo_coeff[0], mo1a, orboa.conj())
                dm1b = lib.einsum('pj,xji,qi->xpq', mo_coeff[1], mo1b, orbob.conj())
            dm1a += dm1a.transpose(0,2,1).conj()
            dm1b += dm1b.transpose(0,2,1).conj()
        
        else: # mo1s[0] = Us(w), mo1s[1] = Us*(-w)
            if mo1a.shape[-2] == orbva.shape[-1]:
                dm1a = lib.einsum('pj,xji,qi->xpq', orbva, mo1a[0], orboa.conj())
                dm1a+= lib.einsum('pi,xji,qj->xpq', orboa, mo1a[1], orbva.conj())
                dm1b = lib.einsum('pj,xji,qi->xpq', orbvb, mo1b[0], orbob.conj())
                dm1b+= lib.einsum('pi,xji,qj->xpq', orbob, mo1b[1], orbvb.conj())
            elif mo1a.shape[-2] == orboa.shape[-1]: # mo1s = s1oos * -.5
                dm1a = lib.einsum('pj,xji,qi->xpq', orboa, mo1a, orboa.conj())
                dm1a+= dm1a.transpose(0,2,1).conj()
                dm1b = lib.einsum('pj,xji,qi->xpq', orbob, mo1b, orbob.conj())
                dm1b+= dm1b.transpose(0,2,1).conj()
            else: # mo1.shape[-2] == mo_coeff.shape[-1]
                dm1a = lib.einsum('pj,xji,qi->xpq', mo_coeff[0], mo1a[0], orboa.conj())
                dm1a+= lib.einsum('pi,xji,qj->xpq', orboa, mo1a[1], mo_coeff[0].conj())
                dm1b = lib.einsum('pj,xji,qi->xpq', mo_coeff[1], mo1b[0], orbob.conj())
                dm1b+= lib.einsum('pi,xji,qj->xpq', orbob, mo1b[1], mo_coeff[1].conj())

        return numpy.array((dm1a, dm1b))

    def get_dm2(self, mo2=None, freq=(0,0), with_mo1=True, with_mo2=True, **kwargs):
        '''The second-order density matrix in AO basis.'''
        if mo2 is None and with_mo2:
            try: mo2 = self.mo2[freq]
            except KeyError: mo2 = self.solve_mo2(freq=freq, **kwargs)
        mf = self.mf
        mo_coeff = mf.mo_coeff
        occidxa = mf.mo_occ[0] > 0
        occidxb = mf.mo_occ[1] > 0
        orbva = mo_coeff[0][:,~occidxa]
        orboa = mo_coeff[0][:, occidxa]
        orbvb = mo_coeff[1][:,~occidxb]
        orbob = mo_coeff[1][:, occidxb]
        dm2a = 0
        dm2b = 0

        if freq == (0,0) and (mo2 is None or mo2[0].ndim == 3): # D(0,0)
            if with_mo2:
                mo2a, mo2b = mo2
                if mo2a.shape[-2] == orbva.shape[-1]:
                    dm22a = lib.einsum('pj,xji,qi->xpq', orbva, mo2a, orboa.conj())
                    dm22b = lib.einsum('pj,xji,qi->xpq', orbvb, mo2b, orbob.conj())
                elif mo2a.shape[-2] == orboa.shape[-1]:
                    dm22a = lib.einsum('pj,xji,qi->xpq', orboa, mo2a, orboa.conj())
                    dm22b = lib.einsum('pj,xji,qi->xpq', orbob, mo2b, orbob.conj())
                else:
                    dm22a = lib.einsum('pj,xji,qi->xpq', mo_coeff[0], mo2a, orboa.conj())
                    dm22b = lib.einsum('pj,xji,qi->xpq', mo_coeff[1], mo2b, orbob.conj())
                dm2a += dm22a
                dm2b += dm22b
            
            if with_mo1:
                try: mo1 = self.mo1[0]
                except KeyError: mo1 = self.solve_mo1(freq=0, **kwargs)
                mo1a, mo1b = mo1
                if self.with_s1: # mo1.shape = (3,ntot,nocc)
                    dm21a = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff[0],
                                       mo1a, mo1a.conj(), mo_coeff[0].conj())
                    dm21b = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff[1],
                                       mo1b, mo1b.conj(), mo_coeff[1].conj())
                else:            # mo1.shape = (3,nvir,nocc)
                    dm21a = lib.einsum('pj,xji,yki,qk->xypq', orbva,
                                       mo1a, mo1a.conj(), orbva.conj())
                    dm21b = lib.einsum('pj,xji,yki,qk->xypq', orbvb,
                                       mo1b, mo1b.conj(), orbvb.conj())
                dm2a += dm21a.reshape(9,*dm21a.shape[-2:])
                dm2b += dm21b.reshape(9,*dm21b.shape[-2:])

            try:
                dm2a += dm2a.transpose(0,2,1).conj()
                dm2b += dm2b.transpose(0,2,1).conj()
            except SyntaxError: pass
        
        else: # mo2s[0] = Us(w1,w2), mo2s[1] = Us*(-w1,-w2)
            if with_mo2:
                mo2a, mo2b = mo2
                if mo2a.shape[-2] == orbva.shape[-1]:
                    dm22a = lib.einsum('pj,xji,qi->xpq', orbva, mo2a[0], orboa.conj())
                    dm22a+= lib.einsum('pi,xji,qj->xpq', orboa, mo2a[1], orbva.conj())
                    dm22b = lib.einsum('pj,xji,qi->xpq', orbvb, mo2b[0], orbob.conj())
                    dm22b+= lib.einsum('pi,xji,qj->xpq', orbob, mo2b[1], orbvb.conj())
                elif mo2a.shape[-2] == orboa.shape[-1]:
                    dm22a = lib.einsum('pj,xji,qi->xpq', orboa, mo2a[0], orboa.conj())
                    dm22a+= lib.einsum('pi,xji,qj->xpq', orboa, mo2a[1], orboa.conj())
                    dm22b = lib.einsum('pj,xji,qi->xpq', orbob, mo2b[0], orbob.conj())
                    dm22b+= lib.einsum('pi,xji,qj->xpq', orbob, mo2b[1], orbob.conj())
                else:
                    dm22a = lib.einsum('pj,xji,qi->xpq', mo_coeff[0],
                                       mo2a[0], orboa.conj())
                    dm22a+= lib.einsum('pi,xji,qj->xpq', orboa,
                                       mo2a[1], mo_coeff[0].conj())
                    dm22b = lib.einsum('pj,xji,qi->xpq', mo_coeff[1],
                                       mo2b[0], orbob.conj())
                    dm22b+= lib.einsum('pi,xji,qj->xpq', orbob,
                                       mo2b[1], mo_coeff[1].conj())
                dm2a += dm22a
                dm2b += dm22b
            
            if with_mo1:
                if freq[0] == freq[1]: # D(w,w)
                    try: mo1 = self.mo1[freq[0]]
                    except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
                    mo1a, mo1b = mo1
                    if self.with_s1:
                        dm21a = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff[0],
                                           mo1a[0], mo1a[1], mo_coeff[0].conj())
                        dm21b = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff[1],
                                           mo1b[0], mo1b[1], mo_coeff[1].conj())
                    else:
                        dm21a = lib.einsum('pj,xji,yki,qk->xypq', orbva,
                                           mo1a[0], mo1a[1], orbva.conj())
                        dm21b = lib.einsum('pj,xji,yki,qk->xypq', orbvb,
                                           mo1b[0], mo1b[1], orbvb.conj())
                    dm21a += dm21a.transpose(1,0,2,3)
                    dm21b += dm21b.transpose(1,0,2,3)
                
                elif freq[0] == -freq[1]: # D(w,-w)
                    try: mo1 = self.mo1[freq[0]]
                    except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
                    mo1a, mo1b = mo1
                    if self.with_s1:
                        dm21a = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff[0],
                                           mo1a[0], mo1a[0].conj(), mo_coeff[0].conj())
                        dm21a+= lib.einsum('pj,xki,yji,qk->xypq', mo_coeff[0],
                                           mo1a[1], mo1a[1].conj(), mo_coeff[0].conj())
                        dm21b = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff[1],
                                           mo1b[0], mo1b[0].conj(), mo_coeff[1].conj())
                        dm21b+= lib.einsum('pj,xki,yji,qk->xypq', mo_coeff[1],
                                           mo1b[1], mo1b[1].conj(), mo_coeff[1].conj())
                    else:
                        dm21a = lib.einsum('pj,xji,yki,qk->xypq', orbva,
                                           mo1a[0], mo1a[0].conj(), orbva.conj())
                        dm21a+= lib.einsum('pj,xki,yji,qk->xypq', orbva,
                                           mo1a[1], mo1a[1].conj(), orbva.conj())
                        dm21b = lib.einsum('pj,xji,yki,qk->xypq', orbvb,
                                           mo1b[0], mo1b[0].conj(), orbvb.conj())
                        dm21b+= lib.einsum('pj,xki,yji,qk->xypq', orbvb,
                                           mo1b[1], mo1b[1].conj(), orbvb.conj())
                
                elif 0 in freq: # D(0,w) / D(w,0)
                    w = freq[0] if freq[0] != 0 else freq[1]
                    try: mo10 = self.mo1[0]
                    except KeyError: mo10 = self.solve_mo1(freq=0, **kwargs)
                    try: mo11 = self.mo1[w]
                    except KeyError: mo11 = self.solve_mo1(freq=w, **kwargs)
                    mo10a, mo10b = mo10
                    mo11a, mo11b = mo11
                    if self.with_s1:
                        dm21a = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff[0],
                                           mo10a, mo11a[1], mo_coeff[0].conj())
                        dm21a+= lib.einsum('pk,xji,yki,qj->xypq', mo_coeff[0],
                                           mo10a.conj(), mo11a[0], mo_coeff[0].conj())
                        dm21b = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff[1],
                                           mo10b, mo11b[1], mo_coeff[1].conj())
                        dm21b+= lib.einsum('pk,xji,yki,qj->xypq', mo_coeff[1],
                                           mo10b.conj(), mo11b[0], mo_coeff[1].conj())
                    else:
                        dm21a = lib.einsum('pj,xji,yki,qk->xypq', orbva,
                                           mo10a, mo11a[1], orbva.conj())
                        dm21a+= lib.einsum('pk,xji,yki,qj->xypq', orbva,
                                           mo10a.conj(), mo11a[0], orbva.conj())
                        dm21b = lib.einsum('pj,xji,yki,qk->xypq', orbvb,
                                           mo10b, mo11b[1], orbvb.conj())
                        dm21b+= lib.einsum('pk,xji,yki,qj->xypq', orbvb,
                                           mo10b.conj(), mo11b[0], orbvb.conj())
                    if freq.index(0) == 1:
                        dm21a = dm21a.transpose(1,0,2,3)
                        dm21b = dm21b.transpose(1,0,2,3)
                
                else: # D(w1,w2)
                    w0 = freq[0]
                    w1 = freq[1]
                    try: mo10 = self.mo1[w0]
                    except KeyError: mo10 = self.solve_mo1(freq=w0, **kwargs)
                    try: mo11 = self.mo1[w1]
                    except KeyError: mo11 = self.solve_mo1(freq=w1, **kwargs)
                    mo10a, mo10b = mo10
                    mo11a, mo11b = mo11
                    if self.with_s1:
                        dm21a = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff[0],
                                           mo10a[0], mo11a[1], mo_coeff[0].conj())
                        dm21a+= lib.einsum('pk,xji,yki,qj->xypq', mo_coeff[0],
                                           mo10a[1], mo11a[0], mo_coeff[0].conj())
                        dm21b = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff[1],
                                           mo10b[0], mo11b[1], mo_coeff[1].conj())
                        dm21b+= lib.einsum('pk,xji,yki,qj->xypq', mo_coeff[1],
                                           mo10b[1], mo11b[0], mo_coeff[1].conj())
                    else:
                        dm21a = lib.einsum('pj,xji,yki,qk->xypq', orbva,
                                           mo10a[0], mo11a[1], orbva.conj())
                        dm21a+= lib.einsum('pk,xji,yki,qj->xypq', orbva,
                                           mo10a[1], mo11a[0], orbva.conj())
                        dm21b = lib.einsum('pj,xji,yki,qk->xypq', orbvb,
                                           mo10b[0], mo11b[1], orbvb.conj())
                        dm21b+= lib.einsum('pk,xji,yki,qj->xypq', orbvb,
                                           mo10b[1], mo11b[0], orbvb.conj())
                
                dm2a += dm21a.reshape(9,*dm21a.shape[-2:])
                dm2b += dm21b.reshape(9,*dm21b.shape[-2:])
        
        return numpy.array((dm2a, dm2b))

    def get_e1(self, mo1=None, freq=0, **kwargs):
        '''The first-order energy for the occ.-occ. block.'''
        if mo1 is None:
            try: mo1 = self.mo1[freq]
            except KeyError: mo1 = self.solve_mo1(freq=freq, **kwargs)
        e1 = self.get_f1(mo1, freq, **kwargs)
        if self.with_s1:
            if freq != 0:
                e1 -= self.get_t1(freq)
            e1a, e1b = self._to_oo(e1)
            mf = self.mf
            e0a = mf.mo_energy[0][mf.mo_occ[0]>0]
            e0b = mf.mo_energy[1][mf.mo_occ[1]>0]
            e0ooa = (e0a[:,None] - e0a + freq) * .5
            e0oob = (e0b[:,None] - e0b + freq) * .5
            s1ooa, s1oob = self._to_oo(self.get_s1())
            e1a -= s1ooa * e0ooa
            e1b -= s1oob * e0oob
            return (e1a, e1b)
        else:
            return self._to_oo(e1)
    
    def get_e0vo(self):
        '''e0vo = e0v - e0o.'''
        mf = self.mf
        occidxa = mf.mo_occ[0] > 0
        occidxb = mf.mo_occ[1] > 0
        e0a, e0b = mf.mo_energy
        e0va = e0a[~occidxa]
        e0oa = e0a[ occidxa]
        e0vb = e0b[~occidxb]
        e0ob = e0b[ occidxb]
        e0voa = e0va[:,None] - e0oa
        e0vob = e0vb[:,None] - e0ob
        return (e0voa, e0vob)
    
    def get_e0to(self):
        '''e0to = e0t - e0o.'''
        mf = self.mf
        e0a, e0b = mf.mo_energy
        e0oa = e0a[mf.mo_occ[0]>0]
        e0ob = e0b[mf.mo_occ[1]>0]
        e0toa = e0a[:,None] - e0oa
        e0tob = e0b[:,None] - e0ob
        return (e0toa, e0tob)

    def _mo2oo(self, freq=(0,0), **kwargs):
        '''The second-order MO response $U$ in the occ.-occ. block.'''
        if freq == (0,0): # U(0,0)
            try: mo1 = self.mo1[0]
            except KeyError: mo1 = self.solve_mo1(freq=0, **kwargs)
            mo1a, mo1b = mo1
            mo2ooa = lib.einsum('xji,yjk->xyik', mo1a.conj(), mo1a)
            mo2oob = lib.einsum('xji,yjk->xyik', mo1b.conj(), mo1b)
            if self.with_s1:
                s2a, s2b = self._to_oo(self.get_s2())
                s1a, s1b = self._to_to(self.get_s1())
                usa = lib.einsum('xji,yjk->xyik', mo1a.conj(), s1a)
                usb = lib.einsum('xji,yjk->xyik', mo1b.conj(), s1b)
                usa+= usa.transpose(0,1,3,2).conj()
                usb+= usb.transpose(0,1,3,2).conj()
                mo2ooa += usa + s2a*.5
                mo2oob += usb + s2b*.5
            mo2ooa += mo2ooa.transpose(1,0,2,3)
            mo2oob += mo2oob.transpose(1,0,2,3)
            mo2ooa *= -.5
            mo2oob *= -.5
            mo2ooa = mo2ooa.reshape(9,*mo2ooa.shape[-2:])
            mo2oob = mo2oob.reshape(9,*mo2oob.shape[-2:])
        
        else:
            if freq[0] == freq[1]: # U(w,w)
                try: mo1 = self.mo1[freq[0]]
                except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
                mo1a, mo1b = mo1
                mo2ooa = lib.einsum('xji,yjk->xyik', mo1a[1], mo1a[0])
                mo2oob = lib.einsum('xji,yjk->xyik', mo1b[1], mo1b[0])
                if self.with_s1:
                    s2a, s2b = self._to_oo(self.get_s2())
                    s1a, s1b = self._to_to(self.get_s1())
                    usa = lib.einsum('xji,yjk->xyik', mo1a[1], s1a)
                    usb = lib.einsum('xji,yjk->xyik', mo1b[1], s1b)
                    usa+= lib.einsum('xjk,yji->xyik', mo1a[0], s1a.conj())
                    usb+= lib.einsum('xjk,yji->xyik', mo1b[0], s1b.conj())
                    mo2ooa += usa + s2a*.5
                    mo2oob += usb + s2b*.5
                mo2ooa += mo2ooa.transpose(1,0,2,3)
                mo2oob += mo2oob.transpose(1,0,2,3)
            
            elif freq[0] == -freq[1]: # U(w,-w)
                try: mo1 = self.mo1[freq[0]]
                except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
                mo1a, mo1b = mo1
                mo2ooa = lib.einsum('xji,yjk->xyik', mo1a[1], mo1a[0])
                mo2ooa+= lib.einsum('xjk,yji->xyik', mo1a[0], mo1a[1])
                mo2oob = lib.einsum('xji,yjk->xyik', mo1b[1], mo1b[0])
                mo2oob+= lib.einsum('xjk,yji->xyik', mo1b[0], mo1b[1])
                if self.with_s1:
                    s2a, s2b = self._to_oo(self.get_s2())
                    s1a, s1b = self._to_to(self.get_s1())
                    usa = lib.einsum('xji,yjk->xyik', mo1a[1], s1a)
                    usb = lib.einsum('xji,yjk->xyik', mo1b[1], s1b)
                    usa+= lib.einsum('xjk,yji->xyik', mo1a[0], s1a.conj())
                    usb+= lib.einsum('xjk,yji->xyik', mo1b[0], s1b.conj())
                    usa+= usa.transpose(1,0,3,2).conj()
                    usb+= usb.transpose(1,0,3,2).conj()
                    mo2ooa += usa + s2a
                    mo2oob += usb + s2b
            
            elif 0 in freq: # U(0,w) / U(w,0)
                w = freq[0] if freq[0] != 0 else freq[1]
                try: mo10 = self.mo1[0]
                except KeyError: mo10 = self.solve_mo1(freq=0, **kwargs)
                try: mo11 = self.mo1[w]
                except KeyError: mo11 = self.solve_mo1(freq=w, **kwargs)
                mo10a, mo10b = mo10
                mo11a, mo11b = mo11
                mo2ooa = lib.einsum('xji,yjk->xyik', mo10a.conj(), mo11a[0])
                mo2ooa+= lib.einsum('xjk,yji->xyik', mo10a, mo11a[1])
                mo2oob = lib.einsum('xji,yjk->xyik', mo10b.conj(), mo11b[0])
                mo2oob+= lib.einsum('xjk,yji->xyik', mo10b, mo11b[1])
                if self.with_s1:
                    s2a, s2b = self._to_oo(self.get_s2())
                    s1a, s1b = self._to_to(self.get_s1())
                    usa = lib.einsum('xji,yjk->xyik', mo10a.conj(), s1a)
                    usb = lib.einsum('xji,yjk->xyik', mo10b.conj(), s1b)
                    usa+= lib.einsum('xjk,yji->xyik', mo10a, s1a.conj())
                    usb+= lib.einsum('xjk,yji->xyik', mo10b, s1b.conj())
                    usa+= lib.einsum('xji,yjk->xyik', s1a.conj(), mo11a[0])
                    usb+= lib.einsum('xji,yjk->xyik', s1b.conj(), mo11b[0])
                    usa+= lib.einsum('xjk,yji->xyik', s1a, mo11a[1])
                    usb+= lib.einsum('xjk,yji->xyik', s1b, mo11b[1])
                    mo2ooa += usa + s2a
                    mo2oob += usb + s2b
                if freq.index(0) == 1:
                    mo2ooa = mo2ooa.transpose(1,0,2,3)
                    mo2oob = mo2oob.transpose(1,0,2,3)
            
            else: # U(w1,w2)
                w0 = freq[0]
                w1 = freq[1]
                try: mo10 = self.mo1[w0]
                except KeyError: mo10 = self.solve_mo1(freq=w0, **kwargs)
                try: mo11 = self.mo1[w1]
                except KeyError: mo11 = self.solve_mo1(freq=w1, **kwargs)
                mo10a, mo10b = mo10
                mo11a, mo11b = mo11
                mo2ooa = lib.einsum('xji,yjk->xyik', mo10a[1], mo11a[0])
                mo2ooa+= lib.einsum('xjk,yji->xyik', mo10a[0], mo11a[1])
                mo2oob = lib.einsum('xji,yjk->xyik', mo10b[1], mo11b[0])
                mo2oob+= lib.einsum('xjk,yji->xyik', mo10b[0], mo11b[1])
                if self.with_s1:
                    s2a, s2b = self._to_oo(self.get_s2())
                    s1a, s1b = self._to_to(self.get_s1())
                    usa = lib.einsum('xji,yjk->xyik', mo10a[1], s1a)
                    usb = lib.einsum('xji,yjk->xyik', mo10b[1], s1b)
                    usa+= lib.einsum('xjk,yji->xyik', mo10a[0], s1a.conj())
                    usb+= lib.einsum('xjk,yji->xyik', mo10b[0], s1b.conj())
                    usa+= lib.einsum('xji,yjk->xyik', s1a.conj(), mo11a[0])
                    usb+= lib.einsum('xji,yjk->xyik', s1b.conj(), mo11b[0])
                    usa+= lib.einsum('xjk,yji->xyik', s1a, mo11a[1])
                    usb+= lib.einsum('xjk,yji->xyik', s1b, mo11b[1])
                    mo2ooa += usa + s2a
                    mo2oob += usb + s2b
            
            mo2ooa = mo2ooa.reshape(9,*mo2ooa.shape[-2:])
            mo2oob = mo2oob.reshape(9,*mo2oob.shape[-2:])
            mo2ooa = numpy.array((mo2ooa, mo2ooa.transpose(0,2,1))) * -.5
            mo2oob = numpy.array((mo2oob, mo2oob.transpose(0,2,1))) * -.5
        
        return (mo2ooa, mo2oob)

    def _rhs1(self, freq=0, **kwargs):
        '''The right-hand side of the working first-order CP-HF/KS equation.'''
        rhs = -self.get_h1(**kwargs)
        if self.with_s1:
            mf = self.mf
            e0a = mf.mo_energy[0,mf.mo_occ[0]>0]
            e0b = mf.mo_energy[1,mf.mo_occ[1]>0]
            s1 = self.get_s1()
            rhs += self.get_vind(self._to_oo(s1)*.5, freq, **kwargs)
            rhs -= self.get_jk1(mf.make_rdm1())
            if freq != 0:
                rhs += self.get_t1(freq)
            rhsa, rhsb = self._to_vo(rhs)
            s1voa, s1vob = self._to_vo(s1)
            rhsa += s1voa * e0a
            rhsb += s1vob * e0b
            return (rhsa, rhsb)
        else:
            return self._to_vo(rhs)

    def _rhs2(self, freq=(0,0), **kwargs):
        '''The right-hand side of the working second-order CP-HF/KS equation.'''
        rhs = -self.get_f2(freq=freq, with_mo2=False, **kwargs)
        rhs -= self.get_vind(self._mo2oo(freq, **kwargs), freq, **kwargs)
        rhsa, rhsb = self._to_vo(rhs)

        if freq[0] == freq[1]:
            try: mo1 = self.mo1[freq[0]]
            except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
            f1 = self.get_f1(mo1, freq[0], **kwargs)
            e1a, e1b = self.get_e1(mo1, freq[0], **kwargs)
            mo1a, mo1b = mo1
            if self.with_s1: # mo1.shape = (3,ntot,nocc)
                mf = self.mf
                occidxa = mf.mo_occ[0] > 0
                occidxb = mf.mo_occ[1] > 0
                e0a = mf.mo_energy[0, occidxa]
                e0b = mf.mo_energy[1, occidxb]
                s1 = self.get_s1()
                f1vta, f1vtb = self._to_vt(f1)
                s1voa, s1vob = self._to_vo(s1)
                s1vta, s1vtb = self._to_vt(s1)
                s2voa, s2vob = self._to_vo(self.get_s2())

                syma = lib.einsum('xjk,yki->xyji', s1voa, e1a)
                symb = lib.einsum('xjk,yki->xyji', s1vob, e1b)
                if freq[0] != 0:
                    mo1a = mo1a[0]
                    mo1b = mo1b[0]
                    t1vta, t1vtb = self._to_vt(self.get_t1(freq[0]))
                    syma -= lib.einsum('xjl,yli->xyji', s1vta, mo1a) * freq[0]
                    symb -= lib.einsum('xjl,yli->xyji', s1vtb, mo1b) * freq[0]
                    syma += lib.einsum('xjl,yli->xyji', t1vta, mo1a)
                    symb += lib.einsum('xjl,yli->xyji', t1vtb, mo1b)
                    t2a, t2b = self._to_vo(self.get_t2(freq))
                    rhsa += t2a
                    rhsb += t2b
                syma += lib.einsum('xjl,yli,i->xyji', s1vta, mo1a, e0a)
                symb += lib.einsum('xjl,yli,i->xyji', s1vtb, mo1b, e0b)
                syma -= lib.einsum('xjl,yli->xyji', f1vta, mo1a)
                symb -= lib.einsum('xjl,yli->xyji', f1vtb, mo1b)
                syma += lib.einsum('xjk,yki->xyji', mo1a[:,~occidxa], e1a)
                symb += lib.einsum('xjk,yki->xyji', mo1b[:,~occidxb], e1b)
                syma += syma.transpose(1,0,2,3)
                symb += symb.transpose(1,0,2,3)
                rhsa += syma.reshape(9,*syma.shape[-2:]) + s2voa * e0a
                rhsb += symb.reshape(9,*symb.shape[-2:]) + s2vob * e0b
            else:
                if freq[0] != 0:
                    mo1a = mo1a[0]
                    mo1b = mo1b[0]
                f1vva, f1vvb = self._to_vv(f1)
                syma = -lib.einsum('xjl,yli->xyji', f1vva, mo1a)
                symb = -lib.einsum('xjl,yli->xyji', f1vvb, mo1b)
                syma += lib.einsum('xjk,yki->xyji', mo1a, e1a)
                symb += lib.einsum('xjk,yki->xyji', mo1b, e1b)
                syma += syma.transpose(1,0,2,3)
                symb += symb.transpose(1,0,2,3)
                rhsa += syma.reshape(9,*syma.shape[-2:])
                rhsb += symb.reshape(9,*symb.shape[-2:])

        elif freq[0] == -freq[1]:
            try: mo1 = self.mo1[freq[0]]
            except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
            f1p = self.get_f1(mo1, freq[0], **kwargs)
            f1m = f1p.transpose(0,1,3,2).conj()
            e1pa, e1pb = self.get_e1(mo1, freq[0], **kwargs)
            e1ma = e1pa.transpose(0,2,1).conj()
            e1mb = e1pb.transpose(0,2,1).conj()
            mo1a, mo1b = mo1
            mo1a[1] = mo1a[1].conj()
            mo1b[1] = mo1b[1].conj()
            if self.with_s1: # mo1.shape = (3,ntot,nocc)
                mf = self.mf
                occidxa = mf.mo_occ[0] > 0
                occidxb = mf.mo_occ[1] > 0
                e0a = mf.mo_energy[0, occidxa]
                e0b = mf.mo_energy[1, occidxb]
                s1 = self.get_s1()
                f1pa, f1pb = self._to_vt(f1p)
                f1ma, f1mb = self._to_vt(f1m)
                s1vta, s1vtb = self._to_vt(s1)
                s1voa, s1vob = self._to_vo(s1)
                t1vta, t1vtb = self._to_vt(self.get_t1(freq[0]))
                s2voa, s2vob = self._to_vo(self.get_s2())
                t2voa, t2vob = self._to_vo(self.get_t2(freq))
                
                rhsa += s2voa * e0a
                rhsb += s2vob * e0b
                tmpa  = lib.einsum('xjl,yli,i->xyji', s1vta, mo1a[1], e0a)
                tmpb  = lib.einsum('xjl,yli,i->xyji', s1vtb, mo1b[1], e0b)
                tmpa += lib.einsum('yjl,xli,i->xyji', s1vta, mo1a[0], e0a)
                tmpb += lib.einsum('yjl,xli,i->xyji', s1vtb, mo1b[0], e0b)
                tmpa += lib.einsum('xjk,yki->xyji', s1voa, e1ma)
                tmpb += lib.einsum('xjk,yki->xyji', s1vob, e1mb)
                tmpa += lib.einsum('yjk,xki->xyji', s1voa, e1pa)
                tmpb += lib.einsum('yjk,xki->xyji', s1vob, e1pb)

                rhsa += t2voa
                rhsb += t2vob
                tmpa -= lib.einsum('xjl,yli->xyji', s1vta, mo1a[1]) * freq[1]
                tmpb -= lib.einsum('xjl,yli->xyji', s1vtb, mo1b[1]) * freq[1]
                tmpa -= lib.einsum('yjl,xli->xyji', s1vta, mo1a[0]) * freq[0]
                tmpb -= lib.einsum('yjl,xli->xyji', s1vtb, mo1b[0]) * freq[0]
                tmpa += lib.einsum('xjl,yli->xyji', t1vta, mo1a[1])
                tmpb += lib.einsum('xjl,yli->xyji', t1vtb, mo1b[1])
                tmpa -= lib.einsum('yjl,xli->xyji', t1vta, mo1a[0])
                tmpb -= lib.einsum('yjl,xli->xyji', t1vtb, mo1b[0])

                tmpa += lib.einsum('xjk,yki->xyji', mo1a[0,:,~occidxa], e1ma)
                tmpb += lib.einsum('xjk,yki->xyji', mo1b[0,:,~occidxb], e1mb)
                tmpa += lib.einsum('yjk,xki->xyji', mo1a[1,:,~occidxa], e1pa)
                tmpb += lib.einsum('yjk,xki->xyji', mo1b[1,:,~occidxb], e1pb)
            else:
                f1pa, f1pb = self._to_vv(f1p)
                f1ma, f1mb = self._to_vv(f1m)
                tmpa  = lib.einsum('xjk,yki->xyji', mo1a[0], e1ma)
                tmpb  = lib.einsum('xjk,yki->xyji', mo1b[0], e1mb)
                tmpa += lib.einsum('yjk,xki->xyji', mo1a[1], e1pa)
                tmpb += lib.einsum('yjk,xki->xyji', mo1b[1], e1pb)
            tmpa -= lib.einsum('xjl,yli->xyji', f1pa, mo1a[1])
            tmpb -= lib.einsum('xjl,yli->xyji', f1pb, mo1b[1])
            tmpa -= lib.einsum('yjl,xli->xyji', f1ma, mo1a[0])
            tmpb -= lib.einsum('yjl,xli->xyji', f1mb, mo1b[0])
            rhsa += tmpa.reshape(9,*tmpa.shape[-2:])
            rhsb += tmpb.reshape(9,*tmpb.shape[-2:])
                
        elif 0 in freq:
            w = freq[0] if freq[0] != 0 else freq[1]
            try: mo10 = self.mo1[0]
            except KeyError: mo10 = self.solve_mo1(freq=0, **kwargs)
            try: mo11 = self.mo1[w]
            except KeyError: mo11 = self.solve_mo1(freq=w, **kwargs)
            f10 = self.get_f1(mo10, 0, **kwargs)
            f11 = self.get_f1(mo11, w, **kwargs)
            e10a, e10b = self.get_e1(mo10, 0, **kwargs)
            e11a, e11b = self.get_e1(mo11, w, **kwargs)
            mo10a, mo10b = mo10
            mo11a, mo11b = mo11
            mo11a = mo11a[0]
            mo11b = mo11b[0]
            if self.with_s1: # mo1.shape = (3,ntot,nocc)
                mf = self.mf
                occidxa = mf.mo_occ[0] > 0
                occidxb = mf.mo_occ[1] > 0
                e0a = mf.mo_energy[0, occidxa]
                e0b = mf.mo_energy[1, occidxb]
                s1 = self.get_s1()
                f10a, f10b = self._to_vt(f10)
                f11a, f11b = self._to_vt(f11)
                s1vta, s1vtb = self._to_vt(s1)
                s1voa, s1vob = self._to_vo(s1)
                t1vta, t1vtb = self._to_vt(self.get_t1(freq=w))
                s2voa, s2vob = self._to_vo(self.get_s2())
                t2voa, t2vob = self._to_vo(self.get_t2(freq=(0,w)))
                
                tmpa  = lib.einsum('xjl,yli,i->xyji', s1vta, mo11a, e0a)
                tmpb  = lib.einsum('xjl,yli,i->xyji', s1vtb, mo11b, e0b)
                tmpa += lib.einsum('yjl,xli,i->xyji', s1vta, mo10a, e0a)
                tmpb += lib.einsum('yjl,xli,i->xyji', s1vtb, mo10b, e0b)
                tmpa += lib.einsum('xjk,yki->xyji', s1voa, e11a)
                tmpb += lib.einsum('xjk,yki->xyji', s1vob, e11b)
                tmpa += lib.einsum('yjk,xki->xyji', s1voa, e10a)
                tmpb += lib.einsum('yjk,xki->xyji', s1vob, e10b)

                rhsa += s2voa * e0a
                rhsb += s2vob * e0b
                rhsa += t2voa
                rhsb += t2vob
                tmpa -= lib.einsum('xjl,yli->xyji', s1vta, mo11a) * w
                tmpb -= lib.einsum('xjl,yli->xyji', s1vtb, mo11b) * w
                tmpa += lib.einsum('yjl,xli->xyji', t1vta, mo10a)
                tmpb += lib.einsum('yjl,xli->xyji', t1vtb, mo10b)
                
                tmpa += lib.einsum('xjk,yki->xyji', mo10a[:,~occidxa], e11a)
                tmpb += lib.einsum('xjk,yki->xyji', mo10b[:,~occidxb], e11b)
                tmpa += lib.einsum('yjk,xki->xyji', mo11a[:,~occidxa], e10a)
                tmpb += lib.einsum('yjk,xki->xyji', mo11b[:,~occidxb], e10b)
            else:
                f10a, f10b = self._to_vv(f10)
                f11a, f11b = self._to_vv(f11)
                tmpa  = lib.einsum('xjk,yki->xyji', mo10a, e11a)
                tmpb  = lib.einsum('xjk,yki->xyji', mo10b, e11b)
                tmpa += lib.einsum('yjk,xki->xyji', mo11a, e10a)
                tmpb += lib.einsum('yjk,xki->xyji', mo11b, e10b)
            tmpa -= lib.einsum('xjl,yli->xyji', f10a, mo11a)
            tmpb -= lib.einsum('xjl,yli->xyji', f10b, mo11b)
            tmpa -= lib.einsum('yjl,xli->xyji', f11a, mo10a)
            tmpb -= lib.einsum('yjl,xli->xyji', f11b, mo10b)
            
            if freq.index(0) == 1:
                rhsa += tmpa.transpose(1,0,2,3).reshape(9,*tmpa.shape[-2:])
                rhsb += tmpb.transpose(1,0,2,3).reshape(9,*tmpb.shape[-2:])
            else:
                rhsa += tmpa.reshape(9,*tmpa.shape[-2:])
                rhsb += tmpb.reshape(9,*tmpb.shape[-2:])

        else:
            w0 = freq[0]
            w1 = freq[1]
            try: mo10 = self.mo1[w0]
            except KeyError: mo10 = self.solve_mo1(freq=w0, **kwargs)
            try: mo11 = self.mo1[w1]
            except KeyError: mo11 = self.solve_mo1(freq=w1, **kwargs)
            f10 = self.get_f1(mo10, w0, **kwargs)
            f11 = self.get_f1(mo11, w1, **kwargs)
            e10a, e10b = self.get_e1(mo10, w0, **kwargs)
            e11a, e11b = self.get_e1(mo11, w1, **kwargs)
            mo10a, mo10b = mo10
            mo11a, mo11b = mo11
            mo10a = mo10a[0]
            mo10b = mo10b[0]
            mo11a = mo11a[0]
            mo11b = mo11b[0]
            if self.with_s1: # mo1.shape = (3,ntot,nocc)
                mf = self.mf
                occidxa = mf.mo_occ[0] > 0
                occidxb = mf.mo_occ[1] > 0
                e0a = mf.mo_energy[0, occidxa]
                e0b = mf.mo_energy[1, occidxb]
                s1 = self.get_s1()
                f10a, f10b = self._to_vt(f10)
                f11a, f11b = self._to_vt(f11)
                t10a, t10b = self._to_vt(self.get_t1(freq=w0))
                t11a, t11b = self._to_vt(self.get_t1(freq=w1))
                s1vta, s1vtb = self._to_vt(s1)
                s1voa, s1vob = self._to_vo(s1)
                s2voa, s2vob = self._to_vo(self.get_s2())
                t2voa, t2vob = self._to_vo(self.get_t2(freq))
                
                rhsa += s2voa * e0a
                rhsb += s2vob * e0b
                tmpa  = lib.einsum('xjl,yli,i->xyji', s1vta, mo11a, e0a)
                tmpb  = lib.einsum('xjl,yli,i->xyji', s1vtb, mo11b, e0b)
                tmpa += lib.einsum('yjl,xli,i->xyji', s1vta, mo10a, e0a)
                tmpb += lib.einsum('yjl,xli,i->xyji', s1vtb, mo10b, e0b)
                tmpa += lib.einsum('xjk,yki->xyji', s1voa, e11a)
                tmpb += lib.einsum('xjk,yki->xyji', s1vob, e11b)
                tmpa += lib.einsum('yjk,xki->xyji', s1voa, e10a)
                tmpb += lib.einsum('yjk,xki->xyji', s1vob, e10b)

                rhsa += t2voa
                rhsb += t2vob
                tmpa -= lib.einsum('xjl,yli->xyji', s1vta, mo11a) * w1
                tmpb -= lib.einsum('xjl,yli->xyji', s1vtb, mo11b) * w1
                tmpa -= lib.einsum('yjl,xli->xyji', s1vta, mo10a) * w0
                tmpb -= lib.einsum('yjl,xli->xyji', s1vtb, mo10b) * w0
                tmpa += lib.einsum('xjl,yli->xyji', t10a, mo11a)
                tmpb += lib.einsum('xjl,yli->xyji', t10b, mo11b)
                tmpa += lib.einsum('yjl,xli->xyji', t11a, mo10a)
                tmpb += lib.einsum('yjl,xli->xyji', t11b, mo10b)

                tmpa += lib.einsum('xjk,yki->xyji', mo10a[:,~occidxa], e11a)
                tmpb += lib.einsum('xjk,yki->xyji', mo10b[:,~occidxb], e11b)
                tmpa += lib.einsum('yjk,xki->xyji', mo11a[:,~occidxa], e10a)
                tmpb += lib.einsum('yjk,xki->xyji', mo11b[:,~occidxb], e10b)
            else:
                f10a, f10b = self._to_vv(f10)
                f11a, f11b = self._to_vv(f11)
                tmpa  = lib.einsum('xjk,yki->xyji', mo10a, e11a)
                tmpb  = lib.einsum('xjk,yki->xyji', mo10b, e11b)
                tmpa += lib.einsum('yjk,xki->xyji', mo11a, e10a)
                tmpb += lib.einsum('yjk,xki->xyji', mo11b, e10b)
            tmpa -= lib.einsum('xjl,yli->xyji', f10a, mo11a)
            tmpb -= lib.einsum('xjl,yli->xyji', f10b, mo11b)
            tmpa -= lib.einsum('yjl,xli->xyji', f11a, mo10a)
            tmpb -= lib.einsum('yjl,xli->xyji', f11b, mo10b)

            rhsa += tmpa.reshape(9,*tmpa.shape[-2:])
            rhsb += tmpb.reshape(9,*tmpb.shape[-2:])
        
        return (rhsa, rhsb)

    def _krylov_solver(self, freq=0, verbose=logger.WARN, **kwargs):
        '''UCP-HF/KS subspace projection solver by `pyscf.lib.krylov`.'''
        log = logger.new_logger(verbose=verbose)
        t0 = (logger.process_clock(), logger.perf_counter())
        e0voa, e0vob = self.get_e0vo()
        nvira, nocca = e0voa.shape
        nvirb, noccb = e0vob.shape
        # first-order solver
        if isinstance(freq, (int, float)): # mo1[0] = Ua, mo1[1] = Ub
            if freq == 0:
                def lhs(mo1): # U(0)
                    mo1 = mo1.reshape(3,-1)
                    mo1a, mo1b = numpy.hsplit(mo1, [nvira*nocca])
                    mo1 = (mo1a.reshape(3,nvira,nocca),
                           mo1b.reshape(3,nvirb,noccb))
                    v1 = self.get_vind(mo1, freq, **kwargs)
                    v1a, v1b = self._to_vo(v1)
                    v1a /= e0voa
                    v1b /= e0vob
                    v1 = numpy.hstack((v1a.reshape(3,-1),
                                       v1b.reshape(3,-1)))
                    return v1.ravel()
                
                rhsa, rhsb = self._rhs1(freq, **kwargs)
                rhsa /= e0voa
                rhsb /= e0vob
                rhs = numpy.hstack((rhsa.reshape(3,-1),
                                    rhsb.reshape(3,-1)))
                rhs = rhs.ravel()
                # casting multiple vectors into Krylov solver yields poor and inefficient results
                mo1 = lib.krylov(lhs, rhs, max_cycle=self.max_cycle,
                                 tol=self.conv_tol, verbose=log)
                mo1 = mo1.reshape(3,-1)
                mo1a, mo1b = numpy.hsplit(mo1, [nvira*nocca])
                mo1a = mo1a.reshape(3,nvira,nocca)
                mo1b = mo1b.reshape(3,nvirb,noccb)
                if self.with_s1:
                    mo1ooa, mo1oob = self._to_oo(self.get_s1()) * -.5
                    mo1a = numpy.concatenate((mo1ooa, mo1a), axis=-2)
                    mo1b = numpy.concatenate((mo1oob, mo1b), axis=-2)
            
            else:
                def lhs(mo1): # mo1[0] = U(w), mo1[1] = U*(-w)
                    mo1 = mo1.reshape(3,-1)
                    mo1a, mo1b = numpy.hsplit(mo1, [nvira*nocca*2])
                    mo1 = (mo1a.reshape(3,2,nvira,nocca).swapaxes(0,1),
                           mo1b.reshape(3,2,nvirb,noccb).swapaxes(0,1))
                    v1 = self.get_vind(mo1, freq, **kwargs)
                    v1pa, v1pb = self._to_vo(v1) # v1 shape: (2,3,nao,nao)
                    v1ma, v1mb = self._to_vo(v1.transpose(0,1,3,2).conj())
                    v1pa /= (e0voa + freq)
                    v1ma /= (e0voa - freq)
                    v1pb /= (e0vob + freq)
                    v1mb /= (e0vob - freq)
                    v1 = numpy.hstack((v1pa       .reshape(3,-1),
                                       v1ma.conj().reshape(3,-1),
                                       v1pb       .reshape(3,-1),
                                       v1mb.conj().reshape(3,-1)))
                    return v1.ravel()
                
                if not self.with_s1:
                    rhs = self._rhs1(freq, **kwargs)
                    rhs = numpy.hstack(((rhs[0]/(e0voa+freq)).reshape(3,-1),
                                        (rhs[0]/(e0voa-freq)).reshape(3,-1).conj(),
                                        (rhs[1]/(e0vob+freq)).reshape(3,-1),
                                        (rhs[1]/(e0vob-freq)).reshape(3,-1).conj()))
                else:
                    rhs = (self._rhs1( freq, **kwargs),
                           self._rhs1(-freq, **kwargs))
                    rhs = numpy.hstack(((rhs[0][0]/(e0voa+freq)).reshape(3,-1),
                                        (rhs[1][0]/(e0voa-freq)).reshape(3,-1).conj(),
                                        (rhs[0][1]/(e0vob+freq)).reshape(3,-1),
                                        (rhs[1][1]/(e0vob-freq)).reshape(3,-1).conj()))
                rhs = rhs.ravel()
                # casting multiple vectors into Krylov solver yields poor and inefficient results
                mo1 = lib.krylov(lhs, rhs, max_cycle=self.max_cycle,
                                 tol=self.conv_tol, verbose=log)
                mo1 = mo1.reshape(3,-1)
                mo1a, mo1b = numpy.hsplit(mo1, [nvira*nocca*2])
                mo1a = mo1a.reshape(3,2,nvira,nocca).swapaxes(0,1)
                mo1b = mo1b.reshape(3,2,nvirb,noccb).swapaxes(0,1)
                if self.with_s1:
                    mo1ooa, mo1oob = self._to_oo(self.get_s1()) * -.5
                    mo1ooa = numpy.array((mo1ooa, mo1ooa.transpose(0,2,1)))
                    mo1oob = numpy.array((mo1oob, mo1oob.transpose(0,2,1)))
                    # transpose() is more efficient than conj() for a hermitian matrix
                    mo1a = numpy.concatenate((mo1ooa, mo1a), axis=-2)
                    mo1b = numpy.concatenate((mo1oob, mo1b), axis=-2)
            
            log.timer('Krylov solver for the first-order UCP-HF/KS', *t0)
            return (mo1a, mo1b)
        # second-order solver
        elif len(freq) == 2: # mo2[0] = Ua, mo2[1] = Ub
            if freq == (0,0):
                def lhs(mo2): # U(0,0)
                    mo2 = mo2.reshape(9,-1)
                    mo2a, mo2b = numpy.hsplit(mo2, [nvira*nocca])
                    mo2 = (mo2a.reshape(9,nvira,nocca),
                           mo2b.reshape(9,nvirb,noccb))
                    v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                    v2a, v2b = self._to_vo(v2)
                    v2a /= e0voa
                    v2b /= e0vob
                    v2 = numpy.hstack((v2a.reshape(9,-1),
                                       v2b.reshape(9,-1)))
                    return v2.ravel()
                
                rhsa, rhsb = self._rhs2(freq, **kwargs)
                rhsa /= e0voa
                rhsb /= e0vob
                rhs = numpy.hstack((rhsa.reshape(9,-1),
                                    rhsb.reshape(9,-1)))
                rhs = rhs.ravel()
                # casting multiple vectors into Krylov solver yields poor and inefficient results
                mo2 = lib.krylov(lhs, rhs, max_cycle=self.max_cycle,
                                 tol=self.conv_tol, verbose=log)
                mo2 = mo2.reshape(9,-1)
                mo2a, mo2b = numpy.hsplit(mo2, [nvira*nocca])
                mo2a = mo2a.reshape(9,nvira,nocca)
                mo2b = mo2b.reshape(9,nvirb,noccb)
            
            else:
                def lhs(mo2): # mo2[0] = U(w1,w2), mo2[1] = U*(-w1,-w2)
                    mo2 = mo2.reshape(9,-1)
                    mo2a, mo2b = numpy.hsplit(mo2, [nvira*nocca*2])
                    mo2 = (mo2a.reshape(9,2,nvira,nocca).swapaxes(0,1),
                           mo2b.reshape(9,2,nvirb,noccb).swapaxes(0,1))
                    v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                    v2pa, v2pb = self._to_vo(v2) # v2 shape: (2,9,nao,nao)
                    v2ma, v2mb = self._to_vo(v2.transpose(0,1,3,2).conj())
                    v2pa /= (e0voa + freq[0] + freq[1])
                    v2ma /= (e0voa - freq[0] - freq[1])
                    v2pb /= (e0vob + freq[0] + freq[1])
                    v2mb /= (e0vob - freq[0] - freq[1])
                    v2 = numpy.hstack((v2pa       .reshape(9,-1),
                                       v2ma.conj().reshape(9,-1),
                                       v2pb       .reshape(9,-1),
                                       v2mb.conj().reshape(9,-1)))
                    return v2.ravel()
                
                rhs = (self._rhs2(freq, **kwargs),
                       self._rhs2((-freq[0],-freq[1]), **kwargs))
                rhs = numpy.hstack(((rhs[0][0]       /(e0voa+freq[0]+freq[1])).reshape(9,-1),
                                    (rhs[1][0].conj()/(e0voa-freq[0]-freq[1])).reshape(9,-1),
                                    (rhs[0][1]       /(e0vob+freq[0]+freq[1])).reshape(9,-1),
                                    (rhs[1][1].conj()/(e0vob-freq[0]-freq[1])).reshape(9,-1)))
                rhs = rhs.ravel()
                # casting multiple vectors into Krylov solver yields poor and inefficient results
                mo2 = lib.krylov(lhs, rhs, max_cycle=self.max_cycle,
                                 tol=self.conv_tol, verbose=log)
                mo2 = mo2.reshape(9,-1)
                mo2a, mo2b = numpy.hsplit(mo2, [nvira*nocca*2])
                mo2a = mo2a.reshape(9,2,nvira,nocca).swapaxes(0,1)
                mo2b = mo2b.reshape(9,2,nvirb,noccb).swapaxes(0,1)
            
            mo2ooa, mo2oob = self._mo2oo(freq, **kwargs)
            mo2a = numpy.concatenate((mo2ooa, mo2a), axis=-2)
            mo2b = numpy.concatenate((mo2oob, mo2b), axis=-2)

            log.timer('Krylov solver for the second-order UCPHF/KS', *t0)
            return (mo2a, mo2b)
        
        else:
            raise NotImplementedError(freq)

    def _newton_solver(self, freq=0, verbose=logger.WARN, **kwargs):
        '''CP-HF/KS iterative solver by `scipy.optimize.newton_krylov`.'''
        log = logger.new_logger(verbose=verbose)
        t0 = (logger.process_clock(), logger.perf_counter())

        e0voa, e0vob = self.get_e0vo()
        nvira, nocca = e0voa.shape
        nvirb, noccb = e0vob.shape
        # first-order solver
        if isinstance(freq, (int, float)):
            if freq == 0:
                rhsa, rhsb = self._rhs1(freq, **kwargs)
                rhs = numpy.hstack((rhsa.reshape(3,-1), rhsb.reshape(3,-1)))
                
                def lhs(mo1): # U(0)
                    mo1a, mo1b = numpy.hsplit(mo1, [nvira*nocca])
                    mo1 = (mo1a.reshape(3,nvira,nocca),
                           mo1b.reshape(3,nvirb,noccb))
                    v1 = self.get_vind(mo1, freq, **kwargs)
                    v1a, v1b = self._to_vo(v1)
                    v1a += e0voa * mo1[0]
                    v1b += e0vob * mo1[1]
                    v1 = numpy.hstack((v1a.reshape(3,-1),
                                       v1b.reshape(3,-1)))
                    return v1 - rhs
                
                mo1 = newton_krylov(lhs, rhs, maxiter=self.max_cycle, f_tol=self.conv_tol)
                mo1a, mo1b = numpy.hsplit(mo1, [nvira*nocca])
                mo1a = mo1a.reshape(3,nvira,nocca)
                mo1b = mo1b.reshape(3,nvirb,noccb)

            else:
                if not self.with_s1:
                    rhs = self._rhs1(freq, **kwargs)
                    rhs = numpy.hstack((rhs[0].reshape(3,-1),
                                        rhs[0].reshape(3,-1).conj(),
                                        rhs[1].reshape(3,-1),
                                        rhs[1].reshape(3,-1).conj()))
                else:
                    rhs = (self._rhs1( freq, **kwargs),
                           self._rhs1(-freq, **kwargs))
                    rhs = numpy.hstack((rhs[0][0].reshape(3,-1),
                                        rhs[1][0].reshape(3,-1).conj(),
                                        rhs[0][1].reshape(3,-1),
                                        rhs[1][1].reshape(3,-1).conj())) 
                
                def lhs(mo1): # mo1[0] = U(w), mo1[1] = U*(-w)
                    mo1a, mo1b = numpy.hsplit(mo1, [nvira*nocca*2])
                    mo1 = (mo1a.reshape(3,2,nvira,nocca).swapaxes(0,1),
                           mo1b.reshape(3,2,nvirb,noccb).swapaxes(0,1))
                    v1 = self.get_vind(mo1, freq, **kwargs)
                    v1pa, v1pb = self._to_vo(v1) # v1 shape: (2,3,nao,nao)
                    v1ma, v1mb = self._to_vo(v1.transpose(0,1,3,2).conj())
                    v1pa += (e0voa + freq) * mo1[0][0]
                    v1ma += (e0voa - freq) * mo1[0][1].conj()
                    v1pb += (e0vob + freq) * mo1[1][0]
                    v1mb += (e0vob - freq) * mo1[1][1].conj()
                    v1 = numpy.hstack((v1pa       .reshape(3,-1),
                                       v1ma.conj().reshape(3,-1),
                                       v1pb       .reshape(3,-1),
                                       v1mb.conj().reshape(3,-1)))
                    return v1 - rhs
                
                mo1 = newton_krylov(lhs, rhs, maxiter=self.max_cycle, f_tol=self.conv_tol)
                mo1a, mo1b = numpy.hsplit(mo1, [nvira*nocca*2])
                mo1a = mo1a.reshape(3,2,nvira,nocca).swapaxes(0,1)
                mo1b = mo1b.reshape(3,2,nvirb,noccb).swapaxes(0,1)
            
            if self.with_s1:
                mo1ooa, mo1oob = self._to_oo(self.get_s1()) * -.5
                if freq != 0:
                    mo1ooa = numpy.array((mo1ooa, mo1ooa.transpose(0,2,1)))
                    mo1oob = numpy.array((mo1oob, mo1oob.transpose(0,2,1)))
                mo1a = numpy.concatenate((mo1ooa, mo1a), axis=-2)
                mo1b = numpy.concatenate((mo1oob, mo1b), axis=-2)
        
            log.timer('Newton-Krylov solver for the first-order UCP-HF/KS', *t0)
            return (mo1a, mo1b)
        # second-order solver
        elif len(freq) == 2:
            if freq == (0,0):
                rhsa, rhsb = self._rhs2(freq, **kwargs)
                rhs = numpy.hstack((rhsa.reshape(9,-1), rhsb.reshape(9,-1)))
                
                def lhs(mo2): # U(0,0)
                    mo2a, mo2b = numpy.hsplit(mo2, [nvira*nocca])
                    mo2 = (mo2a.reshape(9,nvira,nocca),
                           mo2b.reshape(9,nvirb,noccb))
                    v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                    v2a, v2b = self._to_vo(v2)
                    v2a += e0voa * mo2[0]
                    v2b += e0vob * mo2[1]
                    v2 = numpy.hstack((v2a.reshape(9,-1),
                                       v2b.reshape(9,-1)))
                    return v2 - rhs
                
                mo2 = newton_krylov(lhs, rhs, maxiter=self.max_cycle, f_tol=self.conv_tol)
                mo2a, mo2b = numpy.hsplit(mo2, [nvira*nocca])
                mo2a = mo2a.reshape(9,nvira,nocca)
                mo2b = mo2b.reshape(9,nvirb,noccb)
                
            else:
                rhs = (self._rhs2(freq, **kwargs),
                       self._rhs2((-freq[0],-freq[1]), **kwargs))
                rhs = numpy.hstack((rhs[0][0].reshape(9,-1),
                                    rhs[1][0].reshape(9,-1).conj(),
                                    rhs[0][1].reshape(9,-1),
                                    rhs[1][1].reshape(9,-1).conj()))
                
                def lhs(mo2): # mo2[0] = U(w1,w2), mo2[1] = U*(-w1,-w2)
                    mo2a, mo2b = numpy.hsplit(mo2, [nvira*nocca*2])
                    mo2 = (mo2a.reshape(9,2,nvira,nocca).swapaxes(0,1),
                           mo2b.reshape(9,2,nvirb,noccb).swapaxes(0,1))
                    v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                    v2pa, v2pb = self._to_vo(v2) # v2 shape: (2,9,nao,nao)
                    v2ma, v2mb = self._to_vo(v2.transpose(0,1,3,2).conj())
                    v2pa += (e0voa + freq[0] + freq[1]) * mo2[0][0]
                    v2ma += (e0voa - freq[0] - freq[1]) * mo2[0][1].conj()
                    v2pb += (e0vob + freq[0] + freq[1]) * mo2[1][0]
                    v2mb += (e0vob - freq[0] - freq[1]) * mo2[1][1].conj()
                    v2 = numpy.hstack((v2pa       .reshape(9,-1),
                                       v2ma.conj().reshape(9,-1),
                                       v2pb       .reshape(9,-1),
                                       v2mb.conj().reshape(9,-1)))
                    return v2 - rhs
                
                mo2 = newton_krylov(lhs, rhs, maxiter=self.max_cycle, f_tol=self.conv_tol)
                mo2a, mo2b = numpy.hsplit(mo2, [nvira*nocca*2])
                mo2a = mo2a.reshape(9,2,nvira,nocca).swapaxes(0,1)
                mo2b = mo2b.reshape(9,2,nvirb,noccb).swapaxes(0,1)
            
            mo2ooa, mo2oob = self._mo2oo(freq, **kwargs)
            mo2a = numpy.concatenate((mo2ooa, mo2a), axis=-2)
            mo2b = numpy.concatenate((mo2oob, mo2b), axis=-2)

            log.timer('Newton-Krylov solver for the second-order UCP-HF/KS', *t0)
            return (mo2a, mo2b)
        
        else:
            raise NotImplementedError(freq)

    def _direct_solver(self, freq=0, verbose=logger.WARN, **kwargs):
        '''CP-HF/KS direct solver by `numpy.linalg.solve`.'''
        log = logger.new_logger(verbose=verbose)
        t0 = (logger.process_clock(), logger.perf_counter())

        e0voa, e0vob = self.get_e0vo()
        nvira, nocca = e0voa.shape
        nvirb, noccb = e0vob.shape
        # first-order solver
        if isinstance(freq, (int, float)):
            def lhs(mo1): # mo1[0] = U(w), mo1[1] = U*(-w)
                mo1a, mo1b = numpy.split(mo1, [nvira*nocca*2])
                mo1 = (mo1a.reshape(2,1,nvira,nocca),
                       mo1b.reshape(2,1,nvirb,noccb))
                v1 = self.get_vind(mo1, freq, **kwargs)
                v1pa, v1pb = self._to_vo(v1) # v1 shape: (2,1,nao,nao)
                v1ma, v1mb = self._to_vo(v1.transpose(0,1,3,2).conj())
                v1pa += (e0voa + freq) * mo1[0][0]
                v1ma += (e0voa - freq) * mo1[0][1].conj()
                v1pb += (e0vob + freq) * mo1[1][0]
                v1mb += (e0vob - freq) * mo1[1][1].conj()
                v1 = numpy.concatenate((v1pa       .ravel(),
                                        v1ma.conj().ravel(),
                                        v1pb       .ravel(),
                                        v1mb.conj().ravel()))
                return v1
            
            if not self.with_s1:
                rhs = self._rhs1(freq, **kwargs)
                rhs = numpy.hstack((rhs[0].reshape(3,-1),
                                    rhs[0].reshape(3,-1).conj(),
                                    rhs[1].reshape(3,-1),
                                    rhs[1].reshape(3,-1).conj()))
            else:
                rhs = (self._rhs1( freq, **kwargs),
                       self._rhs1(-freq, **kwargs))
                rhs = numpy.hstack((rhs[0][0].reshape(3,-1),
                                    rhs[1][0].reshape(3,-1).conj(),
                                    rhs[0][1].reshape(3,-1),
                                    rhs[1][1].reshape(3,-1).conj()))
            size = rhs[0].size
            operator = numpy.empty((size, size))
            iden = numpy.eye(size)
            for i, row in enumerate(iden):
                operator[:,i] = lhs(row)
            
            mo1 = numpy.linalg.solve(operator, rhs.T).T
            mo1a, mo1b = numpy.hsplit(mo1, [nvira*nocca*2])
            mo1a = mo1a.reshape(3,2,nvira,nocca).swapaxes(0,1)
            mo1b = mo1b.reshape(3,2,nvirb,noccb).swapaxes(0,1)
            if self.with_s1:
                mo1ooa, mo1oob = self._to_oo(self.get_s1()) * -.5
                mo1ooa = numpy.array((mo1ooa, mo1ooa.transpose(0,2,1)))
                mo1oob = numpy.array((mo1oob, mo1oob.transpose(0,2,1)))
                # transpose() is more efficient than conj() for a hermitian matrix
                mo1a = numpy.concatenate((mo1ooa, mo1a), axis=-2)
                mo1b = numpy.concatenate((mo1oob, mo1b), axis=-2)
            
            log.timer('Direct solver for the first-order UCP-HF/KS', *t0)
            return (mo1a[0], mo1b[0]) if freq == 0 else (mo1a, mo1b)
        # second-order solver
        elif len(freq) == 2:
            def lhs(mo2): # mo2[0] = U(w1,w2), mo2[1] = U*(-w1,-w2)
                mo2a, mo2b = numpy.split(mo2, [nvira*nocca*2])
                mo2 = (mo2a.reshape(2,1,nvira,nocca),
                       mo2b.reshape(2,1,nvirb,noccb))
                v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                v2pa, v2pb = self._to_vo(v2) # v2 shape: (2,1,nao,nao)
                v2ma, v2mb = self._to_vo(v2.transpose(0,1,3,2).conj())
                v2pa += (e0voa + freq[0] + freq[1]) * mo2[0][0]
                v2ma += (e0voa - freq[0] - freq[1]) * mo2[0][1].conj()
                v2pb += (e0vob + freq[0] + freq[1]) * mo2[1][0]
                v2mb += (e0vob - freq[0] - freq[1]) * mo2[1][1].conj()
                v2 = numpy.concatenate((v2pa       .ravel(),
                                        v2ma.conj().ravel(),
                                        v2pb       .ravel(),
                                        v2mb.conj().ravel()))
                return v2
            
            rhs = (self._rhs2(freq, **kwargs),
                   self._rhs2((-freq[0],-freq[1]), **kwargs))
            rhs = numpy.hstack((rhs[0][0].reshape(9,-1),
                                rhs[1][0].reshape(9,-1).conj(),
                                rhs[0][1].reshape(9,-1),
                                rhs[1][1].reshape(9,-1).conj()))
            size = rhs[0].size
            operator = numpy.empty((size, size))
            iden = numpy.eye(size)
            for i, row in enumerate(iden):
                operator[:,i] = lhs(row)
            
            mo2 = numpy.linalg.solve(operator, rhs.T).T
            mo2a, mo2b = numpy.hsplit(mo2, [nvira*nocca*2])
            mo2a = mo2a.reshape(9,2,nvira,nocca).swapaxes(0,1)
            mo2b = mo2b.reshape(9,2,nvirb,noccb).swapaxes(0,1)
            if freq == (0,0):
                mo2a = mo2a[0]
                mo2b = mo2b[0]
            mo2ooa, mo2oob = self._mo2oo(freq, **kwargs)
            mo2a = numpy.concatenate((mo2ooa, mo2a), axis=-2)
            mo2b = numpy.concatenate((mo2oob, mo2b), axis=-2)

            log.timer('Direct solver for the second-order UCP-HF/KS', *t0)
            return (mo2a, mo2b)
        
        else:
            raise NotImplementedError(freq)            

    def _to_vo(self, ao):
        '''Convert some quantity in AO basis to that in vir.-occ. MO basis.'''
        mf = self.mf
        mo_coeff = mf.mo_coeff
        occidxa = mf.mo_occ[0] > 0
        occidxb = mf.mo_occ[1] > 0
        orbva = mo_coeff[0][:,~occidxa]
        orboa = mo_coeff[0][:, occidxa]
        orbvb = mo_coeff[1][:,~occidxb]
        orbob = mo_coeff[1][:, occidxb]
        
        if ao.ndim == 2:
            voa = orbva.T.conj() @ ao @ orboa
            vob = orbvb.T.conj() @ ao @ orbob
        elif ao.ndim == 3:
            if len(ao) == 2:
                voa = lib.einsum('pj,pq,qi->ji', orbva.conj(), ao[0], orboa)
                vob = lib.einsum('pj,pq,qi->ji', orbvb.conj(), ao[1], orbob)
            else:
                voa = lib.einsum('pj,xpq,qi->xji', orbva.conj(), ao, orboa)
                vob = lib.einsum('pj,xpq,qi->xji', orbvb.conj(), ao, orbob)
        elif ao.ndim == 4:
            if len(ao) == 2:
                voa = lib.einsum('pj,xpq,qi->xji', orbva.conj(), ao[0], orboa)
                vob = lib.einsum('pj,xpq,qi->xji', orbvb.conj(), ao[1], orbob)
            else:
                voa = lib.einsum('pj,xypq,qi->xyji', orbva.conj(), ao, orboa)
                vob = lib.einsum('pj,xypq,qi->xyji', orbvb.conj(), ao, orbob)
        elif ao.ndim == 5:
            if len(ao) == 2:
                voa = lib.einsum('pj,xypq,qi->xyji', orbva.conj(), ao[0], orboa)
                vob = lib.einsum('pj,xypq,qi->xyji', orbvb.conj(), ao[1], orbob)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        return (voa, vob)
    
    def _to_vv(self, ao):
        '''Convert some quantity in AO basis to that in vir.-occ. MO basis.'''
        mf = self.mf
        mo_coeff = mf.mo_coeff
        viridxa = mf.mo_occ[0] == 0
        viridxb = mf.mo_occ[1] == 0
        orbva = mo_coeff[0][:, viridxa]
        orbvb = mo_coeff[1][:, viridxb]

        if ao.ndim == 2:
            vva = orbva.T.conj() @ ao @ orbva
            vvb = orbvb.T.conj() @ ao @ orbvb
        elif ao.ndim == 3:
            if len(ao) == 2:
                vva = lib.einsum('pj,pq,qi->ji', orbva.conj(), ao[0], orbva)
                vvb = lib.einsum('pj,pq,qi->ji', orbvb.conj(), ao[1], orbvb)
            else:
                vva = lib.einsum('pj,xpq,qi->xji', orbva.conj(), ao, orbva)
                vvb = lib.einsum('pj,xpq,qi->xji', orbvb.conj(), ao, orbvb)
        elif ao.ndim == 4:
            if len(ao) == 2:
                vva = lib.einsum('pj,xpq,qi->xji', orbva.conj(), ao[0], orbva)
                vvb = lib.einsum('pj,xpq,qi->xji', orbvb.conj(), ao[1], orbvb)
            else:
                vva = lib.einsum('pj,xypq,qi->xyji', orbva.conj(), ao, orbva)
                vvb = lib.einsum('pj,xypq,qi->xyji', orbvb.conj(), ao, orbvb)
        elif ao.ndim == 5:
            if len(ao) == 2:
                vva = lib.einsum('pj,xypq,qi->xyji', orbva.conj(), ao[0], orbva)
                vvb = lib.einsum('pj,xypq,qi->xyji', orbvb.conj(), ao[1], orbvb)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        return (vva, vvb)
    
    def _to_oo(self, ao):
        '''Convert some quantity in AO basis to that in occ.-occ. MO basis.'''
        mf = self.mf
        mo_coeff = mf.mo_coeff
        occidxa = mf.mo_occ[0] > 0
        occidxb = mf.mo_occ[1] > 0
        orboa = mo_coeff[0][:, occidxa]
        orbob = mo_coeff[1][:, occidxb]

        if ao.ndim == 2:
            ooa = orboa.T.conj() @ ao @ orboa
            oob = orbob.T.conj() @ ao @ orbob
        elif ao.ndim == 3:
            if len(ao) == 2:
                ooa = lib.einsum('pj,pq,qi->ji', orboa.conj(), ao[0], orboa)
                oob = lib.einsum('pj,pq,qi->ji', orbob.conj(), ao[1], orbob)
            else:
                ooa = lib.einsum('pj,xpq,qi->xji', orboa.conj(), ao, orboa)
                oob = lib.einsum('pj,xpq,qi->xji', orbob.conj(), ao, orbob)
        elif ao.ndim == 4:
            if len(ao) == 2:
                ooa = lib.einsum('pj,xpq,qi->xji', orboa.conj(), ao[0], orboa)
                oob = lib.einsum('pj,xpq,qi->xji', orbob.conj(), ao[1], orbob)
            else:
                ooa = lib.einsum('pj,xypq,qi->xyji', orboa.conj(), ao, orboa)
                oob = lib.einsum('pj,xypq,qi->xyji', orbob.conj(), ao, orbob)
        elif ao.ndim == 5:
            if len(ao) == 2:
                ooa = lib.einsum('pj,xypq,qi->xyji', orboa.conj(), ao[0], orboa)
                oob = lib.einsum('pj,xypq,qi->xyji', orbob.conj(), ao[1], orbob)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        return (ooa, oob)
    
    def _to_to(self, ao):
        '''Convert some quantity in AO basis to that in tot.-occ. MO basis.'''
        mf = self.mf
        mo_coeff = mf.mo_coeff
        occidxa = mf.mo_occ[0] > 0
        occidxb = mf.mo_occ[1] > 0
        orboa = mo_coeff[0][:, occidxa]
        orbob = mo_coeff[1][:, occidxb]

        if ao.ndim == 2:
            toa = mo_coeff[0].T.conj() @ ao @ orboa
            tob = mo_coeff[1].T.conj() @ ao @ orbob
        elif ao.ndim == 3:
            if len(ao) == 2:
                toa = lib.einsum('pj,pq,qi->ji', mo_coeff[0].conj(), ao[0], orboa)
                tob = lib.einsum('pj,pq,qi->ji', mo_coeff[1].conj(), ao[1], orbob)
            else:
                toa = lib.einsum('pj,xpq,qi->xji', mo_coeff[0].conj(), ao, orboa)
                tob = lib.einsum('pj,xpq,qi->xji', mo_coeff[1].conj(), ao, orbob)
        elif ao.ndim == 4:
            if len(ao) == 2:
                toa = lib.einsum('pj,xpq,qi->xji', mo_coeff[0].conj(), ao[0], orboa)
                tob = lib.einsum('pj,xpq,qi->xji', mo_coeff[1].conj(), ao[1], orbob)
            else:
                toa = lib.einsum('pj,xypq,qi->xyji', mo_coeff[0].conj(), ao, orboa)
                tob = lib.einsum('pj,xypq,qi->xyji', mo_coeff[1].conj(), ao, orbob)
        elif ao.ndim == 5:
            if len(ao) == 2:
                toa = lib.einsum('pj,xypq,qi->xyji', mo_coeff[0].conj(), ao[0], orboa)
                tob = lib.einsum('pj,xypq,qi->xyji', mo_coeff[1].conj(), ao[1], orbob)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        return (toa, tob)
    
    def _to_vt(self, ao):
        '''Convert some quantity in AO basis to that in vir.-tot. MO basis.'''
        mf = self.mf
        mo_coeff = mf.mo_coeff
        viridxa = mf.mo_occ[0] == 0
        viridxb = mf.mo_occ[1] == 0
        orbva = mo_coeff[0][:, viridxa]
        orbvb = mo_coeff[1][:, viridxb]

        if ao.ndim == 2:
            vta = orbva.T.conj() @ ao @ mo_coeff[0]
            vtb = orbvb.T.conj() @ ao @ mo_coeff[1]
        elif ao.ndim == 3:
            if len(ao) == 2:
                vta = lib.einsum('pj,pq,qi->ji', orbva.conj(), ao[0], mo_coeff[0])
                vtb = lib.einsum('pj,pq,qi->ji', orbvb.conj(), ao[1], mo_coeff[1])
            else:
                vta = lib.einsum('pj,xpq,qi->xji', orbva.conj(), ao, mo_coeff[0])
                vtb = lib.einsum('pj,xpq,qi->xji', orbvb.conj(), ao, mo_coeff[1])
        elif ao.ndim == 4:
            if len(ao) == 2:
                vta = lib.einsum('pj,xpq,qi->xji', orbva.conj(), ao[0], mo_coeff[0])
                vtb = lib.einsum('pj,xpq,qi->xji', orbvb.conj(), ao[1], mo_coeff[1])
            else:
                vta = lib.einsum('pj,xypq,qi->xyji', orbva.conj(), ao, mo_coeff[0])
                vtb = lib.einsum('pj,xypq,qi->xyji', orbvb.conj(), ao, mo_coeff[1])
        elif ao.ndim == 5:
            if len(ao) == 2:
                vta = lib.einsum('pj,xypq,qi->xyji', orbva.conj(), ao[0], mo_coeff[0])
                vtb = lib.einsum('pj,xypq,qi->xyji', orbvb.conj(), ao[1], mo_coeff[1])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        return (vta, vtb)
