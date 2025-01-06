import numpy
from scipy.optimize import newton_krylov
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, _response_functions
from pyscf.x2c.x2c import X2C1E_GSCF

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
        
        if freq == 0 and len(mo1) != 2:
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
        dm22 = dm21 = 0

        if freq == (0,0) and (mo2 is None or len(mo2) != 2): # D(0,0)
            if with_mo2:
                if mo2.shape[-2] == orbv.shape[-1]:
                    dm22 = lib.einsum('pj,xyji,qi->xypq', orbv, mo2, orbo.conj())
                elif mo2.shape[-2] == orbo.shape[-1]:
                    dm22 = lib.einsum('pj,xyji,qi->xypq', orbo, mo2, orbo.conj())
                else:
                    dm22 = lib.einsum('pj,xyji,qi->xypq', mo_coeff, mo2, orbo.conj())
            if with_mo1:
                try: mo1 = self.mo1[0]
                except KeyError: mo1 = self.solve_mo1(freq=0, **kwargs)
                if self.with_s1: # mo1.shape = (3,ntot,nocc)
                    dm21 = lib.einsum('pj,xji,yki,qk->xypq', mo_coeff, mo1, mo1.conj(),
                                                             mo_coeff.conj())
                else:            # mo1.shape = (3,nvir,nocc)
                    dm21 = lib.einsum('pj,xji,yki,qk->xypq', orbv, mo1, mo1.conj(),
                                                             orbv.conj())
            dm2 = dm22 + dm21
            try: dm2 += dm2.transpose(0,1,3,2).conj()
            except SyntaxError: pass
        
        else: # mo2[0] = U(w1,w2), mo2[1] = U*(-w1,-w2)
            if with_mo2:
                if mo2.shape[-2] == orbv.shape[-1]:
                    dm22 = lib.einsum('pj,xyji,qi->xypq', orbv, mo2[0], orbo.conj())
                    dm22+= lib.einsum('pi,xyji,qj->xypq', orbo, mo2[1], orbv.conj())
                elif mo2.shape[-2] == orbo.shape[-1]:
                    dm22 = lib.einsum('pj,xyji,qi->xypq', orbo, mo2[0], orbo.conj())
                    dm22+= lib.einsum('pi,xyji,qj->xypq', orbo, mo2[1], orbo.conj())
                else:
                    dm22 = lib.einsum('pj,xyji,qi->xypq', mo_coeff, mo2[0], orbo.conj())
                    dm22+= lib.einsum('pi,xyji,qj->xypq', orbo, mo2[1], mo_coeff.conj())
            if with_mo1:
                if freq[0] == freq[1]: # D(w,w)
                    try: mo1 = self.mo1[freq[0]]
                    except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
                    if self.with_s1:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq',
                                          mo_coeff, mo1[0], mo1[1], mo_coeff.conj())
                    else:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq',
                                          orbv, mo1[0], mo1[1], orbv.conj())
                    dm21 += dm21.transpose(1,0,2,3)
                elif freq[0] == -freq[1]: # D(w,-w)
                    try: mo1 = self.mo1[freq[0]]
                    except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
                    if self.with_s1:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq',
                                          mo_coeff, mo1[0], mo1[0].conj(), mo_coeff.conj())
                        dm21+= lib.einsum('pj,xki,yji,qk->xypq',
                                          mo_coeff, mo1[1], mo1[1].conj(), mo_coeff.conj())
                    else:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq',
                                          orbv, mo1[0], mo1[0].conj(), orbv.conj())
                        dm21+= lib.einsum('pj,xki,yji,qk->xypq',
                                          orbv, mo1[1], mo1[1].conj(), orbv.conj())
                elif 0 in freq: # D(0,w) / D(w,0)
                    w = freq[0] if freq[0] != 0 else freq[1]
                    try: mo10 = self.mo1[0]
                    except KeyError: mo10 = self.solve_mo1(freq=0, **kwargs)
                    try: mo11 = self.mo1[w]
                    except KeyError: mo11 = self.solve_mo1(freq=w, **kwargs)
                    if self.with_s1:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq',
                                          mo_coeff, mo10, mo11[1], mo_coeff.conj())
                        dm21+= lib.einsum('pk,xji,yki,qj->xypq',
                                          mo_coeff, mo10.conj(), mo11[0], mo_coeff.conj())
                    else:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq',
                                          orbv, mo10, mo11[1], orbv.conj())
                        dm21+= lib.einsum('pk,xji,yki,qj->xypq',
                                          orbv, mo10.conj(), mo11[0], orbv.conj())
                    if freq.index(0) == 1: dm21 = dm21.transpose(1,0,2,3)
                else: # D(w1,w2)
                    w0 = freq[0]; w1 = freq[1]
                    try: mo10 = self.mo1[w0]
                    except KeyError: mo10 = self.solve_mo1(freq=w0, **kwargs)
                    try: mo11 = self.mo1[w1]
                    except KeyError: mo11 = self.solve_mo1(freq=w1, **kwargs)
                    if self.with_s1:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq',
                                          mo_coeff, mo10[0], mo11[1], mo_coeff.conj())
                        dm21+= lib.einsum('pk,xji,yki,qj->xypq',
                                          mo_coeff, mo10[1], mo11[0], mo_coeff.conj())
                    else:
                        dm21 = lib.einsum('pj,xji,yki,qk->xypq',
                                          orbv, mo10[0], mo11[1], orbv.conj())
                        dm21+= lib.einsum('pk,xji,yki,qj->xypq',
                                          orbv, mo10[1], mo11[0], orbv.conj())
            dm2 = dm22 + dm21
        
        return dm2*2 if isinstance(mf, hf.RHF) else dm2

    def get_vind(self, mo_deriv, freq, **kwargs):
        '''The induced potential matrix in AO basis.'''
        if isinstance(freq, (int, float)):
            dm_deriv = self.get_dm1(mo_deriv, freq, **kwargs)
            hermi = 1 if freq == 0 else 0
        elif len(freq) == 2:
            dm_deriv = self.get_dm2(mo_deriv, freq, **kwargs)
            hermi = 1 if all(f == 0 for f in freq) else 0
        else:
            raise NotImplementedError(freq)
        
        mf = self.mf
        vind = mf.gen_response(hermi=hermi)
        # GKS collinear response function does not support complex density matrices
        if not hermi and isinstance(mf, X2C1E_GSCF) and \
            hasattr(mf, 'collinear') and mf.collinear == 'col':
            dmr_deriv = dm_deriv.real
            dmi_deriv = dm_deriv.imag
            return vind(dmr_deriv) + vind(dmi_deriv)*1j
        
        return vind(dm_deriv)

    def get_f1(self, mo1=None, freq=0, **kwargs):
        '''The first-order Fock matrix in AO basis.'''
        if mo1 is None:
            try: mo1 = self.mo1[freq]
            except KeyError: mo1 = self.solve_mo1(freq=freq, **kwargs)
        f1 = self.get_h1(**kwargs)
        f1+= self.get_vind(mo1, freq, **kwargs)
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
                e1 -= self.get_t1(freq) * 1j
            e1 = self._to_oo(e1)
            mf = self.mf
            e0 = mf.mo_energy[mf.mo_occ>0]
            e0oo = (e0[:,None] - e0 + freq) * .5
            e1 -= self._to_oo(self.get_s1()) * e0oo
            return e1
        else:
            return self._to_oo(e1)
        
    def get_s1(self):
        '''The first-order overlap matrix in AO basis.'''
        mf = self.mf
        nao = mf.mo_coeff.shape[0]
        return numpy.zeros((3,nao,nao))
    
    def get_t1(self, freq):
        r'''( \mu | \partial_t | \nu )^(1).'''
        mf = self.mf
        nao = mf.mo_coeff.shape[0]
        return numpy.zeros((3,nao,nao))
    
    def get_f2(self, mo2=None, freq=(0,0), with_mo2=True, **kwargs):
        '''The second-order Fock matrix in AO basis.'''
        if mo2 is None and with_mo2:
            try: mo2 = self.mo2[freq]
            except KeyError: mo2 = self.solve_mo2(freq=freq, **kwargs)
        f2 = self.get_vind(mo2, freq, with_mo2=with_mo2, **kwargs)
        if hasattr(self, 'get_h2'):
            f2 += self.get_h2(**kwargs)
        if self.with_s1:
            f2 += self.get_jk2()
            f2 += self.get_jk11(freq)
        return f2

    def get_s2(self):
        '''The second-order overlap matrix in AO basis.'''
        mf = self.mf
        nao = mf.mo_coeff.shape[0]
        return numpy.zeros((3,3,nao,nao))
    
    def get_t2(self, freq):
        r'''( \mu | \partial_t | \nu )^(2).'''
        mf = self.mf
        nao = mf.mo_coeff.shape[0]
        return numpy.zeros((3,3,nao,nao))

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
        mo_energy = mf.mo_energy
        e0o = mo_energy[mf.mo_occ>0]
        e0to = mo_energy[:,None] - e0o
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
        if freq != 0: self.mo1[-freq] = mo1[[1,0]].conj()
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
        if freq != (0,0): self.mo2[(-freq[0],-freq[1])] = mo2[[1,0]].conj()
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
                w0 = freq[0]; w1 = freq[1]
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
            mo2oo = numpy.stack((mo2oo, mo2oo.transpose(0,1,3,2))) * -.5
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
                rhs += self.get_t1(freq) * 1j
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
                sym = lib.einsum('xjk,yki->xyji', self._to_vo(s1), e1)
                if freq[0] != 0:
                    mo1 = mo1[0]
                    t1 = self._to_vt(self.get_t1(freq[0]))
                    sym -= lib.einsum('xjl,yli->xyji', self._to_vt(s1), mo1) * freq[0]
                    sym += lib.einsum('xjl,yli->xyji', t1, mo1) * 1j
                    rhs += self._to_vo(self.get_t2(freq)) * 1j
                sym += lib.einsum('xjl,yli,i->xyji', self._to_vt(s1), mo1, e0)
                sym -= lib.einsum('xjl,yli->xyji', self._to_vt(f1), mo1)
                sym += lib.einsum('xjk,yki->xyji', mo1[:,~occidx], e1)
                sym += sym.transpose(1,0,2,3)
                rhs += sym + self._to_vo(self.get_s2()) * e0
            else:
                if freq[0] != 0:
                    mo1 = mo1[0]
                sym = -lib.einsum('xjl,yli->xyji', self._to_vv(f1), mo1)
                sym += lib.einsum('xjk,yki->xyji', mo1, e1)
                sym += sym.transpose(1,0,2,3)
                rhs += sym

        elif freq[0] == -freq[1]:
            try: mo1 = self.mo1[freq[0]]
            except KeyError: mo1 = self.solve_mo1(freq=freq[0], **kwargs)
            f1_p = self.get_f1(mo1, freq[0], **kwargs)
            f1_m = f1_p.transpose(0,2,1).conj()
            e1_p = self.get_e1(mo1, freq[0], **kwargs)
            e1_m = e1_p.transpose(0,2,1).conj()
            mo1[1] = mo1[1].conj()
            if self.with_s1: # mo1.shape = (3,ntot,nocc)
                mf = self.mf
                occidx = mf.mo_occ > 0
                e0 = mf.mo_energy[occidx]
                s1 = self.get_s1()
                t1_p = self.get_t1(freq[0]) * 1j
                t1_m = t1_p.transpose(0,2,1).conj()
                
                rhs += self._to_vo(self.get_s2()) * e0
                rhs += lib.einsum('xjl,yli,i->xyji', self._to_vt(s1), mo1[1], e0)
                rhs += lib.einsum('yjl,xli,i->xyji', self._to_vt(s1), mo1[0], e0)
                rhs += lib.einsum('xjk,yki->xyji', self._to_vo(s1), e1_m)
                rhs += lib.einsum('yjk,xki->xyji', self._to_vo(s1), e1_p)

                rhs -= lib.einsum('xjl,yli->xyji', self._to_vt(s1), mo1[1]) * freq[1]
                rhs -= lib.einsum('yjl,xli->xyji', self._to_vt(s1), mo1[0]) * freq[0]
                rhs += lib.einsum('xjl,yli->xyji', self._to_vt(t1_p), mo1[1])
                rhs += lib.einsum('yjl,xli->xyji', self._to_vt(t1_m), mo1[0])
                rhs += self._to_vo(self.get_t2(freq)) * 1j

                rhs -= lib.einsum('xjl,yli->xyji', self._to_vt(f1_p), mo1[1])
                rhs -= lib.einsum('yjl,xli->xyji', self._to_vt(f1_m), mo1[0])
                rhs += lib.einsum('xjk,yki->xyji', mo1[0,:,~occidx], e1_m)
                rhs += lib.einsum('yjk,xki->xyji', mo1[1,:,~occidx], e1_p)
            else:
                rhs -= lib.einsum('xjl,yli->xyji', self._to_vv(f1_p), mo1[1])
                rhs -= lib.einsum('yjl,xli->xyji', self._to_vv(f1_m), mo1[0])
                rhs += lib.einsum('xjk,yki->xyji', mo1[0], e1_m)
                rhs += lib.einsum('yjk,xki->xyji', mo1[1], e1_p)
                
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
                t1 = self.get_t1(freq=w)
                
                tmp = self._to_vo(self.get_s2()) * e0
                tmp += lib.einsum('xjl,yli,i->xyji', self._to_vt(s1), mo11, e0)
                tmp += lib.einsum('yjl,xli,i->xyji', self._to_vt(s1), mo10, e0)
                tmp += lib.einsum('xjk,yki->xyji', self._to_vo(s1), e11)
                tmp += lib.einsum('yjk,xki->xyji', self._to_vo(s1), e10)

                tmp -= lib.einsum('xjl,yli->xyji', self._to_vt(s1), mo11) * w
                tmp += lib.einsum('yjl,xli->xyji', self._to_vt(t1), mo10) * 1j
                tmp += self._to_vo(self.get_t2(freq=(0,w))) * 1j

                tmp -= lib.einsum('xjl,yli->xyji', self._to_vt(f10), mo11)
                tmp -= lib.einsum('yjl,xli->xyji', self._to_vt(f11), mo10)
                tmp += lib.einsum('xjk,yki->xyji', mo10[:,~occidx], e11)
                tmp += lib.einsum('yjk,xki->xyji', mo11[:,~occidx], e10)
            else:
                tmp = -lib.einsum('xjl,yli->xyji', self._to_vv(f10), mo11)
                tmp -= lib.einsum('yjl,xli->xyji', self._to_vv(f11), mo10)
                tmp += lib.einsum('xjk,yki->xyji', mo10, e11)
                tmp += lib.einsum('yjk,xki->xyji', mo11, e10)
            rhs += tmp.transpose(1,0,2,3) if freq.index(0) == 1 else tmp

        else:
            w0 = freq[0]; w1 = freq[1]
            try: mo10 = self.mo1[w0]
            except KeyError: mo10 = self.solve_mo1(freq=w0, **kwargs)
            try: mo11 = self.mo1[w1]
            except KeyError: mo11 = self.solve_mo1(freq=w1, **kwargs)
            f10 = self.get_f1(mo10, w0, **kwargs)
            f11 = self.get_f1(mo11, w1, **kwargs)
            e10 = self.get_e1(mo10, w0, **kwargs)
            e11 = self.get_e1(mo11, w1, **kwargs)
            mo10 = mo10[0]; mo11 = mo11[0]
            if self.with_s1: # mo1.shape = (3,ntot,nocc)
                mf = self.mf
                occidx = mf.mo_occ > 0
                e0 = mf.mo_energy[occidx]
                s1 = self.get_s1()
                t10 = self.get_t1(freq=w0)
                t11 = self.get_t1(freq=w1)
                
                tmp = self._to_vo(self.get_s2()) * e0
                tmp += lib.einsum('xjl,yli,i->xyji', self._to_vt(s1), mo11, e0)
                tmp += lib.einsum('yjl,xli,i->xyji', self._to_vt(s1), mo10, e0)
                tmp += lib.einsum('xjk,yki->xyji', self._to_vo(s1), e11)
                tmp += lib.einsum('yjk,xki->xyji', self._to_vo(s1), e10)

                tmp -= lib.einsum('xjl,yli->xyji', self._to_vt(s1), mo11) * w1
                tmp -= lib.einsum('yjl,xli->xyji', self._to_vt(s1), mo10) * w0
                tmp += lib.einsum('xjl,yli->xyji', self._to_vt(t10), mo11) * 1j
                tmp += lib.einsum('yjl,xli->xyji', self._to_vt(t11), mo10) * 1j
                tmp += self._to_vo(self.get_t2(freq)) * 1j

                tmp -= lib.einsum('xjl,yli->xyji', self._to_vt(f10), mo11)
                tmp -= lib.einsum('yjl,xli->xyji', self._to_vt(f11), mo10)
                tmp += lib.einsum('xjk,yki->xyji', mo10[:,~occidx], e11)
                tmp += lib.einsum('yjk,xki->xyji', mo11[:,~occidx], e10)
            else:
                tmp = -lib.einsum('xjl,yli->xyji', self._to_vv(f10), mo11)
                tmp -= lib.einsum('yjl,xli->xyji', self._to_vv(f11), mo10)
                tmp += lib.einsum('xjk,yki->xyji', mo10, e11)
                tmp += lib.einsum('yjk,xki->xyji', mo11, e10)
        
        return rhs

    def _krylov_solver(self, freq=0, verbose=logger.WARN, **kwargs):
        '''CP-HF/KS subspace projection solver by `pyscf.lib.krylov`.'''
        log = logger.new_logger(verbose=verbose)
        t0 = (logger.process_clock(), logger.perf_counter())
        e0vo = self.get_e0vo()
        nvir, nocc = e0vo.shape
        # first-order solver
        if isinstance(freq, (int, float)):
            rhs = self._rhs1(freq, **kwargs)
            
            if freq == 0:
                def lhs(mo1): # U(0)
                    mo1 = mo1.reshape(3,nvir,nocc)
                    v1 = self._to_vo(self.get_vind(mo1, freq, **kwargs)) / e0vo
                    return v1.ravel()
                # accurate enough
                rhs /= e0vo
                mo1 = lib.krylov(lhs, rhs.ravel(), max_cycle=self.max_cycle,
                                 tol=self.conv_tol, verbose=log).reshape(3,nvir,nocc)
                if self.with_s1:
                    mo1oo = self._to_oo(self.get_s1()) * -.5
                    mo1 = numpy.concatenate((mo1oo, mo1), axis=-2)
            
            else:
                def lhs(mo1): # mo1[0] = U(w), mo1[1] = U*(-w)
                    mo1 = mo1.reshape(2,-1,nvir,nocc)
                    v1 = self.get_vind(mo1, freq, **kwargs)
                    v1_p = self._to_vo(v1) # shape: (1,nvir,nocc)
                    v1_m = self._to_vo(v1.transpose(0,2,1).conj())
                    v1_p /= (e0vo + freq)
                    v1_m /= (e0vo - freq)
                    v1 = numpy.stack((v1_p, v1_m.conj()))
                    return v1.ravel()
                # calculating each direction separately can give a more exact result
                rhs = numpy.stack((rhs       /(e0vo+freq),
                                   rhs.conj()/(e0vo-freq))) # shape: (2,3,nvir,nocc)
                mo1 = numpy.empty_like(rhs)
                for i in range(3):
                    mo1[:,i] = lib.krylov(lhs, rhs[:,i].ravel(), max_cycle=self.max_cycle,
                                          tol=self.conv_tol, verbose=log).reshape(2,nvir,nocc)
                if self.with_s1:
                    mo1oo = self._to_oo(self.get_s1()) * -.5
                    mo1oo = numpy.stack((mo1oo, mo1oo.transpose(0,2,1)))
                    # transpose() is more efficient than conj() for a hermitian matrix
                    mo1 = numpy.concatenate((mo1oo, mo1), axis=-2)
                
            log.timer('Krylov solver for the first-order CP-HF/KS', *t0)
            return mo1
        # second-order solver
        elif len(freq) == 2:
            assert isinstance(freq, tuple)
            rhs = self._rhs2(freq, **kwargs)
            
            if freq == (0,0):
                def lhs(mo2): # U(0,0)
                    mo2 = mo2.reshape(1,1,nvir,nocc)
                    v2 = self._to_vo(self.get_vind(mo2, freq, with_mo1=False, **kwargs)) / e0vo
                    return v2.ravel()
                # Combined calculations can be efficient, but solving each direction individually
                # yields results very closely aligned with those obtained from the direct solver.
                rhs /= e0vo
                mo2 = numpy.empty_like(rhs)
                for i, j in numpy.ndindex(3,3):
                    mo2[i,j] = lib.krylov(lhs, rhs[i,j].ravel(), max_cycle=self.max_cycle,
                                          tol=self.conv_tol, verbose=log).reshape(nvir,nocc)
                
            else:
                def lhs(mo2): # mo2[0] = U(w1,w2), mo2[1] = U*(-w1,-w2)
                    mo2 = mo2.reshape(2,1,1,nvir,nocc)
                    v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                    v2_p = self._to_vo(v2)
                    v2_m = self._to_vo(v2.transpose(0,1,3,2).conj())
                    v2_p /= (e0vo + freq[0] + freq[1])
                    v2_m /= (e0vo - freq[0] - freq[1])
                    v2 = numpy.stack((v2_p, v2_m.conj()))
                    return v2.ravel()
                # calculating each direction separately can give a more exact result
                rhs = numpy.stack((rhs       /(e0vo+freq[0]+freq[1]),
                                   rhs.conj()/(e0vo-freq[0]-freq[1])))
                mo2 = numpy.empty_like(rhs) # shape: (2,3,3,nvir,nocc)
                for i, j in numpy.ndindex(3,3):
                    mo2[:,i,j] = lib.krylov(lhs, rhs[:,i,j].ravel(), max_cycle=self.max_cycle,
                                            tol=self.conv_tol, verbose=log).reshape(2,nvir,nocc)
            
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
            rhs = self._rhs1(freq, **kwargs)

            if freq == 0:
                def lhs(mo1): # U(0)
                    v1 = self._to_vo(self.get_vind(mo1, freq, **kwargs))
                    v1 += e0vo * mo1
                    return v1 - rhs

            else:
                rhs = numpy.stack((rhs, rhs.conj()))
                def lhs(mo1): # mo1[0] = U(w), mo1[1] = U*(-w)
                    v1 = self.get_vind(mo1, freq, **kwargs)
                    v1_p = self._to_vo(v1)
                    v1_m = self._to_vo(v1.transpose(0,2,1).conj())
                    v1_p += (e0vo + freq) * mo1[0]
                    v1_m += (e0vo - freq) * mo1[1].conj()
                    v1 = numpy.stack((v1_p, v1_m.conj()))
                    return v1 - rhs

            mo1 = newton_krylov(lhs, rhs, maxiter=self.max_cycle, f_tol=self.conv_tol)
            if self.with_s1:
                mo1oo = self._to_oo(self.get_s1()) * -.5
                if freq != 0:
                    mo1oo = numpy.stack((mo1oo, mo1oo.transpose(0,2,1)))
                mo1 = numpy.concatenate((mo1oo, mo1), axis=-2)
        
            log.timer('Newton-Krylov solver for the first-order CP-HF/KS', *t0)
            return mo1
        # second-order solver
        elif len(freq) == 2:
            rhs = self._rhs2(freq, **kwargs)

            if freq == (0,0):
                def lhs(mo2): # U(0,0)
                    v2 = self._to_vo(self.get_vind(mo2, freq, with_mo1=False, **kwargs))
                    v2 += e0vo * mo2
                    return v2 - rhs
                
            else:
                rhs = numpy.stack((rhs, rhs.conj()))
                def lhs(mo2): # mo2[0] = U(w1,w2), mo2[1] = U*(-w1,-w2)
                    v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                    v2_p = self._to_vo(v2)
                    v2_m = self._to_vo(v2.transpose(0,1,3,2).conj())
                    v2_p += (e0vo + freq[0] + freq[1]) * mo2[0]
                    v2_m += (e0vo - freq[0] - freq[1]) * mo2[1].conj()
                    v2 = numpy.stack((v2_p, v2_m.conj()))
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
            rhs = self._rhs1(freq, **kwargs)
            
            def lhs(mo1): # mo1[0] = U(w), mo1[1] = U*(-w)
                mo1 = mo1.reshape(2,1,nvir,nocc)
                v1 = self.get_vind(mo1, freq, **kwargs)
                v1_p = self._to_vo(v1)
                v1_m = self._to_vo(v1.transpose(0,2,1).conj())
                v1_p += (e0vo + freq) * mo1[0]
                v1_m += (e0vo - freq) * mo1[1].conj()
                v1 = numpy.stack((v1_p, v1_m.conj())) # shape: (2,1,nvir,nocc)
                return v1.ravel()
        # second-order solver
        elif len(freq) == 2:
            rhs = self._rhs2(freq, **kwargs).reshape(9,nvir,nocc)
            
            def lhs(mo2): # mo2[0] = U(w1,w2), mo2[1] = U*(-w1,-w2)
                mo2 = mo2.reshape(2,1,1,nvir,nocc)
                v2 = self.get_vind(mo2, freq, with_mo1=False, **kwargs)
                v2_p = self._to_vo(v2)
                v2_m = self._to_vo(v2.transpose(0,1,3,2).conj())
                v2_p += (e0vo + freq[0] + freq[1]) * mo2[0]
                v2_m += (e0vo - freq[0] - freq[1]) * mo2[1].conj()
                v2 = numpy.stack((v2_p, v2_m.conj()))
                return v2.ravel()
            
        else:
            raise NotImplementedError(freq)
        
        rhs = numpy.stack((rhs, rhs.conj()), axis=1)
        size = rhs[0].size
        operator = numpy.empty((size, size))
        iden = numpy.eye(size)
        for i, row in enumerate(iden):
            operator[:,i] = lhs(row)
        
        if isinstance(freq, (int, float)):
            mo1 = numpy.linalg.solve(operator, rhs.reshape(3,-1).T).T
            mo1 = mo1.reshape(rhs.shape).transpose(1,0,2,3)
            if self.with_s1:
                mo1oo = self._to_oo(self.get_s1()) * -.5
                mo1oo = numpy.stack((mo1oo, mo1oo.transpose(0,2,1)))
                # transpose() is more efficient than conj() for a hermitian matrix
                mo1 = numpy.concatenate((mo1oo, mo1), axis=-2)
            
            log.timer('Direct solver for the first-order CP-HF/KS', *t0)
            return mo1[0] if freq == 0 else mo1
        
        else:
            mo2 = numpy.linalg.solve(operator, rhs.reshape(9,-1).T).T
            mo2 = mo2.reshape(rhs.shape).transpose(1,0,2,3).reshape(2,3,3,nvir,nocc)
            if freq == (0,0): mo2 = mo2[0]
            mo2 = numpy.concatenate((self._mo2oo(freq, **kwargs), mo2), axis=-2)

            log.timer('Direct solver for the second-order CP-HF/KS', *t0)
            return mo2

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