#!/usr/bin/env python

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import dks
from pyscf.scf import hf, ghf
from pyscf.x2c.x2c import X2C1E_GSCF
from .rhf import RHFPolar


def rho_eff(rho, collinear):
    if collinear[0] == 'c':  # collinear
        n = rho[0]
        s = rho[3]
    elif collinear[0] == 'n':  # ncol
        n = rho[0]
        m = rho[1:4]
        s = lib.norm(m, axis=0)
    elif collinear[0] == 'm':  # mcol
        return rho
    else:
        raise RuntimeError(f'Unknown collinear scheme `{collinear}`')
    rho = numpy.array([n + s, n - s]) * .5
    return rho

def hyperpolar(pl:'GHFPolar', freq=(0,0,0), **kwargs):
    '''The hyperpolarizability tensor (with picture change correction if in X2C).

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
    log = logger.new_logger(pl, pl.verbose)
    mf = pl.mf

    if freq[0] == freq[1] == freq[2]: # b(0,0,0) / b(w,w,w)
        try: mo1 = pl.mo1[freq[0]]
        except KeyError: mo1 = pl.solve_mo1(freq[0], **kwargs)
        f1 = pl.get_f1(mo1, freq[0], **kwargs)
        f1vv = pl._to_vv(f1)
        f1oo = pl._to_oo(f1)
        
        if freq[0] == 0:
            e3 = -lib.einsum('xjl,yli,zji->xyz', f1vv, mo1, mo1.conj())
            e3 += lib.einsum('xik,yji,zjk->xyz', f1oo, mo1, mo1.conj())
        else:
            e3 = -lib.einsum('xjl,yli,zji->xyz', f1vv, mo1[0], mo1[1])
            e3 += lib.einsum('xik,yji,zjk->xyz', f1oo, mo1[0], mo1[1])

            try: mo2 = pl.mo2[freq[1:]]
            except KeyError: mo2 = pl.solve_mo2(freq[1:], **kwargs)
            if isinstance(mf, dks.dhf.DHF):
                n2c = mf.mol.nao_2c()
                mo2 = mo2[...,mf.mo_occ[n2c:]==0,:]
            else:
                mo2 = mo2[...,mf.mo_occ==0,:]
            mo2 = mo2.reshape(2,3,3,*mo2.shape[-2:])
            mo2[0] *= -1
            e3 += lib.einsum('syzji,sxji->xyz', mo2, mo1[[1,0]]) * freq[0] * .5
        
        e3 += e3.transpose(0,2,1) + e3.transpose(1,0,2) + \
              e3.transpose(1,2,0) + e3.transpose(2,0,1) + e3.transpose(2,1,0)
        
        if isinstance(mf, hf.KohnShamDFT):
            mol = mf.mol
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            col = ni.collinear
            xctype = ni._xc_type(mf.xc)
            dm0 = mf.make_rdm1()
            dm1 = pl.get_dm1(mo1, freq[0])
            nao = dm0.shape[-1]
            cur_mem = lib.current_memory()[0]
            max_mem = max(2000, mf.max_memory*.8 - cur_mem)

            if xctype == 'LDA':
                deriv = 0
            elif xctype in ('GGA', 'MGGA'):
                deriv = 1
            else:
                raise NotImplementedError(xctype)
                
            if isinstance(mf, dks.DKS):
                block_loop = ni.block_loop(mol, mf.grids, nao, deriv, max_mem,
                                           with_s=True)
            else:
                block_loop = ni.block_loop(mol, mf.grids, nao, deriv, max_mem)
            
            if col[0] == 'm': 
                eval_xc = ni.mcfun_eval_xc_adapter(mf.xc)
            else:
                eval_xc = ni.eval_xc_eff
            
            for ao, mask, weight, coords in block_loop:
                rho0 = ni.eval_rho(mol, ao, dm0, mask, xctype,
                                   hermi=1, with_lapl=False).real
                rho1 = numpy.array([rho_eff(ni.eval_rho(mol, ao, dm, mask, xctype,
                                    hermi=freq[0]==0, with_lapl=False), col)
                                    for dm in dm1])
                kxc = eval_xc(mf.xc, rho0, deriv=3, xctype=xctype)[3] * weight

                if xctype == 'LDA':
                    kxc = kxc[:,0,:,0,:,0]
                    e3 -= lib.einsum('xag,ybg,zcg,abcg->xyz',
                                       rho1, rho1, rho1, kxc)
                else:
                    e3 -= lib.einsum('xaig,ybjg,zckg,aibjckg->xyz',
                                       rho1, rho1, rho1, kxc)

    elif len(set(freq)) == 2: # b(we,we,wi) -> b(we,wi,we) / b(wi,we,we)
        we = next(f for f in set(freq) if freq.count(f) == 2)
        wi = next(f for f in set(freq) if f != we)
        try: mo1e = pl.mo1[we]
        except KeyError: mo1e = pl.solve_mo1(we, **kwargs)
        try: mo1i = pl.mo1[wi]
        except KeyError: mo1i = pl.solve_mo1(wi, **kwargs)
        f1e = pl.get_f1(mo1e, we, **kwargs)
        f1i = pl.get_f1(mo1i, wi, **kwargs)
        f1vve = pl._to_vv(f1e)
        f1ooe = pl._to_oo(f1e)
        f1vvi = pl._to_vv(f1i)
        f1ooi = pl._to_oo(f1i)

        if we == 0: mo1e = (mo1e, mo1e.conj())
        if wi == 0: mo1i = (mo1i, mo1i.conj())
        
        e3 = -lib.einsum('xjl,yli,zji->xyz', f1vve, mo1e[0], mo1i[1])
        e3 -= lib.einsum('yjl,zli,xji->xyz', f1vve, mo1i[0], mo1e[1])
        e3 -= lib.einsum('zjl,xli,yji->xyz', f1vvi, mo1e[0], mo1e[1])
        e3 += lib.einsum('xik,yji,zjk->xyz', f1ooe, mo1e[0], mo1i[1])
        e3 += lib.einsum('yik,zji,xjk->xyz', f1ooe, mo1i[0], mo1e[1])
        e3 += lib.einsum('zik,xji,yjk->xyz', f1ooi, mo1e[0], mo1e[1])

        if we == 0:
            mo1e = mo1e[0]
        else:
            try: mo2 = pl.mo2[(we,wi)]
            except KeyError: mo2 = pl.solve_mo2((we,wi), **kwargs)
            if isinstance(mf, dks.dhf.DHF):
                n2c = mf.mol.nao_2c()
                mo2 = mo2[...,mf.mo_occ[n2c:]==0,:]
            else:
                mo2 = mo2[...,mf.mo_occ==0,:]
            mo2 = mo2.reshape(2,3,3,*mo2.shape[-2:])
            mo2[0] *= -1
            e3 += lib.einsum('syzji,sxji->xyz', mo2, mo1e[[1,0]]) * we
        if wi == 0:
            mo1i = mo1i[0]
        else:
            try: mo2 = pl.mo2[(we,we)]
            except KeyError: mo2 = pl.solve_mo2((we,we), **kwargs)
            if isinstance(mf, dks.dhf.DHF):
                n2c = mf.mol.nao_2c()
                mo2 = mo2[...,mf.mo_occ[n2c:]==0,:]
            else:
                mo2 = mo2[...,mf.mo_occ==0,:]
            if we == 0: mo2 = numpy.array((mo2, mo2.conj()))
            mo2 = mo2.reshape(2,3,3,*mo2.shape[-2:])
            mo2[0] *= -1
            e3 += lib.einsum('sxyji,szji->xyz', mo2, mo1i[[1,0]]) * wi * .5

        e3 += e3.transpose(1,0,2)

        if isinstance(mf, hf.KohnShamDFT):
            mol = mf.mol
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            col = ni.collinear
            xctype = ni._xc_type(mf.xc)
            dm0 = mf.make_rdm1()
            dm1e = pl.get_dm1(mo1e, we)
            dm1i = pl.get_dm1(mo1i, wi)
            nao = dm0.shape[-1]
            cur_mem = lib.current_memory()[0]
            max_mem = max(2000, mf.max_memory*.8 - cur_mem)

            if xctype == 'LDA':
                deriv = 0
            elif xctype in ('GGA', 'MGGA'):
                deriv = 1
            else:
                raise NotImplementedError(xctype)

            if isinstance(mf, dks.DKS):
                block_loop = ni.block_loop(mol, mf.grids, nao, deriv, max_mem,
                                           with_s=True)
            else:
                block_loop = ni.block_loop(mol, mf.grids, nao, deriv, max_mem)

            if col[0] == 'm': 
                eval_xc = ni.mcfun_eval_xc_adapter(mf.xc)
            else:
                eval_xc = ni.eval_xc_eff
                
            for ao, mask, weight, coords in block_loop:
                rho0 = ni.eval_rho(mol, ao, dm0, mask, xctype,
                                   hermi=1, with_lapl=False).real
                rho1e = numpy.array([rho_eff(ni.eval_rho(mol, ao, dm, mask, xctype,
                                     hermi=we==0, with_lapl=False), col)
                                     for dm in dm1e])
                rho1i = numpy.array([rho_eff(ni.eval_rho(mol, ao, dm, mask, xctype,
                                     hermi=wi==0, with_lapl=False), col)
                                     for dm in dm1i])
                kxc = eval_xc(mf.xc, rho0, deriv=3, xctype=xctype)[3] * weight

                if xctype == 'LDA':
                    kxc = kxc[:,0,:,0,:,0]
                    e3 -= lib.einsum('xag,ybg,zcg,abcg->xyz',
                                       rho1e, rho1e, rho1i, kxc)
                else:
                    e3 -= lib.einsum('xaig,ybjg,zckg,aibjckg->xyz',
                                       rho1e, rho1e, rho1i, kxc)

        if   freq.index(wi) == 1: e3 = e3.transpose(0,2,1)
        elif freq.index(wi) == 0: e3 = e3.transpose(2,1,0)

    else: # b(w1,w2,w3)
        try: mo10 = pl.mo1[freq[0]]
        except KeyError: mo10 = pl.solve_mo1(freq[0], **kwargs)
        try: mo11 = pl.mo1[freq[1]]
        except KeyError: mo11 = pl.solve_mo1(freq[1], **kwargs)
        try: mo12 = pl.mo1[freq[2]]
        except KeyError: mo12 = pl.solve_mo1(freq[2], **kwargs)
        f10 = pl.get_f1(mo10, freq[0], **kwargs)
        f11 = pl.get_f1(mo11, freq[1], **kwargs)
        f12 = pl.get_f1(mo12, freq[2], **kwargs)
        f1vv0 = pl._to_vv(f10)
        f1oo0 = pl._to_oo(f10)
        f1vv1 = pl._to_vv(f11)
        f1oo1 = pl._to_oo(f11)
        f1vv2 = pl._to_vv(f12)
        f1oo2 = pl._to_oo(f12)

        if freq[0] == 0: mo10 = (mo10, mo10.conj())
        if freq[1] == 0: mo11 = (mo11, mo11.conj())
        if freq[2] == 0: mo12 = (mo12, mo12.conj())
        
        e3 = -lib.einsum('xjl,yli,zji->xyz', f1vv0, mo11[0], mo12[1])
        e3 -= lib.einsum('xjl,zli,yji->xyz', f1vv0, mo12[0], mo11[1])
        e3 -= lib.einsum('yjl,xli,zji->xyz', f1vv1, mo10[0], mo12[1])
        e3 -= lib.einsum('yjl,zli,xji->xyz', f1vv1, mo12[0], mo10[1])
        e3 -= lib.einsum('zjl,xli,yji->xyz', f1vv2, mo10[0], mo11[1])
        e3 -= lib.einsum('zjl,yli,xji->xyz', f1vv2, mo11[0], mo10[1])
        e3 += lib.einsum('xik,yji,zjk->xyz', f1oo0, mo11[0], mo12[1])
        e3 += lib.einsum('xik,zji,yjk->xyz', f1oo0, mo12[0], mo11[1])
        e3 += lib.einsum('yik,xji,zjk->xyz', f1oo1, mo10[0], mo12[1])
        e3 += lib.einsum('yik,zji,xjk->xyz', f1oo1, mo12[0], mo10[1])
        e3 += lib.einsum('zik,xji,yjk->xyz', f1oo2, mo10[0], mo11[1])
        e3 += lib.einsum('zik,yji,xjk->xyz', f1oo2, mo11[0], mo10[1])
        
        if freq[0] == 0:
            mo10 = mo10[0]
        else:
            try: mo2 = pl.mo2[freq[1:]]
            except KeyError: mo2 = pl.solve_mo2(freq[1:], **kwargs)
            if isinstance(mf, dks.dhf.DHF):
                n2c = mf.mol.nao_2c()
                mo2 = mo2[...,mf.mo_occ[n2c:]==0,:]
            else:
                mo2 = mo2[...,mf.mo_occ==0,:]
            mo2 = mo2.reshape(2,3,3,*mo2.shape[-2:])
            mo2[0] *= -1
            e3 += lib.einsum('syzji,sxji->xyz', mo2, mo10[[1,0]]) * freq[0]
        if freq[1] == 0:
            mo11 = mo11[0]
        else:
            try: mo2 = pl.mo2[(freq[0],freq[2])]
            except KeyError: mo2 = pl.solve_mo2((freq[0],freq[2]), **kwargs)
            if isinstance(mf, dks.dhf.DHF):
                n2c = mf.mol.nao_2c()
                mo2 = mo2[...,mf.mo_occ[n2c:]==0,:]
            else:
                mo2 = mo2[...,mf.mo_occ==0,:]
            mo2 = mo2.reshape(2,3,3,*mo2.shape[-2:])
            mo2[0] *= -1
            e3 += lib.einsum('sxzji,syji->xyz', mo2, mo11[[1,0]]) * freq[1]
        if freq[2] == 0:
            mo12 = mo12[0]
        else:
            try: mo2 = pl.mo2[freq[:-1]]
            except KeyError: mo2 = pl.solve_mo2(freq[:-1], **kwargs)
            if isinstance(mf, dks.dhf.DHF):
                n2c = mf.mol.nao_2c()
                mo2 = mo2[...,mf.mo_occ[n2c:]==0,:]
            else:
                mo2 = mo2[...,mf.mo_occ==0,:]
            mo2 = mo2.reshape(2,3,3,*mo2.shape[-2:])
            mo2[0] *= -1
            e3 += lib.einsum('sxyji,szji->xyz', mo2, mo12[[1,0]]) * freq[2]
        
        if isinstance(mf, hf.KohnShamDFT):
            mol = mf.mol
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            col = ni.collinear
            xctype = ni._xc_type(mf.xc)
            dm0 = mf.make_rdm1()
            dm10 = pl.get_dm1(mo10, freq[0])
            dm11 = pl.get_dm1(mo11, freq[1])
            dm12 = pl.get_dm1(mo12, freq[2])
            nao = dm0.shape[-1]
            cur_mem = lib.current_memory()[0]
            max_mem = max(2000, mf.max_memory*.8 - cur_mem)

            if xctype == 'LDA':
                deriv = 0
            elif xctype in ('GGA', 'MGGA'):
                deriv = 1
            else:
                raise NotImplementedError(xctype)
            
            if isinstance(mf, dks.DKS):
                block_loop = ni.block_loop(mol, mf.grids, nao, deriv, max_mem,
                                           with_s=True)
            else:
                block_loop = ni.block_loop(mol, mf.grids, nao, deriv, max_mem)
            
            if col[0] == 'm': 
                eval_xc = ni.mcfun_eval_xc_adapter(mf.xc)
            else:
                eval_xc = ni.eval_xc_eff
                
            for ao, mask, weight, coords in block_loop:
                rho0 = ni.eval_rho(mol, ao, dm0, mask, xctype,
                                   hermi=1, with_lapl=False).real
                rho10 = numpy.array([rho_eff(ni.eval_rho(mol, ao, dm, mask, xctype,
                                     hermi=freq[0]==0, with_lapl=False), col)
                                     for dm in dm10])
                rho11 = numpy.array([rho_eff(ni.eval_rho(mol, ao, dm, mask, xctype,
                                     hermi=freq[1]==0, with_lapl=False), col)
                                     for dm in dm11])
                rho12 = numpy.array([rho_eff(ni.eval_rho(mol, ao, dm, mask, xctype,
                                     hermi=freq[2]==0, with_lapl=False), col)
                                     for dm in dm12])
                kxc = eval_xc(mf.xc, rho0, deriv=3, xctype=xctype)[3] * weight

                if xctype == 'LDA':
                    kxc = kxc[:,0,:,0,:,0]
                    e3 -= lib.einsum('xag,ybg,zcg,abcg->xyz',
                                       rho10, rho11, rho12, kxc)
                else:
                    e3 -= lib.einsum('xaig,ybjg,zckg,aibjckg->xyz',
                                       rho10, rho11, rho12, kxc)

    e3 = e3.real

    if isinstance(pl, RHFPolar): pm = "hyperpolarizability"
    else: pm = "paramagnetic hypermagnetizability"
    log.info(f'The {pm} with frequencies {freq} in A.U.')
    log.info(f'{e3}')
    if pl.verbose >= logger.DEBUG:
        mgn1 = lib.einsum('ijj->i', e3)
        mgn2 = lib.einsum('jij->i', e3)
        mgn3 = lib.einsum('jji->i', e3)
        mgn = lib.norm((mgn1 + mgn2 + mgn3)/3)
        log.debug(f'The magnitude of the {pm}: {mgn:.6g}')
    
    return e3

def _dirac_relation(mat):
    '''[[ W0 + iWz], [Wy + iWx],
        [-Wy + iWx], [W0 - iWz]].

    np.einsum performs better than np.kron in most cases.'''
    quaternion = numpy.vstack([1j * lib.PauliMatrices, numpy.eye(2)[None]])
    qshape = quaternion.shape # shape: (4,2,2)
    mshape = mat.shape # shape: (3,4,nao,nao)
    tenpro = lib.einsum('sij,xspq->xipjq', quaternion, mat)
    return tenpro.reshape(mshape[0], qshape[1]*mshape[2], qshape[2]*mshape[3])


class GHFPolar(RHFPolar):
    def get_h1(self, picture_change=True, **kwargs):
        '''The dipole matrix in AO basis.'''
        mf = self.mf
        mol = mf.mol
        with mol.with_common_orig((0,0,0)):
            if isinstance(mf, X2C1E_GSCF) and picture_change:
                xmol = mf.with_x2c.get_xmol()[0]
                nao = xmol.nao
                ao_dip = xmol.intor_symmetric('int1e_r')
                ao_dip = lib.einsum('ij,xpq->xipjq', numpy.eye(2), ao_dip)
                ao_dip = ao_dip.reshape(3,2*nao,2*nao)
                prp = xmol.intor_symmetric('int1e_sprsp').reshape(3,4,nao,nao)
                prp = _dirac_relation(prp)
                c = 0.5/lib.param.LIGHT_SPEED
                ao_dip = mf.with_x2c.picture_change((ao_dip, prp*c**2))
            else:
                ao_dip = mol.intor_symmetric('int1e_r', comp=3)
                nao = ao_dip.shape[-1]
                ao_dip = lib.einsum('ij,xpq->xipjq', numpy.eye(2), ao_dip)
                ao_dip = ao_dip.reshape(3,2*nao,2*nao)
        return ao_dip

    hyperpol = hyperpolar = hyperpolar

ghf.GHF.Polarizability = lib.class_as_method(GHFPolar)


if __name__ == '__main__':
    # static polarizabilities computed via analytical gradient vs. finite field
    from pyscf import gto
    mol = gto.M(atom = '''H    0.   0.   0.
                          F    0.   0.   0.917''')

    mf = mol.GHF().run(conv_tol=1e-14)
    hcore = mf.get_hcore()
    pl = GHFPolar(mf)
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
        return GHFPolar(mf).polarizability()
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
