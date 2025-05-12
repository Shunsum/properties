#!/usr/bin/env python

from pyscf import lib, numpy
from pyscf.x2c import x2c, _response_functions
from .ghf import GHFMagnet


class X2CMagnet(GHFMagnet):
    def get_h1(self, picture_change=True, **kwargs):
        '''The angular momentum matrix in AO basis.'''
        mf = self.mf
        mol = mf.mol
        with mol.with_common_orig((0,0,0)):
            if picture_change:
                xmol = mf.with_x2c.get_xmol()[0]
                c = 0.5/lib.param.LIGHT_SPEED
                t1 = xmol.intor_asymmetric('int1e_cg_irxp_spinor') * -.5j
                t1+= xmol.intor_symmetric('int1e_sigma_spinor') * .5
                w1 = xmol.intor('int1e_cg_sa10nucsp_spinor') * c**2 * -1j
                w1+= w1.transpose(0,2,1).conj()
                w1-= t1
                ang_mom = mf.with_x2c.picture_change((None, w1), t1)
            else:
                ang_mom = mol.intor_asymmetric('int1e_cg_irxp_spinor') * -.5j
                ang_mom+= mol.intor_symmetric('int1e_sigma_spinor') * .5
        return ang_mom
    
    def get_h2(self, picture_change=True, **kwargs):
        '''The diamagnetic Hamiltonian in AO basis.'''
        mf = self.mf
        mol = mf.mol
        with mol.with_common_orig((0,0,0)):
            if picture_change:
                raise NotImplementedError('X2C h2 integrals not implemented')
            else:
                n2c = mol.nao_2c()
                h2 = mol.intor_symmetric('int1e_rr_spinor').reshape(3,3,n2c,n2c)
                h2 = lib.einsum('xy,zzuv->xyuv', numpy.eye(3), h2) - h2
        return h2.reshape(9,n2c,n2c)

x2c.SCF.Magnetizability = lib.class_as_method(X2CMagnet)


if __name__ == '__main__':
    # static polarizabilities computed via analytical gradient vs. finite field
    import numpy
    from functools import reduce
    from pyscf import gto
    mol = gto.M(atom = '''H    0.   0.   0.
                          F    0.   0.   0.917''')

    mf = mol.X2C_HF().run(conv_tol=1e-14)
    pl = X2CPolar(mf)
    h1 = pl.get_h1()
    polar = pl.polar()
    hyperpolar = pl.hyperpolar()
    
    def get_hcore(mf, mol=None, E=None):
        '''2-component X2c Foldy-Wouthuysen (FW) Hamiltonian (including
        spin-free and spin-dependent terms) in the j-adapted spinor basis.
        '''
        if mol is None: mol = mf.mol
        if mol.has_ecp():
            raise NotImplementedError

        xmol, contr_coeff_nr = mf.with_x2c.get_xmol(mol)
        c = lib.param.LIGHT_SPEED
        assert ('1E' in mf.with_x2c.approx.upper())
        s = xmol.intor_symmetric('int1e_ovlp_spinor')
        t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
        v = xmol.intor_symmetric('int1e_nuc_spinor')
        v-= lib.einsum('xuv,x->uv', xmol.intor_symmetric('int1e_r_spinor'), E)
        w = xmol.intor_symmetric('int1e_spnucsp_spinor')
        w-= lib.einsum('xuv,x->uv', xmol.intor_symmetric('int1e_sprsp_spinor'), E)
        if 'get_xmat' in mf.with_x2c.__dict__:
            # If the get_xmat method is overwritten by user, build the X
            # matrix with the external get_xmat method
            x = mf.with_x2c.get_xmat(xmol)
            h1 = x2c._get_hcore_fw(t, v, w, s, x, c)

        elif 'ATOM' in mf.with_x2c.approx.upper():
            atom_slices = xmol.offset_2c_by_atom()
            n2c = xmol.nao_2c()
            x = numpy.zeros((n2c,n2c), dtype=numpy.complex128)
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                s1 = xmol.intor('int1e_ovlp_spinor', shls_slice=shls_slice)
                t1 = xmol.intor('int1e_spsp_spinor', shls_slice=shls_slice) * .5
                with xmol.with_rinv_at_nucleus(ia):
                    z = -xmol.atom_charge(ia)
                    v1 = z*xmol.intor('int1e_rinv_spinor', shls_slice=shls_slice)
                    w1 = z*xmol.intor('int1e_sprinvsp_spinor', shls_slice=shls_slice)
                x[p0:p1,p0:p1] = x2c._x2c1e_xmatrix(t1, v1, w1, s1, c)
            h1 = x2c._get_hcore_fw(t, v, w, s, x, c)

        else:
            h1 = x2c._x2c1e_get_hcore(t, v, w, s, c)

        if mf.with_x2c.basis is not None:
            s22 = xmol.intor_symmetric('int1e_ovlp_spinor')
            s21 = gto.mole.intor_cross('int1e_ovlp_spinor', xmol, mol)
            c = lib.cho_solve(s22, s21)
            h1 = reduce(numpy.dot, (c.T.conj(), h1, c))
        elif mf.with_x2c.xuncontract:
            np, nc = contr_coeff_nr.shape
            contr_coeff = numpy.zeros((np*2,nc*2))
            contr_coeff[0::2,0::2] = contr_coeff_nr
            contr_coeff[1::2,1::2] = contr_coeff_nr
            h1 = reduce(numpy.dot, (contr_coeff.T.conj(), h1, contr_coeff))
        return h1
    
    def apply_E(E):
        mf.get_hcore = lambda *args, **kwargs: get_hcore(mf, mol, E)
        mf.run(conv_tol=1e-14)
        return -mf.dip_moment(mol, mf.make_rdm1(), unit_symbol='AU', verbose=0)
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
        mf.get_hcore = lambda *args, **kwargs: get_hcore(mf, mol, E)
        mf.run(conv_tol=1e-14)
        return X2CPolar(mf).polarizability()
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
