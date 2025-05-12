#!/usr/bin/env python

import numpy
from pyscf import lib
from pyscf.scf import ghf
from pyscf.x2c.x2c import X2C1E_GSCF
from pyscf.prop.polarizability.ghf import hyperpolar, _dirac_relation
from .rhf import RHFMagnet


class GHFMagnet(RHFMagnet):
    def get_h1(self, picture_change=True, **kwargs):
        '''The angular momentum matrix in AO basis.'''
        mf = self.mf
        mol = mf.mol
        with mol.with_common_orig((0,0,0)):
            if isinstance(mf, X2C1E_GSCF) and picture_change:
                xmol = mf.with_x2c.get_xmol()[0]
                nao = xmol.nao
                c = 0.5/lib.param.LIGHT_SPEED
                # 'int1e_sigma' differs from sigma x 'int1e_ovlp' by a negative sign
                t1 = -xmol.intor_symmetric('int1e_sigma').reshape(3,4,nao,nao)
                t1[:,3] = xmol.intor_asymmetric('int1e_cg_irxp')
                t1 = _dirac_relation(t1) * -.5j
                w1 = xmol.intor('int1e_cg_sa10nucsp').reshape(3,4,nao,nao)
                w1 = _dirac_relation(w1) * c**2 * -1j
                w1+= w1.transpose(0,2,1).conj()
                w1-= t1
                ang_mom = mf.with_x2c.picture_change((None, w1), t1)
            else:
                nao = mol.nao
                ang_mom = -mol.intor_symmetric('int1e_sigma').reshape(3,4,nao,nao)
                ang_mom[:,3] = mol.intor_asymmetric('int1e_cg_irxp')
                ang_mom = _dirac_relation(ang_mom) * -.5j
        return ang_mom
    
    def get_h2(self, picture_change=True, **kwargs):
        '''The diamagnetic Hamiltonian in AO basis.'''
        h2 = super().get_h2(picture_change, **kwargs)
        nao = h2.shape[-1]
        h2 = lib.einsum('ij,xpq->xipqj', numpy.eye(2), h2)
        return h2.reshape(9,nao*2,nao*2)

    hyperpmag = hyperpolar
    hyperpmag.__doc__ = hyperpmag.__doc__.replace('hyperpolarizability',
                                                  'para-hypermagnetizability')

ghf.GHF.Magnetizability = lib.class_as_method(GHFMagnet)


if __name__ == '__main__':
    # static polarizabilities computed via analytical gradient vs. finite field
    from pyscf import gto
    mol = gto.M(atom = '''H    0.   0.   0.
                          F    0.   0.   0.917''')

    mf = mol.GHF().run(conv_tol=1e-14)
    hcore = mf.get_hcore()
    pl = GHFMagnet(mf)
    h1 = pl.get_h1()
    polar = pl.polar()
    hyperpolar = pl.hyperpolar()
    
    def apply_E(E):
        mf.get_hcore = lambda *args, **kwargs: hcore + lib.einsum('x,xuv->uv', E, h1)
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
        mf.get_hcore = lambda *args, **kwargs: hcore + lib.einsum('x,xuv->uv', E, h1)
        mf.run(conv_tol=1e-14)
        return GHFMagnet(mf).polarizability()
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
