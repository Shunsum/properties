#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
See also pyscf/hessian/rks.py
'''

from pyscf.hessian import rks as rks_hess
from pyscf.prop.freq.rhf import kernel

class Frequency(rks_hess.Hessian):
    def __init__(self, mf):
        self.nroots = None
        self.freq = None
        self.mode = None
        self.conv_tol = 1e-3
        rks_hess.Hessian.__init__(self, mf)

    kernel = kernel

Freq = Frequency


if __name__ == '__main__':
    import numpy
    from pyscf import lib
    from pyscf import gto
    from pyscf import scf, dft

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)] ]
    mol.basis = '631g'
    mol.unit = 'B'
    mol.build()
    mf = dft.RKS(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    hobj = rks_hess.Hessian(mf)
    e2 = hobj.kernel()
    numpy.random.seed(1)
    x = numpy.random.random((mol.natm,3))
    e2x = numpy.einsum('abxy,ax->by', e2, x)
    print(lib.finger(e2x) - -0.20252942721146411)
    hop = Freq(mf).gen_hop()[0]
    print(lib.finger(hop(x)) - -0.20252942721146411)
    print(abs(e2x-hop(x).reshape(mol.natm,3)).sum())
    print(Freq(mf).set(nroots=1).kernel()[0])
    print(numpy.linalg.eigh(e2.transpose(0,2,1,3).reshape(n3,n3))[0])
