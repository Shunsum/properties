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

from pyscf import lib
from pyscf.dft import uks
from .uhf import UHFPolar


class UKSPolar(UHFPolar):
    pass

uks.UKS.Polarizability = lib.class_as_method(UKSPolar)

if __name__ == '__main__':
    # static polarizabilities computed via analytical gradient vs. finite field
    from pyscf import gto
    mol = gto.M(atom = '''O    0.    0.       0.
                          H    0.   -0.757    0.587
                          H    0.    0.757    0.587''',
                charge = 1,
                spin = 1)

    mf = mol.UKS(xc='b3lyp').run(conv_tol=1e-14)
    hcore = mf.get_hcore()
    pl = UKSPolar(mf)
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
        return UKSPolar(mf).polarizability()
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
