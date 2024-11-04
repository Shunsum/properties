#!/usr/bin/python3

from pyscf import lib
from . import dhf


class Polarizability(dhf.Polarizability):
    pass


from pyscf.dft import dks
dks.DKS.Polarizability = lib.class_as_method(Polarizability)