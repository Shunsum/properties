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
(In testing)
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, uhf, _response_functions
from . import rhf


def get_dm1(polobj, mo1):
    '''
    Generate the 1st-order density matrix in AO basis.
    '''
    assert isinstance(mo1, tuple)
    mf = polobj.mf
    mo_coeff = mf.mo_coeff
    occidxa = mf.mo_occ[0] > 0
    occidxb = mf.mo_occ[1] > 0
    orbva = mo_coeff[0][:,~occidxa]
    orboa = mo_coeff[0][:, occidxa]
    orbvb = mo_coeff[1][:,~occidxb]
    orbob = mo_coeff[1][:, occidxb]
    if len(mo1) == 4:
        dm1a = lib.einsum('pj,xji,qi->xpq', orbva, mo1[0]       , orboa.conj())
        dm1a+= lib.einsum('pi,xji,qj->xpq', orboa, mo1[1].conj(), orbva.conj())
        dm1b = lib.einsum('pj,xji,qi->xpq', orbvb, mo1[2]       , orbob.conj())
        dm1b+= lib.einsum('pi,xji,qj->xpq', orbob, mo1[3].conj(), orbvb.conj())
    elif len(mo1) == 2:
        dm1a = lib.einsum('pj,xji,qi->xpq', orbva, mo1[0], orboa.conj())
        dm1a+= dm1a.transpose(0,2,1).conj()
        dm1b = lib.einsum('pj,xji,qi->xpq', orbvb, mo1[1], orbob.conj())
        dm1b+= dm1b.transpose(0,2,1).conj()
    else:
        raise ValueError('The number of mo1\'s should be 2 for freq = 0'
                         ' and 4 for freq != 0 in unrestricted CPHF/KS.')
    return (dm1a, dm1b)

def get_e_vo(mf):
    '''
    Generate $e_vo = e_v - e_o$.
    '''
    occidxa = mf.mo_occ[0] > 0
    occidxb = mf.mo_occ[1] > 0
    mo_ea, mo_eb = mf.mo_energy
    ea_v = mo_ea[~occidxa]
    ea_o = mo_ea[ occidxa]
    ea_vo = ea_v[:,None] - ea_o
    eb_v = mo_eb[~occidxb]
    eb_o = mo_eb[ occidxb]
    eb_vo = eb_v[:,None] - eb_o
    return (ea_vo, eb_vo)

def _to_vo(mf, ao):
    '''
    Convert some quantity in AO basis to that in vir-occ MO basis.
    '''
    mo_coeff = mf.mo_coeff
    occidxa = mf.mo_occ[0] > 0
    occidxb = mf.mo_occ[1] > 0
    orbva = mo_coeff[0][:,~occidxa]
    orboa = mo_coeff[0][:, occidxa]
    orbvb = mo_coeff[1][:,~occidxb]
    orbob = mo_coeff[1][:, occidxb]
    if len(ao) == 3:
        voa = lib.einsum('pj,xpq,qi->xji', orbva.conj(), ao, orboa)
        vob = lib.einsum('pj,xpq,qi->xji', orbvb.conj(), ao, orbob)
    elif len(ao) == 2:
        voa = lib.einsum('pj,xpq,qi->xji', orbva.conj(), ao[0], orboa)
        vob = lib.einsum('pj,xpq,qi->xji', orbvb.conj(), ao[1], orbob)
    return (voa, vob)

def _to_vv(mf, ao):
    '''
    Convert some quantity in AO basis to that in vir-occ MO basis.
    '''
    mo_coeff = mf.mo_coeff
    viridxa = mf.mo_occ[0] == 0
    viridxb = mf.mo_occ[1] == 0
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]
    if len(ao) == 3:
        vva = lib.einsum('pj,xpq,qi->xji', orbva.conj(), ao, orbva)
        vvb = lib.einsum('pj,xpq,qi->xji', orbvb.conj(), ao, orbvb)
    elif len(ao) == 2:
        vva = lib.einsum('pj,xpq,qi->xji', orbva.conj(), ao[0], orbva)
        vvb = lib.einsum('pj,xpq,qi->xji', orbvb.conj(), ao[1], orbvb)
    return (vva, vvb)

def _to_oo(mf, ao):
    '''
    Convert some quantity in AO basis to that in vir-occ MO basis.
    '''
    mo_coeff = mf.mo_coeff
    occidxa = mf.mo_occ[0] > 0
    occidxb = mf.mo_occ[1] > 0
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    if len(ao) == 3:
        ooa = lib.einsum('pj,xpq,qi->xji', orboa.conj(), ao, orboa)
        oob = lib.einsum('pj,xpq,qi->xji', orbob.conj(), ao, orbob)
    elif len(ao) == 2:
        ooa = lib.einsum('pj,xpq,qi->xji', orboa.conj(), ao[0], orboa)
        oob = lib.einsum('pj,xpq,qi->xji', orbob.conj(), ao[1], orbob)
    return (ooa, oob)

# Note: polarizability and relevant properties are demanding on basis sets.
# ORCA recommends to use Sadlej basis for these properties.
def polarizability(polobj, freq=0, picture_change=True, solver='krylov'):
    '''
    Polarizability (with picture change correction if in SFX2C).
    '''
    log = logger.new_logger(polobj)
    mf = polobj.mf
    h1 = polobj.get_h1vo(picture_change)
    mo1 = polobj.get_mo1(freq, picture_change, solver)

    if freq == 0:
        # alpha(0;0)
        alpha  = -lib.einsum('xji,yji->xy', h1[0].conj(), mo1[0])
        alpha += -lib.einsum('xji,yji->xy', h1[1].conj(), mo1[1])
        alpha += alpha.conj()
    
    else:
        # alpha(-omega;omega)
        alpha  = -lib.einsum('xji,yji->xy', h1[0].conj(), mo1[0])
        alpha += -lib.einsum('xji,yji->xy', h1[0], mo1[1].conj())
        alpha += -lib.einsum('xji,yji->xy', h1[1].conj(), mo1[2])
        alpha += -lib.einsum('xji,yji->xy', h1[1], mo1[3].conj())

    if mf.verbose >= logger.INFO:
        xx, yy, zz = alpha.diagonal()
        log.note('Isotropic polarizability %.12g', (xx+yy+zz)/3)
        log.note('Polarizability anisotropy %.12g',
                 (.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2))**.5)
        log.debug(f'Polarizability tensor alpha({-freq};{freq})')
        log.debug(f'{alpha}')

    return alpha

def hyperpolarizability(polobj, freq=0, type='SHG', picture_change=True, solver='krylov'):
    '''
    Hyperpolarizability (with picture change correction if in SFX2C).
    '''
    log = logger.new_logger(polobj)
    mf = polobj.mf

    if freq == 0:
        mo1 = polobj.get_mo1(freq, picture_change, solver)
        f1 = polobj.get_f1vv(mo1, picture_change)
        e1 = polobj.get_e1(mo1, picture_change)

        # beta(0;0,0)
        beta = -lib.einsum('xjl,yji,zli->xyz', f1[0], mo1[0].conj(), mo1[0])
        beta +=-lib.einsum('xjl,yji,zli->xyz', f1[1], mo1[1].conj(), mo1[1])
        beta += lib.einsum('xik,yjk,zji->xyz', e1[0], mo1[0].conj(), mo1[0])
        beta += lib.einsum('xik,yjk,zji->xyz', e1[1], mo1[1].conj(), mo1[1])
        beta += beta.transpose(0,2,1) + beta.transpose(1,0,2) + \
                beta.transpose(1,2,0) + beta.transpose(2,0,1) + beta.transpose(2,1,0)
        
        if isinstance(mf, hf.KohnShamDFT):
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm = mf.make_rdm1()
            dm1 = polobj.get_dm1(mo1=mo1)

            if xctype == 'LDA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords)
                rho = numpy.stack([ni.eval_rho(mf.mol, ao, dms, xctype=xctype) for dms in dm])
                rho1a = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                     for dmx in dm1[0]])
                rho1b = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                     for dmx in dm1[1]])
                rho1 = numpy.stack((rho1a, rho1b), axis=1).reshape(3,2,1,-1) # reshape to match kxc
                
            elif xctype == 'GGA' or 'MGGA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1)
                rho = numpy.stack([ni.eval_rho(mf.mol, ao, dms, xctype=xctype) for dms in dm])
                rho1a = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                 with_lapl=False) for dmx in dm1[0]])
                rho1b = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                 with_lapl=False) for dmx in dm1[1]])
                rho1 = numpy.stack((rho1a, rho1b), axis=1)

            else:
                raise RuntimeError(f"Unknown xctype '{xctype}'")
            
            kxc = ni.eval_xc_eff(mf.xc, rho, deriv=3)[3] * mf.grids.weights
            beta -= lib.einsum('xaig,ybjg,zckg,aibjckg->xyz', rho1, rho1, rho1, kxc)
        
        if mf.verbose >= logger.INFO:
            log.debug(f'Static hyperpolarizability tensor beta({freq};{freq},{freq})')
            log.debug(f'{beta}')
    
    elif type.upper() == 'SHG':
        mo1_2o = polobj.get_mo1(freq*2, picture_change, solver)
        f1_p2o = polobj.get_f1vv(mo1_2o, picture_change)
        f1_m2o = (f1_p2o[0].transpose(0,2,1).conj(),
                  f1_p2o[1].transpose(0,2,1).conj())
        e1_p2o = polobj.get_e1(mo1_2o, picture_change)
        e1_m2o = (e1_p2o[0].transpose(0,2,1).conj(),
                  e1_p2o[1].transpose(0,2,1).conj())
        
        mo1_1o = polobj.get_mo1(freq, picture_change, solver)
        f1_p1o = polobj.get_f1vv(mo1_1o, picture_change)
        e1_p1o = polobj.get_e1(mo1_1o, picture_change)
        
        # beta(-2omega;omega,omega)
        beta = -lib.einsum('xjl,yji,zli->xyz', f1_m2o[0], mo1_1o[1].conj(), mo1_1o[0])
        beta +=-lib.einsum('xjl,yji,zli->xyz', f1_m2o[1], mo1_1o[3].conj(), mo1_1o[2])
        beta +=-lib.einsum('yjl,xji,zli->xyz', f1_p1o[0], mo1_2o[0].conj(), mo1_1o[0])
        beta +=-lib.einsum('yjl,xji,zli->xyz', f1_p1o[1], mo1_2o[2].conj(), mo1_1o[2])
        beta +=-lib.einsum('zjl,yji,xli->xyz', f1_p1o[0], mo1_1o[1].conj(), mo1_2o[1])
        beta +=-lib.einsum('zjl,yji,xli->xyz', f1_p1o[1], mo1_1o[3].conj(), mo1_2o[3])
        beta += lib.einsum('xji,yki,zkj->xyz', e1_m2o[0], mo1_1o[1].conj(), mo1_1o[0])
        beta += lib.einsum('xji,yki,zkj->xyz', e1_m2o[1], mo1_1o[3].conj(), mo1_1o[2])
        beta += lib.einsum('yji,xki,zkj->xyz', e1_p1o[0], mo1_2o[0].conj(), mo1_1o[0])
        beta += lib.einsum('yji,xki,zkj->xyz', e1_p1o[1], mo1_2o[2].conj(), mo1_1o[2])
        beta += lib.einsum('zji,yki,xkj->xyz', e1_p1o[0], mo1_1o[1].conj(), mo1_2o[1])
        beta += lib.einsum('zji,yki,xkj->xyz', e1_p1o[1], mo1_1o[3].conj(), mo1_2o[3])
        beta += beta.transpose(0,2,1)
        
        if isinstance(mf, hf.KohnShamDFT):
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm = mf.make_rdm1()
            dm1_p2o = polobj.get_dm1(mo1=mo1_2o)
            dm1_m2o = (dm1_p2o[0].transpose(0,2,1).conj(),
                       dm1_p2o[1].transpose(0,2,1).conj())
            dm1_p1o = polobj.get_dm1(mo1=mo1_1o)

            if xctype == 'LDA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords)
                rho = numpy.stack([ni.eval_rho(mf.mol, ao, dms, xctype=xctype) for dms in dm])
                rho1a_m2o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                         for dmx in dm1_m2o[0]])
                rho1b_m2o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                         for dmx in dm1_m2o[1]])
                rho1a_p1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                         for dmx in dm1_p1o[0]])
                rho1b_p1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                         for dmx in dm1_p1o[1]])
                rho1_m2o = numpy.stack((rho1a_m2o, rho1b_m2o), axis=1).reshape(3,2,1,-1) # reshape to match kxc
                rho1_p1o = numpy.stack((rho1a_p1o, rho1b_p1o), axis=1).reshape(3,2,1,-1) # reshape to match kxc
                
            elif xctype == 'GGA' or 'MGGA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1)
                rho = numpy.stack([ni.eval_rho(mf.mol, ao, dms, xctype=xctype) for dms in dm])
                rho1a_m2o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                     with_lapl=False) for dmx in dm1_m2o[0]])
                rho1b_m2o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                     with_lapl=False) for dmx in dm1_m2o[1]])
                rho1a_p1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                     with_lapl=False) for dmx in dm1_p1o[0]])
                rho1b_p1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                     with_lapl=False) for dmx in dm1_p1o[1]])
                rho1_m2o = numpy.stack((rho1a_m2o, rho1b_m2o), axis=1)
                rho1_p1o = numpy.stack((rho1a_p1o, rho1b_p1o), axis=1)
                
            else:
                raise RuntimeError(f"Unknown xctype '{xctype}'")
            
            kxc = ni.eval_xc_eff(mf.xc, rho, deriv=3)[3] * mf.grids.weights
            beta -= lib.einsum('xaig,ybjg,zckg,aibjckg->xyz', rho1_m2o, rho1_p1o, rho1_p1o, kxc)

        if mf.verbose >= logger.INFO:
            log.debug(f'{type} hyperpolarizability tensor beta({-2*freq};{freq},{freq})')
            log.debug(f'{beta}')
    
    elif type.upper() == 'EOPE' or 'OR':
        mo1_1o = polobj.get_mo1(freq, picture_change, solver)
        f1_p1o = polobj.get_f1vv(mo1_1o, picture_change)
        f1_m1o = (f1_p1o[0].transpose(0,2,1).conj(),
                  f1_p1o[1].transpose(0,2,1).conj())
        e1_p1o = polobj.get_e1(mo1_1o, picture_change)
        e1_m1o = (e1_p1o[0].transpose(0,2,1).conj(),
                  e1_p1o[1].transpose(0,2,1).conj())

        mo1 = polobj.get_mo1(0, picture_change, solver)
        f1 = polobj.get_f1vv(mo1, 0, picture_change)
        e1 = polobj.get_e1(mo1, 0, picture_change)

        # beta(-omega;omega,0)
        beta = -lib.einsum('xjl,yji,zli->xyz', f1_m1o[0], mo1_1o[1].conj(), mo1[0])
        beta +=-lib.einsum('xjl,yji,zli->xyz', f1_m1o[1], mo1_1o[3].conj(), mo1[1])
        beta +=-lib.einsum('xjl,zji,yli->xyz', f1_m1o[0], mo1[0].conj(), mo1_1o[0])
        beta +=-lib.einsum('xjl,zji,yli->xyz', f1_m1o[1], mo1[1].conj(), mo1_1o[2])
        beta +=-lib.einsum('yjl,xji,zli->xyz', f1_p1o[0], mo1_1o[0].conj(), mo1[0])
        beta +=-lib.einsum('yjl,xji,zli->xyz', f1_p1o[1], mo1_1o[2].conj(), mo1[1])
        beta +=-lib.einsum('yjl,zji,xli->xyz', f1_p1o[0], mo1[0].conj(), mo1_1o[1])
        beta +=-lib.einsum('yjl,zji,xli->xyz', f1_p1o[1], mo1[1].conj(), mo1_1o[3])
        beta +=-lib.einsum('zjl,xji,yli->xyz', f1[0], mo1_1o[0].conj(), mo1_1o[0])
        beta +=-lib.einsum('zjl,xji,yli->xyz', f1[1], mo1_1o[2].conj(), mo1_1o[2])
        beta +=-lib.einsum('zjl,yji,xli->xyz', f1[0], mo1_1o[1].conj(), mo1_1o[1])
        beta +=-lib.einsum('zjl,yji,xli->xyz', f1[1], mo1_1o[3].conj(), mo1_1o[3])
        beta += lib.einsum('xji,yki,zkj->xyz', e1_m1o[0], mo1_1o[1].conj(), mo1[0])
        beta += lib.einsum('xji,yki,zkj->xyz', e1_m1o[1], mo1_1o[3].conj(), mo1[1])
        beta += lib.einsum('xji,zki,ykj->xyz', e1_m1o[0], mo1[0].conj(), mo1_1o[0])
        beta += lib.einsum('xji,zki,ykj->xyz', e1_m1o[1], mo1[1].conj(), mo1_1o[2])
        beta += lib.einsum('yji,xki,zkj->xyz', e1_p1o[0], mo1_1o[0].conj(), mo1[0])
        beta += lib.einsum('yji,xki,zkj->xyz', e1_p1o[1], mo1_1o[2].conj(), mo1[1])
        beta += lib.einsum('yji,zki,xkj->xyz', e1_p1o[0], mo1[0].conj(), mo1_1o[1])
        beta += lib.einsum('yji,zki,xkj->xyz', e1_p1o[1], mo1[1].conj(), mo1_1o[3])
        beta += lib.einsum('zji,xki,ykj->xyz', e1[0], mo1_1o[0].conj(), mo1_1o[0])
        beta += lib.einsum('zji,xki,ykj->xyz', e1[1], mo1_1o[2].conj(), mo1_1o[2])
        beta += lib.einsum('zji,yki,xkj->xyz', e1[0], mo1_1o[1].conj(), mo1_1o[1])
        beta += lib.einsum('zji,yki,xkj->xyz', e1[1], mo1_1o[3].conj(), mo1_1o[3])
        
        if isinstance(mf, hf.KohnShamDFT):
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm = mf.make_rdm1()
            dm1 = polobj.get_dm1(mo1=mo1)
            dm1_p1o = polobj.get_dm1(mo1=mo1_1o)
            dm1_m1o = (dm1_p1o[0].transpose(0,2,1).conj(),
                       dm1_p1o[1].transpose(0,2,1).conj())

            if xctype == 'LDA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords)
                rho = numpy.stack([ni.eval_rho(mf.mol, ao, dms, xctype=xctype) for dms in dm])
                rho1a = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                     for dmx in dm1[0]])
                rho1b = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                     for dmx in dm1[1]])
                rho1a_m1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                         for dmx in dm1_m1o[0]])
                rho1b_m1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                         for dmx in dm1_m1o[1]])
                rho1a_p1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                         for dmx in dm1_p1o[0]])
                rho1b_p1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype)
                                         for dmx in dm1_p1o[1]])
                rho1 = numpy.stack((rho1a, rho1b), axis=1).reshape(3,2,1,-1) # reshape to match kxc
                rho1_m1o = numpy.stack((rho1a_m1o, rho1b_m1o), axis=1).reshape(3,2,1,-1) # reshape to match kxc
                rho1_p1o = numpy.stack((rho1a_p1o, rho1b_p1o), axis=1).reshape(3,2,1,-1) # reshape to match kxc
                
            elif xctype == 'GGA' or 'MGGA':
                ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1)
                rho = numpy.stack([ni.eval_rho(mf.mol, ao, dms, xctype=xctype) for dms in dm])
                rho1a = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                 with_lapl=False) for dmx in dm1[0]])
                rho1b = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                 with_lapl=False) for dmx in dm1[1]])
                rho1a_m1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                     with_lapl=False) for dmx in dm1_m1o[0]])
                rho1b_m1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                     with_lapl=False) for dmx in dm1_m1o[1]])
                rho1a_p1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                     with_lapl=False) for dmx in dm1_p1o[0]])
                rho1b_p1o = numpy.stack([ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                     with_lapl=False) for dmx in dm1_p1o[1]])
                rho1 = numpy.stack((rho1a, rho1b), axis=1)
                rho1_m1o = numpy.stack((rho1a_m1o, rho1b_m1o), axis=1)
                rho1_p1o = numpy.stack((rho1a_p1o, rho1b_p1o), axis=1)
                
            else:
                raise RuntimeError(f"Unknown xctype '{xctype}'")
            
            kxc = ni.eval_xc_eff(mf.xc, rho, deriv=3)[3] * mf.grids.weights
            beta -= lib.einsum('xaig,ybjg,zckg,aibjckg->xyz', rho1_m1o, rho1_p1o, rho1, kxc)

        if type.upper() == 'EOPE':
            if mf.verbose >= logger.INFO:
                log.debug(f'{type} hyperpolarizability tensor beta({-freq};{freq},0)')
                log.debug(f'{beta}')
        
        if type.upper() == 'OR':
            if mf.verbose >= logger.INFO:
                log.debug(f'{type} hyperpolarizability tensor beta(0;{freq},{-freq})')
                log.debug(f'{beta.transpose(2,1,0)}')
                
            beta = beta.transpose(2,1,0)

    else:
        raise NotImplementedError(f'{type}')
    
    return beta

# Solve the frequency-dependent CPHF problem
# [A-wI, B   ] [X] + [h1] = [0]
# [B   , A+wI] [Y]   [h1]   [0]
def krylov_solver(polobj, freq=0, picture_change=True, verbose=logger.WARN):
    '''
    1st-order UCPHF/KS iterative solver by `pyscf.lib.krylov`.
    '''
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    ea_vo, eb_vo = polobj._get_e_vo()
    nvira, nocca = ea_vo.shape
    nvirb, noccb = eb_vo.shape
    h1a, h1b = polobj.get_h1vo(picture_change)
    
    if freq == 0:
        def vind(mo1):
            mo1 = mo1.reshape(3,-1)
            mo1a, mo1b = numpy.split(mo1, [nvira*nocca], axis=1)
            mo1 = (mo1a.reshape(3,nvira,nocca), mo1b.reshape(3,nvirb,noccb))

            v1a, v1b = polobj.get_v1vo(mo1)
            v1 = numpy.hstack(((v1a/ea_vo).reshape(3,-1),
                               (v1b/eb_vo).reshape(3,-1)))
            return v1.ravel()
        
        mo1base = -numpy.hstack(((h1a/ea_vo).reshape(3,-1),
                                 (h1b/eb_vo).reshape(3,-1)))
        mo1 = lib.krylov(vind, mo1base.ravel(), max_cycle=polobj.max_cycle_cphf,
                         tol=polobj.conv_tol, verbose=log).reshape(3,-1)
    
        mo1a, mo1b = numpy.split(mo1, [nvira*nocca], axis=1)
        mo1 = (mo1a.reshape(3,nvira,nocca), mo1b.reshape(3,nvirb,noccb))
        
    else:
        sep = numpy.cumsum((nocca*nvira, nocca*nvira, noccb*nvirb))
        def vind(mo1):
            mo1 = mo1.reshape(3,-1)
            mo1_pa, mo1_ma, mo1_pb, mo1_mb = numpy.split(mo1, sep, axis=1)
            mo1 = (mo1_pa.reshape(3,nvira,nocca), mo1_ma.reshape(3,nvira,nocca),
                   mo1_pb.reshape(3,nvirb,noccb), mo1_mb.reshape(3,nvirb,noccb))

            v1 = rhf.get_v1(polobj, mo1)
            v1_pa, v1_pb = polobj._to_vo(v1)
            v1_ma, v1_mb = polobj._to_vo((v1[0].transpose(0,2,1).conj(),
                                         v1[1].transpose(0,2,1).conj()))

            v1 = numpy.hstack(((v1_pa/(ea_vo+freq)).reshape(3,-1),
                               (v1_ma/(ea_vo-freq)).reshape(3,-1),
                               (v1_pb/(eb_vo+freq)).reshape(3,-1),
                               (v1_mb/(eb_vo-freq)).reshape(3,-1)))
            return v1.ravel()

        mo1base = -numpy.hstack(((h1a/(ea_vo+freq)).reshape(3,-1),
                                 (h1a/(ea_vo-freq)).reshape(3,-1),
                                 (h1b/(eb_vo+freq)).reshape(3,-1),
                                 (h1b/(eb_vo-freq)).reshape(3,-1)))
        mo1 = lib.krylov(vind, mo1base.ravel(), max_cycle=polobj.max_cycle_cphf,
                         tol=polobj.conv_tol, verbose=log).reshape(3,-1)
        
        mo1_pa, mo1_ma, mo1_pb, mo1_mb = numpy.split(mo1, sep, axis=1)
        mo1 = (mo1_pa.reshape(3,nvira,nocca), mo1_ma.reshape(3,nvira,nocca),
               mo1_pb.reshape(3,nvirb,noccb), mo1_mb.reshape(3,nvirb,noccb))
    
    log.timer('Krylov solver in UCPHF/KS', *t0)
    return mo1

def newton_solver(polobj, freq=0, picture_change=True, verbose=logger.WARN):
    '''
    1st-order CPHF/KS iterative solver by `scipy.optimize.newton_krylov`.
    '''
    from scipy.optimize import newton_krylov
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    ea_vo, eb_vo = polobj._get_e_vo()
    nvira, nocca = ea_vo.shape
    nvirb, noccb = eb_vo.shape
    h1a, h1b = polobj.get_h1vo(picture_change)
    h1a = h1a.reshape(3,-1)
    h1b = h1b.reshape(3,-1)

    if freq == 0:
        h1 = numpy.hstack((h1a, h1b))
        def vind(mo1):
            mo1a, mo1b = numpy.split(mo1, [nvira*nocca], axis=1)
            mo1 = (mo1a.reshape(3,nvira,nocca), mo1b.reshape(3,nvirb,noccb))

            v1a, v1b = polobj.get_v1vo(mo1)
            v1a += ea_vo * mo1[0]
            v1b += eb_vo * mo1[1]

            v1 = numpy.hstack((v1a.reshape(3,-1), v1b.reshape(3,-1)))
            return v1 + h1
        
        mo1 = newton_krylov(vind, -h1, maxiter=polobj.max_cycle_cphf, f_tol=polobj.conv_tol)
    
        mo1a, mo1b = numpy.split(mo1, [nvira*nocca], axis=1)
        mo1 = (mo1a.reshape(3,nvira,nocca), mo1b.reshape(3,nvirb,noccb))
        
    else:
        h1 = numpy.hstack((h1a, h1a, h1b, h1b))
        sep = numpy.cumsum((nocca*nvira, nocca*nvira, noccb*nvirb))
        def vind(mo1):
            mo1_pa, mo1_ma, mo1_pb, mo1_mb = numpy.split(mo1, sep, axis=1)
            mo1 = (mo1_pa.reshape(3,nvira,nocca), mo1_ma.reshape(3,nvira,nocca),
                   mo1_pb.reshape(3,nvirb,noccb), mo1_mb.reshape(3,nvirb,noccb))

            v1 = rhf.get_v1(polobj, mo1)
            v1_pa, v1_pb = polobj._to_vo(v1)
            v1_ma, v1_mb = polobj._to_vo((v1[0].transpose(0,2,1).conj(),
                                         v1[1].transpose(0,2,1).conj()))

            v1_pa += (ea_vo + freq) * mo1[0]
            v1_ma += (ea_vo - freq) * mo1[1]
            v1_pb += (eb_vo + freq) * mo1[2]
            v1_mb += (eb_vo - freq) * mo1[3]

            v1 = numpy.hstack((v1_pa.reshape(3,-1), v1_ma.reshape(3,-1),
                               v1_pb.reshape(3,-1), v1_mb.reshape(3,-1)))
            return v1 + h1

        mo1 = newton_krylov(vind, -h1, maxiter=polobj.max_cycle_cphf, f_tol=polobj.conv_tol)
        
        mo1_pa, mo1_ma, mo1_pb, mo1_mb = numpy.split(mo1, sep, axis=1)
        mo1 = (mo1_pa.reshape(3,nvira,nocca), mo1_ma.reshape(3,nvira,nocca),
               mo1_pb.reshape(3,nvirb,noccb), mo1_mb.reshape(3,nvirb,noccb))

    log.timer('Newton-Krylov solver in UCPHF/KS', *t0)
    return mo1

def direct_solver(polobj, freq=0, picture_change=True, verbose=logger.WARN):
    '''
    1st-order CPHF/KS direct solver by `numpy.linalg.solve`.
    '''
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    ea_vo, eb_vo = polobj._get_e_vo()
    nvira, nocca = ea_vo.shape
    nvirb, noccb = eb_vo.shape
    h1a, h1b = polobj.get_h1vo(picture_change)
    h1a = h1a.reshape(3,-1)
    h1b = h1b.reshape(3,-1)
    h1 = numpy.hstack((h1a, h1a.conj(), h1b, h1b.conj()))
    sep = numpy.cumsum((nocca*nvira, nocca*nvira, noccb*nvirb))
    
    def vind(mo1):
        mo1_pa, mo1_ma, mo1_pb, mo1_mb = numpy.split(mo1, sep)
        mo1 = (mo1_pa.reshape(1,nvira,nocca), mo1_ma.reshape(1,nvira,nocca).conj(),
               mo1_pb.reshape(1,nvirb,noccb), mo1_mb.reshape(1,nvirb,noccb).conj())

        v1 = rhf.get_v1(polobj, mo1)
        v1_pa, v1_pb = polobj._to_vo(v1)
        v1_ma, v1_mb = polobj._to_vo((v1[0].transpose(0,2,1).conj(),
                                     v1[1].transpose(0,2,1).conj()))

        v1_pa += (ea_vo + freq) * mo1[0]
        v1_ma += (ea_vo - freq) * mo1[1]
        v1_pb += (eb_vo + freq) * mo1[2]
        v1_mb += (eb_vo - freq) * mo1[3]

        v1 = numpy.concatenate((v1_pa.ravel(), v1_ma.ravel().conj(),
                                v1_pb.ravel(), v1_mb.ravel().conj()))
        return v1

    size = h1[0].size
    operator = numpy.empty((size, size))
    iden = numpy.eye(size)
    for i, row in enumerate(iden):
        operator[:,i] = vind(row)
    
    mo1 = numpy.linalg.solve(operator, -h1.T).T
    mo1_pa, mo1_ma, mo1_pb, mo1_mb = numpy.split(mo1, sep, axis=1)
    mo1_pa = mo1_pa.reshape(3,nvira,nocca)
    mo1_ma = mo1_ma.reshape(3,nvira,nocca).conj()
    mo1_pb = mo1_pb.reshape(3,nvirb,noccb)
    mo1_mb = mo1_mb.reshape(3,nvirb,noccb).conj()
    
    if freq == 0: mo1 = (mo1_pa, mo1_pb)
    else: mo1 = (mo1_pa, mo1_ma, mo1_pb, mo1_mb)
    
    log.timer('Direct solver in UCPHF/KS', *t0)
    return mo1


class Polarizability(rhf.Polarizability):
    def get_mo1(self, freq=0, picture_change=True, solver='krylov'):
        '''
        Generate the transformation matrix $U(0)$ or ($U(+\omega)$, $U(-\omega)$),
        which is the coefficient matrix of the 1st-order derivative of MO.
        '''
        solver = solver.lower()
        if   'direct' in solver: # exact solver
            mo1 = direct_solver(self, freq, picture_change)
        elif 'newton' in solver: # only newton-krylov recommended
            mo1 = newton_solver(self, freq, picture_change)
        elif 'krylov' in solver: # not recommended
            mo1 = krylov_solver(self, freq, picture_change)
        else:
            raise NotImplementedError(solver)
        return mo1
    
    @lib.with_doc(get_dm1.__doc__)
    def get_dm1(self, freq=None, mo1=None, picture_change=True, solver='krylov'):
        if freq is None: freq = 0
        if mo1 is None: mo1 = self.get_mo1(freq, picture_change, solver)
        return get_dm1(self, mo1)

    def _get_e_vo(self) : return get_e_vo(self.mf)
    def _to_vo(self, ao): return _to_vo(self.mf, ao)
    def _to_vv(self, ao): return _to_vv(self.mf, ao)
    def _to_oo(self, ao): return _to_oo(self.mf, ao)

    polarizability = polarizability
    hyperpolarizability = hyperpolarizability


uhf.UHF.Polarizability = lib.class_as_method(Polarizability)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    # Disagreement between analytical results and finite difference found for
    # linear molecule
    #mol.atom = '''h  ,  0.   0.   0.
    #              F  ,  0.   0.   .917'''

    mol.atom='''O      0.   0.       0.
                        H      0.  -0.757    0.587
                        H      0.   0.757    0.587'''
    mol.spin = 2
    mol.basis = '631g'
    mol.build()

    mf = scf.UHF(mol).run(conv_tol=1e-14)
    polobj = mf.Polarizability().polarizability()
    hpol = mf.Polarizability().hyper_polarizability()
    print(polobj)

    mf.verbose = 0
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    h1 = mf.get_hcore()
    def apply_E(E):
        mf.get_hcore = lambda *args, **kwargs: h1 + numpy.einsum('x,xij->ij', E, ao_dip)
        mf.run(conv_tol=1e-14)
        return mf.dip_moment(mol, mf.make_rdm1(), unit='AU', verbose=0)
    e1 = apply_E([ 0.0001, 0, 0])
    e2 = apply_E([-0.0001, 0, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([0, 0.0001, 0])
    e2 = apply_E([0,-0.0001, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([0, 0, 0.0001])
    e2 = apply_E([0, 0,-0.0001])
    print((e1 - e2) / 0.0002)

    # Small discrepancy found between analytical derivatives and finite
    # differences
    print(hpol)
    def apply_E(E):
        mf.get_hcore = lambda *args, **kwargs: h1 + numpy.einsum('x,xij->ij', E, ao_dip)
        mf.run(conv_tol=1e-14)
        return Polarizability(mf).polarizability()
    e1 = apply_E([ 0.0001, 0, 0])
    e2 = apply_E([-0.0001, 0, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([0, 0.0001, 0])
    e2 = apply_E([0,-0.0001, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([0, 0, 0.0001])
    e2 = apply_E([0, 0,-0.0001])
    print((e1 - e2) / 0.0002)

    print(Polarizability(mf).polarizability())
    print(Polarizability(mf).polarizability_with_freq(freq= 0.))

    print(Polarizability(mf).polarizability_with_freq(freq= 0.1))
    print(Polarizability(mf).polarizability_with_freq(freq=-0.1))

