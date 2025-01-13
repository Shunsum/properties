#!/usr/bin/env python

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, ghf
from pyscf.x2c import x2c, _response_functions
from . import rhf


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
        raise RuntimeError(f"Unknown collinear scheme '{collinear}'")
    rho = numpy.stack([n + s, n - s]) * .5
    return rho

def get_dm1(polobj, mo1):
    '''
    Generate the 1st-order density matrix in AO basis.
    '''
    mf = polobj.mf
    mo_coeff = mf.mo_coeff
    occidx = mf.mo_occ > 0
    orbv = mo_coeff[:,~occidx]
    orbo = mo_coeff[:, occidx]
    if len(mo1) == 2:
        dm1  = lib.einsum('pj,xji,qi->xpq', orbv, mo1[0]       , orbo.conj())
        dm1 += lib.einsum('pi,xji,qj->xpq', orbo, mo1[1].conj(), orbv.conj())
    elif len(mo1) == 3:
        dm1  = lib.einsum('pj,xji,qi->xpq', orbv, mo1, orbo.conj())
        dm1 += dm1.transpose(0,2,1).conj()
    else:
        raise ValueError('mo1 does not have the correct shape (3,nvir,nocc)'
                         ' for freq = 0 or (2,3,nvir,nocc) for freq != 0.')
    return dm1

def get_h1(polobj, picture_change=True):
    '''
    Generate the dipole matrix in AO basis.
    '''
    mf = polobj.mf
    mol = mf.mol
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = lib.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        ao_dip = numpy.kron(numpy.eye(2), ao_dip)
        if isinstance(mf, x2c.X2C1E_GSCF) and picture_change:
            xmol = mf.with_x2c.get_xmol()[0]
            ao_dip = xmol.intor_symmetric('int1e_r')
            ao_dip = numpy.kron(numpy.eye(2), ao_dip)
            nao = xmol.nao
            prp = xmol.intor_symmetric('int1e_sprsp').reshape(3,4,nao,nao)
            prp = _dirac_relation(prp)
            c = lib.param.LIGHT_SPEED
            ao_dip = mf.with_x2c.picture_change((ao_dip, prp/(2*c)**2))
    return -ao_dip

# Note: polarizability and relevant properties are demanding on basis sets.
# ORCA recommends to use Sadlej basis for these properties.
def polarizability(polobj, freq=0, picture_change=True, solver='krylov'):
    '''
    Polarizability with picture change correction.
    '''
    log = logger.new_logger(polobj)
    mf = polobj.mf
    h1 = polobj.get_h1vo(picture_change)
    mo1 = polobj.get_mo1(freq, picture_change, solver)
    
    if freq == 0:
        # alpha(0;0)
        alpha  = -lib.einsum('xji,yji->xy', h1.conj(), mo1)
        alpha += alpha.conj()

    else:
        # alpha(-omega;omega)
        alpha  = -lib.einsum('xji,yji->xy', h1.conj(), mo1[0])
        alpha += -lib.einsum('xji,yji->xy', h1, mo1[1].conj())

    if mf.verbose >= logger.INFO:
        xx, yy, zz = alpha.diagonal()
        log.note('Isotropic polarizability %.12g', (xx+yy+zz)/3)
        log.note('Polarizability anisotropy %.12g',
                 (.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2))**.5)
        log.debug(f'Dynamic polarizability tensor alpha({-freq};{freq})')
        log.debug(f'{alpha}')

    return alpha.real

def hyperpolarizability(polobj, freq=0, type='SHG', picture_change=True, solver='krylov'):
    '''
    Hyperpolarizability with picture change correction.
    '''
    log = logger.new_logger(polobj)
    mf = polobj.mf

    if freq == 0:
        mo1 = polobj.get_mo1(freq, picture_change, solver)
        f1 = polobj.get_f1vv(mo1, picture_change)
        e1 = polobj.get_e1(mo1, picture_change)

        # beta(0;0,0)
        beta = -lib.einsum('xjl,yji,zli->xyz', f1, mo1.conj(), mo1)
        beta += lib.einsum('xik,yjk,zji->xyz', e1, mo1.conj(), mo1)
        beta += beta.transpose(0,2,1) + beta.transpose(1,0,2) + \
                beta.transpose(1,2,0) + beta.transpose(2,0,1) + beta.transpose(2,1,0)
                
        if isinstance(mf, hf.KohnShamDFT):
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm = mf.make_rdm1()
            dm1 = polobj.get_dm1(mo1=mo1)

            if xctype == 'LDA':
                if isinstance(mf, x2c.SCF):
                    ao = ni.eval_ao(mf.mol, mf.grids.coords, with_s=False)
                else:
                    ao = ni.eval_ao(mf.mol, mf.grids.coords)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype).real
                rho1 = numpy.stack([rho_eff(ni.eval_rho(mf.mol, ao, dmx, xctype=xctype),
                                            ni.collinear) for dmx in dm1])
                rho1 = rho1[:,:,None,:] # reshape to match kxc
                
            elif xctype == 'GGA' or 'MGGA':
                if isinstance(mf, x2c.SCF):
                    ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1, with_s=False)
                else:
                    ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype, with_lapl=False).real
                rho1 = numpy.stack([rho_eff(ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                        with_lapl=False),
                                            ni.collinear) for dmx in dm1])
                
            else:
                raise RuntimeError(f"Unknown xctype '{xctype}'")
            
            if ni.collinear[0] == 'm': 
                eval_xc = ni.mcfun_eval_xc_adapter(mf.xc)
            else:
                eval_xc = ni.eval_xc_eff
            
            kxc = eval_xc(mf.xc, rho, deriv=3)[3] * mf.grids.weights
            beta -= lib.einsum('xaig,ybjg,zckg,aibjckg->xyz', rho1, rho1, rho1, kxc)

        if mf.verbose >= logger.INFO:
            log.debug(f'Static hyperpolarizability tensor beta({freq};{freq},{freq})')
            log.debug(f'{beta}')
    
    elif type.upper() == 'SHG':
        mo1_2o = polobj.get_mo1(freq*2, picture_change, solver)
        f1_m2o = polobj.get_f1vv(mo1_2o, picture_change).transpose(0,2,1).conj()
        e1_m2o = polobj.get_e1(mo1_2o, picture_change).transpose(0,2,1).conj()

        mo1_1o = polobj.get_mo1(freq, picture_change, solver)
        f1_p1o = polobj.get_f1vv(mo1_1o, picture_change)
        e1_p1o = polobj.get_e1(mo1_1o, picture_change)

        # beta(-2omega;omega,omega)
        beta = -lib.einsum('xjl,yji,zli->xyz', f1_m2o, mo1_1o[1].conj(), mo1_1o[0])
        beta +=-lib.einsum('yjl,xji,zli->xyz', f1_p1o, mo1_2o[0].conj(), mo1_1o[0])
        beta +=-lib.einsum('zjl,yji,xli->xyz', f1_p1o, mo1_1o[1].conj(), mo1_2o[1])
        beta += lib.einsum('xik,yjk,zji->xyz', e1_m2o, mo1_1o[1].conj(), mo1_1o[0])
        beta += lib.einsum('yik,xjk,zji->xyz', e1_p1o, mo1_2o[0].conj(), mo1_1o[0])
        beta += lib.einsum('zik,yjk,xji->xyz', e1_p1o, mo1_1o[1].conj(), mo1_2o[1])
        beta += beta.transpose(0,2,1)
            
        if isinstance(mf, hf.KohnShamDFT):
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm = mf.make_rdm1()
            dm1_m2o = polobj.get_dm1(mo1=mo1_2o).transpose(0,2,1).conj()
            dm1_p1o = polobj.get_dm1(mo1=mo1_1o)

            if xctype == 'LDA':
                if isinstance(mf, x2c.SCF):
                    ao = ni.eval_ao(mf.mol, mf.grids.coords, with_s=False)
                else:
                    ao = ni.eval_ao(mf.mol, mf.grids.coords)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype).real
                rho1_m2o = numpy.stack([rho_eff(ni.eval_rho(mf.mol, ao, dmx, xctype=xctype),
                                                ni.collinear) for dmx in dm1_m2o])
                rho1_m2o = rho1_m2o[:,:,None,:] # reshape to match kxc
                rho1_p1o = numpy.stack([rho_eff(ni.eval_rho(mf.mol, ao, dmx, xctype=xctype),
                                                ni.collinear) for dmx in dm1_p1o])
                rho1_p1o = rho1_p1o[:,:,None,:] # reshape to match kxc
            
            elif xctype == 'GGA' or 'MGGA':
                if isinstance(mf, x2c.SCF):
                    ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1, with_s=False)
                else:
                    ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype, with_lapl=False).real
                rho1_m2o = numpy.stack([rho_eff(ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                            with_lapl=False),
                                                ni.collinear) for dmx in dm1_m2o])
                rho1_p1o = numpy.stack([rho_eff(ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                            with_lapl=False),
                                                ni.collinear) for dmx in dm1_p1o])

            else:
                raise RuntimeError(f"Unknown xctype '{xctype}'")
        
            if ni.collinear[0] == 'm': 
                eval_xc = ni.mcfun_eval_xc_adapter(mf.xc)
            else:
                eval_xc = ni.eval_xc_eff
            
            kxc = eval_xc(mf.xc, rho, deriv=3)[3] * mf.grids.weights
            beta -= lib.einsum('xaig,ybjg,zckg,aibjckg->xyz',
                                rho1_m2o, rho1_p1o, rho1_p1o, kxc)
        
        if mf.verbose >= logger.INFO:
            log.debug(f'{type} hyperpolarizability tensor beta({-2*freq};{freq},{freq})')
            log.debug(f'{beta}')
        
    elif type.upper() == 'EOPE' or 'OR':    
        mo1_1o = polobj.get_mo1(freq, picture_change, solver)
        f1_p1o = polobj.get_f1vv(mo1_1o, picture_change)
        f1_m1o = f1_p1o.transpose(0,2,1).conj()
        e1_p1o = polobj.get_e1(mo1_1o, picture_change)
        e1_m1o = e1_p1o.transpose(0,2,1).conj()

        mo1 = polobj.get_mo1(0, picture_change, solver)
        f1 = polobj.get_f1vv(mo1, 0, picture_change)
        e1 = polobj.get_e1(mo1, 0, picture_change)

        # beta(-omega;omega,0)
        beta = -lib.einsum('xjl,yji,zli->xyz', f1_m1o, mo1_1o[1].conj(), mo1)
        beta +=-lib.einsum('xjl,zji,yli->xyz', f1_m1o, mo1.conj(), mo1_1o[0])
        beta +=-lib.einsum('yjl,xji,zli->xyz', f1_p1o, mo1_1o[0].conj(), mo1)
        beta +=-lib.einsum('yjl,zji,xli->xyz', f1_p1o, mo1.conj(), mo1_1o[1])
        beta +=-lib.einsum('zjl,xji,yli->xyz', f1, mo1_1o[0].conj(), mo1_1o[0])
        beta +=-lib.einsum('zjl,yji,xli->xyz', f1, mo1_1o[1].conj(), mo1_1o[1])
        beta += lib.einsum('xik,yjk,zji->xyz', e1_m1o, mo1_1o[1].conj(), mo1)
        beta += lib.einsum('xik,zjk,yji->xyz', e1_m1o, mo1.conj(), mo1_1o[0])
        beta += lib.einsum('yik,xjk,zji->xyz', e1_p1o, mo1_1o[0].conj(), mo1)
        beta += lib.einsum('yik,zjk,xji->xyz', e1_p1o, mo1.conj(), mo1_1o[1])
        beta += lib.einsum('zik,xjk,yji->xyz', e1, mo1_1o[0].conj(), mo1_1o[0])
        beta += lib.einsum('zik,yjk,xji->xyz', e1, mo1_1o[1].conj(), mo1_1o[1])
            
        if isinstance(mf, hf.KohnShamDFT):
            ni = mf._numint
            ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
            xctype = ni._xc_type(mf.xc)
            dm = mf.make_rdm1()
            dm1 = polobj.get_dm1(mo1=mo1)
            dm1_p1o = polobj.get_dm1(mo1=mo1_1o)
            dm1_m1o = dm1_p1o.transpose(0,2,1).conj()

            if xctype == 'LDA':
                if isinstance(mf, x2c.SCF):
                    ao = ni.eval_ao(mf.mol, mf.grids.coords, with_s=False)
                else:
                    ao = ni.eval_ao(mf.mol, mf.grids.coords)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype).real
                rho1 = numpy.stack([rho_eff(ni.eval_rho(mf.mol, ao, dmx, xctype=xctype),
                                            ni.collinear) for dmx in dm1])
                rho1 = rho1[:,:,None,:] # reshape to match kxc
                rho1_m1o = numpy.stack([rho_eff(ni.eval_rho(mf.mol, ao, dmx, xctype=xctype),
                                                ni.collinear) for dmx in dm1_m1o])
                rho1_m1o = rho1_m1o[:,:,None,:] # reshape to match kxc
                rho1_p1o = numpy.stack([rho_eff(ni.eval_rho(mf.mol, ao, dmx, xctype=xctype),
                                                ni.collinear) for dmx in dm1_p1o])
                rho1_p1o = rho1_p1o[:,:,None,:] # reshape to match kxc
                
            elif xctype == 'GGA' or 'MGGA':
                if isinstance(mf, x2c.SCF):
                    ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1, with_s=False)
                else:
                    ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=1)
                rho = ni.eval_rho(mf.mol, ao, dm, xctype=xctype, with_lapl=False).real
                rho1 = numpy.stack([rho_eff(ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                        with_lapl=False),
                                            ni.collinear) for dmx in dm1])
                rho1_m1o = numpy.stack([rho_eff(ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                            with_lapl=False),
                                                ni.collinear) for dmx in dm1_m1o])
                rho1_p1o = numpy.stack([rho_eff(ni.eval_rho(mf.mol, ao, dmx, xctype=xctype,
                                                            with_lapl=False),
                                                ni.collinear) for dmx in dm1_p1o])
            
            else:
                raise RuntimeError(f"Unknown xctype '{xctype}'")
            
            if ni.collinear[0] == 'm': 
                eval_xc = ni.mcfun_eval_xc_adapter(mf.xc)
            else:
                eval_xc = ni.eval_xc_eff
            
            kxc = eval_xc(mf.xc, rho, deriv=3)[3] * mf.grids.weights
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
    
    return beta.real

def _dirac_relation(mat):
    '''Make a tensor product --- sprsp = is prp + I2 prp'''
    quaternion = numpy.vstack([1j * lib.PauliMatrices, numpy.eye(2)[None,:,:]])
    qshape = quaternion.shape
    mshape = mat.shape
    return lib.einsum('sxy,aspq->axpyq', quaternion, mat).reshape(mshape[0],
                                                                  qshape[1]*mshape[2],
                                                                  qshape[2]*mshape[3])


class Polarizability(rhf.Polarizability):
    @lib.with_doc(get_dm1.__doc__)
    def get_dm1(self, freq=None, mo1=None, picture_change=True, solver='krylov'):
        if freq is None: freq = 0
        if mo1 is None: mo1 = self.get_mo1(freq, picture_change, solver)
        return get_dm1(self, mo1)
    
    get_h1 = get_h1

    polarizability = polarizability
    hyperpolarizability = hyperpolarizability


ghf.GHF.Polarizability = x2c.X2C1E_GSCF.Polarizability = lib.class_as_method(Polarizability)
    

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    from pyscf import gto
    mol = gto.M(atom ='''O     0.000000     0.000000     0.123323
                         H     0.000000     0.757497    -0.493291
                         H     0.000000    -0.757497    -0.493291''',
                #charge = 1,
                #spin = 1,
                basis = 'ccpvdz')
    
    xc = 'TPSS'
    '''
    mf1 = mol.GHF()
    mf1.conv_tol = 1e-11
    mf1.kernel()
    polar1 = Polarizability(mf1)
    #polar1.conv_tol = 1e-11
    print(polar1.polarizability())
    print(polar1.hyper_polarizability())
    '''
    mfg = mol.GKS()
    mfg.xc = xc
    mfg.conv_tol = 1e-11
    mfg.kernel()
    plg = Polarizability(mfg)
    print(plg.polarizability())

    mfg.collinear = 'mcol'
    mfg.kernel()
    plgm = Polarizability(mfg)
    print(plgm.polarizability())

    mfx = mol.GKS().x2c()
    mfx.xc = xc
    mfx.conv_tol = 1e-11
    mfx.kernel()
    plx = Polarizability(mfx)
    print(plx.polarizability())

    mfx.collinear = 'mcol'
    mfx.kernel()
    plxm = Polarizability(mfx)
    print(plxm.polarizability())