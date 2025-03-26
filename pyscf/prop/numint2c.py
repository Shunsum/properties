from . import numint
from pyscf.dft.numint2c import *
from pyscf.dft.numint2c import _mcol_lda_fxc_mat, _mcol_gga_fxc_mat, _mcol_mgga_fxc_mat

def nr_gks_kxc(ni, mol, grids, xc_code, dm0, dms, hermi=(0,0),
               max_memory=2000, verbose=None):
    if ni.collinear[0] not in ('c', 'm'):  # col or mcol
        raise NotImplementedError('non-collinear kxc')
    if ni.collinear[0] == 'm':  # mcol
        kxcmat = _gks_mcol_kxc(ni, mol, grids, xc_code, dm0, dms,
                               hermi, max_memory, verbose)
    else:
        dms = np.asarray(dms) # shape: (2,3,nao,nao)
        nao = dms.shape[-1] // 2
        dm0a = dm0[:nao,:nao].real.copy('C')
        dm0b = dm0[nao:,nao:].real.copy('C')
        dm0 = (dm0a, dm0b)
        # dms_a and dms_b may be complex if they are TDDFT amplitudes
        dms_a = dms[...,:nao,:nao].copy()
        dms_b = dms[...,nao:,nao:].copy()
        dm1 = np.stack((dms_a, dms_b), axis=1)
        ni = ni._to_numint1c()
        vmat = ni.nr_uks_kxc(mol, grids, xc_code, dm0, dm1,
                             hermi, max_memory, verbose)
        kxcmat = np.zeros((9,nao*2,nao*2), dtype=vmat.dtype)
        kxcmat[...,:nao,:nao] = vmat[0]
        kxcmat[...,nao:,nao:] = vmat[1]
    return kxcmat

def _gks_mcol_kxc(ni, mol, grids, xc_code, dm0, dms, hermi=(0,0),
                  max_memory=2000, verbose=None):
    assert ni.collinear[0] == 'm'  # mcol
    xctype = ni._xc_type(xc_code)

    if xctype == 'MGGA':
        fmat, ao_deriv = (_mcol_mgga_fxc_mat , 1)
    elif xctype == 'GGA':
        fmat, ao_deriv = (_mcol_gga_fxc_mat  , 1)
    else:
        fmat, ao_deriv = (_mcol_lda_fxc_mat  , 0)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    make_rho10, nset, n2c = ni._gen_rho_evaluator(mol, dms[0], hermi[0], False)
    make_rho11 = ni._gen_rho_evaluator(mol, dms[1], hermi[1], False)[0]
    nao = n2c // 2
    vmat = np.zeros((nset,nset,n2c,n2c), dtype=np.complex128)

    if xctype in ('LDA', 'GGA', 'MGGA'):
        for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho0 = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False).real
            eval_xc_eff = ni.mcfun_eval_xc_adapter(xc_code)
            kxc = eval_xc_eff(xc_code, rho0, deriv=3, xctype=xctype)[3]
            for i, j in np.ndindex(nset, nset):
                rho10 = make_rho10(i, ao, mask, xctype)
                rho11 = make_rho11(j, ao, mask, xctype)
                if xctype == 'LDA':
                    _fxc = lib.einsum('cg,aibjcg->aibjg', rho11, kxc[...,0,:])
                else:
                    _fxc = lib.einsum('ckg,aibjckg->aibjg', rho11, kxc)
                vmat[i,j] += fmat(mol, ao, weight, rho0, rho10, _fxc,
                                  mask, shls_slice, ao_loc, all(hermi))
    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint2c.get_kxc for functional {xc_code}')
    
    vmat = vmat.reshape(nset*nset,n2c,n2c)

    if all(hermi):
        vmat = vmat + vmat.conj().transpose(0,2,1)
    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
    return vmat

NumInt2C.get_kxc = nr_gks_kxc