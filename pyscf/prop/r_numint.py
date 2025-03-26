from pyscf.dft.r_numint import *
from pyscf.dft.r_numint import (_col_rho_tm2ud,
    _col_lda_fxc_mat, _col_gga_fxc_mat, _col_mgga_fxc_mat,
    _mcol_lda_fxc_mat, _mcol_gga_fxc_mat, _mcol_mgga_fxc_mat)

def r_kxc(ni, mol, grids, xc_code, dm0, dms, hermi=(0,0),
          max_memory=2000, verbose=None):
    '''Calculate 2-component or 4-component Kxc matrix in j-adapted basis.
    '''
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()
    n2c = ao_loc[-1]
    if ni.collinear[0] not in ('c', 'm'):  # col or mcol
        raise NotImplementedError('non-collinear fxc')

    make_rho10, nset, nao = ni._gen_rho_evaluator(mol, dms[0], hermi[0])
    make_rho11 = ni._gen_rho_evaluator(mol, dms[1], hermi[1])[0]
    with_s = (nao == n2c*2)  # 4C DM

    matLL = numpy.zeros((nset,nset,n2c,n2c), dtype=numpy.complex128)
    matSS = numpy.zeros_like(matLL)
    if xctype in ('LDA', 'GGA', 'MGGA'):
        f_eval_mat = {
            ('LDA' , 'c'): (_col_lda_fxc_mat   , 0),
            ('GGA' , 'c'): (_col_gga_fxc_mat   , 1),
            ('MGGA', 'c'): (_col_mgga_fxc_mat  , 1),
            ('LDA' , 'm'): (_mcol_lda_fxc_mat  , 0),
            ('GGA' , 'm'): (_mcol_gga_fxc_mat  , 1),
            ('MGGA', 'm'): (_mcol_mgga_fxc_mat , 1),
        }
        fmat, ao_deriv = f_eval_mat[(xctype, ni.collinear[0])]

        if ni.collinear[0] == 'm':
            eval_xc = ni.mcfun_eval_xc_adapter(xc_code)
        else:
            eval_xc = ni.eval_xc_eff
        
        for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory,
                             with_s=with_s):
            rho0 = ni.eval_rho(mol, ao, dm0, mask, xctype,
                               hermi=1, with_lapl=False).real
            kxc = eval_xc(xc_code, rho0, deriv=3, xctype=xctype)[3]
            for i, j in numpy.ndindex(nset, nset):
                rho10 = make_rho10(i, ao, mask, xctype)
                rho11 = make_rho11(j, ao, mask, xctype)
                if ni.collinear[0] == 'c':
                    rho11 = _col_rho_tm2ud(rho11)
                if xctype == 'LDA':
                    _fxc = lib.einsum('cg,aibjcg->aibjg', rho11, kxc[...,0,:])
                else:
                    _fxc = lib.einsum('ckg,aibjckg->aibjg', rho11, kxc)
                matLL[i,j] += fmat(mol, ao[:2], weight, rho0, rho10, _fxc,
                                   mask, shls_slice, ao_loc, all(hermi))
                if with_s:
                    matSS[i,j] += fmat(mol, ao[2:], weight, rho0, rho10, _fxc,
                                       mask, shls_slice, ao_loc, all(hermi), False)
        matLL = matLL.reshape(nset*nset,n2c,n2c)
        matSS = matSS.reshape(nset*nset,n2c,n2c)

        if all(hermi):
            # for (\nabla\mu) \nu + \mu (\nabla\nu)
            matLL = matLL + matLL.conj().transpose(0,2,1)
            if with_s:
                matSS = matSS + matSS.conj().transpose(0,2,1)

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'r_kxc for functional {xc_code}')

    if with_s:
        matSS *= (.5 / lib.param.LIGHT_SPEED)**2
        vmat = numpy.zeros((nset*nset,nao,nao), dtype=numpy.complex128)
        vmat[:,:n2c,:n2c] = matLL
        vmat[:,n2c:,n2c:] = matSS
    else:
        vmat = matLL

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
    return vmat

RNumInt.get_kxc = r_kxc