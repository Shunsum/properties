from pyscf.dft.numint import *
from pyscf.dft.numint import _scale_ao_sparse, _dot_ao_ao_sparse, _tau_dot_sparse

# ValueError: operands could not be broadcast together with remapped shapes [original->remapped]
def __nr_rks_kxc1(ni, mol, grids, xc_code, dm0, dms, hermi=0,
                rho1=None, kxc=None, max_memory=2000, verbose=None):
    fxc = lib.einsum('xjg,ijkg->xikg', rho1, kxc)
    kxcmat = numpy.array([
        nr_rks_fxc(ni, mol, grids, xc_code, dm0, dms, hermi=hermi, fxc=fxc[x], max_memory=max_memory, verbose=verbose)
        for x in range(3)])
    return kxcmat.reshape(-1,*kxcmat.shape[-2:])

def nr_rks_kxc(ni, mol, grids, xc_code, dm0, dms, hermi=(0,0),
               max_memory=2000, verbose=None):
    '''Contract RKS KXC matrix with given density matrices

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm0 : 2D array
            Zeroth order density matrix
        dms : 2D array a list of 2D arrays
            First order density matrix or density matrices

    Kwargs:
        hermi : (int,int)
            First order density matrix symmetric or not. It also indicates
            whether the matrices in return are symmetric or not.
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        kxc-vmat.
        kxc-vmat is the KXC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.

    Examples:

    '''
    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)
    if not all(hermi) and dtype != numpy.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)

    make_rho10, nset, nao = ni._gen_rho_evaluator(mol, dms[0], hermi[0], False, grids)
    make_rho11 = ni._gen_rho_evaluator(mol, dms[1], hermi[1], False, grids)[0]

    def block_loop(ao_deriv):
        for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            rho0 = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
            kxc = ni.eval_xc_eff(xc_code, rho0, deriv=3, xctype=xctype)[3]
            for i, j in numpy.ndindex(nset, nset):
                rho10 = make_rho10(i, ao, mask, xctype)
                rho11 = make_rho11(j, ao, mask, xctype)
                if xctype == 'LDA':
                    wv = rho10 * rho11 * kxc[0,0,0] * weight
                else:
                    wv = lib.einsum('jg,kg,ijkg,g->ig', rho10, rho11, kxc, weight)
                yield i, j, ao, mask, wv

    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    vmat = numpy.zeros((nset,nset,nao,nao))
    aow = None

    if xctype == 'LDA':
        ao_deriv = 0
        for i, j, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv, nbins, mask, pair_mask, ao_loc,
                              all(hermi), vmat[i,j])
        vmat = vmat.reshape(nset*nset,nao,nao)

    elif xctype == 'GGA':
        ao_deriv = 1
        for i, j, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5  # *.5 for v+v.conj().T
            aow = _scale_ao_sparse(ao, wv, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              all(hermi), out=vmat[i,j])
        # For real orbitals, K_{ia,bj} = K_{ia,jb}. It simplifies real fxc_jb
        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat.reshape(nset*nset,nao,nao), axes=(0,2,1))

    elif xctype == 'MGGA':
        assert not MGGA_DENSITY_LAPL
        ao_deriv = 2 if MGGA_DENSITY_LAPL else 1
        v1 = numpy.zeros_like(vmat)
        for i, j, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5  # *.5 for v+v.conj().T
            wv[4] *= .5  # *.5 for 1/2 in tau
            aow = _scale_ao_sparse(ao, wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              all(hermi), out=vmat[i,j])
            _tau_dot_sparse(ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i,j])
        vmat = lib.hermi_sum(vmat.reshape(nset*nset,nao,nao), axes=(0,2,1))
        vmat += v1.reshape(nset*nset,nao,nao)

    if isinstance(dms[0], numpy.ndarray) and dms[0].ndim == 2:
        vmat = vmat[0]
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return vmat

def nr_uks_kxc(ni, mol, grids, xc_code, dm0, dms, hermi=(0,0),
               max_memory=2000, verbose=None):
    '''Contract UKS KXC matrix with given density matrices

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm0 : (2, N, N) array
            Zeroth order density matrices
        dms : 2D array a list of 2D arrays
            First order density matrices

    Kwargs:
        hermi : (int,int)
            First order density matrix symmetric or not. It also indicates
            whether the matrices in return are symmetric or not.
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        kxc-vmat.
        kxc-vmat is the KXC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.

    Examples:

    '''
    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms[0])
    if not all(hermi) and dtype != numpy.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)
    nset, nao = dms[0][0].shape[-3:-1]

    def block_loop(ao_deriv):
        for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            rho0 = numpy.array([ni.eval_rho(mol, ao, dm, mask, xctype, hermi=1,
                                with_lapl=False) for dm in dm0])
            kxc = ni.eval_xc_eff(xc_code, rho0, deriv=3, xctype=xctype)[3]
            for i, j in numpy.ndindex(nset, nset):
                rho10 = numpy.array([ni.eval_rho(mol, ao, dm[i], mask, xctype,
                                     hermi[0], False) for dm in dms[0]])
                rho11 = numpy.array([ni.eval_rho(mol, ao, dm[j], mask, xctype,
                                     hermi[1], False) for dm in dms[1]])
                if xctype == 'LDA':
                    wv = lib.einsum('bg,cg,abcg,g->ag', rho10, rho11,
                                    kxc[:,0,:,0,:,0], weight)
                else:
                    wv = lib.einsum('bjg,ckg,aibjckg,g->aig', rho10, rho11, kxc, weight)
                yield i, j, ao, mask, wv

    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    vmat = numpy.zeros((2,nset,nset,nao,nao))
    aow = None

    if xctype == 'LDA':
        ao_deriv = 0
        for i, j, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv[0], nbins, mask, pair_mask, ao_loc,
                              all(hermi), vmat[0,i,j])
            _dot_ao_ao_sparse(ao, ao, wv[1], nbins, mask, pair_mask, ao_loc,
                              all(hermi), vmat[1,i,j])
        vmat = vmat.reshape(2,nset*nset,nao,nao)

    elif xctype == 'GGA':
        ao_deriv = 1
        for i, j, ao, mask, wv in block_loop(ao_deriv):
            wv[:,0] *= .5
            aow = _scale_ao_sparse(ao, wv[0], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              all(hermi), out=vmat[0,i,j])
            aow = _scale_ao_sparse(ao, wv[1], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              all(hermi), out=vmat[1,i,j])
        # For real orbitals, K_{ia,bj} = K_{ia,jb}. It simplifies real fxc_jb
        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(2,nset*nset,nao,nao)

    elif xctype == 'MGGA':
        assert not MGGA_DENSITY_LAPL
        ao_deriv = 1
        v1 = numpy.zeros_like(vmat)
        for i, j, ao, mask, wv in block_loop(ao_deriv):
            wv[:,0] *= .5
            wv[:,4] *= .5
            aow = _scale_ao_sparse(ao[:4], wv[0,:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              all(hermi), out=vmat[0,i,j])
            _tau_dot_sparse(ao, ao, wv[0,4], nbins, mask, pair_mask, ao_loc, out=v1[0,i,j])
            aow = _scale_ao_sparse(ao[:4], wv[1,:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              all(hermi), out=vmat[1,i,j])
            _tau_dot_sparse(ao, ao, wv[1,4], nbins, mask, pair_mask, ao_loc, out=v1[1,i,j])
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(2,nset*nset,nao,nao)
        vmat += v1.reshape(2,nset*nset,nao,nao)

    if isinstance(dms[0], numpy.ndarray) and dms[0].ndim == 3:
        vmat = vmat[:,0]
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return vmat

def nr_kxc(ni, mol, grids, xc_code, dm0, dms, hermi=(0,0),
           max_memory=2000, verbose=None):
    if mol.spin == 0:
        return ni.nr_rks_kxc(mol, grids, xc_code, dm0, dms,
                             hermi, max_memory, verbose)
    else:
        return ni.nr_uks_kxc(mol, grids, xc_code, dm0, dms,
                             hermi, max_memory, verbose)

NumInt.nr_rks_kxc = nr_rks_kxc
NumInt.nr_uks_kxc = nr_uks_kxc
NumInt.get_kxc = nr_kxc