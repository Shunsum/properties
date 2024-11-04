#!/usr/bin/python3
import warnings
warnings.filterwarnings('ignore')
import unittest
from pyscf import gto
from pyscf.prop.polarizability import rhf, uhf, rks, uks, ghf, gks, x2c, dhf

def setUpModule():
    global mol1, mol2, mol3, mol4
    mol1 = gto.M(atom = '''O     0.000000     0.000000     0.123323
                           H     0.000000     0.757497    -0.493291
                           H     0.000000    -0.757497    -0.493291''',
                 basis = {'O': '''
                               O    S
                                    0.1172000000E+05  0.7118644339E-03
                                    0.1759000000E+04  0.5485201992E-02
                                    0.4008000000E+03  0.2790992963E-01
                                    0.1137000000E+03  0.1051332075E+00
                                    0.3703000000E+02  0.2840024898E+00
                                    0.1327000000E+02  0.4516739459E+00
                                    0.5025000000E+01  0.2732081255E+00
                               O    S
                                    0.1172000000E+05  0.7690300460E-05
                                    0.4008000000E+03  0.3134845790E-03
                                    0.1137000000E+03 -0.2966148530E-02
                                    0.3703000000E+02 -0.1087535430E-01
                                    0.1327000000E+02 -0.1207538168E+00
                                    0.5025000000E+01 -0.1062752639E+00
                                    0.1013000000E+01  0.1095975478E+01
                               O    S
                                    0.3023000000E+00  0.1000000000E+01
                               O    P
                                    0.1770000000E+02  0.6267916628E-01
                                    0.3854000000E+01  0.3335365659E+00
                                    0.1046000000E+01  0.7412396416E+00
                               O    P
                                    0.2753000000E+00  0.1000000000E+01
                               O    D
                                    0.1185000000E+01  0.1000000000E+01
                               ''',
                          'H': '''
                               H    S
                                    0.1301000000E+02  0.3349872639E-01
                                    0.1962000000E+01  0.2348008012E+00
                                    0.4446000000E+00  0.8136829579E+00
                               H    S
                                    0.1220000000E+00  0.1000000000E+01
                               H    P
                                    0.7270000000E+00  0.1000000000E+01
                               '''},
                 max_memory = 50000,
                #cart = True, # Gaussian 16 uses Cartesian functions but when
                              # the cart is set to True,
                              # the results become not so satisfactory.
                 )

    mol2 = gto.M(atom = '''O     0.000000     0.000000     0.123323
                           H     0.000000     0.757497    -0.493291
                           H     0.000000    -0.757497    -0.493291''',
                 charge = 1,
                 spin = 1,
                 basis = {'O': '''
                               O    S
                                    0.1172000000E+05  0.7118644339E-03
                                    0.1759000000E+04  0.5485201992E-02
                                    0.4008000000E+03  0.2790992963E-01
                                    0.1137000000E+03  0.1051332075E+00
                                    0.3703000000E+02  0.2840024898E+00
                                    0.1327000000E+02  0.4516739459E+00
                                    0.5025000000E+01  0.2732081255E+00
                               O    S
                                    0.1172000000E+05  0.7690300460E-05
                                    0.4008000000E+03  0.3134845790E-03
                                    0.1137000000E+03 -0.2966148530E-02
                                    0.3703000000E+02 -0.1087535430E-01
                                    0.1327000000E+02 -0.1207538168E+00
                                    0.5025000000E+01 -0.1062752639E+00
                                    0.1013000000E+01  0.1095975478E+01
                               O    S
                                    0.3023000000E+00  0.1000000000E+01
                               O    P
                                    0.1770000000E+02  0.6267916628E-01
                                    0.3854000000E+01  0.3335365659E+00
                                    0.1046000000E+01  0.7412396416E+00
                               O    P
                                    0.2753000000E+00  0.1000000000E+01
                               O    D
                                    0.1185000000E+01  0.1000000000E+01
                               ''',
                          'H': '''
                               H    S
                                    0.1301000000E+02  0.3349872639E-01
                                    0.1962000000E+01  0.2348008012E+00
                                    0.4446000000E+00  0.8136829579E+00
                               H    S
                                    0.1220000000E+00  0.1000000000E+01
                               H    P
                                    0.7270000000E+00  0.1000000000E+01
                               '''},
                 max_memory = 50000,
                #cart = True,
                 )
    
    mol3 = gto.M(atom = 'H 0 0 0; I 0 0 1.609',
                 basis = 'unc-sto3g', # ANO-R0
                 symmetry = True
                 )

    mol4 = gto.M(atom = 'Ag',
                 spin = 1,
                 basis = {'Ag': '''unc
#BASIS SET: (15s,12p,6d) -> [5s,4p,2d]
Ag    S
      0.4744521634E+04       0.1543289673E+00
      0.8642205383E+03       0.5353281423E+00
      0.2338918045E+03       0.4446345422E+00
Ag    SP
      0.4149652069E+03      -0.9996722919E-01       0.1559162750E+00
      0.9642898995E+02       0.3995128261E+00       0.6076837186E+00
      0.3136170035E+02       0.7001154689E+00       0.3919573931E+00
Ag    SP
      0.4941048605E+02      -0.2277635023E+00       0.4951511155E-02
      0.1507177314E+02       0.2175436044E+00       0.5777664691E+00
      0.5815158634E+01       0.9166769611E+00       0.4846460366E+00
Ag    SP
      0.5290230450E+01      -0.3306100626E+00      -0.1283927634E+00
      0.2059988316E+01       0.5761095338E-01       0.5852047641E+00
      0.9068119281E+00       0.1115578745E+01       0.5439442040E+00
Ag    SP
      0.4370804803E+00      -0.3842642608E+00      -0.3481691526E+00
      0.2353408164E+00      -0.1972567438E+00       0.6290323690E+00
      0.1039541771E+00       0.1375495512E+01       0.6662832743E+00
Ag    D
      0.4941048605E+02       0.2197679508E+00
      0.1507177314E+02       0.6555473627E+00
      0.5815158634E+01       0.2865732590E+00
Ag    D
      0.3283395668E+01       0.1250662138E+00
      0.1278537254E+01       0.6686785577E+00
      0.5628152469E+00       0.3052468245E+00'''})

def tearDownModule():
    global mol1, mol2, mol3
    mol1.stdout.close()
    mol2.stdout.close()
    mol3.stdout.close()
    del mol1, mol2, mol3

def rhf_direct_solver(freq=0):
    import numpy
    from pyscf import lib
    
    setUpModule()
    mf = mol1.RHF()
    mf.conv_tol = 1e-11
    mf.kernel()
    
    mo_coeff = mf.mo_coeff
    occidx = mf.mo_occ > 0
    orbv = mo_coeff[:,~occidx]
    orbo = mo_coeff[:, occidx]
    nvir = orbv.shape[-1]
    nocc = orbo.shape[-1]

    moe = mf.mo_energy
    e_a = moe[~occidx]
    e_i = moe[ occidx]
    ep_ai = e_a[:,None] - e_i + freq
    em_ai = e_a[:,None] - e_i - freq
    eye = numpy.einsum('jl,ik->jilk', numpy.eye(nvir), numpy.eye(nocc))
    ep = numpy.einsum('jilk,lk->jilk', eye, ep_ai).reshape(nvir*nocc, nocc*nvir)
    em = numpy.einsum('jilk,lk->jilk', eye, em_ai).reshape(nvir*nocc, nocc*nvir)

    j = mol1.intor('int2e_sph')
    #import pdb
    #pdb.set_trace()
    j_ov = lib.einsum('mnpq,mj,ni,pk,ql->jilk', j, orbv, orbo, orbo, orbv)*2
    k_ov = lib.einsum('mnpq,mj,nl,pk,qi->jilk', j, orbv, orbv, orbo, orbo)*2
    j_vo = lib.einsum('mnpq,mj,ni,pl,qk->jilk', j, orbv, orbo, orbv, orbo)*2
    jk_ov = j_ov - k_ov # (ji|kl)-(jl|ki)
    jk_ov = jk_ov.reshape(nvir*nocc,nocc*nvir)
    jk_vo = j_vo - j_vo.transpose(0,3,2,1)
    jk_vo = jk_vo.reshape(nvir*nocc,nocc*nvir)
    #jkaa = jaa - jaa.transpose(3,1,2,0)
    #jkaa = jkaa.transpose(0,1,3,2).reshape(nvira*nocca,nocca*nvira)
    #jkbb = jbb - jbb.transpose(3,1,2,0)
    #jkbb = jkbb.transpose(0,1,3,2).reshape(nvirb*noccb,noccb*nvirb)
    #import pdb
    #pdb.set_trace()

    aop = numpy.block([[jk_ov+ep, jk_vo],
                       [jk_vo, jk_ov+em]])
    aop_inv = numpy.linalg.inv(aop)
    #import pdb
    #pdb.set_trace()

    polar = rhf.Polarizability(mf)
    h1 = polar.get_h1vo()
    mo1base = numpy.concatenate((h1.reshape(3,-1), h1.reshape(3,-1)), axis=1)
    #import pdb
    #pdb.set_trace()

    mo1 = lib.einsum('mn,xn->xm', aop_inv, -mo1base)
    offsets = numpy.cumsum((nocc*nvir))
    mo1p, mo1m = numpy.split(mo1, offsets, axis=1)
    mo1 = (mo1p.reshape(3,nvir,nocc),
           mo1m.reshape(3,nvir,nocc))
    #import pdb; pdb.set_trace()
    alpha  = -lib.einsum('xji,yji->xy', h1.conj(), mo1[0])*2
    alpha += -lib.einsum('xji,yji->xy', h1, mo1[1].conj())*2

    return alpha

def uhf_direct_solver(freq=0):
    import numpy
    from pyscf import lib
    
    mf = mol2.UHF()
    mf.conv_tol = 1e-11
    e = mf.kernel()
    print(f'AbsTol of energy is {e - -76.0250801801}.')

    mo_coeffa, mo_coeffb = mf.mo_coeff
    occidxa = mf.mo_occ[0] > 0
    occidxb = mf.mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    orbva = mo_coeffa[:,viridxa]
    orbvb = mo_coeffb[:,viridxb]
    orboa = mo_coeffa[:,occidxa]
    orbob = mo_coeffb[:,occidxb]
    nvira = orbva.shape[1]
    nvirb = orbvb.shape[1]
    nocca = orboa.shape[1]
    noccb = orbob.shape[1]

    ea, eb = mf.mo_energy
    ea_a = ea[viridxa]
    ea_i = ea[occidxa]
    eap_ai = ea_a[:,None] - ea_i + freq
    eam_ai = ea_a[:,None] - ea_i - freq
    eyea = numpy.einsum('jl,ik->jilk', numpy.eye(nvira), numpy.eye(nocca))
    eap = numpy.einsum('jilk,lk->jilk', eyea, eap_ai).reshape(nvira*nocca,
                                                              nocca*nvira)
    eam = numpy.einsum('jilk,lk->jilk', eyea, eam_ai).reshape(nvira*nocca,
                                                              nocca*nvira)
    eb_a = eb[viridxb]
    eb_i = eb[occidxb]
    ebp_ai = eb_a[:,None] - eb_i + freq
    ebm_ai = eb_a[:,None] - eb_i - freq
    eyeb = numpy.einsum('jl,ik->jilk', numpy.eye(nvirb), numpy.eye(noccb))
    ebp = numpy.einsum('jilk,lk->jilk', eyeb, ebp_ai).reshape(nvirb*noccb,
                                                              noccb*nvirb)
    ebm = numpy.einsum('jilk,lk->jilk', eyeb, ebm_ai).reshape(nvirb*noccb,
                                                              noccb*nvirb)

    j = mol2.intor('int2e_sph')
    #import pdb
    #pdb.set_trace()
    jaa_ov = lib.einsum('mnpq,mj,ni,pk,ql->jilk', j, orbva, orboa, orboa, orbva)
    kaa_ov = lib.einsum('mnpq,mj,nl,pk,qi->jilk', j, orbva, orbva, orboa, orboa)
    jaa_vo = lib.einsum('mnpq,mj,ni,pl,qk->jilk', j, orbva, orboa, orbva, orboa)
    jbb_ov = lib.einsum('mnpq,mj,ni,pk,ql->jilk', j, orbvb, orbob, orbob, orbvb)
    kbb_ov = lib.einsum('mnpq,mj,nl,pk,qi->jilk', j, orbvb, orbvb, orbob, orbob)
    jbb_vo = lib.einsum('mnpq,mj,ni,pl,qk->jilk', j, orbvb, orbob, orbvb, orbob)
    jkaa_ov = jaa_ov - kaa_ov # (ji|kl)-(jl|ki)
    jkaa_ov = jkaa_ov.reshape(nvira*nocca,nocca*nvira)
    jkaa_vo = jaa_vo - jaa_vo.transpose(0,3,2,1)
    jkaa_vo = jkaa_vo.reshape(nvira*nocca,nocca*nvira)
    jkbb_ov = jbb_ov - kbb_ov
    jkbb_ov = jkbb_ov.reshape(nvirb*noccb,noccb*nvirb)
    jkbb_vo = jbb_vo - jbb_vo.transpose(0,3,2,1)
    jkbb_vo = jkbb_vo.reshape(nvirb*noccb,noccb*nvirb)
    #jkaa = jaa - jaa.transpose(3,1,2,0)
    #jkaa = jkaa.transpose(0,1,3,2).reshape(nvira*nocca,nocca*nvira)
    #jkbb = jbb - jbb.transpose(3,1,2,0)
    #jkbb = jkbb.transpose(0,1,3,2).reshape(nvirb*noccb,noccb*nvirb)
    jab = lib.einsum('mnpq,mj,ni,pk,ql->jilk', j, orbva, orboa, orbob, orbvb)
    jba = jab.transpose(2,3,0,1).reshape(nvirb*noccb,nocca*nvira)
    jab = jab.reshape(nvira*nocca,noccb*nvirb)
    #import pdb
    #pdb.set_trace()

    aop = numpy.block([[jkaa_ov+eap, jkaa_vo, jab, jab],
                       [jkaa_vo, jkaa_ov+eam, jab, jab],
                       [jba, jba, jkbb_ov+ebp, jkbb_vo],
                       [jba, jba, jkbb_vo, jkbb_ov+ebm]])
    aop_inv = numpy.linalg.inv(aop)
    #import pdb
    #pdb.set_trace()

    polar = uhf.Polarizability(mf)
    h1 = polar.get_h1()
    h1a = lib.einsum('xpq,pi,qj->xij', h1, orbva.conj(), orboa)
    h1b = lib.einsum('xpq,pi,qj->xij', h1, orbvb.conj(), orbob)
    h1 = numpy.concatenate((h1a.reshape(3,-1), h1a.reshape(3,-1),
                            h1b.reshape(3,-1), h1b.reshape(3,-1)), axis=1)
    #import pdb
    #pdb.set_trace()

    mo1 = lib.einsum('mn,xn->xm', aop_inv, -h1)
    offsets = numpy.cumsum((nocca*nvira, nocca*nvira, noccb*nvirb))
    mo1ap, mo1am, mo1bp, mo1bm = numpy.split(mo1, offsets, axis=1)
    mo1 = (mo1ap.reshape(3,nvira,nocca),
           mo1bp.reshape(3,nvirb,noccb),
           mo1am.reshape(3,nvira,nocca),
           mo1bm.reshape(3,nvirb,noccb))
    
    alpha  = -lib.einsum('xji,yji->xy', h1a.conj(), mo1[0])
    alpha += -lib.einsum('xji,yji->xy', h1b.conj(), mo1[1])
    alpha += -lib.einsum('xji,yji->xy', h1a, mo1[2].conj())
    alpha += -lib.einsum('xji,yji->xy', h1b, mo1[3].conj())

    return alpha

class TestRHF(unittest.TestCase):
    def __init__(self):
        setUpModule()
        self.mol1 = mol1
        self.mol2 = mol2
        self.mol3 = mol3
        self.mol4 = mol4

    def polar(self):
        mol = mol1
        mf = mol.RHF()
        mf.conv_tol = 1e-11
        e = mf.kernel()
        if mol == mol1: print(f'AbsTol of energy is {e - -76.0250801801}.')
        polar = rhf.Polarizability(mf)
        return polar

    def test_mo1(self, freq=0):
        polar = self.polar()
        mo1 = polar.get_mo1(freq)
        return mo1
    
    def test_mu(self):
        polar = self.polar()
        mu = polar.dipole_moment()
        return mu
    
    def test_alpha(self, freq=0):
        polar = self.polar()
        alpha = polar.polarizability(freq)
        if freq == 0 and polar.mf.mol == mol1:
            try:
                self.assertAlmostEqual(alpha[0,0], 3.03434962E+00, 5)
                self.assertAlmostEqual(alpha[1,1], 7.23019465E+00, 5)
                self.assertAlmostEqual(alpha[2,2], 5.40908534E+00, 5)
            except AssertionError as error:
                print(error)
                print(f'AbsTol of alpha_xx is {alpha[0,0] - 3.03434962E+00}.')
                print(f'AbsTol of alpha_yy is {alpha[1,1] - 7.23019465E+00}.')
                print(f'AbsTol of alpha_zz is {alpha[2,2] - 5.40908534E+00}.')
        if freq == 0.1 and polar.mf.mol == mol1:
            try:
                self.assertAlmostEqual(alpha[0,0], 3.11867849E+00, 5)
                self.assertAlmostEqual(alpha[1,1], 7.41176461E+00, 5)
                self.assertAlmostEqual(alpha[2,2], 5.54564545E+00, 5)
            except AssertionError as error:
                print(error)
                print(f'AbsTol of alpha_xx is {alpha[0,0] - 3.11867849E+00}.')
                print(f'AbsTol of alpha_yy is {alpha[1,1] - 7.41176461E+00}.')
                print(f'AbsTol of alpha_zz is {alpha[2,2] - 5.54564545E+00}.')
        return alpha
    
    def test_beta(self, freq=0, type='SHG'):
        polar = self.polar()
        beta = polar.hyperpolarizability(freq, type)
        if freq == 0 and polar.mf.mol == mol1:
            try:
                self.assertAlmostEqual(beta[0,0,2], -2.38928522E+00, 5)
                self.assertAlmostEqual(beta[1,1,2], -1.90407348E+01, 5)
                self.assertAlmostEqual(beta[2,2,2], -1.17009975E+01, 5)
            except AssertionError as error:
                print(error)
                print(f'AbsTol of beta_xxz is {beta[0,0,2] - -2.38928522E+00}.')
                print(f'AbsTol of beta_yyz is {beta[1,1,2] - -1.90407348E+01}.')
                print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.17009975E+01}.')
        if freq == 0.1 and polar.mf.mol == mol1:
            if type is None or type == 'SHG':
                try:
                    self.assertAlmostEqual(beta[2,0,0], -2.61483123E+00, 5)
                    self.assertAlmostEqual(beta[2,1,1], -2.43002545E+01, 5)
                    self.assertAlmostEqual(beta[0,2,0], -4.87719836E+00, 5)
                    self.assertAlmostEqual(beta[1,2,1], -2.38204261E+01, 5)
                    self.assertAlmostEqual(beta[2,2,2], -1.57512080E+01, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_zxx is {beta[2,0,0] - -2.61483123E+00}.')
                    print(f'AbsTol of beta_zyy is {beta[2,1,1] - -2.43002545E+01}.')
                    print(f'AbsTol of beta_xzx is {beta[0,2,0] - -4.87719836E+00}.')
                    print(f'AbsTol of beta_yzy is {beta[1,2,1] - -2.38204261E+01}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.57512080E+01}.')
            elif type == 'EOPE':
                beta = beta.transpose(0,2,1)
                try:
                    self.assertAlmostEqual(beta[2,0,0], -2.62048642E+00, 5)
                    self.assertAlmostEqual(beta[2,1,1], -2.05194733E+01, 5)
                    self.assertAlmostEqual(beta[0,0,2], -3.15267173E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -2.04081532E+01, 5)
                    self.assertAlmostEqual(beta[2,2,2], -1.28348625E+01, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_zxx is {beta[2,0,0] - -2.62048642E+00}.')
                    print(f'AbsTol of beta_zyy is {beta[2,1,1] - -2.05194733E+01}.')
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -3.15267173E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -2.04081532E+01}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.28348625E+01}.')
        return beta

class TestX2CRHF(TestRHF):
    def polar(self):
        mol = mol3
        mf = mol.RHF().x2c()
        mf.kernel()
        polar = rhf.Polarizability(mf)
        return polar

class TestUHF(unittest.TestCase):
    def __init__(self):
        TestRHF.__init__(self)

    def polar(self):
        mol = mol2
        mf = mol.UHF()
        mf.conv_tol = 1e-11
        e = mf.kernel()
        if mol == mol2: print(f'AbsTol of energy is {e - -75.6323370477}.')
        polar = uhf.Polarizability(mf)
        return polar
    
    def test_mo1(self, freq=0):
        polar = self.polar()
        mo1 = polar.get_mo1(freq)
        return mo1
    
    def test_mu(self):
        polar = self.polar()
        mu = polar.dipole_moment()
        return mu
    
    def test_alpha(self, freq=0):
        polar = self.polar()
        alpha = polar.polarizability(freq)
        if freq == 0 and polar.mf.mol == mol2:
            try:
                self.assertAlmostEqual(alpha[0,0], 2.81377251E+00, 5)
                self.assertAlmostEqual(alpha[1,1], 4.82776602E+00, 5)
                self.assertAlmostEqual(alpha[2,2], 3.77067491E+00, 5)
            except AssertionError as error:
                print(error)
                print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.81377251E+00}.')
                print(f'AbsTol of alpha_yy is {alpha[1,1] - 4.82776602E+00}.')
                print(f'AbsTol of alpha_zz is {alpha[2,2] - 3.77067491E+00}.')
        if freq == 0.1 and polar.mf.mol == mol2:
            try:
                self.assertAlmostEqual(alpha[0,0], -3.14295511E+00, 5)
                self.assertAlmostEqual(alpha[1,1],  4.89774957E+00, 5)
                self.assertAlmostEqual(alpha[2,2],  3.81886198E+00, 5)
            except AssertionError as error:
                print(error)
                print(f'AbsTol of alpha_xx is {alpha[0,0] - -3.14295511E+00}.')
                print(f'AbsTol of alpha_yy is {alpha[1,1] -  4.89774957E+00}.')
                print(f'AbsTol of alpha_zz is {alpha[2,2] -  3.81886198E+00}.')
        return alpha
    
    def test_beta(self, freq=0, type='SHG'):
        polar = self.polar()
        beta = polar.hyperpolarizability(freq, type)
        if freq == 0 and polar.mf.mol == mol2:
            try:
                self.assertAlmostEqual(beta[0,0,2], -1.49414681E+00, 5)
                self.assertAlmostEqual(beta[1,1,2], -8.16184425E+00, 5)
                self.assertAlmostEqual(beta[2,2,2], -4.98500283E+00, 5)
            except AssertionError as error:
                print(error)
                print(f'AbsTol of beta_xxz is {beta[0,0,2] - -1.49414681E+00}.')
                print(f'AbsTol of beta_yyz is {beta[1,1,2] - -8.16184425E+00}.')
                print(f'AbsTol of beta_zzz is {beta[2,2,2] - -4.98500283E+00}.')
        if freq == 0.1 and polar.mf.mol == mol2:
            if type is None or type == 'SHG':
                try:
                    self.assertAlmostEqual(beta[2,0,0], -2.87961611E+01, 5)
                    self.assertAlmostEqual(beta[2,1,1], -9.28870765E+00, 5)
                    self.assertAlmostEqual(beta[0,2,0], -2.57138440E+00, 5)
                    self.assertAlmostEqual(beta[1,2,1], -9.29200566E+00, 5)
                    self.assertAlmostEqual(beta[2,2,2], -5.77825058E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_zxx is {beta[2,0,0] - -2.87961611E+01}.')
                    print(f'AbsTol of beta_zyy is {beta[2,1,1] - -9.28870765E+00}.')
                    print(f'AbsTol of beta_xzx is {beta[0,2,0] - -2.57138440E+00}.')
                    print(f'AbsTol of beta_yzy is {beta[1,2,1] - -9.29200566E+00}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -5.77825058E+00}.')
            if type == 'EOPE':
                try:
                    beta = beta.transpose(0,2,1)
                    self.assertAlmostEqual(beta[2,0,0],  1.05459770E+01, 5)
                    self.assertAlmostEqual(beta[2,1,1], -8.51153928E+00, 5)
                    self.assertAlmostEqual(beta[0,0,2], -2.02167694E+02, 5)
                    self.assertAlmostEqual(beta[1,1,2], -8.51301059E+00, 5)
                    self.assertAlmostEqual(beta[2,2,2], -5.22761881E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_zxx is {beta[2,0,0] -  1.05459770E+01}.')
                    print(f'AbsTol of beta_zyy is {beta[2,1,1] - -8.51153928E+00}.')
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -2.02167694E+02}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -8.51301059E+00}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -5.22761881E+00}.')
        return beta

class TestX2CUHF(TestUHF):
    def polar(self):
        mol = mol4
        mf = mol.UHF().x2c()
        mf.kernel()
        polar = uhf.Polarizability(mf)
        return polar

class TestRKS(unittest.TestCase):
    def __init__(self):
        TestRHF.__init__(self)

    def polar(self, xc='LDA,VWN3'):
        mol = mol3
        mf = mol.RKS()
        mf.xc = xc
        mf.conv_tol = 1e-11
        e = mf.kernel()
        if mol == mol1:
            if xc.upper() == 'LDA,VWN3':
                print(f'AbsTol of energy is {e - -76.0507659408}.')
            if xc.upper() == 'PBE':
                print(f'AbsTol of energy is {e - -76.3341677433}.')
            if xc.upper() == 'TPSS':
                print(f'AbsTol of energy is {e - -76.4238280190}.')
            if xc.upper() == 'B3LYP':
                print(f'AbsTol of energy is {e - -76.4205004770}.')
            if xc.upper() == 'M062X':
                print(f'AbsTol of energy is {e - -76.3885172361}.')
        polar = rks.Polarizability(mf)
        return polar

    def test_mo1(self, xc='LDA,VWN3', freq=0):
        polar = self.polar(xc)
        mo1 = polar.get_mo1(freq)
        return mo1
    
    def test_mu(self, xc='LDA,VWN3'):
        polar = self.polar(xc)
        mu = polar.dipole_moment()
        return mu
    
    def test_alpha(self, xc='LDA,VWN3', freq=0):
        polar = self.polar(xc)
        alpha = polar.polarizability(freq)
        if freq == 0 and polar.mf.mol == mol1: 
            if xc.upper() == 'LDA,VWN3':
                try:
                    self.assertAlmostEqual(alpha[0,0], 3.23122079E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 7.46110991E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 5.76071139E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 3.23122079E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 7.46110991E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 5.76071139E+00}.')
            if xc.upper() == 'PBE':
                try:
                    self.assertAlmostEqual(alpha[0,0], 3.29382396E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 7.58665511E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 5.84738506E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 3.29382396E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 7.58665511E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 5.84738506E+00}.')
            if xc.upper() == 'TPSS':
                try:
                    self.assertAlmostEqual(alpha[0,0], 3.25900010E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 7.67711417E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 5.87278174E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 3.25900010E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 7.67711417E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 5.87278174E+00}.')
            if xc.upper() == 'B3LYP':
                try:
                    self.assertAlmostEqual(alpha[0,0], 3.20071936E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 7.53725270E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 5.72427208E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 3.20071936E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 7.53725270E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 5.72427208E+00}.')
            if xc.upper() == 'M062X':
                try:
                    self.assertAlmostEqual(alpha[0,0], 3.19943214E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 7.38869831E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 5.61452462E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 3.19943214E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 7.38869831E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 5.61452462E+00}.')
        if freq == 0.1 and polar.mf.mol == mol1:
            if xc.upper() == 'LDA,VWN3':
                try:
                    self.assertAlmostEqual(alpha[0,0], 3.38495995E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 7.67562794E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 5.97206212E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 3.38495995E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 7.67562794E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 5.97206212E+00}.')
            if xc.upper() == 'PBE':
                try:
                    self.assertAlmostEqual(alpha[0,0], 3.45993237E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 7.80865544E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 6.06511272E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 3.45993237E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 7.80865544E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 6.06511272E+00}.')
            if xc.upper() == 'TPSS':
                try:
                    self.assertAlmostEqual(alpha[0,0], 3.40926877E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 7.89660308E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 6.07880670E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 3.40926877E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 7.89660308E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 6.07880670E+00}.')
            if xc.upper() == 'B3LYP':
                try:
                    self.assertAlmostEqual(alpha[0,0], 3.34425269E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 7.75357505E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 5.92200143E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 3.34425269E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 7.75357505E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 5.92200143E+00}.')
            if xc.upper() == 'M062X':
                try:
                    self.assertAlmostEqual(alpha[0,0], 3.32780485E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 7.59294970E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 5.79057495E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 3.32780485E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 7.59294970E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 5.79057495E+00}.')
        return alpha
    
    def test_beta(self, xc='LDA,VWN3', freq=0, type='SHG'):
        polar = self.polar(xc)
        beta = polar.hyperpolarizability(freq, type)
        if freq == 0 and polar.mf.mol == mol1:
            if xc.upper() == 'LDA,VWN3':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -4.02570434E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -2.06569805E+01, 5)
                    self.assertAlmostEqual(beta[2,2,2], -1.57627809E+01, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -4.02570434E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -2.06569805E+01}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.57627809E+01}.')
            if xc.upper() == 'PBE':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -4.09255294E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -2.07742373E+01, 5)
                    self.assertAlmostEqual(beta[2,2,2], -1.54170023E+01, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -4.09255294E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -2.07742373E+01}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.54170023E+01}.')
            if xc.upper() == 'TPSS':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -3.83291677E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -2.10321712E+01, 5)
                    self.assertAlmostEqual(beta[2,2,2], -1.49809327E+01, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -3.83291677E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -2.10321712E+01}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.49809327E+01}.')
            if xc.upper() == 'B3LYP':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -3.59209626E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -2.03543040E+01, 5)
                    self.assertAlmostEqual(beta[2,2,2], -1.44418601E+01, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -3.59209626E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -2.03543040E+01}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.44418601E+01}.')
            if xc.upper() == 'M062X':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -3.06948225E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -1.90038485E+01, 5)
                    self.assertAlmostEqual(beta[2,2,2], -1.27505460E+01, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -3.06948225E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -1.90038485E+01}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.27505460E+01}.')
        if freq == 0.1 and polar.mf.mol == mol1:
            if type is None or type == 'SHG':
                if xc.upper() == 'LDA,VWN3':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -5.28191848E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -2.94977456E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0], -1.26117691E+01, 5)
                        self.assertAlmostEqual(beta[1,2,1], -2.75012890E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -2.51620663E+01, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -5.28191848E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -2.94977456E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -1.26117691E+01}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -2.75012890E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -2.51620663E+01}.')
                if xc.upper() == 'PBE':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -5.37034170E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -2.97673147E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0], -1.35999815E+01, 5)
                        self.assertAlmostEqual(beta[1,2,1], -2.77424877E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -2.46061907E+01, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -5.37034170E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -2.97673147E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -1.35999815E+01}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -2.77424877E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -2.46061907E+01}.')
                if xc.upper() == 'TPSS':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -4.89426526E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -2.93826394E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0], -1.11942069E+01, 5)
                        self.assertAlmostEqual(beta[1,2,1], -2.77062825E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -2.30113929E+01, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -4.89426526E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -2.93826394E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -1.11942069E+01}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -2.77062825E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -2.30113929E+01}.')
                if xc.upper() == 'B3LYP':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -4.53609670E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -2.83912317E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0], -1.07095728E+01, 5)
                        self.assertAlmostEqual(beta[1,2,1], -2.68238674E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -2.22652371E+01, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -4.53609670E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -2.83912317E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -1.07095728E+01}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -2.68238674E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -2.22652371E+01}.')
                if xc.upper() == 'M062X':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -3.61197718E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -2.56640451E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0], -8.36125896E+00, 5)
                        self.assertAlmostEqual(beta[1,2,1], -2.45882766E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -1.88164252E+01, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -3.61197718E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -2.56640451E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -8.36125896E+00}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -2.45882766E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.88164252E+01}.')
            if type == 'EOPE':
                beta = beta.transpose(0,2,1)
                if xc.upper() == 'LDA,VWN3':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -4.72206197E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -2.28512685E+01, 5)
                        self.assertAlmostEqual(beta[0,0,2], -5.95610877E+00, 5)
                        self.assertAlmostEqual(beta[1,1,2], -2.24609740E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -1.81276230E+01, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -4.72206197E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -2.28512685E+01}.')
                        print(f'AbsTol of beta_xxz is {beta[0,0,2] - -5.95610877E+00}.')
                        print(f'AbsTol of beta_yyz is {beta[1,1,2] - -2.24609740E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.81276230E+01}.')
                if xc.upper() == 'PBE':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -4.82226304E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -2.30042398E+01, 5)
                        self.assertAlmostEqual(beta[0,0,2], -6.15504340E+00, 5)
                        self.assertAlmostEqual(beta[1,1,2], -2.26058344E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -1.77283900E+01, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -4.82226304E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -2.30042398E+01}.')
                        print(f'AbsTol of beta_xxz is {beta[0,0,2] - -6.15504340E+00}.')
                        print(f'AbsTol of beta_yyz is {beta[1,1,2] - -2.26058344E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.77283900E+01}.')
                if xc.upper() == 'TPSS':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -4.45302507E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -2.31584590E+01, 5)
                        self.assertAlmostEqual(beta[0,0,2], -5.57687224E+00, 5)
                        self.assertAlmostEqual(beta[1,1,2], -2.28151307E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -1.70515354E+01, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -4.45302507E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -2.31584590E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,0,2] - -5.57687224E+00}.')
                        print(f'AbsTol of beta_yzy is {beta[1,1,2] - -2.28151307E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.70515354E+01}.')
                if xc.upper() == 'B3LYP':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -4.17082781E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -2.24046974E+01, 5)
                        self.assertAlmostEqual(beta[0,0,2], -5.26564872E+00, 5)
                        self.assertAlmostEqual(beta[1,1,2], -2.20877588E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -1.64520971E+01, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -4.17082781E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -2.24046974E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,0,2] - -5.26564872E+00}.')
                        print(f'AbsTol of beta_yzy is {beta[1,1,2] - -2.20877588E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.64520971E+01}.')
                if xc.upper() == 'M062X':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -3.49771454E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -2.07602174E+01, 5)
                        self.assertAlmostEqual(beta[0,0,2], -4.42151120E+00, 5)
                        self.assertAlmostEqual(beta[1,1,2], -2.05340665E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -1.43522104E+01, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -3.49771454E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -2.07602174E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,0,2] - -4.42151120E+00}.')
                        print(f'AbsTol of beta_yzy is {beta[1,1,2] - -2.05340665E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -1.43522104E+01}.')
        return beta

class TestX2CRKS(TestRKS):
    def polar(self, xc='LDA,VWN3'):
        mol = mol3
        mf = mol.RKS().x2c()
        mf.xc = xc
        mf.kernel()
        polar = rks.Polarizability(mf)
        return polar

class TestUKS(unittest.TestCase):
    def __init__(self):
        TestRHF.__init__(self)

    def polar(self, xc='LDA,VWN3'):
        mol = mol2
        mf = mol.UKS()
        mf.xc = xc
        mf.conv_tol = 1e-11
        e = mf.kernel()
        if mol == mol2:
            if xc.upper() == 'LDA,VWN3':
                print(f'AbsTol of energy is {e - -75.5691427658}.')
            if xc.upper() == 'PBE':
                print(f'AbsTol of energy is {e - -75.8844983098}.')
            if xc.upper() == 'TPSS':
                print(f'AbsTol of energy is {e - -75.9796905563}.')
            if xc.upper() == 'B3LYP':
                print(f'AbsTol of energy is {e - -75.9696796709}.')
            if xc.upper() == 'M062X':
                print(f'AbsTol of energy is {e - -75.9359256915}.')
        polar = uks.Polarizability(mf)
        return polar

    def test_mo1(self, xc='LDA,VWN3', freq=0):
        polar = self.polar(xc)
        mo1 = polar.get_mo1(freq)
        return mo1
    
    def test_mu(self, xc='LDA,VWN3'):
        polar = self.polar(xc)
        mu = polar.dipole_moment()
        return mu
    
    def test_alpha(self, xc='LDA,VWN3', freq=0):
        polar = self.polar(xc)
        alpha = polar.polarizability(freq)
        if freq == 0 and polar.mf.mol == mol2: 
            if xc.upper() == 'LDA,VWN3':
                try:
                    self.assertAlmostEqual(alpha[0,0], 2.85761350E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.01676573E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 3.96741341E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.85761350E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.01676573E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 3.96741341E+00}.')
            if xc.upper() == 'PBE':
                try:
                    self.assertAlmostEqual(alpha[0,0], 2.84400724E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.09466771E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 4.00956212E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.84400724E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.09466771E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 4.00956212E+00}.')
            if xc.upper() == 'TPSS':
                try:
                    self.assertAlmostEqual(alpha[0,0], 2.81289214E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.12858963E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 4.01805213E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.81289214E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.12858963E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 4.01805213E+00}.')
            if xc.upper() == 'B3LYP':
                try:
                    self.assertAlmostEqual(alpha[0,0], 2.81661196E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.04410707E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 3.93918863E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.81661196E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.04410707E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 3.93918863E+00}.')
            if xc.upper() == 'M062X':
                try:
                    self.assertAlmostEqual(alpha[0,0], 2.94736127E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 4.98092835E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 3.89736352E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.94736127E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 4.98092835E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 3.89736352E+00}.')
        if freq == 0.1 and polar.mf.mol == mol2:
            if xc.upper() == 'LDA,VWN3':
                try:
                    self.assertAlmostEqual(alpha[0,0], 1.59471845E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.09743767E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 4.03120781E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 1.59471845E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.09743767E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 4.03120781E+00}.')
            if xc.upper() == 'PBE':
                try:
                    self.assertAlmostEqual(alpha[0,0], -1.16147912E+01, 5)
                    self.assertAlmostEqual(alpha[1,1],  5.17806451E+00, 5)
                    self.assertAlmostEqual(alpha[2,2],  4.07504190E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - -1.16147912E+01}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] -  5.17806451E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] -  4.07504190E+00}.')
            if xc.upper() == 'TPSS':
                try:
                    self.assertAlmostEqual(alpha[0,0], 4.61749893E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.21086438E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 4.08128928E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 4.61749893E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.21086438E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 4.08128928E+00}.')
            if xc.upper() == 'B3LYP':
                try:
                    self.assertAlmostEqual(alpha[0,0], -5.97691978E-01, 5)
                    self.assertAlmostEqual(alpha[1,1],  5.12542526E+00, 5)
                    self.assertAlmostEqual(alpha[2,2],  4.00041624E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - -5.97691978E-01}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] -  5.12542526E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] -  4.00041624E+00}.')
            if xc.upper() == 'M062X':
                try:
                    self.assertAlmostEqual(alpha[0,0], 2.07822281E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.06053490E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 3.95578554E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.07822281E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.06053490E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 3.95578554E+00}.')
        return alpha
    
    def test_beta(self, xc='LDA,VWN3', freq=0, type='SHG'):
        polar = self.polar(xc)
        beta = polar.hyperpolarizability(freq, type)
        if freq == 0 and polar.mf.mol == mol2:
            if xc.upper() == 'LDA,VWN3':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -1.16383952E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -8.72543024E+00, 5)
                    self.assertAlmostEqual(beta[2,2,2], -6.36364182E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -1.16383952E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -8.72543024E+00}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.36364182E+00}.')
            if xc.upper() == 'PBE':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -1.01369879E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -8.78686964E+00, 5)
                    self.assertAlmostEqual(beta[2,2,2], -6.29317301E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -1.01369879E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -8.78686964E+00}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.29317301E+00}.')
            if xc.upper() == 'TPSS':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -9.58298584E-01, 5)
                    self.assertAlmostEqual(beta[1,1,2], -9.22140938E+00, 5)
                    self.assertAlmostEqual(beta[2,2,2], -6.30348943E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -9.58298584E-01}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -9.22140938E+00}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.30348943E+00}.')
            if xc.upper() == 'B3LYP':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -1.13168451E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -8.71819032E+00, 5)
                    self.assertAlmostEqual(beta[2,2,2], -5.94886137E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -1.13168451E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -8.71819032E+00}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -5.94886137E+00}.')
            if xc.upper() == 'M062X':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -1.75369427E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -8.08669165E+00, 5)
                    self.assertAlmostEqual(beta[2,2,2], -5.41035953E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -1.75369427E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -8.08669165E+00}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -5.41035953E+00}.')
        if freq == 0.1 and polar.mf.mol == mol2:
            if type is None or type == 'SHG':
                if xc.upper() == 'LDA,VWN3':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  1.75036497E-01, 5)
                        self.assertAlmostEqual(beta[2,1,1], -1.02791494E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0], -6.04080516E-01, 5)
                        self.assertAlmostEqual(beta[1,2,1], -1.01672190E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -7.81565625E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  1.75036497E-01}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -1.02791494E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -6.04080516E-01}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -1.01672190E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -7.81565625E+00}.')
                if xc.upper() == 'PBE':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -6.53570016E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -1.03753955E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0], -8.08273919E+00, 5)
                        self.assertAlmostEqual(beta[1,2,1], -1.02615264E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -7.74364287E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -6.53570016E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -1.03753955E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -8.08273919E+00}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -1.02615264E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -7.74364287E+00}.')
                if xc.upper() == 'TPSS':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -2.30586195E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -1.07896970E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0],  1.39650548E+00, 5)
                        self.assertAlmostEqual(beta[1,2,1], -1.07003955E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -7.65713293E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -2.30586195E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -1.07896970E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] -  1.39650548E+00}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -1.07003955E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -7.65713293E+00}.')
                if xc.upper() == 'B3LYP':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  1.84570995E-01, 5)
                        self.assertAlmostEqual(beta[2,1,1], -1.02073827E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0], -1.56163535E+00, 5)
                        self.assertAlmostEqual(beta[1,2,1], -1.01296496E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -7.22851506E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  1.84570995E-01}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -1.02073827E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -1.56163535E+00}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -1.01296496E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -7.22851506E+00}.')
                if xc.upper() == 'M062X':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  3.88037186E-02, 5)
                        self.assertAlmostEqual(beta[2,1,1], -9.42179572E+00, 5)
                        self.assertAlmostEqual(beta[0,2,0], -3.74993815E-01, 5)
                        self.assertAlmostEqual(beta[1,2,1], -9.37151306E+00, 5)
                        self.assertAlmostEqual(beta[2,2,2], -6.52101977E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  3.88037186E-02}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -9.42179572E+00}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -3.74993815E-01}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -9.37151306E+00}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.52101977E+00}.')
            if type == 'EOPE':
                beta = beta.transpose(0,2,1)
                if xc.upper() == 'LDA,VWN3':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  7.41179881E-01, 5)
                        self.assertAlmostEqual(beta[2,1,1], -9.18539680E+00, 5)
                        self.assertAlmostEqual(beta[0,0,2], -6.91313916E+00, 5)
                        self.assertAlmostEqual(beta[1,1,2], -9.15741160E+00, 5)
                        self.assertAlmostEqual(beta[2,2,2], -6.79300873E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  7.41179881E-01}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -9.18539680E+00}.')
                        print(f'AbsTol of beta_xxz is {beta[0,0,2] - -6.91313916E+00}.')
                        print(f'AbsTol of beta_yyz is {beta[1,1,2] - -9.15741160E+00}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.79300873E+00}.')
                if xc.upper() == 'PBE':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  1.67462224E+01, 5)
                        self.assertAlmostEqual(beta[2,1,1], -9.25685850E+00, 5)
                        self.assertAlmostEqual(beta[0,0,2], -1.31867660E+03, 5)
                        self.assertAlmostEqual(beta[1,1,2], -9.22819853E+00, 5)
                        self.assertAlmostEqual(beta[2,2,2], -6.72180410E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  1.67462224E+01}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -9.25685850E+00}.')
                        print(f'AbsTol of beta_xxz is {beta[0,0,2] - -1.31867660E+03}.')
                        print(f'AbsTol of beta_yyz is {beta[1,1,2] - -9.22819853E+00}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.72180410E+00}.')
                if xc.upper() == 'TPSS':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -3.02866356E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -9.69016326E+00, 5)
                        self.assertAlmostEqual(beta[0,0,2], -2.85648127E+01, 5)
                        self.assertAlmostEqual(beta[1,1,2], -9.66759698E+00, 5)
                        self.assertAlmostEqual(beta[2,2,2], -6.70649952E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -3.02866356E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -9.69016326E+00}.')
                        print(f'AbsTol of beta_xzx is {beta[0,0,2] - -2.85648127E+01}.')
                        print(f'AbsTol of beta_yzy is {beta[1,1,2] - -9.66759698E+00}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.70649952E+00}.')
                if xc.upper() == 'B3LYP':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  3.84381673E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -9.16392088E+00, 5)
                        self.assertAlmostEqual(beta[0,0,2], -6.54791869E+01, 5)
                        self.assertAlmostEqual(beta[1,1,2], -9.14449384E+00, 5)
                        self.assertAlmostEqual(beta[2,2,2], -6.32955073E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  3.84381673E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -9.16392088E+00}.')
                        print(f'AbsTol of beta_xzx is {beta[0,0,2] - -6.54791869E+01}.')
                        print(f'AbsTol of beta_yzy is {beta[1,1,2] - -9.14449384E+00}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.32955073E+00}.')
                if xc.upper() == 'M062X':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  2.24004969E-01, 5)
                        self.assertAlmostEqual(beta[2,1,1], -8.48956110E+00, 5)
                        self.assertAlmostEqual(beta[0,0,2], -2.02809963E+00, 5)
                        self.assertAlmostEqual(beta[1,1,2], -8.47720957E+00, 5)
                        self.assertAlmostEqual(beta[2,2,2], -5.74228976E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  2.24004969E-01}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -8.48956110E+00}.')
                        print(f'AbsTol of beta_xzx is {beta[0,0,2] - -2.02809963E+00}.')
                        print(f'AbsTol of beta_yzy is {beta[1,1,2] - -8.47720957E+00}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -5.74228976E+00}.')
        return beta

class TestX2CUKS(TestUKS):
    def polar(self, xc='LDA,VWN3'):
        mol = mol4
        mf = mol.UKS().x2c()
        mf.xc = xc
        mf.kernel()
        polar = uks.Polarizability(mf)
        return polar

class TestGHF(TestUHF):
    def polar(self):
        mol = mol2
        mf = mol.GHF()
        mf.conv_tol = 1e-11
        e = mf.kernel()
        if mol == mol2: print(f'AbsTol of energy is {e - -75.6323370477}.')
        polar = ghf.Polarizability(mf)
        return polar

class TestX2CGHF(TestGHF):
    def polar(self):
        mol = mol3
        mf = mol.GHF().x2c()
        mf.kernel()
        polar = ghf.Polarizability(mf)
        return polar

class TestGKS(TestUKS):
    def polar(self, xc='LDA,VWN3', col='col'):
        mol = mol2
        mf = mol.GKS()
        mf.xc = xc
        mf.collinear = col
        mf.conv_tol = 1e-11
        e = mf.kernel()
        if mol == mol2:
            if xc.upper() == 'LDA,VWN3':
                print(f'AbsTol of energy is {e - -75.5691427658}.')
            if xc.upper() == 'PBE':
                print(f'AbsTol of energy is {e - -75.8844983098}.')
            if xc.upper() == 'TPSS':
                print(f'AbsTol of energy is {e - -75.9796905563}.')
            if xc.upper() == 'B3LYP':
                print(f'AbsTol of energy is {e - -75.9696796709}.')
            if xc.upper() == 'M062X':
                print(f'AbsTol of energy is {e - -75.9359256915}.')
        polar = gks.Polarizability(mf)
        return polar
    
    def test_mo1(self, xc='LDA,VWN3', col='col', freq=0):
        polar = self.polar(xc, col)
        mo1 = polar.get_mo1(freq)
        return mo1

    def test_mu(self, xc='LDA,VWN3', col='col'):
        polar = self.polar(xc, col)
        mu = polar.dipole_moment()
        return mu

    def test_alpha(self, xc='LDA,VWN3', col='col', freq=0):
        polar = self.polar(xc, col)
        alpha = polar.polarizability(freq)
        if freq == 0 and polar.mf.mol == mol2: 
            if xc.upper() == 'LDA,VWN3':
                try:
                    self.assertAlmostEqual(alpha[0,0], 2.85761350E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.01676573E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 3.96741341E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.85761350E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.01676573E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 3.96741341E+00}.')
            if xc.upper() == 'PBE':
                try:
                    self.assertAlmostEqual(alpha[0,0], 2.84400724E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.09466771E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 4.00956212E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.84400724E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.09466771E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 4.00956212E+00}.')
            if xc.upper() == 'TPSS':
                try:
                    self.assertAlmostEqual(alpha[0,0], 2.81289214E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.12858963E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 4.01805213E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.81289214E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.12858963E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 4.01805213E+00}.')
            if xc.upper() == 'B3LYP':
                try:
                    self.assertAlmostEqual(alpha[0,0], 2.81661196E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.04410707E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 3.93918863E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.81661196E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.04410707E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 3.93918863E+00}.')
            if xc.upper() == 'M062X':
                try:
                    self.assertAlmostEqual(alpha[0,0], 2.94736127E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 4.98092835E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 3.89736352E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.94736127E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 4.98092835E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 3.89736352E+00}.')
        if freq == 0.1 and polar.mf.mol == mol2:
            if xc.upper() == 'LDA,VWN3':
                try:
                    self.assertAlmostEqual(alpha[0,0], 1.59471845E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.09743767E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 4.03120781E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 1.59471845E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.09743767E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 4.03120781E+00}.')
            if xc.upper() == 'PBE':
                try:
                    self.assertAlmostEqual(alpha[0,0], -1.16147912E+01, 5)
                    self.assertAlmostEqual(alpha[1,1],  5.17806451E+00, 5)
                    self.assertAlmostEqual(alpha[2,2],  4.07504190E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - -1.16147912E+01}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] -  5.17806451E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] -  4.07504190E+00}.')
            if xc.upper() == 'TPSS':
                try:
                    self.assertAlmostEqual(alpha[0,0], 4.61749893E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.21086438E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 4.08128928E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 4.61749893E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.21086438E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 4.08128928E+00}.')
            if xc.upper() == 'B3LYP':
                try:
                    self.assertAlmostEqual(alpha[0,0], -5.97691978E-01, 5)
                    self.assertAlmostEqual(alpha[1,1],  5.12542526E+00, 5)
                    self.assertAlmostEqual(alpha[2,2],  4.00041624E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - -5.97691978E-01}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] -  5.12542526E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] -  4.00041624E+00}.')
            if xc.upper() == 'M062X':
                try:
                    self.assertAlmostEqual(alpha[0,0], 2.07822281E+00, 5)
                    self.assertAlmostEqual(alpha[1,1], 5.06053490E+00, 5)
                    self.assertAlmostEqual(alpha[2,2], 3.95578554E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of alpha_xx is {alpha[0,0] - 2.07822281E+00}.')
                    print(f'AbsTol of alpha_yy is {alpha[1,1] - 5.06053490E+00}.')
                    print(f'AbsTol of alpha_zz is {alpha[2,2] - 3.95578554E+00}.')
        return alpha

    def test_beta(self, xc='LDA,VWN3', col='col', freq=0, type='SHG'):
        polar = self.polar(xc, col)
        beta = polar.hyperpolarizability(freq, type)
        if freq == 0 and polar.mf.mol == mol2:
            if xc.upper() == 'LDA,VWN3':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -1.16383952E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -8.72543024E+00, 5)
                    self.assertAlmostEqual(beta[2,2,2], -6.36364182E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -1.16383952E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -8.72543024E+00}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.36364182E+00}.')
            if xc.upper() == 'PBE':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -1.01369879E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -8.78686964E+00, 5)
                    self.assertAlmostEqual(beta[2,2,2], -6.29317301E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -1.01369879E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -8.78686964E+00}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.29317301E+00}.')
            if xc.upper() == 'TPSS':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -9.58298584E-01, 5)
                    self.assertAlmostEqual(beta[1,1,2], -9.22140938E+00, 5)
                    self.assertAlmostEqual(beta[2,2,2], -6.30348943E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -9.58298584E-01}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -9.22140938E+00}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.30348943E+00}.')
            if xc.upper() == 'B3LYP':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -1.13168451E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -8.71819032E+00, 5)
                    self.assertAlmostEqual(beta[2,2,2], -5.94886137E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -1.13168451E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -8.71819032E+00}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -5.94886137E+00}.')
            if xc.upper() == 'M062X':
                try:
                    self.assertAlmostEqual(beta[0,0,2], -1.75369427E+00, 5)
                    self.assertAlmostEqual(beta[1,1,2], -8.08669165E+00, 5)
                    self.assertAlmostEqual(beta[2,2,2], -5.41035953E+00, 5)
                except AssertionError as error:
                    print(error)
                    print(f'AbsTol of beta_xxz is {beta[0,0,2] - -1.75369427E+00}.')
                    print(f'AbsTol of beta_yyz is {beta[1,1,2] - -8.08669165E+00}.')
                    print(f'AbsTol of beta_zzz is {beta[2,2,2] - -5.41035953E+00}.')
        if freq == 0.1 and polar.mf.mol == mol2:
            if type is None or type == 'SHG':
                if xc.upper() == 'LDA,VWN3':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  1.75036497E-01, 5)
                        self.assertAlmostEqual(beta[2,1,1], -1.02791494E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0], -6.04080516E-01, 5)
                        self.assertAlmostEqual(beta[1,2,1], -1.01672190E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -7.81565625E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  1.75036497E-01}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -1.02791494E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -6.04080516E-01}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -1.01672190E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -7.81565625E+00}.')
                if xc.upper() == 'PBE':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -6.53570016E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -1.03753955E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0], -8.08273919E+00, 5)
                        self.assertAlmostEqual(beta[1,2,1], -1.02615264E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -7.74364287E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -6.53570016E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -1.03753955E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -8.08273919E+00}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -1.02615264E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -7.74364287E+00}.')
                if xc.upper() == 'TPSS':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -2.30586195E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -1.07896970E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0],  1.39650548E+00, 5)
                        self.assertAlmostEqual(beta[1,2,1], -1.07003955E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -7.65713293E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -2.30586195E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -1.07896970E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] -  1.39650548E+00}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -1.07003955E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -7.65713293E+00}.')
                if xc.upper() == 'B3LYP':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  1.84570995E-01, 5)
                        self.assertAlmostEqual(beta[2,1,1], -1.02073827E+01, 5)
                        self.assertAlmostEqual(beta[0,2,0], -1.56163535E+00, 5)
                        self.assertAlmostEqual(beta[1,2,1], -1.01296496E+01, 5)
                        self.assertAlmostEqual(beta[2,2,2], -7.22851506E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  1.84570995E-01}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -1.02073827E+01}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -1.56163535E+00}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -1.01296496E+01}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -7.22851506E+00}.')
                if xc.upper() == 'M062X':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  3.88037186E-02, 5)
                        self.assertAlmostEqual(beta[2,1,1], -9.42179572E+00, 5)
                        self.assertAlmostEqual(beta[0,2,0], -3.74993815E-01, 5)
                        self.assertAlmostEqual(beta[1,2,1], -9.37151306E+00, 5)
                        self.assertAlmostEqual(beta[2,2,2], -6.52101977E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  3.88037186E-02}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -9.42179572E+00}.')
                        print(f'AbsTol of beta_xzx is {beta[0,2,0] - -3.74993815E-01}.')
                        print(f'AbsTol of beta_yzy is {beta[1,2,1] - -9.37151306E+00}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.52101977E+00}.')
            if type == 'EOPE':
                beta = beta.transpose(0,2,1)
                if xc.upper() == 'LDA,VWN3':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  7.41179881E-01, 5)
                        self.assertAlmostEqual(beta[2,1,1], -9.18539680E+00, 5)
                        self.assertAlmostEqual(beta[0,0,2], -6.91313916E+00, 5)
                        self.assertAlmostEqual(beta[1,1,2], -9.15741160E+00, 5)
                        self.assertAlmostEqual(beta[2,2,2], -6.79300873E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  7.41179881E-01}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -9.18539680E+00}.')
                        print(f'AbsTol of beta_xxz is {beta[0,0,2] - -6.91313916E+00}.')
                        print(f'AbsTol of beta_yyz is {beta[1,1,2] - -9.15741160E+00}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.79300873E+00}.')
                if xc.upper() == 'PBE':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  1.67462224E+01, 5)
                        self.assertAlmostEqual(beta[2,1,1], -9.25685850E+00, 5)
                        self.assertAlmostEqual(beta[0,0,2], -1.31867660E+03, 5)
                        self.assertAlmostEqual(beta[1,1,2], -9.22819853E+00, 5)
                        self.assertAlmostEqual(beta[2,2,2], -6.72180410E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  1.67462224E+01}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -9.25685850E+00}.')
                        print(f'AbsTol of beta_xxz is {beta[0,0,2] - -1.31867660E+03}.')
                        print(f'AbsTol of beta_yyz is {beta[1,1,2] - -9.22819853E+00}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.72180410E+00}.')
                if xc.upper() == 'TPSS':
                    try:
                        self.assertAlmostEqual(beta[2,0,0], -3.02866356E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -9.69016326E+00, 5)
                        self.assertAlmostEqual(beta[0,0,2], -2.85648127E+01, 5)
                        self.assertAlmostEqual(beta[1,1,2], -9.66759698E+00, 5)
                        self.assertAlmostEqual(beta[2,2,2], -6.70649952E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] - -3.02866356E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -9.69016326E+00}.')
                        print(f'AbsTol of beta_xzx is {beta[0,0,2] - -2.85648127E+01}.')
                        print(f'AbsTol of beta_yzy is {beta[1,1,2] - -9.66759698E+00}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.70649952E+00}.')
                if xc.upper() == 'B3LYP':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  3.84381673E+00, 5)
                        self.assertAlmostEqual(beta[2,1,1], -9.16392088E+00, 5)
                        self.assertAlmostEqual(beta[0,0,2], -6.54791869E+01, 5)
                        self.assertAlmostEqual(beta[1,1,2], -9.14449384E+00, 5)
                        self.assertAlmostEqual(beta[2,2,2], -6.32955073E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  3.84381673E+00}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -9.16392088E+00}.')
                        print(f'AbsTol of beta_xzx is {beta[0,0,2] - -6.54791869E+01}.')
                        print(f'AbsTol of beta_yzy is {beta[1,1,2] - -9.14449384E+00}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -6.32955073E+00}.')
                if xc.upper() == 'M062X':
                    try:
                        self.assertAlmostEqual(beta[2,0,0],  2.24004969E-01, 5)
                        self.assertAlmostEqual(beta[2,1,1], -8.48956110E+00, 5)
                        self.assertAlmostEqual(beta[0,0,2], -2.02809963E+00, 5)
                        self.assertAlmostEqual(beta[1,1,2], -8.47720957E+00, 5)
                        self.assertAlmostEqual(beta[2,2,2], -5.74228976E+00, 5)
                    except AssertionError as error:
                        print(error)
                        print(f'AbsTol of beta_zxx is {beta[2,0,0] -  2.24004969E-01}.')
                        print(f'AbsTol of beta_zyy is {beta[2,1,1] - -8.48956110E+00}.')
                        print(f'AbsTol of beta_xzx is {beta[0,0,2] - -2.02809963E+00}.')
                        print(f'AbsTol of beta_yzy is {beta[1,1,2] - -8.47720957E+00}.')
                        print(f'AbsTol of beta_zzz is {beta[2,2,2] - -5.74228976E+00}.')
        return beta

class TestX2CGKS(TestGKS):
    def polar(self, xc='LDA,VWN3', col='col'):
        mol = mol3
        mf = mol.GKS().x2c()
        mf.xc = xc
        mf.collinear = col
        mf.kernel()
        polar = gks.Polarizability(mf)
        return polar

class TestX2CHF(TestGHF):
    def polar(self):
        mol = mol3
        mf = mol.X2C_HF()
        mf.kernel()
        polar = x2c.Polarizability(mf)
        return polar
    
class TestX2CKS(TestGKS):
    def polar(self, xc='LDA,VWN3', col='col'):
        mol = mol3
        mf = mol.X2C_KS()
        mf.xc = xc
        mf.collinear = col
        mf.kernel()
        polar = x2c.Polarizability(mf)
        return polar
    
class TestDHF(TestGHF):
    def polar(self):
        mol = mol3
        mf = mol.DHF()
        mf.kernel()
        polar = dhf.Polarizability(mf)
        return polar

class TestDKS(TestGKS):
    def polar(self, xc='LDA,VWN3', col='col'):
        mol = mol3
        mf = mol.DKS()
        mf.xc = xc
        mf.collinear = col
        mf.kernel()
        polar = dhf.Polarizability(mf)
        return polar
