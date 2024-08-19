from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
from scipy.interpolate import interp1d
from classy import Class
import numpy as np

def _cleft_pk(k, p_lin):
    '''
    Returns a spline object which computes the cleft component spectra.
    Computed either in "full" CLEFT or in "k-expanded" CLEFT (kecleft)
    which allows for faster redshift dependence.
    Args:
        k: array-like
            Array of wavevectors to compute power spectra at (in h/Mpc).
        p_lin: array-like
            Linear power spectrum to produce velocileptors predictions for.
            Note that we require p_lin at the redshift that you wish
            to make predictions for, because in cosmologies with neutrinos
            a constant linear growth rescaling no longer works.
    Returns:
        cleft_aem : InterpolatedUnivariateSpline
            Spline that computes basis spectra as a function of k.
        cleftobt: CLEFT object
            CLEFT object used to compute basis spectra.
    '''
    cleftobj = RKECLEFT(k, p_lin)
    cleftobj.make_ptable(D=1, kmin=k[0], kmax=k[-1], nk=1000)
    cleftpk = cleftobj.pktable.T

    cleftpk[2, :] /= 2 #(1 d)
    cleftpk[6, :] /= 0.25 # (d2 d2)
    cleftpk[7, :] /= 2 #(1 s)
    cleftpk[8, :] /= 2 #(d s)

    cleftspline = interp1d(cleftpk[0], cleftpk, fill_value='extrapolate')

    return cleftspline, cleftobj

def lpt_spectra(k, z, cosmo, pkclass=None):
    """_summary_

    Args:
        k (array-like): 1d vector of wave-numbers. Maximum k cannot be larger than
                    self.kmax. For k < self.kmin, predictions will be made using
                    velocileptors, for self.kmin <= k < self.kmax predictions
                    use the emulator.
        z (float): redshift
        cosmo (array-like): Vector containing cosmology in the order
                            (ombh2, omch2, w0, ns, 10^9 As, H0, mnu).  
        pkclass (Class object, optional): Class for this cosmology if already run.
                                          Defaults to None.
    """
    h = cosmo[5]/100
    cosmo_dict = {
        'h': h,
        'Omega_b': cosmo[0] / h**2,
        'Omega_cdm': cosmo[1] / h**2,
        'N_ur': 0.00641,
        'N_ncdm': 1,
        'output': 'mPk mTk',
        'z_pk': '0.0,99',
        'P_k_max_h/Mpc': 20.,
        'm_ncdm': cosmo[6]/3,
        'deg_ncdm': 3,
        'T_cmb': 2.7255,
        'A_s': cosmo[4] * 10**-9,
        'n_s': cosmo[3],
        'Omega_Lambda': 0.0,
        'w0_fld': cosmo[2],
        'wa_fld': 0.0,
        'cs2_fld': 1.0,
        'fluid_equation_of_state': "CLP"
    }
    
    if pkclass == None:
        pkclass = Class()
        pkclass.set(cosmo_dict)
        pkclass.compute()

    sigma8z = pkclass.sigma(8 / h, z)#, h_units=True)
    kt = np.logspace(-3, 1, 400)

    pk_m_lin = np.array(
        [
            pkclass.pk_lin(ki, np.array([z])) * h ** 3
            for ki in kt * h
        ]
    )
    pk_cb_lin = np.array(
        [
            pkclass.pk_cb_lin(ki, np.array([z])) * h ** 3
            for ki in kt * h
        ]
    )
    
    pk_cb_m_lin = np.sqrt(pk_m_lin * pk_cb_lin)

    cleft_m_spline, _ = _cleft_pk(kt, pk_m_lin)
    cleft_cb_spline, _ = _cleft_pk(kt, pk_cb_lin)
    cleft_cb_m_spline, _ = _cleft_pk(kt, pk_cb_m_lin)
    
    pk_m_cleft = cleft_m_spline(k)[1:]
    pk_cb_cleft = cleft_cb_spline(k)[1:]
    pk_cb_m_cleft = cleft_cb_m_spline(k)[1:]
    s_m_map = {1:0, 3: 1, 6: 3, 10: 6}
    s_cb_map = {2: 0, 4: 1, 5: 2, 7: 3, 8: 4, 9: 5, 11: 6, 12: 7, 13: 8, 14: 9}
        
    pk_cleft = np.zeros((15,len(k)))
    for s in np.arange(15):
        if s==0:
            pk_cleft[s, :] = pk_m_cleft[0]
        elif s in [1, 3, 6, 10]:
            pk_cleft[s, :] = pk_cb_m_cleft[s_m_map[s]]
        else:
            pk_cleft[s, :] = pk_cb_cleft[s_cb_map[s]]

    return pk_cleft, sigma8z

