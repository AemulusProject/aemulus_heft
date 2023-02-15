import numpy as np
import json
from scipy.interpolate import interp1d
import os

class HEFTEmulator(object):

    """Main emulator object"""

    def __init__(
        self,
        forceLPT=True,
        training_file='training_data.json',
    ):
        """
        Initialize the emulator object. Default values for all kwargs were
        used for fiducial results in 2101.11014, so don't change these unless
        you have a good reason!

        Kwargs:
            forceLPT : bool
                Whether to transition to pure LPT at low k.
            training_file: str
                File that the training data is stored in.
        """
        self.nspec = 14
        self.forceLPT = forceLPT
        
        training_file_abspath = "/".join(
            [
                os.path.dirname(os.path.realpath(__file__)),
                "data",
                training_file,
            ]
        )

        with open(training_file_abspath, 'r') as f:
            fp = json.load(f)
            self.coeff = np.array(fp['pce_coefficients'])
            self.exp = np.array(fp['pce_exponents'])
            self.param_mean = np.array(fp['param_mean'])
            self.param_mult = np.array(fp['param_mult'])
            self.pcs_mean = np.array(fp['pcs_mean'])
            self.pcs_mult = np.array(fp['pcs_mult'])
            self.evec_spec = np.array(fp['evec_spec'])
            self.k = np.array(fp['k'])
            self.kmin = np.array(fp['k_min'])
            self.kmax = np.array(fp['k_max'])

        self.evec_spline = interp1d(
            self.k, self.evec_spec, axis=1, fill_value=0, bounds_error=False
        )

    def predict(self, k, cosmo, spec_lpt, k_lpt=None):
        """
        Make predictions from a trained emulator given a vector of wavenumbers and
        a cosmology.

        Args:
            k : array-like
                1d vector of wave-numbers. Maximum k cannot be larger than
                self.kmax. For k < self.kmin, predictions will be made using
                velocileptors, for self.kmin <= k < self.kmax predictions
                use the emulator.
            cosmo : array-like
                Vector containing cosmology/scale factor in the order
                (ombh2, omch2, w0, ns, 10^9 As, H0, mnu, sigma8(z)).
            spec_lpt : array-like
                LPT predictions for spectra from velocileptors at the specified cosmology
                call.
            k_lpt: array-like
                k values LPT predictions are evaluated at if different than k.

        Output:
            Emulator predictions for the basis spectra of the 2nd order lagrangian bias expansion.
            Since we are treating neutrinos, the lensing and clustering spectra trace the matter field ('1') and
            the cdm+baryon field ('cb') respectively. This means we have, in fact, 14 basis spectra.  

            Order of spectra is 1-1, cb-cb, delta-1, delta-cb, delta-delta, delta2-1, delta2-cb, delta2-delta,
            delta2-delta2, s2-1, s2-cb, s2-delta, s2-delta2, s2-s2.
        """

        if len(cosmo.shape) == 1:
            x = cosmo[:, np.newaxis]
        else:
            # to keep API same as before for aemulus alpha            
            x = cosmo.T

        if np.any(k > np.max(self.kmax)):
            if np.all(k > np.max(self.kmax)):
                raise (
                    ValueError(
                        "Trying to compute spectra beyond the maximum value of the emulator!"
                    )
                )
            else:
                print(
                    "{} is greater than k_max for at least one spectrum. Setting P(k>k_max)=0".format(
                        np.max(k)
                    )
                )

        # scale input variables
        x_n = (x - self.param_mean[:, np.newaxis]) * self.param_mult[:, np.newaxis]
        
        in_domain = (-1.0001 <= x_n) & (x_n <= 1.0001)

        #allow for zero neutrino mass extrapolation
        if not (in_domain.all(axis=1) | ((x[6,:]>=0) & (x_n[6,:] <= 1.0001))).all():
            raise (ValueError("{} is not in training domain".format(x[~in_domain])))

        # evaluate lpt spectra at correct k if not already
        if (k_lpt is not None) & (np.sum(k != k_lpt) > 0):
            lpt_interp = interp1d(k_lpt, spec_lpt, axis=-1, fill_value="extrapolate")
            spectra_lpt = lpt_interp(k)
        else:
            spectra_lpt = spec_lpt

        if spectra_lpt.shape[-1] != len(k):
            raise (
                ValueError(
                    "Trying to feed in lpt spectra computed at different k than the desired outcome!"
                )
            )

        # interpolate PCs
        evecs = self.evec_spline(k)

        lambda_surr_normed = np.sum(
            self.coeff[..., np.newaxis]
            * np.prod(x_n ** self.exp[..., np.newaxis], axis=-2)[:, np.newaxis, :],
            axis=-2,
        )
        lambda_surr = (
            lambda_surr_normed / self.pcs_mult[..., np.newaxis]
            + self.pcs_mean[..., np.newaxis]
        )
        simoverlpt_emu = np.einsum("bkp, bpc->cbk", evecs, lambda_surr)

        pk_emu = np.zeros_like(spectra_lpt)
        pk_emu[:] = spectra_lpt
        # set spectra above kmax to 0
        pk_emu[..., k[np.newaxis, :] > self.kmax[:, np.newaxis]] = 0

        # Enforce agreement with LPT
        if self.forceLPT:
            pk_emu[..., k[np.newaxis, :] > self.kmin[:, np.newaxis]] = (
                10 ** (simoverlpt_emu) * pk_emu
            )[..., k[np.newaxis, :] > self.kmin[:, np.newaxis]]
        else:
            pk_emu[...] = 10 ** (simoverlpt_emu) * pk_emu[...]

        return pk_emu

    def basis_to_full(self, k, btheta, emu_spec, cross=True):
        """
        Take an LPTemulator.predict() array and combine with bias parameters to obtain predictions for P_hh and P_hm.


        Inputs:
        -k: set of wavenumbers used to generate emu_spec.
        -btheta: vector of bias + shot noise. See notes below for structure of terms
        -emu_spec: output of LPTemu.predict() at a cosmology / set of k values
        -halomatter: whether we compute only P_hh or also P_hm

        Outputs:
        -pfull: P_hh (k) or a flattened [P_hh (k),P_hm (k)] for given spectrum + bias params.


        Notes:
        Bias parameters can either be

        btheta = [b1, b2, bs2, SN]

        or

        btheta = [b1, b2, bs2, bnabla2, SN]

        Where SN is a constant term, and the bnabla2 terms follow the approximation

        <X, nabla^2 delta> ~ -k^2 <X, 1>.

        Note the term <nabla^2, nabla^2> isn't included in the prediction since it's degenerate with even higher deriv
        terms such as <nabla^4, 1> which in principle have different parameters.

        """
        if len(btheta) == 4:
            b1, b2, bs, sn = btheta
            # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
            bterms_hh = [
                0,
                1,
                0,
                2 * b1,
                b1**2,
                0,
                b2,
                b2 * b1,
                0.25 * b2**2,
                0,
                2 * bs,
                2 * bs * b1,
                bs * b2,
                bs**2,
            ]

            # hm correlations only have one kind of <1,delta_i> correlation
            bterms_hm = [1, 0, b1, 0, 0, b2 / 2, 0, 0, 0, bs, 0, 0, 0, 0]

            pkvec = emu_spec

        else:
            b1, b2, bs, bk2, sn = btheta
            # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
            bterms_hh = [
                0,
                1,
                0,
                2 * b1,
                b1**2,
                0,
                b2,
                b2 * b1,
                0.25 * b2**2,
                0,
                2 * bs,
                2 * bs * b1,
                bs * b2,
                bs**2, 
                #the bnabla spectra start here
                2 * bk2,
                2 * bk2 * b1,
                bk2 * b2,
                2 * bk2 * bs,
            ]

            # hm correlations only have one kind of <1,delta_i> correlation
            bterms_hm = [
                1,
                0,
                b1,
                0,
                0,
                b2 / 2,
                0,
                0,
                0,
                bs,
                0,
                0,
                0,
                0,
                bk2,
                0,
                0, 
                0,
            ]

            pkvec = np.zeros(shape=(self.nspec + 4, len(k)))
            pkvec[: self.nspec] = emu_spec

            # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
            if cross:
                nabla_idx = [0, 2, 5, 9]
            else:
                nabla_idx = [1, 3, 6, 10]

            # Higher derivative terms
            pkvec[self.nspec :] = -(k**2) * pkvec[nabla_idx]

        if cross:
            bterms_hm = np.array(bterms_hm)
            pfull = np.einsum("b, bk->k", bterms_hm, pkvec)

        else:
            bterms_hh = np.array(bterms_hh)
            pfull = np.einsum("b, bk->k", bterms_hh, pkvec) + sn

        return pfull
