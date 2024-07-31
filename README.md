# aemulus_heft
The emulator formerly known as anzu. A hybrid effective field theory (and matter power spectrum) emulator built from the Aemulus nu simulations 
described in [arxiv:2303.09762](https://arxiv.org/abs/2303.09762). 


# Installation

The code has the following dependencies:

`numpy`, `scipy`, `json`, and [`velocileptors`](https://github.com/sfschen/velocileptors)

After you have installed these, `aemulus_heft` can be installed via

`python3 -m pip install -v git+https://github.com/AemulusProject/aemulus_heft`

# Basic usage

Making predictions for a single cosmology and redshift can be accomplished via:

```python
import numpy as np
from aemulus_heft.heft_emu import HEFTEmulator
from aemulus_heft.utils import lpt_spectra

emu = HEFTEmulator()
cosmo = [0.0223, 0.12, -1, 0.97, 2.1, 67, 0.06]
k = np.logspace(-2, np.log10(0.9), 100)
z=0

#first need 1-loop predictions
spec_lpt, sigma8z = lpt_spectra(k, z, cosmo)
cosmo.append(sigma8z)

spec_heft = emu.predict(k, np.array(cosmo), spec_lpt)
```

This returns the HEFT basis spectra, including the matter power spectrum, evaluated at the provided wavenumbers.

Alternatively, you can circumvent calling velocileptors by making predictions with the provided neural network emulator. 
In this case, redshift should be passed as the last element of the input vector. This outputs emulator 
predictions at a pre-specified set of wavenumbers, which can then be interpolated over.

```python
from aemulus_heft.heft_emu import NNHEFTEmulator
from scipy.interpolate import interp1d
# (ombh2, omch2, w0, ns, 10^9 As, H0, mnu, z)
cosmo = [0.0223, 0.12, -1, 0.97, 2.1, 67, 0.06, 0]
nnemu = NNHEFTEmulator()
k_nn, spec_heft_nn = nnemu.predict(np.atleast_2d(cosmo))

k = np.logspace(-2, np.log10(0.9), 100)
spec_heft_nn= interp1d(k_nn, spec_heft_nn, kind='cubic', fill_value='extrapolate')(k)
```

These can then be used to predict galaxy auto-spectra and galaxy-matter cross-spectra, provided a set of bias parameters via
```python
bvec = [0.786, 0.583, -0.406, -0.512, 1755]

P_gg = emu.basis_to_full(k, bvec, spec_heft, cross=False)
P_gm = emu.basis_to_full(k, bvec, spec_heft)
P_mm = spec_heft[0,:]
``` 

Finally, we provide functionality to produce emulator error covariances using
```python
cov_heft = emu.error_covariance(spec_heft, k, z, cov)
```
where `cov` is our measured fractional error covariances, available upon request (they are too large to host in github).

See `notebooks/predict_spectra.ipynb` for more worked examples for how to use the code.
