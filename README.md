# Learnable Harmonic Variable-Q Transform (LHVQT)
Implements a frontend filterbank learning module which can be initialized with complex weights for a Variable-Q Transform (lvqt_orig.py). Several techniques and variations of the module are also implemented, including:
 - Multi-channel (harmonic) structure (lhvqt.py)
 - Real-only weights (lvqt_real.py)
 - Hilbert Transform for analytic filters (lvqt_hilb.py)
 - Harmonic comb initialization (lhvqt_comb.py)
 - Variational dropout for 1D convolutional layer (variational.py)

The repository was created for my Master's Thesis, [End-to-End Music Transcription Using Fine-Tuned Variable-Q Filterbanks](https://scholarworks.rit.edu/theses/10143/).
It has since been updated with various improvements, and to support my new work, [Stub](Stub).

# Installation
##### Standard (PyPI)
Recommended for standard/quick usage
```
pip install lhvqt
```

##### Cloning Repository
Recommended for running examples or making experimental changes.
```
git clone https://github.com/cwitkowitz/lhvqt
pip install -e lhvqt
```

# Usage
Several examples of instantiation, inference, and visualization are provided under the ```examples``` sub-directory. A full-blown training, visualization, and evaluation example for stub can be found at https://github.com/cwitkowitz/stub.

## Cite
Please cite whichever is more relevant to your usage.

##### Stub 2021 Paper
```
@inproceedings{
  stub2021
}
```

##### Master's Thesis
```
@mastersthesis{
  cwitkowitz2019end,
  author  = {Cwitkowitz, Frank},
  title   = {End-to-End Music Transcription Using Fine-Tuned Variable-Q Filterbanks},
  school  = {Rochester Institute of Technology},
  year    = {2019}
}
```
