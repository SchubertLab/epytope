# epytope - An Immunoinformatics Framework for Python

![PyPi](https://github.com/KohlbacherLab/epytope/actions/workflows/pypi-publish.yml/badge.svg)
![Tests](https://github.com/KohlbacherLab/epytope/actions/workflows/python-test-conda.yml/badge.svg)
![Tests external](https://github.com/KohlbacherLab/epytope/actions/workflows/python-test-conda-external.yml/badge.svg)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/epytope/badges/version.svg)](https://anaconda.org/bioconda/epytope)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/epytope/badges/latest_release_date.svg)](https://anaconda.org/bioconda/epytope)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/epytope/badges/platforms.svg
)](https://anaconda.org/bioconda/epytope)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/epytope/badges/downloads.svg)](https://anaconda.org/bioconda/epytope)

Copyright 2014 by Benjamin Schuber, Mathias Walzer, Philipp Brachvogel, Andras Szolek, Christopher Mohr, and Oliver Kohlbacher

**epytope** is a framework for T-cell epitope detection, and vaccine design. It offers consistent, easy, and simultaneous access to well established prediction methods of computational immunology. **epytope** can handle polymorphic proteins and offers analysis tools to select, assemble, and design linker sequences for string-of-beads epitope-based vaccines. It is implemented in Python in a modular way and can easily be extended by user defined methods.

## Copyright

epytope is released under the three clause BSD license.

## Installation

use the following commands:

    pip install git+https://github.com/KohlbacherLab/epytope

## Dependencies

### Python Packages

- pandas
- pyomo>=4.0
- svmlight
- PyMySQL
- biopython
- pyVCF
- h5py<=2.10.0

### Third-Party Software (not installed through pip)

- NetMHC predictor family (NetMHC(pan)-(I/II), NetChop, NetCTL) (<http://www.cbs.dtu.dk/services/software.php>)
- PickPocket (<http://www.cbs.dtu.dk/services/software.php>)
- Integer Linear Programming Solver (recommended CBC: <https://projects.coin-or.org/Cbc>)

Please pay attention to the different licensing of third party tools.

## Framework summary

Currently **epytope** provides implementations of several prediction methods or interfaces to external prediction tools.

- Cleavage Prediction
  - Proteasomal cleavage matrix-based prediction by [Dönnes et al.](https://pubmed.ncbi.nlm.nih.gov/15987883/)
  - ProteaSMM by [Tenzer et al.](https://pubmed.ncbi.nlm.nih.gov/15868101/)
  - [NetChop](https://pubmed.ncbi.nlm.nih.gov/15744535/) 3.1
- Epitope Assembly
  - Approach by [Toussaint et al.](https://pubmed.ncbi.nlm.nih.gov/21875632/)
  - Bi-objective extension of approach by [Toussaint et al.](https://pubmed.ncbi.nlm.nih.gov/21875632/)
  - Assembly with spacers by [Schubert et al.](https://pubmed.ncbi.nlm.nih.gov/26813686/)
- Epitope Prediction
  - [SYFPEITHI](https://link.springer.com/article/10.1007/s002510050595)
  - [MHCNuggets](https://pubmed.ncbi.nlm.nih.gov/31871119/) 2.0, 2.3.2
  - [MHCflurry](https://pubmed.ncbi.nlm.nih.gov/29960884/) 1.2.2, 1.4.3
  - [NetMHC](https://pubmed.ncbi.nlm.nih.gov/26515819/) 3.0, 3.4, 4.0
  - [NetMHCII](https://pubmed.ncbi.nlm.nih.gov/29315598/) 2.2, 2.3
  - [NetMHCpan](https://pubmed.ncbi.nlm.nih.gov/28978689/) 2.4, 2.8, 3.0, 4.0, 4.1
  - [NetMHCIIpan](https://pubmed.ncbi.nlm.nih.gov/38000035/) 3.0, 3.1, 4.0, 4.1, 4.2, 4.3
  - [PickPocket](https://pubmed.ncbi.nlm.nih.gov/19297351/) 1.1
  - [NetCTLpan](https://pubmed.ncbi.nlm.nih.gov/20379710/) 1.1
- Epitope Selection
  - [OptiTope](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2703925/)
- Stability Prediction
  - [NetMHCstabpan](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4976001/) 1.0
- TAPP Prediction
  - TAP prediction model by [Doytchinova et al.](https://pubmed.ncbi.nlm.nih.gov/15557175/)
  - [SMMTAP](https://pubmed.ncbi.nlm.nih.gov/12902473/)

## Getting Started

Users and developers should start by reading our [wiki](https://github.com/KohlbacherLab/epytope/wiki) and [IPython tutorials](https://github.com/KohlbacherLab/epytope/tree/master/epytope/tutorials). A reference documentation is also available [online](http://epytope.readthedocs.org/en/latest/).

## How to Cite

Please cite

[Schubert, B., Walzer, M., Brachvogel, H-P., Sozolek, A., Mohr, C., and Kohlbacher, O. (2016). FRED 2 - An Immunoinformatics Framework for Python. Bioinformatics 2016; doi: 10.1093/bioinformatics/btw113](http://bioinformatics.oxfordjournals.org/content/early/2016/02/26/bioinformatics.btw113.short?rss=1)

and the original publications of the used methods.
