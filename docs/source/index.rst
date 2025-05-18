.. ProbNet documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ProbNet's documentation!
===================================

.. image:: https://img.shields.io/badge/release-0.1.0-yellow.svg
   :target: https://github.com/thieu1995/ProbNet/releases

.. image:: https://badge.fury.io/py/probnet.svg
   :target: https://badge.fury.io/py/probnet

.. image:: https://img.shields.io/pypi/pyversions/probnet.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/dm/probnet.svg
   :target: https://img.shields.io/pypi/dm/probnet.svg

.. image:: https://github.com/thieu1995/ProbNet/actions/workflows/publish-package.yaml/badge.svg
   :target: https://github.com/thieu1995/ProbNet/actions/workflows/publish-package.yaml

.. image:: https://pepy.tech/badge/probnet
   :target: https://pepy.tech/project/probnet

.. image:: https://readthedocs.org/projects/probnet/badge/?version=latest
   :target: https://probnet.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.28802531-blue
   :target: https://doi.org/10.6084/m9.figshare.28802435

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


**ProbNet** is a lightweight and extensible Python library that provides a unified implementation of
**Probabilistic Neural Network (PNN)** and its key variant, the **General Regression Neural Network (GRNN)**.
It supports both **classification** and **regression** tasks, making it suitable for a wide range of
supervised learning applications.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimators**: `PnnClassifier`, `GrnnRegressor`
* **Supported Kernel Functions**: Gaussian, Laplace, Triangular, Epanechnikov...
* **Supported Distance Metrics**: Euclidean, Manhattan, Chebyshev, Minkowski, Cosine, ...
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://probnet.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, permetrics


.. toctree::
   :maxdepth: 4
   :caption: Quick Start:

   pages/quick_start.rst

.. toctree::
   :maxdepth: 4
   :caption: Models API:

   pages/probnet.rst

.. toctree::
   :maxdepth: 4
   :caption: Support:

   pages/support.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
