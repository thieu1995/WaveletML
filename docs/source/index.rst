.. WaveletML documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to WaveletML's documentation!
=====================================

.. image:: https://img.shields.io/badge/release-0.1.0-yellow.svg
   :target: https://github.com/thieu1995/WaveletML/releases

.. image:: https://badge.fury.io/py/waveletml.svg
   :target: https://badge.fury.io/py/waveletml

.. image:: https://img.shields.io/pypi/pyversions/waveletml.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/dm/waveletml.svg
   :target: https://img.shields.io/pypi/dm/waveletml.svg

.. image:: https://github.com/thieu1995/WaveletML/actions/workflows/publish-package.yml/badge.svg
   :target: https://github.com/thieu1995/WaveletML/actions/workflows/publish-package.yml

.. image:: https://pepy.tech/badge/waveletml
   :target: https://pepy.tech/project/waveletml

.. image:: https://readthedocs.org/projects/waveletml/badge/?version=latest
   :target: https://waveletml.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.28802531-blue
   :target: https://doi.org/10.6084/m9.figshare.28802435

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


**WaveletML** is an open-source Python framework designed for building, training, and evaluating Wavelet
Neural Networks (WNNs) tailored for supervised learning tasks such as regression and classification. Leveraging the
power of PyTorch and the modularity of scikit-learn, WaveletML provides a unified, extensible, and scalable platform
for researchers and practitioners to explore wavelet-based neural architectures.


* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimators**: `GdWnnClassifier`, `GdWnnRegressor`, `MhaWnnClassifier`, `MhaWnnRegressor`
* **Supported Wavelet Functions**: Morlet, Mexican Hat, Haar, ...
* **Supported Wavelet Layers**:
  - Weighed Linear Layer
  - Product Layer
  - Summation Layer
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://waveletml.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, permetrics, mealpy, torch


.. toctree::
   :maxdepth: 4
   :caption: Quick Start:

   pages/quick_start.rst

.. toctree::
   :maxdepth: 4
   :caption: Models API:

   pages/waveletml.rst

.. toctree::
   :maxdepth: 4
   :caption: Support:

   pages/support.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
