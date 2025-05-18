# WaveletML: A Scalable and Extensible Wavelet Neural Network Framework

[![GitHub release](https://img.shields.io/badge/release-0.1.0-yellow.svg)](https://github.com/thieu1995/WaveletML/releases)
[![PyPI version](https://badge.fury.io/py/waveletml.svg)](https://badge.fury.io/py/waveletml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/waveletml.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/waveletml.svg)
[![Downloads](https://pepy.tech/badge/waveletml)](https://pepy.tech/project/waveletml)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/WaveletML/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/WaveletML/actions/workflows/publish-package.yaml)
[![Documentation Status](https://readthedocs.org/projects/waveletml/badge/?version=latest)](https://waveletml.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.29095376-blue)](https://doi.org/10.6084/m9.figshare.29095376)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---


## 📌 Overview

**WaveletML** is an open-source Python framework designed for building, training, and evaluating Wavelet 
Neural Networks (WNNs) tailored for supervised learning tasks such as regression and classification. Leveraging the 
power of PyTorch and the modularity of scikit-learn, WaveletML provides a unified, extensible, and scalable platform 
for researchers and practitioners to explore wavelet-based neural architectures.

## Key Features

- ⚙️ Built with **PyTorch**, and fully compatible with **Scikit-Learn** pipeline and metrics.
- 🔧 A flexible architecture supporting various WNN configurations.
- 📌 Built-in support for both gradient-based and metaheuristic optimization methods.
- 📌 Seamless integration with the scikit-learn API for model interoperability and pipeline construction.
- 🧠 Easily extendable to accommodate custom wavelet functions, and training algorithms.
- 🔍 Designed for scalability, enabling deployment on CPU or GPU environments.

Whether you're prototyping WNN-based models or conducting advanced experimental research, **WaveletML** aims to bridge 
the gap between theory and practical implementation in wavelet-based learning systems.


## 📦 Installation

You can install the library using `pip` (once published to PyPI):

```bash
pip install waveletml
```

After installation, you can import `WaveletML` as any other Python module:

```sh
$ python
>>> import waveletml
>>> waveletml.__version__
```

## 🚀 Quick Start


### Classification

```python
```

### Regression

```python
```

Please read the [examples](/examples) folder for more use cases.


## 📚 Documentation

Documentation is available at: 👉 https://waveletml.readthedocs.io

You can build the documentation locally:

```shell
cd docs
make html
```

## 🧪 Testing
You can run unit tests using:

```shell
pytest tests/
```

## 🤝 Contributing
We welcome contributions to `WaveletML`! If you have suggestions, improvements, or bug fixes, feel free to fork 
the repository, create a pull request, or open an issue.


## 📄 License
This project is licensed under the GPLv3 License. See the LICENSE file for more details.


## Citation Request
Please include these citations if you plan to use this library:

```bibtex
@software{thieu20250517PyLWL,
  author       = {Nguyen Van Thieu},
  title        = {WaveletML: A Scalable and Extensible Wavelet Neural Network Framework},
  month        = may,
  year         = 2025,
  doi         = {10.6084/m9.figshare.29095376},
  url          = {https://github.com/thieu1995/WaveletML}
}
```

## Official Links 

* Official source code repo: https://github.com/thieu1995/WaveletML
* Official document: https://waveletml.readthedocs.io/
* Download releases: https://pypi.org/project/waveletml/
* Issue tracker: https://github.com/thieu1995/WaveletML/issues
* Notable changes log: https://github.com/thieu1995/WaveletML/blob/master/ChangeLog.md
* Official chat group: https://t.me/+fRVCJGuGJg1mNDg1

---

Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=WaveletML_QUESTIONS) @ 2025
