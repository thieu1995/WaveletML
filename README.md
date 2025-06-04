# WaveletML: A Scalable and Extensible Wavelet Neural Network Framework

[![GitHub release](https://img.shields.io/badge/release-0.1.0-yellow.svg)](https://github.com/thieu1995/WaveletML/releases)
[![PyPI version](https://badge.fury.io/py/waveletml.svg)](https://badge.fury.io/py/waveletml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/waveletml.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/waveletml.svg)
[![Downloads](https://pepy.tech/badge/waveletml)](https://pepy.tech/project/waveletml)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/WaveletML/actions/workflows/publish-package.yml/badge.svg)](https://github.com/thieu1995/WaveletML/actions/workflows/publish-package.yml)
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

## Features

- ✅ Modular Wavelet Neural Network (WNN) architectures
- ✅ Support for multiple wavelet functions (e.g., Morlet, Mexican Hat)
- ✅ Support for multiple wavelet layers (e.g., Weighed Linear, Product, Summation, etc.)
- ✅ Gradient Descent-based training via Pytorch
- ✅ Metaheuristic Algorithm-based training via Mealpy
- ✅ `scikit-learn`-compatible API with `BaseEstimator` support
- ✅ Built-in support for both classification and regression tasks
- ✅ Customizable activation functions, training parameters, and loss functions
- ✅ Designed for scalability, enabling deployment on CPU or GPU environments.

Whether you're prototyping WNN-based models or conducting advanced experimental research, **WaveletML** aims to bridge 
the gap between theory and practical implementation in wavelet-based learning systems.

## Model Types
- `GdWnnClassifier`: Wavelet-based classifier using gradient-based training
- `GdWnnRegressor`: Wavelet-based regressor using gradient-based training
- `MhaWnnClassifier`: Uses metaheuristics (e.g., PSO, GA) for training
- `MhaWnnRegressor`: Wavelet-based regressor with metaheuristic training


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

In this example, we will use `Adam` optimizer to train Wavelet Weighted Linear Neural Network (WNN) for a classification task.

```python
from sklearn.datasets import load_iris
from waveletml import Data, GdWnnClassifier


## Load data object
X, y = load_iris(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True, shuffle=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)
data.y_test = scaler_y.transform(data.y_test)

print(type(data.X_train), type(data.y_train))

## Create model
model = GdWnnClassifier(size_hidden=10, wavelet_fn="morlet", act_output=None,
                        epochs=100, batch_size=16, optim="Adam", optim_params=None,
                        valid_rate=0.1, seed=42, verbose=True, device=None)
## Train the model
model.fit(X=data.X_train, y=data.y_train)

## Test the model
y_pred = model.predict(data.X_test)
print(y_pred)
print(model.predict_proba(data.X_test))

## Calculate some metrics
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS", "PS", "RS", "NPV", "F1S"]))

## Print model parameters
for k, v in model.network.named_parameters():
    print(f"{k}: {v.shape}, {v.data}")
```

### Regression

In this example, we will use `Genetic Algorithm - GA` to train Wavelet Summation Neural Network (WNN) for a regression task.

```python
from sklearn.datasets import load_diabetes
from waveletml import Data, MhaWnnRegressor, CustomWaveletSummationNetwork


## Load data object
X, y = load_diabetes(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", "minmax"))
data.y_test = scaler_y.transform(data.y_test.reshape(-1, 1))

print(type(data.X_train), type(data.y_train))

## Create model
model = MhaWnnRegressor(size_hidden=10, wavelet_fn="morlet", act_output=None,
                        optim="BaseGA", optim_params={"epoch": 40, "pop_size": 20},
                        obj_name="MSE", seed=42, verbose=True, wnn_type=CustomWaveletSummationNetwork)
## Train the model
model.fit(data.X_train, data.y_train)

## Test the model
y_pred = model.predict(data.X_test)
print(y_pred)

## Calculate some metrics
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["R2", "NSE", "MAPE", "NNSE"]))

## Print model parameters
for k, v in model.network.named_parameters():
    print(f"{k}: {v.shape}, {v.data}")
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
@software{thieu20250525WaveletML,
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
