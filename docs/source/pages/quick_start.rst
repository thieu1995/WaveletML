============
Installation
============

* Install the `current PyPI release <https://pypi.python.org/pypi/waveletml />`_::

   $ pip install waveletml==0.1.0


* Install directly from source code::

   $ git clone https://github.com/thieu1995/WaveletML.git
   $ cd WaveletML
   $ python setup.py install

* In case, you want to install the development version from Github::

   $ pip install git+https://github.com/thieu1995/WaveletML


After installation, you can import MetaPerceptron as any other Python module::

   $ python
   >>> import waveletml
   >>> waveletml.__version__

========
Examples
========


Classification
~~~~~~~~~~~~~~

In this example, we will use ``Adam`` optimizer to train Wavelet Weighted Linear Neural Network (WNN) for a classification task.

.. code-block:: python

    from sklearn.datasets import load_iris
    from waveletml import Data, GdWnnClassifier

    # Load data object
    X, y = load_iris(return_X_y=True)
    data = Data(X, y)

    # Split train and test
    data.split_train_test(test_size=0.2, random_state=2, inplace=True, shuffle=True)
    print(data.X_train.shape, data.X_test.shape)

    # Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
    data.X_test = scaler_X.transform(data.X_test)

    data.y_train, scaler_y = data.encode_label(data.y_train)
    data.y_test = scaler_y.transform(data.y_test)

    print(type(data.X_train), type(data.y_train))

    # Create model
    model = GdWnnClassifier(size_hidden=10, wavelet_fn="morlet", act_output=None,
                            epochs=100, batch_size=16, optim="Adam", optim_params=None,
                            valid_rate=0.1, seed=42, verbose=True, device=None)
    # Train the model
    model.fit(X=data.X_train, y=data.y_train)

    # Test the model
    y_pred = model.predict(data.X_test)
    print(y_pred)
    print(model.predict_proba(data.X_test))

    # Calculate some metrics
    print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS", "PS", "RS", "NPV", "F1S"]))

    # Print model parameters
    for k, v in model.network.named_parameters():
        print(f"{k}: {v.shape}, {v.data}")

Regression
~~~~~~~~~~

In this example, we will use ``Genetic Algorithm - GA`` to train Wavelet Summation Neural Network (WNN) for a regression task.

.. code-block:: python

    from sklearn.datasets import load_diabetes
    from waveletml import Data, MhaWnnRegressor, CustomWaveletSummationNetwork

    # Load data object
    X, y = load_diabetes(return_X_y=True)
    data = Data(X, y)

    # Split train and test
    data.split_train_test(test_size=0.2, random_state=2, inplace=True)
    print(data.X_train.shape, data.X_test.shape)

    # Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
    data.X_test = scaler_X.transform(data.X_test)

    data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", "minmax"))
    data.y_test = scaler_y.transform(data.y_test.reshape(-1, 1))

    print(type(data.X_train), type(data.y_train))

    # Create model
    model = MhaWnnRegressor(size_hidden=10, wavelet_fn="morlet", act_output=None,
                            optim="BaseGA", optim_params={"epoch": 40, "pop_size": 20},
                            obj_name="MSE", seed=42, verbose=True, wnn_type=CustomWaveletSummationNetwork,
                            lb=None, ub=None, mode='single', n_workers=None, termination=None)
    # Train the model
    model.fit(data.X_train, data.y_train)

    # Test the model
    y_pred = model.predict(data.X_test)
    print(y_pred)

    # Calculate some metrics
    print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["R2", "NSE", "MAPE", "NNSE"]))

    # Print model parameters
    for k, v in model.network.named_parameters():
        print(f"{k}: {v.shape}, {v.data}")

For more use cases, please read the `examples <https://github.com/thieu1995/WaveletML/tree/main/examples>`_ folder.


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
