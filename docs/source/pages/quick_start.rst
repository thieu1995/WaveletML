============
Installation
============

* Install the `current PyPI release <https://pypi.python.org/pypi/probnet />`_::

   $ pip install probnet==0.1.0


* Install directly from source code::

   $ git clone https://github.com/thieu1995/ProbNet.git
   $ cd ProbNet
   $ python setup.py install

* In case, you want to install the development version from Github::

   $ pip install git+https://github.com/thieu1995/ProbNet


After installation, you can import MetaPerceptron as any other Python module::

   $ python
   >>> import probnet
   >>> probnet.__version__

========
Examples
========

In this section, we will explore the usage of the Adam-based Gradient Optimizer for training ANFIS networks::

    from probnet import PnnClassifier, Data
    from sklearn.datasets import load_breast_cancer


    ## Load data object
    X, y = load_breast_cancer(return_X_y=True)
    data = Data(X, y)

    ## Split train and test
    data.split_train_test(test_size=0.2, random_state=2, inplace=True)
    print(data.X_train.shape, data.X_test.shape)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
    data.X_test = scaler_X.transform(data.X_test)

    data.y_train, scaler_y = data.encode_label(data.y_train)
    data.y_test = scaler_y.transform(data.y_test)

    ## Train and test
    # model = PnnClassifier(sigma=1.0, kernel='laplace', dist='manhattan', normalize_output=True)
    model = PnnClassifier(sigma=1.0)
    model.fit(data.X_train, data.y_train)

    ## Predict
    print("Predicted:", model.predict(data.X_test))
    print("Probabilities:\n", model.predict_proba(data.X_test))

    ## Calculate some metrics
    print(model.score(X=data.X_test, y=data.y_test))
    print(model.scores(X=data.X_test, y=data.y_test, list_metrics=["PS", "RS", "NPV", "F1S", "F2S"]))
    print(model.evaluate(y_true=data.y_test, y_pred=model.predict(data.X_test), list_metrics=["F2S", "CKS", "FBS"]))


A real-world dataset contains features that vary in magnitudes, units, and range. We would suggest performing
normalization when the scale of a feature is irrelevant or misleading. Feature Scaling basically helps to normalize
the data within a particular range.

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
