#!/usr/bin/env python
# Created by "Thieu" at 22:53, 24/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_iris
from waveletml import Data, GdWnnClassifier, CustomWaveletProductNetwork


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
                        valid_rate=0.1, seed=42, verbose=True, device=None, wnn_type=CustomWaveletProductNetwork,
                        lb=None, ub=None, mode='single', n_workers=None, termination=None)
## Train the model
model.fit(X=data.X_train, y=data.y_train)

## Test the model
y_pred = model.predict(data.X_test)
print(y_pred)
print(model.predict_proba(data.X_test))

## Calculate some metrics
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS", "PS", "RS", "NPV", "F1S"]))

for k, v in model.network.named_parameters():
    print(f"{k}: {v.shape}, {v.data}")
