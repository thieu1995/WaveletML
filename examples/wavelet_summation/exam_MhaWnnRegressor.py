#!/usr/bin/env python
# Created by "Thieu" at 22:49, 24/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

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

for k, v in model.network.named_parameters():
    print(f"{k}: {v.shape}, {v.data}")
