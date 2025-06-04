# Changelog

## [0.2.0] – Major Update

### 🔧 Enhancements

* Improved `setup.py` for more robust and maintainable package management.
* Upgraded `mealpy` dependency to **v3.0.2** for better performance and compatibility.
* Updated GitHub Actions workflows for testing and publishing.

### 🧠 Core Module Updates

* Enhanced `BaseMhaWnnModel`:
  * Added support for additional parameters in the `__init__()` method.
  
* Refactored:
  * `data_preparer` and `data_scaler` modules with improved docstrings and minor internal adjustments.

### 📚 Documentation & Testing

* Updated example scripts for clarity and consistency.
* Improved unit tests across modules.
* Revised and expanded documentation.



## [0.1.0] - Initial Release

### 📁 Project Structure
- Added essential project files: `CODE_OF_CONDUCT.md`, `MANIFEST.in`, `LICENSE`, `CITATION.cff`, and `requirements.txt`
- Added structured layout for `examples/`, `tests/`, and `docs/` (documentation site)

### 🧰 Helpers Module
- Added `helpers` package:
  - `verifier`: input and parameter validation
  - `evaluator`: evaluation metrics
  - `data_scaler`: feature normalization
  - `data_preparer`: dataset scaling and splitting
  - `callbacks`: custom callback functionality
  - `wavelet_funcs`: wavelet function definitions and management
  - `wavelet_layers`: PyTorch-based wavelet layer implementations

### 🧠 Models Package
- Added `models` package:
  - `base_model.py`: defines `BaseModel` for consistent design and logic reuse
  - `custom_wnn`: custom wavelet neural network implementations (4 types)
  - `gd_wnn`: fully gradient-descent-based WNNs:
    - `GdWnnClassifier`: for classification tasks
    - `GdWnnRegressor`: for regression tasks
  - `mha_wnn`: fully metaheuristic-optimized WNNs:
    - `MhaWnnClassifier`: for classification
    - `MhaWnnRegressor`: for regression
