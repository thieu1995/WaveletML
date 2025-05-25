# Changelog

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
