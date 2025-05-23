# Cardiovascular Disease Prediction using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🩺 Overview

This repository implements a deep learning approach for cardiovascular disease (CVD) prediction using Bidirectional LSTM (BDLSTM) combined with CatBoost classifier. The research demonstrates how AI can transform healthcare by providing rapid, precise, and cost-effective identification of heart problems.

## 📊 Key Results

- **92%** success rate in diagnosing cardiac disease
- **88%** success rate in diagnosing healthy individuals  
- **93%** precision rate
- **94%** ROC-AUC score
- **8.52 seconds** execution time (fastest among compared models)

## 🏗️ Architecture

The proposed framework combines:
- **Bidirectional LSTM (BDLSTM)** for sequence modeling
- **CatBoost** for gradient boosting classification
- **SHAP** for feature importance analysis
- **Comprehensive preprocessing** pipeline

![image](https://github.com/user-attachments/assets/5c03b601-4a95-48dc-b66f-8ef3dc31d128)



## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Gnanasudharsan/cardiovascular-disease-prediction.git
cd cardiovascular-disease-prediction
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dataset

Download the cardiovascular disease dataset from [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) and place it in the `data/raw/` directory.

### Usage

1. **Data Preprocessing:**
```bash
python src/data_preprocessing.py
```

2. **Feature Selection:**
```bash
python src/feature_selection.py
```

3. **Train Models:**
```bash
python src/training.py --model ensemble --epochs 100
```

4. **Run Complete Pipeline:**
```bash
python main.py
```

## 📊 Model Performance

| Model | Precision | Recall | F1-Score | Accuracy | ROC-AUC | Execution Time (s) |
|-------|-----------|--------|----------|----------|---------|-------------------|
| KNN | 0.76 | 0.78 | 0.80 | 0.70 | 0.67 | 23.44 |
| Logistic Regression | 0.90 | 0.90 | 0.92 | 0.92 | 0.88 | 32.59 |
| Random Forest | 0.84 | 0.82 | 0.90 | 0.80 | 0.79 | 48.95 |
| **BDLSTM+CatBoost** | **0.93** | **0.92** | **0.94** | **0.94** | **0.94** | **8.52** |

## 🔍 Key Features

### Data Preprocessing
- Missing value imputation for cholesterol data
- Feature normalization and standardization
- Categorical variable encoding
- Train-test-validation split (65%-35% split)

### Feature Selection
- SHAP (Shapley Additive Explanations) for feature importance
- Gradient boosting-based feature selection
- Features with Shapley values > 0.1 selected
- Multicollinearity handling

### Model Architecture
- **Bidirectional LSTM**: Captures both forward and backward sequence dependencies
- **CatBoost**: Handles categorical features effectively
- **Ensemble Approach**: Combines strengths of both models

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curve analysis
- Cross-validation (K-fold)
- Execution time comparison

## 📈 Exploratory Data Analysis

Key findings from the dataset:
- **Gender Distribution**: Males have ~2.44x higher prevalence of heart disease
- **Chest Pain Types**: Asymptomatic chest pain shows highest risk (76% of cases)
- **Age Factor**: Strong correlation with cardiovascular disease risk
- **Exercise-Induced Angina**: 2.4x higher risk for affected individuals

## 🛠️ Advanced Usage

### Custom Model Training
```python
from src.models.ensemble_model import EnsembleModel
from src.utils.config import Config

# Initialize model
model = EnsembleModel(Config.MODEL_PARAMS)

# Train with custom parameters
model.train(X_train, y_train, 
           epochs=100, 
           batch_size=32,
           validation_data=(X_val, y_val))
```

### SHAP Analysis
```python
from src.utils.visualization import plot_shap_values
import shap

# Generate SHAP explanations
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Visualize feature importance
plot_shap_values(shap_values, feature_names)
```

## 🔬 Research Implementation

This implementation is based on the research paper:
> "Transforming Healthcare With Deep Learning: Cardiovascular Disease Prediction" 
> Published in 2023 International Conference on Ambient Intelligence, Knowledge Informatics and Industrial Electronics (AIKIIE)

### Key Contributions:
1. Novel combination of BDLSTM and CatBoost for CVD prediction
2. Comprehensive feature selection using SHAP values
3. Cost-effective healthcare solution with high accuracy
4. Extensive comparison with traditional ML approaches

## 📊 Visualization

The repository includes comprehensive visualization tools:
- Performance comparison charts
- ROC curves and confusion matrices
- SHAP feature importance plots
- Data distribution analysis
- Training progress monitoring

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Individual test modules:
```bash
pytest tests/test_preprocessing.py
pytest tests/test_models.py
pytest tests/test_utils.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{sasikala2023transforming,
  title={Transforming Healthcare With Deep Learning: Cardiovascular Disease Prediction},
  author={Sasikala, V and Arunarasi, J and Surya, S and Shivaanivarsha, N and Raghavendra, Guru and Gnanasudharsan, A},
  booktitle={2023 International Conference on Ambient Intelligence, Knowledge Informatics and Industrial Electronics (AIKIIE)},
  year={2023},
  organization={IEEE}
}
```

## 🔗 Links

- [Original Research Paper](https://ieeexplore.ieee.org/document/10390290)
- [Dataset Source](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- [Documentation](docs/)
- [Issues](https://github.com/yourusername/cardiovascular-disease-prediction/issues)

## 👥 Authors

- **V Sasikala** - Department of ECE, Sri Sairam Engineering College
- **J. Arunarasi** - Department of ECE, Sri Sairam Engineering College  
- **S. Surya** - Department of ECE, Sri Sairam Engineering College
- **N. Shivaanivarsha** - Department of ECE, Sri Sairam Engineering College
- **Guru Raghavendra S** - Department of ECE, Sri Sairam Engineering College
- **Gnanasudharsan A** - Department of ECE, Sri Sairam Engineering College

## 📞 Contact

For questions or collaborations, please reach out to [sasikala.ece@sairam.edu.in](mailto:sasikala.ece@sairam.edu.in)

---

⭐ **Star this repository if you found it helpful!**
