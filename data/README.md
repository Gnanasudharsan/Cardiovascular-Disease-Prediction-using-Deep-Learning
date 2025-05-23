# Data Directory

This directory contains the datasets used for cardiovascular disease prediction.

## Directory Structure

```
data/
├── raw/                    # Original, unprocessed data
│   └── cardiovascular_disease_dataset.csv
├── processed/              # Cleaned and preprocessed data
│   ├── train_data.csv
│   ├── test_data.csv
│   └── validation_data.csv
└── README.md              # This file
```

## Dataset Information

### Source Dataset

The primary dataset used in this research is the **Cardiovascular Disease Dataset** from Kaggle:

- **Source**: [Kaggle - Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Size**: ~70,000 records
- **Features**: 12 features including age, gender, blood pressure, cholesterol, etc.
- **Target**: Binary classification (0 = Healthy, 1 = Heart Disease)

### Data Description

| Feature | Description | Type | Range/Values |
|---------|-------------|------|--------------|
| age | Age in days | Continuous | 10,000 - 25,000 |
| gender | Gender | Binary | 1 = Female, 2 = Male |
| height | Height in cm | Continuous | 55 - 250 |
| weight | Weight in kg | Continuous | 10 - 200 |
| ap_hi | Systolic blood pressure | Continuous | 80 - 200 |
| ap_lo | Diastolic blood pressure | Continuous | 50 - 120 |
| cholesterol | Cholesterol level | Ordinal | 1 = Normal, 2 = Above normal, 3 = Well above normal |
| gluc | Glucose level | Ordinal | 1 = Normal, 2 = Above normal, 3 = Well above normal |
| smoke | Smoking status | Binary | 0 = No, 1 = Yes |
| alco | Alcohol intake | Binary | 0 = No, 1 = Yes |
| active | Physical activity | Binary | 0 = No, 1 = Yes |
| cardio | Target variable | Binary | 0 = Healthy, 1 = Heart Disease |

### Data Preprocessing

The preprocessing pipeline includes:

1. **Missing Value Handling**:
   - Cholesterol values recorded as 0 are imputed using group means
   - Other missing values are handled using median/mode imputation

2. **Feature Engineering**:
   - Age conversion from days to years
   - BMI calculation from height and weight
   - Interaction terms between important features
   - Blood pressure categories

3. **Data Validation**:
   - Outlier detection and handling
   - Range validation for physiological parameters
   - Consistency checks

4. **Feature Scaling**:
   - StandardScaler for continuous variables
   - Label encoding for categorical variables

### Data Splits

The data is split following the research paper methodology:

- **Training Set**: 65% of the data
- **Test Set**: 35% of the data
- **Validation Set**: 20% of training data (for hyperparameter tuning)

### Data Quality

#### Initial Data Quality Issues

1. **Missing Values**: Some cholesterol readings recorded as 0
2. **Outliers**: Extreme values in blood pressure and BMI
3. **Inconsistencies**: Some physiologically impossible combinations

#### Post-Processing Quality

After preprocessing:
- **Missing Values**: 0%
- **Outliers**: Handled using IQR method
- **Feature Distribution**: Normalized and balanced

### Usage Instructions

#### Download Dataset

1. Visit [Kaggle Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
2. Download `cardio_train.csv`
3. Rename to `cardiovascular_disease_dataset.csv`
4. Place in `data/raw/` directory

#### Load Data

```python
import pandas as pd

# Load raw data
df = pd.read_csv('data/raw/cardiovascular_disease_dataset.csv')

# Load preprocessed data
train_df = pd.read_csv('data/processed/train_data.csv')
test_df = pd.read_csv('data/processed/test_data.csv')
val_df = pd.read_csv('data/processed/validation_data.csv')
```

#### Data Exploration

```python
# Basic information
print(f"Dataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")
print(f"Target distribution:\n{df['cardio'].value_counts()}")

# Statistical summary
print(df.describe())

# Missing values
print(df.isnull().sum())
```

### Data Privacy and Ethics

- **De-identification**: All data is anonymized
- **Consent**: Dataset is publicly available for research
- **Bias Considerations**: Model performance evaluated across demographic groups
- **Clinical Validation**: Results require clinical validation before medical use

### Data Updates

The preprocessing pipeline automatically:
- Validates data quality
- Generates data quality reports
- Creates visualizations for data distribution
- Saves processed data with timestamps

### Alternative Datasets

For additional validation and comparison, consider:

1. **Cleveland Heart Disease Dataset** (UCI)
2. **Framingham Heart Study Dataset**
3. **MIMIC-III Clinical Database**
4. **UK Biobank Cardiovascular Data**

### Citation

If you use this dataset, please cite:

```bibtex
@dataset{cardiovascular_disease_dataset,
  title={Cardiovascular Disease Dataset},
  author={Svetlana Ulianova},
  year={2019},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset}
}
```

### Research Paper Citation

```bibtex
@inproceedings{sasikala2023transforming,
  title={Transforming Healthcare With Deep Learning: Cardiovascular Disease Prediction},
  author={Sasikala, V and Arunarasi, J and Surya, S and Shivaanivarsha, N and Raghavendra, Guru and Gnanasudharsan, A},
  booktitle={2023 International Conference on Ambient Intelligence, Knowledge Informatics and Industrial Electronics (AIKIIE)},
  year={2023},
  organization={IEEE}
}
```

---

**Note**: Ensure you have proper permissions and follow ethical guidelines when using medical data for research purposes.
