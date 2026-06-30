# House Price Prediction using XGBoost

A beginner-friendly Machine Learning project that predicts house prices using the California Housing Dataset and the XGBoost Regression algorithm.

---

## Overview

This project demonstrates the complete Machine Learning workflow for predicting house prices using the in-built **California Housing Dataset**.

The model is trained using **XGBoost Regressor**, one of the most powerful gradient boosting algorithms used in real-world machine learning applications.

This project is designed to help beginners understand:

- Data Loading
- Data Analysis
- Feature Selection
- Model Training
- Prediction
- Model Evaluation

---

## Features

-  Load California Housing Dataset
-  Perform Correlation Analysis
-  Prepare Features and Target Variables
-  Split Dataset into Training and Testing Sets
-  Train an XGBoost Regression Model
-  Predict House Prices
-  Evaluate Model Performance
-  Data Visualizations

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## Project Structure


House-Price-Prediction/
│
├── house_price_prediction.py
├── requirements.txt
├── README.md
└── .gitignore

---

## Dataset

The project uses the **California Housing Dataset** provided by Scikit-learn.

### Input Features

| Feature | Description |
|----------|-------------|
| MedInc | Median Income |
| HouseAge | Median House Age |
| AveRooms | Average Rooms |
| AveBedrms | Average Bedrooms |
| Population | Population |
| AveOccup | Average Occupancy |
| Latitude | Latitude |
| Longitude | Longitude |

### Target

- Median House Price

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/esistdini/house_price_prediction
```

### 2. Navigate to the Project Folder

```bash
cd house_price_prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python house_price_prediction.py
```

---

## 📦 Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

---

## 📈 Model Performance

The model is evaluated using two regression metrics:

* **R² Score** – Measures how well the model fits the data.

* **Mean Absolute Error (MAE)** – Measures the average prediction error.

Example Output:

```text
R Square Error : 0.94
Mean Absolute Error : 0.18

R Square Error : 0.83
Mean Absolute Error : 0.30
```

*(Results may vary depending on train/test split and library versions.)*

---

## Visualizations

The project contains optional visualization code.

### Correlation Heatmap

```python
plt.figure(figsize=(10,10))
sns.heatmap(correlation,
            cbar=True,
            square=True,
            fmt='.1f',
            annot=True,
            annot_kws={'size':8},
            cmap='Blues')
plt.show()
```

### Actual vs Predicted Prices

```python
plt.scatter(y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
```

---

## Project Goals

This project was created to:

* Practice supervised machine learning
* Learn regression techniques
* Understand model evaluation
* Build a beginner-friendly AI/ML portfolio project

---

## Contributing

Contributions are welcome 🤝!

Feel free to fork this repository and submit a Pull Request.

---

## License

This project is licensed under the MIT License.

---

## Author

**Dinesh Aswin**

Aspiring AI/ML Engineer | Data Science Enthusiast | Python Developer

If you found this project helpful, don't forget to **Star** the repository!

```
```
