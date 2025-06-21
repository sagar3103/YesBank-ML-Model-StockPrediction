# 📊 Yes Bank Stock Price Prediction (Regression Project)

This project aims to predict the **monthly closing stock price** of **Yes Bank** using machine learning techniques, particularly regression models. The dataset includes historical stock prices (Open, High, Low, Close) from July 2005 to November 2020.

---

## 🧠 Objective

To build a machine learning model that can accurately forecast **Yes Bank's monthly closing stock price**, helping investors and analysts better understand price behavior, trends, and volatility.

---

## 📂 Project Structure

- `data_YesBank_StockPrices.csv` - Input dataset containing historical stock prices.
- `yesbank_stock_prediction.ipynb` - Full Google Colab Notebook with EDA, preprocessing, modeling, and evaluation.
- `best_model_random_forest.pkl` - Saved model for deployment.
- `README.md` - Project overview.

---

## 📌 Problem Statement

Yes Bank's stock has experienced major fluctuations, especially post-2018 due to financial frauds and instability. This project analyzes stock behavior over time and predicts closing prices using ML techniques.

---

## 🛠️ Tools & Technologies

- **Languages & Frameworks:** Python, Pandas, NumPy, Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Modeling Techniques:** Linear Regression, Random Forest, XGBoost
- **Optimization:** GridSearchCV, RandomizedSearchCV
- **Model Evaluation:** MAE, RMSE, R² Score

---

## 📊 Exploratory Data Analysis (EDA)

- Univariate Analysis on Open, High, Low, Close prices.
- Multivariate Analysis using Correlation Heatmaps and Time Series plots.
- Outlier Detection & Treatment using IQR capping.
- Feature Extraction: Extracted **Month** and **Year** from the Date.
- Hypothesis Testing:
  - t-test for mean difference (pre/post 2018)
  - Pearson correlation analysis

---

## ⚙️ Data Preprocessing

- **Outliers:** Handled using IQR Capping
- **Missing Values:** None found
- **Feature Engineering:** Added `Month` and `Year` features
- **Scaling:** StandardScaler used for numerical features
- **Train-Test Split:** 80-20 ratio used

---

## 🧪 Models Implemented

| Model               | MAE   | RMSE   | R² Score |
|--------------------|-------|--------|----------|
| Linear Regression  | 5.05  | 8.45   | 0.9914   |
| Random Forest       | 7.85  | 13.27  | 0.9788   |
| XGBoost Regressor  | 3.63  | 5.70   | 0.9752   |

✅ **Final Model Chosen: XGBoost Regressor** due to its lowest error and highest R² value.

---

## 🔍 Model Evaluation

- **MAE (Mean Absolute Error):** Measures average deviation from true values.
- **RMSE (Root Mean Square Error):** Penalizes large errors; important in high volatility scenarios.
- **R² Score:** Indicates how much variance in the target is explained by the model.

📈 **Business Impact:** A reliable forecasting tool helps investors and analysts to manage risk, evaluate volatility, and make informed decisions.

---

## 🗂️ Future Improvements

- Deploy the model via Flask or FastAPI
- Include external factors (news sentiment, macroeconomic data)
- Model deployment on Streamlit for interactive forecasting

---

## 🤝 Contribution

👤 **Author:** Sagar Zujam  
🎓 B.Tech CSE | Nutan College of Engineering & Research | LabMentix
📫 Feel free to connect for collaboration or feedback!

---

## 📌 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
