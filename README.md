# Customer Churn Prediction

This is a practice project I built from scratch while preparing for data analyst roles. I picked a problem businesses actually care about: which telecom customers are likely to leave?

I used the Telco Customer Churn dataset (7,043 customers, 21 columns) and took it end to end: cleaning, exploring, modelling, testing and saving a model that can predict for a new customer.

## Why churn?
Keeping a customer is cheaper than finding a new one. If a company can spot who is likely to leave, it can reach out first with a call or an offer.

## The data
Each row is one customer: their contract, services, payment method, how long they've stayed, and what they're charged. The last column says whether they left.

1,869 out of 7,043 customers left (about 26.5%). That imbalance mattered a lot for how I trained and judged the model.

## What I did

**Cleaning**
- Dropped `customerID`, since it's only a label.
- `TotalCharges` was stored as text and hid 11 blank values. These were customers with zero tenure (new, nothing billed yet), so I set them to 0 and converted the column to numbers.
- No other missing values.

**Exploring**
- Checked the spread of tenure, monthly charges and total charges with histograms and boxplots.
- Looked at how the three numeric columns relate (correlation heatmap).
- Plotted counts for every categorical column.

**Preparing the data**
- Label-encoded the categorical columns and saved the encoders, so new customers can be encoded the same way later.
- Split the data 80/20 (5,634 for training, 1,409 for testing).
- Used SMOTE on the **training set only** to balance the classes. The test set stays untouched.

**Choosing a model**
I compared Decision Tree, Random Forest and XGBoost with default settings using 5-fold cross-validation. Random Forest scored highest, so I went with it.

## Results
Tested on 1,409 customers the model had never seen.

**Accuracy: 77.7%**

| | Precision | Recall | F1 |
|---|---|---|---|
| Stayed | 0.85 | 0.85 | 0.85 |
| Churned | 0.58 | 0.58 | 0.58 |

| | Predicted: stay | Predicted: churn |
|---|---|---|
| **Actually stayed** | 879 | 157 |
| **Actually churned** | 157 | 216 |

The model catches 216 of the 373 customers who really left (58%) and misses 157. It also flags 157 loyal customers who were never going to leave. For churn, recall matters more than accuracy, and 58% leaves room to improve.

## Conclusions
- Random Forest was the best of the three models I compared, and it reached **77.7% accuracy** on the test set.
- The model is much better at recognising customers who stay (85% recall) than customers who leave (58% recall). It still misses about 4 in 10 churners.
- It catches 216 of 373 churners but also raises 157 false alarms. Whether that trade-off is acceptable depends on what a retention offer costs compared with losing a customer.
- In its current form, I'd treat it as a first screening tool to shortlist at-risk customers, not as a final decision maker.
- The biggest improvement to try next is raising churn recall through threshold tuning and model tuning.

## What I learned
- **Accuracy alone can mislead.** With about 26% churn, a model can look fine overall and still miss many of the customers who matter. That's why I looked at recall and the confusion matrix.
- **Be careful when using SMOTE.** I ran cross-validation on SMOTE-balanced data, and the scores (0.78, 0.84, 0.83) came out higher than my test accuracy. Synthetic rows can leak between folds and inflate results. I only used those scores to compare models, and I report the test-set result as the real one. Next time I'll put SMOTE inside a pipeline.
- **Saving a model means saving the preprocessing too.** My first saved encoders were broken because I ran the encoding cell twice. I rebuilt them from the raw data. Now I run encoding once, from the top.

## What I'd do next
- Put SMOTE inside a pipeline so cross-validation scores are fair.
- Tune the model and the decision threshold to catch more churners.
- Check feature importance to see what pushes customers to leave.
- Turn it into a small web app for single predictions.

## Files
| File | What it is |
|---|---|
| `Customer_churn_prediction.ipynb` | The full notebook, from cleaning to prediction |
| `customer_churn_model.pkl` | The trained Random Forest, saved with its feature names |
| `encoder.pkl` | The saved label encoders |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | The dataset |

## Run it yourself
```bash
git clone https://github.com/AnushaaPV/Customer_churn_prediction_using_ML.git
cd Customer_churn_prediction_using_ML
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
jupyter notebook
```
Open the notebook and run the cells from the top.

## Predict for one customer
```python
import pickle
import pandas as pd

with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
feature_names = model_data["features_names"]

with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

customer = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
}

df = pd.DataFrame([customer])
for col, enc in encoders.items():
    df[col] = enc.transform(df[col])
df = df[feature_names]

pred = model.predict(df)[0]
prob = model.predict_proba(df)[0][1]
print("Churn" if pred == 1 else "No churn", f"(churn probability: {prob:.2f})")
```

