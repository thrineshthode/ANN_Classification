# Customer Churn Prediction with ANN

A neural-network classifier that predicts whether a bank customer will churn, served through an interactive Streamlit app. The trained model and all preprocessing artifacts are committed, so the app runs out of the box.

## 📊 The Business Problem

Acquiring a new customer costs far more than retaining one. Given a customer's profile — credit score, geography, balance, activity — this model flags who is likely to leave, so the bank can act before they do.

## 📁 Dataset

| File | Rows | Contents |
|------|------|----------|
| `Churn_Modelling.csv` | 10,000 | Bank customers — demographics, account activity, and churn label (`Exited`) |

**Features:** CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary → **Target:** `Exited` (1 = churned)

## 🔬 Methodology

1. **Data examination** — 10,000 customer records, churn distribution and feature correlations
2. **Preprocessing**
   - Label-encoded `Gender`
   - One-hot encoded `Geography` (France / Spain / Germany)
   - Standardized all numeric features with `StandardScaler`
3. **Model** — feedforward Artificial Neural Network built with TensorFlow/Keras for binary classification
4. **Artifacts** — trained model (`model.h5`) plus fitted encoders and scaler saved for inference
5. **Deployment** — Streamlit app (`app.py`) that loads the artifacts and predicts churn probability for any customer profile

## ▶️ Run It

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter a customer's details in the sidebar form — the app returns the churn probability and a likely-to-churn / not-likely verdict (threshold 0.5).

## 🛠 Tech Stack

TensorFlow/Keras · scikit-learn · Streamlit · Pandas · NumPy

## 📂 Project Structure

```
├── app.py                      # Streamlit churn-prediction app
├── model.h5                    # Trained ANN classifier
├── scaler.pkl                  # Fitted StandardScaler
├── label_encoder_gender.pkl    # Fitted gender encoder
├── onehot_encoder_geo.pkl      # Fitted geography encoder
├── Churn_Modelling.csv         # Training data (10,000 customers)
├── experiments.ipynb           # Model experimentation
├── prediction.ipynb            # Inference walkthrough
├── requirements.txt
├── LICENSE
└── README.md
```

---
*End-to-end ML: raw customer data in, deployable churn predictions out.*
