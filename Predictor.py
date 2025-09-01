# -----------------------
# 0) Import Libraries
# -----------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import messagebox

# -----------------------
# 1) Load & Prepare Data
# -----------------------

df = pd.read_csv("heart.csv")  # Ensure dataset file is in the same folder

# Make sure target is binary: 0 = no disease, 1 = disease
if df['target'].max() > 1:
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Separate features and target
feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols].copy()
y = df["target"].copy()

# Debugging info
print("Feature columns order:", feature_cols)
print("Target value counts:\n", y.value_counts())

# Split dataset (stratify to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# Evaluation metrics
acc = model.score(X_test_scaled, y_test)
print(f"\nTest accuracy: {acc:.4f}")
print("\nClassification report:\n", classification_report(y_test, model.predict(X_test_scaled)))
print("\nConfusion matrix:\n", confusion_matrix(y_test, model.predict(X_test_scaled)))
print("Model classes:", model.classes_)

# -----------------------
# 2) Prepare Default & Example Inputs
# -----------------------

defaults = X_train.median().to_dict()  # Sensible defaults: median values from training set

# Median examples for healthy & disease patients
median_no = X_train[y_train == 0].median().to_dict()
median_yes = X_train[y_train == 1].median().to_dict()

def make_example(median_dict, make_healthier=False, make_sicker=False):
    """
    Build an example input dictionary from medians.
    Optionally nudge values to simulate healthier or sicker patient.
    """
    ex = {}
    for col, med in median_dict.items():
        val = float(med)

        # Nudge numeric vitals for clarity
        if col == 'age':
            val = val - 8 if make_healthier else (val + 8 if make_sicker else val)
        if col in ['trestbps', 'chol']:
            val = val - 15 if make_healthier else (val + 30 if make_sicker else val)
        if col == 'thalach':
            val = val + 15 if make_healthier else (val - 15 if make_sicker else val)
        if col == 'oldpeak':
            val = max(0.0, val - 0.5) if make_healthier else (val + 1.0 if make_sicker else val)

        # Round categorical columns and clip within dataset min/max
        if col in ['sex','cp','fbs','restecg','exang','slope','ca','thal']:
            cand = int(round(val))
            val = max(int(X_train[col].min()), min(int(X_train[col].max()), cand))

        # Final rounding for numeric columns
        if isinstance(val, float) and abs(val - round(val)) < 1e-8 and col != 'oldpeak':
            val = int(round(val))

        ex[col] = str(val)
    return ex

healthy_example = make_example(median_no, make_healthier=True)
disease_example = make_example(median_yes, make_sicker=True)

# -----------------------
# 3) Tkinter GUI Setup
# -----------------------

gui_inputs = feature_cols  # Use all features in GUI

root = tk.Tk()
root.title("❤️ Heart Disease Prediction")
root.geometry("640x760")
root.resizable(False, False)

# Header
header = tk.Frame(root, bg="#d9534f", height=90)
header.pack(fill="x")
tk.Label(header, text="Heart Disease Prediction", bg="#d9534f",
         fg="white", font=("Helvetica", 20, "bold")).pack(pady=18)

# Body
body = tk.Frame(root, bg="#f7f7f7")
body.pack(fill="both", expand=True)

# Input Frame
input_frame = tk.Frame(body, bg="#f7f7f7", padx=12, pady=12)
input_frame.pack(pady=8)

# Feature labels and entry boxes
entries = {}
pretty_names = {
    "age": "Age",
    "sex": "Sex (1=Male,0=Female)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting BP",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar >120 (1=True,0=False)",
    "restecg": "Resting ECG (0-2)",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Induced Angina (1=Yes,0=No)",
    "oldpeak": "ST Depression",
    "slope": "Slope (0-2)",
    "ca": "Major Vessels (0-3)",
    "thal": "Thal (dataset-specific values)"
}

for row, feat in enumerate(gui_inputs):
    tk.Label(input_frame, text=pretty_names.get(feat, feat),
             anchor="w", width=40, bg="#f7f7f7").grid(row=row, column=0, pady=4, sticky="w")
    ent = tk.Entry(input_frame, width=18, bd=1, font=("Arial", 11))
    ent.grid(row=row, column=1, pady=4, padx=6)
    entries[feat] = ent

# Result display
result_frame = tk.Frame(body, bg="#f7f7f7")
result_frame.pack(pady=10)

result_label = tk.Label(result_frame, text="Enter values and click Predict",
                        font=("Arial", 13), bg="#f7f7f7")
result_label.pack()

prob_label = tk.Label(result_frame, text="", font=("Arial", 12, "bold"), bg="#f7f7f7")
prob_label.pack(pady=4)

# -----------------------
# 4) Helper Functions
# -----------------------

def build_sample_row_from_entries():
    """Build a DataFrame row from GUI entry inputs, using defaults for empty fields"""
    sample = {}
    for col in feature_cols:
        val_str = entries[col].get().strip() if col in entries else ""
        sample[col] = float(val_str) if val_str else float(defaults[col])
    return pd.DataFrame([sample], columns=feature_cols)

def build_sample_row_from_dict(dct):
    """Build a DataFrame row from a dictionary"""
    sample = {col: float(dct.get(col, defaults[col])) for col in feature_cols}
    return pd.DataFrame([sample], columns=feature_cols)

def predict_action():
    """Predict heart disease probability from GUI inputs"""
    try:
        X_sample = build_sample_row_from_entries()
        X_sample_scaled = scaler.transform(X_sample)
        probs = model.predict_proba(X_sample_scaled)[0]
        pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else -1
        prob = probs[pos_idx] * 100
        pred = model.predict(X_sample_scaled)[0]

        if pred == 1:
            result_label.config(text="⚠️ Prediction: Likely HEART DISEASE", fg="#a40000")
        else:
            result_label.config(text="✅ Prediction: Unlikely Heart Disease", fg="#0a6a0a")
        prob_label.config(text=f"Risk probability: {prob:.1f}%")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input:\n{e}")

def fill_healthy():
    """Autofill healthy example values"""
    for k, v in healthy_example.items():
        if k in entries:
            entries[k].delete(0, tk.END)
            entries[k].insert(0, v)

def fill_disease():
    """Autofill disease example values"""
    for k, v in disease_example.items():
        if k in entries:
            entries[k].delete(0, tk.END)
            entries[k].insert(0, v)

def verify_examples():
    """Show predicted probabilities for healthy & disease examples"""
    try:
        h_df = build_sample_row_from_dict(healthy_example)
        d_df = build_sample_row_from_dict(disease_example)
        h_prob = model.predict_proba(scaler.transform(h_df))[0][list(model.classes_).index(1)] * 100
        d_prob = model.predict_proba(scaler.transform(d_df))[0][list(model.classes_).index(1)] * 100
        messagebox.showinfo("Verify Examples",
                            f"Healthy example risk: {h_prob:.1f}%\nDisease example risk: {d_prob:.1f}%")
        print(f"Healthy example risk: {h_prob:.1f}%")
        print(f"Disease example risk: {d_prob:.1f}%")
    except Exception as e:
        messagebox.showerror("Error", f"Verification failed:\n{e}")

# -----------------------
# 5) Buttons & Notes
# -----------------------

btn_frame = tk.Frame(body, bg="#f7f7f7")
btn_frame.pack(pady=8)

tk.Button(btn_frame, text="🔍 Predict", width=14, height=2, bg="#d9534f", fg="white",
          font=("Arial", 12, "bold"), command=predict_action).grid(row=0, column=0, padx=8)
tk.Button(btn_frame, text="🙂 Healthy Example", width=16, command=fill_healthy).grid(row=0, column=1, padx=6)
tk.Button(btn_frame, text="⚠️ Disease Example", width=16, command=fill_disease).grid(row=0, column=2, padx=6)
tk.Button(btn_frame, text="🧪 Verify Examples", width=16, command=verify_examples).grid(row=0, column=3, padx=6)

accuracy_label = tk.Label(body, text=f"Model test accuracy: {acc:.3f}", bg="#f7f7f7", font=("Arial", 11))
accuracy_label.pack(pady=6)

note = tk.Label(body, text="Note: Missing fields filled with dataset medians (training set).",
                bg="#f7f7f7", font=("Arial", 9))
note.pack(pady=4)

# -----------------------
# 6) Launch GUI
# -----------------------

root.mainloop()