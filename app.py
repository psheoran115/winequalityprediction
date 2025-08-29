import io
import os
import glob
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Try optional joblib
try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

MODEL_FILE = "winequality.pkl"

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

def _is_git_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(200)
        return "git-lfs.github.com/spec/v1" in head
    except Exception:
        return False

@st.cache_resource
def load_model():
    # 0) Check model file exists
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model file '{MODEL_FILE}' not found. "
            "Place your pickle next to app.py or enable the fallback trainer below."
        )

    # 1) Detect Git LFS pointer files (not the actual model)
    if _is_git_lfs_pointer(MODEL_FILE):
        raise RuntimeError(
            "Your model file appears to be a **Git LFS pointer**, not the real binary.\n\n"
            "Fix: Enable Git LFS on your repo and re-upload the model, or store the model under 25MB directly.\n"
            "Commands:\n"
            "```
"
            "git lfs install
"
            "git lfs track '*.pkl'
"
            "git add .gitattributes winequality.pkl
"
            "git commit -m 'Track model with LFS'
"
            "git push
"
            "```
"
            "Alternatively, rebuild the model at app start from a CSV (see fallback trainer)."
        )

    # 2) Try joblib first if available
    if HAS_JOBLIB:
        try:
            return joblib.load(MODEL_FILE)
        except Exception as e:
            st.warning(f"joblib.load failed: {e}. Trying pickle...", icon="⚠️")

    # 3) Fallback to pickle
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

def _find_training_csv():
    # Try common names or any CSV containing 'quality' column
    candidates = [
        "wineQualityReds.csv",
        "winequality.csv",
        "winequality-red.csv",
        "winequality_red.csv",
        "winequality_dataset.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for path in glob.glob("*.csv"):
        try:
            df = pd.read_csv(path, nrows=50)
            if any(col.lower() == "quality" for col in df.columns):
                return path
        except Exception:
            continue
    return None

@st.cache_resource
def train_fallback_model():
    csv_path = _find_training_csv()
    if not csv_path:
        raise FileNotFoundError(
            "No suitable CSV found to train a fallback model. Place a CSV with a 'quality' column in the repo."
        )
    df = pd.read_csv(csv_path)
    target = [c for c in df.columns if c.lower() == "quality"][0]
    features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]
    # Simple heuristic: regression if target is numeric non-integer spread, else classification
    is_reg = np.issubdtype(y.dtype, np.number) and (y.nunique() > 10 or y.dtype.kind == 'f')
    # Minimal pipeline without external imports
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    if is_reg:
        from sklearn.ensemble import RandomForestRegressor as RF
    else:
        from sklearn.ensemble import RandomForestClassifier as RF

    pre = ColumnTransformer([("num", StandardScaler(), features)], remainder="drop")
    model = RF(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(X, y)
    return pipe, features, is_reg

st.title("🍷 Wine Quality Predictor")

# Sidebar: choose source
source = st.sidebar.radio("Model source", ["Load pickle", "Fallback: train from CSV"], index=0)

model = None
features = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
is_regression = True

try:
    if source == "Load pickle":
        model = load_model()
        # Try to infer features from model if it is a pipeline
        try:
            # If user pickled a sklearn Pipeline with ColumnTransformer
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            if isinstance(model, Pipeline):
                # Find the first transformer with 'transformers' attribute
                for name, step in model.steps:
                    if hasattr(step, "transformers"):
                        # assume numeric only
                        for tname, transformer, cols in step.transformers_:
                            if cols is not None and len(cols) > 0:
                                features = list(cols)
                                break
                        break
        except Exception:
            pass
        # Determine task type
        is_regression = not hasattr(model, "predict_proba")
    else:
        model, features, is_regression = train_fallback_model()
        st.info("Using a fallback model trained at startup from a CSV in your repo.", icon="ℹ️")
except Exception as e:
    st.error(str(e))
    st.stop()

with st.expander("Expected Features", expanded=False):
    st.write(features)

st.subheader("Enter Features")
cols = st.columns(2)
inputs = {}
for i, feat in enumerate(features):
    with cols[i % 2]:
        val = st.number_input(str(feat).title(), value=0.0, step=0.1, format="%.3f", key=f"in_{i}")
        inputs[str(feat)] = float(val)

if st.button("Predict"):
    X = pd.DataFrame([inputs], columns=features)
    try:
        pred = model.predict(X)[0]
        if is_regression:
            st.success(f"Predicted Quality: **{float(pred):.2f}**")
        else:
            st.success(f"Predicted Quality Class: **{pred}**")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                classes = getattr(model, "classes_", list(range(len(proba))))
                dfp = pd.DataFrame({"class": classes, "probability": proba}).sort_values("probability", ascending=False)
                st.write("Class Probabilities")
                st.dataframe(dfp, use_container_width=True)
    except Exception as e:
        st.error("Prediction failed. Check that feature names/order match your training pipeline.")
        st.exception(e)
