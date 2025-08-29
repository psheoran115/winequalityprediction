import os
import glob
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Try to import joblib without hard-crashing
try:
    import joblib  # type: ignore
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

MODEL_FILE = "winequality.pkl"

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

def _is_git_lfs_pointer(path: str) -> bool:
    """Check if a file is a Git LFS pointer instead of the real binary."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(200)
        return "git-lfs.github.com/spec/v1" in head
    except Exception:
        return False

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model file '{MODEL_FILE}' not found next to app.py."
        )
    if _is_git_lfs_pointer(MODEL_FILE):
        raise RuntimeError(
            "Your 'winequality.pkl' file looks like a Git LFS pointer, not the actual model. "
            "Enable Git LFS and push the real file, or avoid LFS if the file is small enough."
        )
    if HAS_JOBLIB:
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

# If we can't infer feature names from the model, we fall back to the common 11 UCI features.
DEFAULT_FEATURES = [
    "fixed acidity","volatile acidity","citric acid","residual sugar",
    "chlorides","free sulfur dioxide","total sulfur dioxide",
    "density","pH","sulphates","alcohol"
]

st.title("🍷 Wine Quality Predictor")
st.caption("Loads your trained model from 'winequality.pkl' and predicts quality.")

# Load model (no sidebar, just do it)
try:
    model = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

# Try to extract feature names if your pickle is a sklearn Pipeline with a ColumnTransformer
features = []
try:
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        for name, step in model.steps:
            if hasattr(step, "transformers_"):
                for _, _, cols in step.transformers_:
                    if cols is not None and len(cols) > 0:
                        features = list(cols)
                        break
                break
except Exception:
    pass

if not features:
    features = DEFAULT_FEATURES

with st.expander("Expected Features", expanded=False):
    st.write(features)

def parse_float(label: str, default: float = 0.0, key: str = "") -> float:
    txt = st.text_input(label, value=str(default), key=key)
    try:
        return float(txt)
    except ValueError:
        st.warning(f"Please enter a valid number for '{label}'. Using {default}.")
        return default

st.subheader("Enter Features")
cols = st.columns(2)
inputs = {}
for i, feat in enumerate(features):
    col = cols[i % 2]
    with col:
        # nicer label
        label = str(feat).replace("_", " ").title()
        inputs[str(feat)] = parse_float(label, default=0.0, key=f"in_{i}")

if st.button("Predict"):
    X = pd.DataFrame([inputs], columns=features)
    try:
        y_pred = model.predict(X)[0]
        # Decide how to display the prediction
        if hasattr(model, "predict_proba"):
            # Classifier
            st.success(f"Predicted Quality Class: {y_pred}")
            try:
                proba = model.predict_proba(X)[0]
                classes = getattr(model, "classes_", list(range(len(proba))))
                dfp = pd.DataFrame({"class": classes, "probability": proba}).sort_values(
                    "probability", ascending=False
                )
                st.write("Class Probabilities")
                st.dataframe(dfp, use_container_width=True)
            except Exception:
                pass
        else:
            # Regressor
            st.success(f"Predicted Quality: {float(y_pred):.2f}")
    except Exception as e:
        st.error("Prediction failed. Check that feature names/order match the training pipeline.")
        st.exception(e)
