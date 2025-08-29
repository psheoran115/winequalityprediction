import os
import glob
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Try to import joblib, but don't hard-crash if unavailable
try:
    import joblib  # type: ignore
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

MODEL_FILE = "winequality.pkl"

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

def _is_git_lfs_pointer(path: str) -> bool:
    """Check if a file is just a Git LFS pointer instead of the real binary."""
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
            "Place your pickle next to app.py or enable the fallback trainer in the sidebar."
        )

    # 1) Detect Git LFS pointer files (not the actual model)
    if _is_git_lfs_pointer(MODEL_FILE):
        raise RuntimeError(
            "Your 'winequality.pkl' file looks like a Git LFS pointer, not the actual model.\n"
            "On Streamlit Cloud, you need to enable Git LFS so the real file is pulled:\n"
            "    git lfs install\n"
            "    git lfs track '*.pkl'\n"
            "    git add .gitattributes winequality.pkl\n"
            "    git commit -m 'Track model with LFS'\n"
            "    git push\n"
            "Or, rebuild the model at startup from a CSV."
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
    """Try to find a CSV with a 'quality' column to rebuild the model if needed."""
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
            "No suitable CSV found to train a fallback model. "
            "Place a CSV with a 'quality' column in the repo."
        )
    df = pd.read_csv(csv_path)
    target = [c for c in df.columns if c.lower() == "quality"][0]
    features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]

    # Simple heuristic: regression if target has many distinct numeric values
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    if np.issubdtype(y.dtype, np.number) and (y.nunique() > 10 or y.dtype.kind == 'f'):
        from sklearn.ensemble import RandomForestRegressor as RF
        is_reg = True
    else:
        from sklearn.ensemble import RandomForestClassifier as RF
        is_reg = False

    pre = ColumnTransformer([("num", StandardScaler(), features)], remainder="drop")
    model = RF(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(X, y)
    return pipe, features, is_reg

st.title("🍷 Wine Quality Predictor")
st.caption("Uses a trained model from `winequality.pkl` or can train a fallback model from CSV.")

# Sidebar option
source = st.sidebar.radio("Model source", ["Load pickle", "Train from CSV"], index=0)

model = None
features = []
is_regression = True

try:
    if source == "Load pickle":
        model = load_model()
        # Try to extract feature names if it's a pipeline
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
            # default features if unknown
            features = [
                "fixed acidity","volatile acidity","citric acid","residual sugar",
                "chlorides","free sulfur dioxide","total sulfur dioxide",
                "density","pH","sulphates","alcohol"
            ]
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
                try:
                    proba = model.predict_proba(X)[0]
                    classes = getattr(model, "classes_", list(range(len(proba))))
                    dfp = pd.DataFrame({"class": classes, "probability": proba}).sort_values("probability", ascending=False)
                    st.write("Class Probabilities")
                    st.dataframe(dfp, use_container_width=True)
                except Exception:
                    pass
    except Exception as e:
        st.error("Prediction failed. Likely because the pickle is a pointer file or not trained.")
        st.exception(e)
"""

with open(APP_PATH, "w", encoding="utf-8") as f:
    f.write(dedent(code))

APP_PATH
