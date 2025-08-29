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

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

@st.cache_resource
def load_model():
    # Prefer joblib if available
    if HAS_JOBLIB:
        try:
            return joblib.load("winequality.pkl")
        except Exception:
            pass
    # Fallback to pickle
    with open("winequality.pkl", "rb") as f:
        return pickle.load(f)

FEATURES = [
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

st.title("🍷 Wine Quality Predictor")
st.caption("Loads your trained model from `winequality.pkl`.")

model = load_model()

with st.expander("Expected Features", expanded=False):
    st.write(FEATURES)

st.subheader("Enter Features")
cols = st.columns(2)
inputs = {}
for i, feat in enumerate(FEATURES):
    with cols[i % 2]:
        val = st.number_input(feat.title(), value=0.0, step=0.1, format="%.3f", key=f"in_{i}")
        inputs[feat] = float(val)

if st.button("Predict"):
    X = pd.DataFrame([inputs], columns=FEATURES)
    try:
        pred = model.predict(X)[0]
        if isinstance(pred, (int, np.integer)):
            st.success(f"Predicted Quality: **{int(pred)}**")
        else:
            try:
                pred_f = float(pred)
                st.success(f"Predicted Quality: **{pred_f:.2f}**")
            except Exception:
                st.success(f"Prediction: **{pred}**")
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0]
                classes = getattr(model, "classes_", list(range(len(proba))))
                st.write("Class Probabilities")
                st.dataframe(
                    pd.DataFrame({"class": classes, "probability": proba}).sort_values("probability", ascending=False),
                    use_container_width=True
                )
            except Exception:
                pass
    except Exception as e:
        st.error(
            "Prediction failed. Common reasons:\n"
            "- Feature names/order don't match the training pipeline.\n"
            "- The pickle expects preprocessing not included here."
        )
        st.exception(e)
