import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from skimage import color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from itertools import combinations

# === Title ===
st.title("🎨 ChromaSense Viewer")

# === Load Models and Matrix ===
MODEL_FILES = {
    "Decision Tree": {
        "X": "tree_x_deep.pkl",
        "Y": "tree_y_deep.pkl",
        "Z": "tree_z_deep.pkl"
    },
    "Random Forest": {
        "X": "forest_x.pkl",
        "Y": "forest_y.pkl",
        "Z": "forest_z.pkl"
    }
}

MODELS = {name: {k: joblib.load(v[k]) for k in ["X", "Y", "Z"]} for name, v in MODEL_FILES.items()}
CUSTOM_MATRIX = np.load("custom_rgb2xyz_matrix.npy")
ERROR_DF = pd.read_csv("validation_df_deep_error.csv")

# === Utility Functions ===
def rgb_to_xyz_custom(rgb, matrix):
    rgb = np.array(rgb) / 255.0
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.dot(rgb, matrix.T)

def predict_xyz(model_type, rgb_df):
    if model_type == "Custom Matrix":
        return rgb_to_xyz_custom(rgb_df.iloc[0], CUSTOM_MATRIX)
    model_set = MODELS[model_type]
    x = model_set["X"].predict(rgb_df)[0]
    y = model_set["Y"].predict(rgb_df)[0]
    z = model_set["Z"].predict(rgb_df)[0]
    return [x, y, z]

def xyz_to_lab(xyz):
    return color.xyz2lab(np.array([xyz]))[0]

def delta_e00(lab1, lab2):
    c1 = LabColor(*lab1, observer='2', illuminant='d65')
    c2 = LabColor(*lab2, observer='2', illuminant='d65')
    delta = delta_e_cie2000(c1, c2)
    return delta.item() if hasattr(delta, "item") else delta

# === Sidebar Input ===
st.sidebar.header("RGB Input")
r = st.sidebar.slider("Red", 0, 255, 128)
g = st.sidebar.slider("Green", 0, 255, 128)
b = st.sidebar.slider("Blue", 0, 255, 128)

model_options = st.sidebar.multiselect(
    "Select model(s) to compare:",
    ["Decision Tree", "Random Forest", "Custom Matrix"],
    default=["Decision Tree"]
)

# === Display Input ===
st.subheader("RGB Input")
rgb_input_df = pd.DataFrame([[r, g, b]], columns=["R", "G", "B"])
st.write(rgb_input_df)

# === Model Predictions ===
st.subheader("XYZ Predictions")
model_predictions = {}
for model in model_options:
    xyz = predict_xyz(model, rgb_input_df)
    model_predictions[model] = xyz
    st.write(f"**{model}:**", pd.DataFrame([xyz], columns=["X", "Y", "Z"]))

# === Compare Selected Models ===
if len(model_predictions) >= 2:
    st.subheader("∆E₀₀ Between Models")
    for m1, m2 in combinations(model_predictions.keys(), 2):
        lab1 = xyz_to_lab(model_predictions[m1])
        lab2 = xyz_to_lab(model_predictions[m2])
        delta = delta_e00(lab1, lab2)
        st.metric(f"∆E₀₀ ({m1} vs {m2})", f"{delta:.2f}")

# === Decision Tree Rules ===
if "Decision Tree" in model_options and st.checkbox("Show Decision Tree Rules (X)"):
    rule_text = export_text(MODELS["Decision Tree"]["X"], feature_names=["R", "G", "B"], decimals=2)
    st.text("Top Decision Tree Rules (X prediction):")
    st.code(rule_text, language="text")

# === ∆E₀₀ Heatmap ===
st.subheader("∆E₀₀ Perceptual Map (R-G plane)")
heatmap_data = ERROR_DF.pivot_table(values="DeltaE00", index="G", columns="R", aggfunc="mean")
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(heatmap_data, cmap="magma", cbar_kws={'label': '∆E₀₀'}, ax=ax)
threshold = 4.0
high_error = ERROR_DF[ERROR_DF["DeltaE00"] > threshold]
ax.scatter(high_error["R"], high_error["G"], color="red", s=10, alpha=0.6, label=f"∆E₀₀ > {threshold}")
ax.legend(loc="upper right")
st.pyplot(fig)

# === Error Histogram ===
if st.checkbox("Show ∆E₀₀ Error Histogram"):
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.histplot(ERROR_DF["DeltaE00"], bins=60, kde=True, ax=ax2)
    ax2.set_title("∆E₀₀ Error Distribution")
    ax2.set_xlabel("∆E₀₀")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

# === Real XYZ Comparison ===
st.markdown("## Compare with Real XYZ")
with st.expander("Enter real XYZ to compare ∆E₀₀ for each model"):
    x_real = st.number_input("Real X", value=0.0)
    y_real = st.number_input("Real Y", value=0.0)
    z_real = st.number_input("Real Z", value=0.0)

    if st.button("Calculate ∆E₀₀ with Real XYZ"):
        lab_real = xyz_to_lab([x_real, y_real, z_real])
        for model, xyz in model_predictions.items():
            lab_pred = xyz_to_lab(xyz)
            delta = delta_e00(lab_real, lab_pred)
            st.metric(f"∆E₀₀ ({model} vs Real)", f"{delta:.2f}")






