# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from st_aggrid import AgGrid, GridOptionsBuilder

# --- Page setup ---
st.set_page_config(page_title="🏡 House Price Predictor", layout="wide")
st.title(":house: House Price Prediction ")

# --- Load model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("DNN.h5", compile=False)

model = load_model()

# --- Load training data ---
@st.cache_data
def load_train():
    df = pd.read_csv("train.csv")
    drop_cols = ['SalePrice', 'LotFrontage', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                 'HalfBath', 'PoolArea', 'OpenPorchSF', 'GarageYrBlt', '1stFlrSF',
                 '2ndFlrSF', 'OpenPorchSF']
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['SalePrice']
    X_encoded = pd.get_dummies(X).fillna(X.median(numeric_only=True))
    return df, X_encoded.columns.tolist(), X

train_df, training_columns, train_X = load_train()

# --- Split columns ---
cat_cols = train_X.select_dtypes(include='object').columns.tolist()
num_cols = train_X.select_dtypes(exclude='object').columns.tolist()

# --- Prepare label encoders ---
label_encoders = {}
encoded_df = train_X.copy()
for col in cat_cols:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
    label_encoders[col] = le

# --- Sidebar Navigation ---
with st.sidebar:
    st.title(":books: Menu")
    section = st.radio("Go to:", [
        "📂 Upload CSV for Prediction",
        "🏡 Predict Single House",
        "📈 Predict All Houses (from train.csv)",
        "📋 Feature Info",
        "📈 Evaluate Model Accuracy"
    ])

# --- Preprocessing Function ---
def preprocess_input(df_input):
    drop_cols = ['LotFrontage', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                 'HalfBath', 'PoolArea', 'OpenPorchSF', 'GarageYrBlt',
                 '1stFlrSF', '2ndFlrSF', 'OpenPorchSF', 'SalePrice']
    df_input = df_input.drop(columns=drop_cols, errors='ignore')
    df_input = pd.get_dummies(df_input)
    for col in training_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[training_columns]
    df_input = df_input.fillna(df_input.median(numeric_only=True))
    return df_input

# --- Show AgGrid ---
def show_aggrid_full(df, label=""):
    if label:
        st.markdown(f"#### {label}")
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, filter=True, sortable=True)
    gb.configure_grid_options(domLayout='normal', paginationAutoPageSize=True)
    gridOptions = gb.build()
    AgGrid(df, gridOptions=gridOptions, height=600, width='100%', fit_columns_on_grid_load=True)

# --- Section 1: Upload CSV for Prediction ---
if section == "📂 Upload CSV for Prediction":
    st.subheader("📁 Upload House Data CSV")
    uploaded_file = st.file_uploader("Upload a test file (e.g., test.csv)", type=["csv"])

    if uploaded_file:
        df_test = pd.read_csv(uploaded_file)
        st.info(f"Uploaded Dataset: **{df_test.shape[0]} rows** × **{df_test.shape[1]} columns**")
        st.markdown("### :bar_chart: Full Dataset Preview")
        st.dataframe(df_test, use_container_width=True)

        df_test_original = df_test.copy()
        df_processed = preprocess_input(df_test)
        predictions = model.predict(df_processed).flatten()
        df_test_original['Predicted_SalePrice'] = predictions

        st.success("✅ Predictions completed!")
        show_aggrid_full(df_test_original[['Id', 'Predicted_SalePrice']], label="📈 Predicted Sale Prices (ID + Price)")

        csv = df_test_original.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, file_name="house_predictions.csv", mime='text/csv')

# --- Section 2: Predict Single House ---
elif section == "🏡 Predict Single House":
    st.subheader(":notebook_with_decorative_cover: Enter House Features")

    single_input = {}
    col1, col2 = st.columns(2)
    input_cols = train_X.columns

    for i, col in enumerate(input_cols):
        if col in cat_cols:
            options = train_X[col].dropna().unique().tolist()
            selected = (col1 if i % 2 == 0 else col2).selectbox(col, options)
            single_input[col] = label_encoders[col].transform([selected])[0]
        else:
            default = float(train_X[col].median())
            val = (col1 if i % 2 == 0 else col2).number_input(col, value=default)
            single_input[col] = val

    if st.button(":mag: Predict Price"):
        input_df = pd.DataFrame([single_input])
        input_processed = preprocess_input(input_df)
        prediction = model.predict(input_processed)[0][0]
        st.success(f":moneybag: Predicted Sale Price: **${prediction:,.2f}**")

# --- Section 3: Predict All Houses (from train.csv) ---
elif section == "📈 Predict All Houses (from train.csv)":
    st.subheader(":bar_chart: Predict Sale Prices for All Houses in train.csv")
    st.info(f"Train.csv shape: **{train_df.shape[0]} rows** × **{train_df.shape[1]} columns**")

    df = pd.read_csv("train.csv")
    test_df = df.drop(columns=['SalePrice'], errors='ignore')
    df_processed = preprocess_input(test_df)
    predictions = model.predict(df_processed).flatten()
    df['Predicted_SalePrice'] = predictions

    show_aggrid_full(df[['Id', 'Predicted_SalePrice']], label="📈 All House Predictions (ID + Price)")

    csv = df[['Id', 'Predicted_SalePrice']].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, file_name="train_predicted_prices.csv", mime='text/csv')

# --- Section 4: Feature Info ---
elif section == "📋 Feature Info":
    st.subheader(":clipboard: Feature Information")
    st.markdown(f"Train CSV shape: **{train_df.shape[0]} rows** × **{train_df.shape[1]} columns**")
    st.markdown(f"Model input features: **{len(training_columns)} columns**")

    st.markdown("### 🧮 Numeric Features")
    for col in num_cols:
        st.markdown(f"- **{col}**")

    st.markdown("### 🔠 Categorical Features")
    for col in cat_cols:
        st.markdown(f"- **{col}**")

    st.success("These features are automatically processed by the system before predictions.")

# --- Section 5: Evaluate Model Accuracy ---
elif section == "📈 Evaluate Model Accuracy":
    st.subheader("📈 Model Evaluation on Training Data")
    df = pd.read_csv("train.csv")
    y_true = df['SalePrice']
    df_input = df.drop(columns=['SalePrice'], errors='ignore')
    df_processed = preprocess_input(df_input)
    y_pred = model.predict(df_processed).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:,.2f}")
    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:,.2f}")
    st.metric(label="R² Score", value=f"{r2:.4f}")

    st.info("✅ A lower MAE and RMSE with a higher R² indicate better model accuracy.")
