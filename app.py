import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# 2. APPLICATION TITLE AND DESCRIPTION

st.title("💳 Credit Card Default Prediction")

st.markdown(
    """
    ### Predict whether a customer is likely to default on the next payment

    This application uses a **Hist Gradient Boosting Classifier** to predict
    credit card payment default based on customer demographic information,
    credit limit, repayment history, bill amounts and previous payments.
    """
)

st.info("Prediction: 0 = No Default | 1 = Default")

# 3. MODEL FILE PATHS

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "credit_card_default_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURE_PATH = BASE_DIR / "feature_names.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"

# 4. CHECK MODEL FILES BEFORE LOADING

required_files = {
    "Model": MODEL_PATH,
    "Scaler": SCALER_PATH,
    "Feature Names": FEATURE_PATH
}

missing_files = [
    name
    for name, path in required_files.items()
    if not path.is_file()
]

if missing_files:
    st.error("Required model files are missing.")

    st.write("Streamlit is looking in this folder:")
    st.code(str(BASE_DIR))

    st.write("Expected files:")

    for name, path in required_files.items():
        if path.is_file():
            st.success(f"Found: {name} → {path.name}")
        else:
            st.error(f"Missing: {name} → {path.name}")

    st.warning(
        "Make sure these files are committed and pushed to the same "
        "GitHub repository and branch used by Streamlit."
    )

    st.stop()

# 5. LOAD SAVED MODEL FILES

@st.cache_resource
def load_model_files():
    """
    Load the trained model, scaler, feature names and optional metadata.
    """

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    metadata = None

    # Metadata is optional.
    # The application should still work if this file is absent.
    if METADATA_PATH.is_file():
        metadata = joblib.load(METADATA_PATH)

    return model, scaler, feature_names, metadata


try:
    model, scaler, feature_names, metadata = load_model_files()

except Exception as e:
    st.error("The model files were found, but they could not be loaded.")

    st.write("This usually happens because of a package/version mismatch "
             "between the environment used for training and Streamlit.")

    st.exception(e)
    st.stop()

# 6. BASIC MODEL VALIDATION

# Convert feature names to a normal Python list where possible.
try:
    feature_names = list(feature_names)
except Exception:
    st.error("feature_names.pkl could not be converted into a feature list.")
    st.stop()


# Number of features expected by the saved scaler.
scaler_feature_count = getattr(scaler, "n_features_in_", None)

if scaler_feature_count is not None:
    if scaler_feature_count != len(feature_names):
        st.error("Feature-count mismatch detected.")

        st.write(
            f"feature_names.pkl contains {len(feature_names)} features, "
            f"but the scaler expects {scaler_feature_count} features."
        )

        st.stop()

# 7. SIDEBAR - MODEL INFORMATION

st.sidebar.header("📊 Model Information")

st.sidebar.write("**Model:** Hist Gradient Boosting")
st.sidebar.write(f"**Features:** {len(feature_names)}")
st.sidebar.write("**SMOTE:** Used during training")
st.sidebar.write("**Scaling:** StandardScaler")

# 8. SIDEBAR - MODEL PERFORMANCE


if metadata is not None and isinstance(metadata, dict):

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Performance")

    if "accuracy" in metadata:
        st.sidebar.write(
            f"Accuracy: {metadata['accuracy']:.4f}"
        )

    if "precision" in metadata:
        st.sidebar.write(
            f"Precision: {metadata['precision']:.4f}"
        )

    if "recall" in metadata:
        st.sidebar.write(
            f"Recall: {metadata['recall']:.4f}"
        )

    if "f1_score" in metadata:
        st.sidebar.write(
            f"F1 Score: {metadata['f1_score']:.4f}"
        )

    if "roc_auc" in metadata:
        st.sidebar.write(
            f"ROC-AUC: {metadata['roc_auc']:.4f}"
        )

# 9. CUSTOMER INPUT FORM

st.header("Enter Customer Details")

st.markdown(
    "Please enter the customer's credit and repayment information."
)

with st.form("prediction_form"):

    # --------------------------------------------------------
    # CUSTOMER DEMOGRAPHIC INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        limit_bal = st.number_input(
            "Credit Limit (LIMIT_BAL)",
            min_value=0.0,
            value=50000.0,
            step=5000.0
        )

    with col2:
        sex = st.selectbox(
            "Gender (SEX)",
            options=[1, 2],
            format_func=lambda x:
                "Male (1)" if x == 1 else "Female (2)"
        )

    with col3:
        education = st.selectbox(
            "Education (EDUCATION)",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Graduate School (1)",
                2: "University (2)",
                3: "High School (3)",
                4: "Others (4)"
            }[x]
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        marriage = st.selectbox(
            "Marital Status (MARRIAGE)",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Married (1)",
                2: "Single (2)",
                3: "Others (3)"
            }[x]
        )

    with col2:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

    with col3:
        st.write("")


    # --------------------------------------------------------
    # REPAYMENT HISTORY
    # --------------------------------------------------------

    st.subheader("📅 Repayment History")

    st.caption(
        "Repayment status: -1 = Pay duly, 1 = 1 month delay, "
        "2 = 2 months delay, etc."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        pay_0 = st.number_input(
            "PAY_0 - September",
            min_value=-2,
            max_value=9,
            value=0,
            step=1
        )

        pay_2 = st.number_input(
            "PAY_2 - August",
            min_value=-2,
            max_value=9,
            value=0,
            step=1
        )

    with col2:
        pay_3 = st.number_input(
            "PAY_3 - July",
            min_value=-2,
            max_value=9,
            value=0,
            step=1
        )

        pay_4 = st.number_input(
            "PAY_4 - June",
            min_value=-2,
            max_value=9,
            value=0,
            step=1
        )

    with col3:
        pay_5 = st.number_input(
            "PAY_5 - May",
            min_value=-2,
            max_value=9,
            value=0,
            step=1
        )

        pay_6 = st.number_input(
            "PAY_6 - April",
            min_value=-2,
            max_value=9,
            value=0,
            step=1
        )

   
    # BILL AMOUNTS
    
    st.subheader("💰 Bill Amounts")

    col1, col2, col3 = st.columns(3)

    with col1:
        bill_amt1 = st.number_input(
            "BILL_AMT1 - September",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )

        bill_amt2 = st.number_input(
            "BILL_AMT2 - August",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )

    with col2:
        bill_amt3 = st.number_input(
            "BILL_AMT3 - July",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )

        bill_amt4 = st.number_input(
            "BILL_AMT4 - June",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )

    with col3:
        bill_amt5 = st.number_input(
            "BILL_AMT5 - May",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )

        bill_amt6 = st.number_input(
            "BILL_AMT6 - April",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )


        # PREVIOUS PAYMENT AMOUNTS
    
    st.subheader("💵 Previous Payment Amounts")

    col1, col2, col3 = st.columns(3)

    with col1:
        pay_amt1 = st.number_input(
            "PAY_AMT1 - September",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        pay_amt2 = st.number_input(
            "PAY_AMT2 - August",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    with col2:
        pay_amt3 = st.number_input(
            "PAY_AMT3 - July",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        pay_amt4 = st.number_input(
            "PAY_AMT4 - June",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    with col3:
        pay_amt5 = st.number_input(
            "PAY_AMT5 - May",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        pay_amt6 = st.number_input(
            "PAY_AMT6 - April",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )


        # PREDICTION BUTTON
   
    st.markdown("---")

    submit = st.form_submit_button(
        "🔮 Predict Credit Default",
        use_container_width=True
    )


# 10. MAKE PREDICTION

if submit:

        # CREATE INPUT DATAFRAME
    
    input_data = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,

        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,

        "BILL_AMT1": bill_amt1,
        "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3,
        "BILL_AMT4": bill_amt4,
        "BILL_AMT5": bill_amt5,
        "BILL_AMT6": bill_amt6,

        "PAY_AMT1": pay_amt1,
        "PAY_AMT2": pay_amt2,
        "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4,
        "PAY_AMT5": pay_amt5,
        "PAY_AMT6": pay_amt6
    }

    input_df = pd.DataFrame([input_data])


        # CHECK THAT ALL TRAINING FEATURES EXIST
    
    missing_input_features = [
        feature
        for feature in feature_names
        if feature not in input_df.columns
    ]

    extra_input_features = [
        feature
        for feature in input_df.columns
        if feature not in feature_names
    ]

    if missing_input_features:

        st.error(
            "Some features required by the trained model are missing "
            "from the Streamlit input."
        )

        st.write("Missing features:")
        st.write(missing_input_features)

        st.stop()


        # ENSURE EXACT FEATURE ORDER
    
    try:
        input_df = input_df[feature_names]

    except Exception as e:

        st.error(
            "Feature mismatch between Streamlit input and trained model."
        )

        st.exception(e)
        st.stop()


        # SCALE INPUT
    
    try:
        input_scaled = scaler.transform(input_df)

    except Exception as e:

        st.error("Error while scaling the input data.")

        st.write(
            f"Input contains {input_df.shape[1]} features."
        )

        if hasattr(scaler, "n_features_in_"):
            st.write(
                f"Scaler expects {scaler.n_features_in_} features."
            )

        st.exception(e)
        st.stop()


        # MODEL PREDICTION
   
    try:
        prediction = model.predict(input_scaled)[0]

    except Exception as e:

        st.error("Error while making the prediction.")
        st.exception(e)
        st.stop()


        # DEFAULT PROBABILITY
    
    probability = None

    if hasattr(model, "predict_proba"):

        try:
            probability = float(
                model.predict_proba(input_scaled)[0][1]
            )

        except Exception:
            probability = None


        # 11. DISPLAY PREDICTION RESULT
   
    st.markdown("---")
    st.header("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ HIGH RISK: Customer is predicted to DEFAULT."
        )

        st.markdown(
            "The model predicts that this customer may fail "
            "to make the next required payment."
        )

    else:

        st.success(
            "✅ LOW RISK: Customer is predicted NOT to DEFAULT."
        )

        st.markdown(
            "The model predicts that this customer is unlikely "
            "to default on the next payment."
        )


       # DEFAULT PROBABILITY
   
    if probability is not None:

        st.subheader("Default Probability")

        probability_percentage = probability * 100

        st.metric(
            "Probability of Default",
            f"{probability_percentage:.2f}%"
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )


        # INPUT SUMMARY
    
    st.subheader("Customer Information Used for Prediction")

    display_df = input_df.T.reset_index()

    display_df.columns = [
        "Feature",
        "Value"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# 12. FOOTER

st.markdown("---")

st.caption(
    "Credit Card Default Prediction | "
    "Machine Learning Classification Project"
)
