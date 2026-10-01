import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib

from src.predict import predict_churn

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon="📊",
    layout="wide"
)
# --------------------------------------------------
# Custom Dashboard Styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Main application */
    .main {
        padding-top: 1rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.20);
        padding: 1rem;
        border-radius: 12px;
    }

    /* Metric value */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Section headers */
    h1 {
        font-weight: 700;
    }

    h2 {
        margin-top: 1.5rem;
    }

    h3 {
        margin-top: 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.20);
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Form submit button */
    button[kind="primaryFormSubmit"] {
        border-radius: 8px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():
    rfm = pd.read_csv(
        "data/processed/customer_segmentation.csv"
    )

    cltv = pd.read_csv(
        "data/processed/customer_cltv.csv"
    )

    telco = pd.read_csv(
        "data/processed/telco_churn_clean.csv"
    )

    return rfm, cltv, telco


rfm, cltv, telco = load_data()

@st.cache_resource
def load_churn_model():
    return joblib.load(
        "data/processed/xgb_churn_pipeline.joblib"
    )


churn_model = load_churn_model()
# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Customer Retention Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Customer Segmentation",
        "CLTV Analysis",
        "Churn Prediction",
        "Retention Strategy"
    ]
)


# --------------------------------------------------
# Executive Overview
# --------------------------------------------------

if page == "Executive Overview":

    st.title("📊 Customer Retention Analytics")

    st.markdown(
        """
        ### From customer behavior to retention decisions

        An end-to-end customer analytics framework combining
        **behavioral segmentation, CLTV analysis, churn prediction,
        explainable machine learning, and retention strategy**.
        """
    )

    st.divider()

    # ---------------------------------------------------------
    # Executive KPIs
    # ---------------------------------------------------------

    total_customers = len(rfm)

    total_cltv = cltv["CLTV"].sum()

    high_value_customers = (
        rfm["BehavioralSegment"]
        .eq("High-Value Champions")
        .sum()
    )

    at_risk_customers = (
        rfm["BehavioralSegment"]
        .eq("At-Risk Valuable Customers")
        .sum()
    )

    churn_rate = (
        (telco["Churn"] == "Yes").mean() * 100
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Retail Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Modeled CLTV",
        f"£{total_cltv:,.0f}"
    )

    col3.metric(
        "High-Value Champions",
        f"{high_value_customers:,}"
    )

    col4.metric(
        "At-Risk Valuable",
        f"{at_risk_customers:,}"
    )

    col5.metric(
        "Telco Churn Rate",
        f"{churn_rate:.1f}%"
    )

    st.divider()

    # ---------------------------------------------------------
    # Project Architecture
    # ---------------------------------------------------------

    st.subheader("🔄 Analytics Framework")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            ### 1️⃣ Understand

            **Customer Behavior**

            Analyze transaction history,
            purchasing frequency, recency,
            monetary value, and engagement.
            """
        )

    with col2:

        st.markdown(
            """
            ### 2️⃣ Segment

            **Customer Groups**

            Apply RFM and behavioral
            segmentation to identify
            meaningful customer groups.
            """
        )

    with col3:

        st.markdown(
            """
            ### 3️⃣ Predict

            **Churn Risk**

            Use XGBoost to estimate
            individual customer churn
            probability.
            """
        )

    with col4:

        st.markdown(
            """
            ### 4️⃣ Act

            **Retention Strategy**

            Combine customer value,
            behavior, and risk to support
            targeted retention actions.
            """
        )

    st.divider()

    # ---------------------------------------------------------
    # Dataset Overview
    # ---------------------------------------------------------

    st.subheader("📁 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### 🛍️ Online Retail

            **Business context:** E-commerce

            **Customers:** 4,338

            **Purpose:**
            - RFM analysis
            - Behavioral segmentation
            - Customer value analysis
            - CLTV estimation

            **Key outputs:**
            - 4 behavioral segments
            - Scenario-based 12-month CLTV
            - Customer value distribution
            """
        )

    with col2:

        st.markdown(
            """
            ### 📡 Telco Customer Churn

            **Business context:** Telecommunications

            **Customers:** 7,043

            **Purpose:**
            - Churn analysis
            - Predictive modeling
            - Risk identification
            - Retention prioritization

            **Key outputs:**
            - XGBoost churn model
            - Probability-based risk scoring
            - SHAP model explanations
            """
        )

    st.divider()

    # ---------------------------------------------------------
    # Key Analytical Findings
    # ---------------------------------------------------------

    st.subheader("📌 Key Analytical Findings")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            **Customer Value**

            - High-Value Champions have the highest average modeled CLTV.
            - At-Risk Valuable Customers represent a substantial amount
              of aggregate modeled customer value.
            - Customer value is highly right-skewed.
            """
        )

    with col2:

        st.markdown(
            """
            **Churn Risk**

            - Tenure, contract type, internet service, and billing-related
              variables are important model features.
            - The XGBoost model can identify customers at different
              estimated levels of churn risk.
            - SHAP provides individual prediction explanations.
            """
        )

    st.divider()

    # ---------------------------------------------------------
    # Model Performance
    # ---------------------------------------------------------

    st.subheader("🤖 Churn Model Performance")

    model_metrics = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "XGBoost"
            ],
            "Precision": [
                0.6604,
                0.4734
            ],
            "Recall": [
                0.5615,
                0.8316
            ],
            "F1-Score": [
                0.6069,
                0.6033
            ],
            "ROC-AUC": [
                0.8422,
                0.8399
            ],
            "PR-AUC": [
                0.6350,
                0.6544
            ]
        }
    )

    st.dataframe(
        model_metrics.style.format(
            {
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-Score": "{:.3f}",
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "XGBoost metrics shown here use the project's 0.20 screening "
        "threshold for classification metrics. ROC-AUC and PR-AUC are "
        "threshold-independent."
    )

    st.divider()

    # ---------------------------------------------------------
    # Dataset Separation Notice
    # ---------------------------------------------------------

    st.info(
        """
        **Important:** The Online Retail and Telco datasets represent
        different business contexts and are intentionally not merged.

        Retail analytics is used for customer behavior, segmentation,
        and CLTV, while Telco data is used for supervised churn
        prediction. The two analyses are connected conceptually through
        a broader customer-retention framework.
        """
    )

    st.caption(
        "This dashboard is an analytical prototype. Model predictions, "
        "CLTV estimates, and retention recommendations should be "
        "validated against real business outcomes before production use."
    )


elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.markdown(
        """
        Customers are grouped into four behavioral segments using
        RFM and additional purchasing behavior features.
        """
    )

    st.divider()

    # ---------------------------------------------------------
    # Segment Summary
    # ---------------------------------------------------------

    segment_summary = (
        rfm.groupby("BehavioralSegment")
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean")
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # KPI Cards
    # ---------------------------------------------------------

    total_customers = len(rfm)

    high_value = (
        rfm["BehavioralSegment"]
        .eq("High-Value Champions")
        .sum()
    )

    at_risk = (
        rfm["BehavioralSegment"]
        .eq("At-Risk Valuable Customers")
        .sum()
    )

    loyal = (
        rfm["BehavioralSegment"]
        .eq("Engaged Loyal Customers")
        .sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "High-Value Champions",
        f"{high_value:,}"
    )

    col3.metric(
        "Engaged Loyal",
        f"{loyal:,}"
    )

    col4.metric(
        "At-Risk Valuable",
        f"{at_risk:,}"
    )

    st.divider()

    # ---------------------------------------------------------
    # Customer Distribution
    # ---------------------------------------------------------

    st.subheader("Customer Distribution by Segment")

    segment_counts = (
        rfm["BehavioralSegment"]
        .value_counts()
    )

    st.bar_chart(segment_counts)

    st.divider()

    # ---------------------------------------------------------
    # Segment Behavioral Profile
    # ---------------------------------------------------------

    st.subheader("Segment Behavioral Profile")

    display_summary = segment_summary.copy()

    display_summary["AvgRecency"] = (
        display_summary["AvgRecency"].round(1)
    )

    display_summary["AvgFrequency"] = (
        display_summary["AvgFrequency"].round(2)
    )

    display_summary["AvgMonetary"] = (
        display_summary["AvgMonetary"].round(2)
    )

    display_summary = display_summary.rename(
        columns={
            "BehavioralSegment": "Segment",
            "AvgRecency": "Avg Recency",
            "AvgFrequency": "Avg Frequency",
            "AvgMonetary": "Avg Monetary (£)"
        }
    )

    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ---------------------------------------------------------
    # CLTV Comparison
    # ---------------------------------------------------------

    st.subheader("Average CLTV by Segment")

    # Merge CLTV with segmentation data
    segment_cltv = (
        cltv.groupby("BehavioralSegment")["CLTV"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(segment_cltv)

    st.divider()

    # ---------------------------------------------------------
    # Segment Explorer
    # ---------------------------------------------------------

    st.subheader("🔎 Segment Explorer")

    selected_segment = st.selectbox(
        "Select a customer segment",
        sorted(rfm["BehavioralSegment"].unique())
    )

    selected = rfm[
        rfm["BehavioralSegment"] == selected_segment
    ]

    selected_cltv = cltv[
        cltv["BehavioralSegment"] == selected_segment
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers",
        f"{len(selected):,}"
    )

    col2.metric(
        "Average Monetary Value",
        f"£{selected['Monetary'].mean():,.2f}"
    )

    col3.metric(
        "Average CLTV",
        f"£{selected_cltv['CLTV'].mean():,.2f}"
    )

    st.markdown("### Segment Characteristics")

    characteristics = {
        "High-Value Champions": """
        These customers have high purchase frequency, high monetary
        value, relatively recent activity, and broad product engagement.
        They represent the strongest historical-value customer group.
        """,

        "Engaged Loyal Customers": """
        These customers show repeated purchasing behavior and relatively
        recent activity, but their monetary value is lower than the
        High-Value Champion segment.
        """,

        "At-Risk Valuable Customers": """
        These customers have relatively high historical monetary value
        but have not purchased recently. Their historical value makes
        them an important group for retention analysis.
        """,

        "Low-Value Inactive Customers": """
        These customers have low purchase frequency, low monetary value,
        limited product engagement, and relatively long periods since
        their last purchase.
        """
    }

    st.info(
        characteristics[selected_segment]
    )
    
elif page == "CLTV Analysis":

    st.title("💰 Customer Lifetime Value Analysis")

    st.markdown(
        """
        Customer Lifetime Value (CLTV) is estimated using historical
        purchasing behavior to create a scenario-based 12-month
        revenue proxy.
        """
    )

    st.divider()

    # ---------------------------------------------------------
    # CLTV Overview
    # ---------------------------------------------------------

    total_cltv = cltv["CLTV"].sum()
    average_cltv = cltv["CLTV"].mean()
    median_cltv = cltv["CLTV"].median()
    customer_count = len(cltv)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Modeled CLTV",
        f"£{total_cltv:,.0f}"
    )

    col2.metric(
        "Average CLTV",
        f"£{average_cltv:,.2f}"
    )

    col3.metric(
        "Median CLTV",
        f"£{median_cltv:,.2f}"
    )

    col4.metric(
        "Customers",
        f"{customer_count:,}"
    )

    st.divider()

    # ---------------------------------------------------------
    # CLTV Distribution
    # ---------------------------------------------------------

    st.subheader("CLTV Distribution")

    st.markdown(
        """
        The distribution is right-skewed, meaning a relatively small
        number of customers have substantially higher modeled CLTV
        than the typical customer.
        """
    )

    cltv_hist = pd.cut(
        cltv["CLTV"],
        bins=[
            0,
            2000,
            4000,
            6000,
            10000,
            20000,
            50000,
            float("inf")
        ],
        labels=[
            "£0–2K",
            "£2K–4K",
            "£4K–6K",
            "£6K–10K",
            "£10K–20K",
            "£20K–50K",
            "£50K+"
        ],
        include_lowest=True
    )

    distribution = (
        cltv_hist
        .value_counts()
        .sort_index()
    )

    st.bar_chart(distribution)

    st.divider()

    # ---------------------------------------------------------
    # CLTV by Behavioral Segment
    # ---------------------------------------------------------

    st.subheader("CLTV by Behavioral Segment")

    segment_cltv = (
        cltv.groupby("BehavioralSegment")
        .agg(
            Customers=("CustomerID", "count"),
            MeanCLTV=("CLTV", "mean"),
            MedianCLTV=("CLTV", "median"),
            TotalCLTV=("CLTV", "sum")
        )
        .reset_index()
        .sort_values("MeanCLTV", ascending=False)
    )

    segment_cltv_display = segment_cltv.copy()

    segment_cltv_display["MeanCLTV"] = (
        segment_cltv_display["MeanCLTV"].round(2)
    )

    segment_cltv_display["MedianCLTV"] = (
        segment_cltv_display["MedianCLTV"].round(2)
    )

    segment_cltv_display["TotalCLTV"] = (
        segment_cltv_display["TotalCLTV"].round(2)
    )

    segment_cltv_display = segment_cltv_display.rename(
        columns={
            "BehavioralSegment": "Segment",
            "MeanCLTV": "Mean CLTV (£)",
            "MedianCLTV": "Median CLTV (£)",
            "TotalCLTV": "Total CLTV (£)"
        }
    )

    st.dataframe(
        segment_cltv_display,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ---------------------------------------------------------
    # Average CLTV Chart
    # ---------------------------------------------------------

    st.subheader("Average CLTV by Segment")

    average_segment_cltv = (
        segment_cltv
        .set_index("BehavioralSegment")["MeanCLTV"]
        .sort_values(ascending=False)
    )

    st.bar_chart(average_segment_cltv)

    st.divider()

    # ---------------------------------------------------------
    # Total CLTV Contribution
    # ---------------------------------------------------------

    st.subheader("Total Modeled CLTV by Segment")

    total_segment_cltv = (
        segment_cltv
        .set_index("BehavioralSegment")["TotalCLTV"]
        .sort_values(ascending=False)
    )

    st.bar_chart(total_segment_cltv)

    st.divider()

    # ---------------------------------------------------------
    # CLTV Share
    # ---------------------------------------------------------

    st.subheader("CLTV Contribution by Segment")

    segment_cltv["CLTVShare"] = (
        segment_cltv["TotalCLTV"]
        / segment_cltv["TotalCLTV"].sum()
        * 100
    )

    share_display = (
        segment_cltv[
            ["BehavioralSegment", "CLTVShare"]
        ]
        .sort_values("CLTVShare", ascending=False)
        .copy()
    )

    share_display["CLTVShare"] = (
        share_display["CLTVShare"].round(2)
    )

    share_display = share_display.rename(
        columns={
            "BehavioralSegment": "Segment",
            "CLTVShare": "CLTV Share (%)"
        }
    )

    st.dataframe(
        share_display,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ---------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------

    st.subheader("Business Interpretation")

    st.markdown(
        """
        **High-Value Champions** have the highest average modeled CLTV,
        reflecting their high purchase frequency and monetary value.

        **At-Risk Valuable Customers** represent an important retention
        opportunity because they combine relatively high historical value
        with long periods since their most recent purchase.

        **Engaged Loyal Customers** contribute substantial aggregate
        modeled value because of their large customer population and
        repeated purchasing behavior.

        **Low-Value Inactive Customers** have the lowest modeled CLTV,
        reflecting limited purchase frequency and lower monetary value.
        """
    )

    st.info(
        """
        CLTV is a scenario-based 12-month revenue proxy derived from
        historical purchasing behavior. It is not a prediction of
        future profit and does not account for costs, margins,
        discounts, retention costs, or customer acquisition costs.
        """
    )


# =========================================================
# Churn Prediction
# =========================================================

elif page == "Churn Prediction":

    st.title("⚠️ Churn Prediction")

    st.markdown(
        """
        Enter customer information below to estimate the probability
        that the customer will churn using the trained XGBoost model.
        """
    )

    st.divider()

    st.subheader("Customer Information")

    # ---------------------------------------------------------
    # Prediction Form
    # ---------------------------------------------------------

    with st.form("churn_prediction_form"):

        # Customer Profile
        st.markdown("### 👤 Customer Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

        with col2:
            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with col3:
            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

        with col2:
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12,
                step=1
            )

        with col3:
            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

        # -----------------------------------------------------
        # Services
        # -----------------------------------------------------

        st.markdown("### 📡 Services")

        col1, col2, col3 = st.columns(3)

        with col1:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

        with col2:
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

        with col3:
            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col2:
            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

        with col3:
            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

        col1, col2 = st.columns(2)

        with col1:
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

        with col2:
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        # -----------------------------------------------------
        # Contract & Billing
        # -----------------------------------------------------

        st.markdown("### 💳 Contract & Billing")

        col1, col2, col3 = st.columns(3)

        with col1:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

        with col2:
            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

        with col3:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        # -----------------------------------------------------
        # Charges
        # -----------------------------------------------------

        st.markdown("### 💰 Charges")

        col1, col2 = st.columns(2)

        with col1:
            monthly_charges = st.number_input(
                "Monthly Charges (£)",
                min_value=0.0,
                max_value=200.0,
                value=80.0,
                step=1.0
            )

        with col2:
            total_charges = st.number_input(
                "Total Charges (£)",
                min_value=0.0,
                max_value=10000.0,
                value=960.0,
                step=10.0
            )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Churn",
            use_container_width=True
        )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    if submitted:

        customer = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        result = predict_churn(customer)

        probability = result["churn_probability"]
        prediction = result["churn_prediction"]
        risk = result["risk_level"]

                # -----------------------------------------------------
        # SHAP Explanation
        # -----------------------------------------------------

        preprocessor = churn_model.named_steps["preprocessor"]
        xgb_classifier = churn_model.named_steps["classifier"]

        customer_df = pd.DataFrame([customer])

        customer_transformed = preprocessor.transform(
            customer_df
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        explainer = shap.TreeExplainer(
            xgb_classifier
        )

        shap_values = explainer.shap_values(
            customer_transformed
        )

        customer_shap = shap_values[0]

        shap_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "SHAP Value": customer_shap
            }
        )

        shap_df["Abs SHAP"] = (
            shap_df["SHAP Value"].abs()
        )

        shap_df = shap_df.sort_values(
            "Abs SHAP",
            ascending=False
        )

        st.divider()

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )

        col2.metric(
            "Risk Level",
            risk
        )

        prediction_text = (
            "Likely Churn"
            if prediction == 1
            else "Likely Retain"
        )

        col3.metric(
            "Prediction",
            prediction_text
        )

        st.divider()

        if risk == "High Risk":

            st.error(
                f"""
                **High-risk customer**

                The model estimates a churn probability of
                **{probability * 100:.1f}%**.

                This customer crosses the 20% screening threshold
                used by the retention model.
                """
            )

        elif risk == "Medium Risk":

            st.warning(
                f"""
                **Medium-risk customer**

                The model estimates a churn probability of
                **{probability * 100:.1f}%**.

                This customer crosses the 20% screening threshold,
                but the estimated probability is below 50%.
                """
            )

        else:

            st.success(
                f"""
                **Low-risk customer**

                The model estimates a churn probability of
                **{probability * 100:.1f}%**.

                This customer is below the 20% screening threshold.
                """
            )

        # -----------------------------------------------------
        # Threshold Explanation
        # -----------------------------------------------------

        st.subheader("Model Decision Threshold")

        st.markdown(
            """
            The dashboard uses a **0.20 probability threshold** for
            churn screening.

            This threshold produced substantially higher recall than
            the default 0.50 threshold during model evaluation under
            the project's illustrative business-cost assumptions.

            The threshold should be recalibrated using actual retention
            costs, customer value, intervention success rates, and
            business objectives before production deployment.
            """
        )

        st.progress(
            min(probability, 1.0),
            text=f"Estimated churn probability: {probability * 100:.1f}%"
        )

        # -----------------------------------------------------
        # Customer Summary
        # -----------------------------------------------------

        st.subheader("Customer Summary")

        summary = pd.DataFrame(
            {
                "Attribute": [
                    "Tenure",
                    "Contract",
                    "Internet Service",
                    "Monthly Charges",
                    "Total Charges",
                    "Payment Method"
                ],
                "Value": [
                    f"{tenure} months",
                    contract,
                    internet_service,
                    f"£{monthly_charges:,.2f}",
                    f"£{total_charges:,.2f}",
                    payment_method
                ]
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Model outputs are estimates based on historical Telco "
            "customer data and should be validated before operational use."
        )
                # -----------------------------------------------------
        # SHAP Model Explanation
        # -----------------------------------------------------

        st.divider()

        st.subheader("🔍 Why Did the Model Make This Prediction?")

        st.markdown(
            """
            SHAP values show which features had the greatest influence
            on this individual model prediction.

            - **Positive SHAP values** increase the model's estimated
              churn probability.
            - **Negative SHAP values** decrease the model's estimated
              churn probability.
            - Larger absolute values indicate stronger influence.
            """
        )

        # Top features by absolute SHAP value
        top_shap = (
            shap_df
            .head(10)
            .copy()
        )

        # Remove preprocessing prefixes
        top_shap["Feature"] = (
            top_shap["Feature"]
            .str.replace(
                "num__",
                "",
                regex=False
            )
            .str.replace(
                "cat__",
                "",
                regex=False
            )
        )

        # -----------------------------------------------------
        # Factors increasing / decreasing risk
        # -----------------------------------------------------

        increasing = (
            top_shap[
                top_shap["SHAP Value"] > 0
            ]
            .sort_values(
                "SHAP Value",
                ascending=False
            )
        )

        decreasing = (
            top_shap[
                top_shap["SHAP Value"] < 0
            ]
            .sort_values(
                "SHAP Value"
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 🔺 Increasing Churn Risk"
            )

            if increasing.empty:

                st.info(
                    "No positive SHAP contributions among the top features."
                )

            else:

                for _, row in increasing.iterrows():

                    st.markdown(
                        f"- **{row['Feature']}** "
                        f"({row['SHAP Value']:+.3f})"
                    )

        with col2:

            st.markdown(
                "### 🔻 Reducing Churn Risk"
            )

            if decreasing.empty:

                st.info(
                    "No negative SHAP contributions among the top features."
                )

            else:

                for _, row in decreasing.iterrows():

                    st.markdown(
                        f"- **{row['Feature']}** "
                        f"({row['SHAP Value']:+.3f})"
                    )

        # -----------------------------------------------------
        # SHAP table
        # -----------------------------------------------------

        st.markdown("### Feature Influence")

        shap_display = top_shap[
            ["Feature", "SHAP Value"]
        ].copy()

        shap_display["SHAP Value"] = (
            shap_display["SHAP Value"].round(4)
        )

                # -----------------------------------------------------
        # SHAP Impact Chart
        # -----------------------------------------------------

        st.markdown("### 📊 Feature Impact on Churn Prediction")

        chart_data = (
            top_shap[
                ["Feature", "SHAP Value"]
            ]
            .sort_values(
                "SHAP Value"
            )
            .set_index("Feature")
        )

        st.bar_chart(
            chart_data,
            horizontal=True
        )

        st.caption(
            "Positive SHAP values indicate features that increased "
            "the model's estimated churn probability, while negative "
            "values indicate features that reduced it. SHAP values "
            "explain the model prediction and should not be interpreted "
            "as causal effects."
        )

        # -----------------------------------------------------
        # SHAP Table
        # -----------------------------------------------------

        st.markdown("### 📋 Feature Influence")

        shap_display = top_shap[
            ["Feature", "SHAP Value"]
        ].copy()

        shap_display["SHAP Value"] = (
            shap_display["SHAP Value"].round(4)
        )

        st.dataframe(
            shap_display,
            use_container_width=True,
            hide_index=True
        )
# =========================================================
# Retention Strategy
# =========================================================

elif page == "Retention Strategy":

    st.title("🎯 Retention Strategy")

    st.markdown(
        """
        This section translates customer segmentation, CLTV, and churn
        analysis into actionable retention strategies.

        The Online Retail and Telco datasets represent different
        business contexts, so the recommendations are developed
        separately rather than treating them as the same customers.
        """
    )

    st.divider()

    # ---------------------------------------------------------
    # Retail Retention Overview
    # ---------------------------------------------------------

    st.subheader("🛍️ Online Retail Retention Strategy")

    st.markdown(
        """
        The retail dataset provides behavioral segments and modeled
        CLTV. These metrics can be used to prioritize customer
        engagement based on historical purchasing behavior.
        """
    )

    # ---------------------------------------------------------
    # Retail Segment Summary
    # ---------------------------------------------------------

    retail_strategy = (
        cltv.groupby("BehavioralSegment")
        .agg(
            Customers=("CustomerID", "count"),
            MeanCLTV=("CLTV", "mean"),
            TotalCLTV=("CLTV", "sum")
        )
        .reset_index()
    )

    retail_strategy["CLTVShare"] = (
        retail_strategy["TotalCLTV"]
        / retail_strategy["TotalCLTV"].sum()
        * 100
    )

    retail_strategy = retail_strategy.sort_values(
        "TotalCLTV",
        ascending=False
    )

    # ---------------------------------------------------------
    # KPI Cards
    # ---------------------------------------------------------

    at_risk_row = retail_strategy[
        retail_strategy["BehavioralSegment"]
        == "At-Risk Valuable Customers"
    ]

    champions_row = retail_strategy[
        retail_strategy["BehavioralSegment"]
        == "High-Value Champions"
    ]

    at_risk_cltv = (
        at_risk_row["TotalCLTV"].iloc[0]
        if not at_risk_row.empty
        else 0
    )

    champion_cltv = (
        champions_row["TotalCLTV"].iloc[0]
        if not champions_row.empty
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customers",
        f"{len(rfm):,}"
    )

    col2.metric(
        "At-Risk Customers",
        f"{at_risk_row['Customers'].iloc[0]:,}"
        if not at_risk_row.empty
        else "0"
    )

    col3.metric(
        "At-Risk Modeled CLTV",
        f"£{at_risk_cltv:,.0f}"
    )

    col4.metric(
        "Champion Modeled CLTV",
        f"£{champion_cltv:,.0f}"
    )

    st.divider()

    # ---------------------------------------------------------
    # Retention Priority Matrix
    # ---------------------------------------------------------

    st.subheader("📌 Retention Priority Matrix")

    priority_data = pd.DataFrame(
        {
            "Segment": [
                "High-Value Champions",
                "At-Risk Valuable Customers",
                "Engaged Loyal Customers",
                "Low-Value Inactive Customers"
            ],
            "Historical Value": [
                "Very High",
                "High",
                "Moderate",
                "Low"
            ],
            "Engagement": [
                "High",
                "Low",
                "Moderate–High",
                "Low"
            ],
            "Retention Objective": [
                "Protect & grow",
                "Win back",
                "Strengthen loyalty",
                "Selective reactivation"
            ]
        }
    )

    st.dataframe(
        priority_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ---------------------------------------------------------
    # Segment Strategies
    # ---------------------------------------------------------

    st.subheader("💡 Segment-Specific Strategies")

    selected_segment = st.selectbox(
        "Select a customer segment",
        [
            "High-Value Champions",
            "At-Risk Valuable Customers",
            "Engaged Loyal Customers",
            "Low-Value Inactive Customers"
        ]
    )

    strategies = {

        "High-Value Champions": {
            "objective": "Protect and increase customer value",
            "actions": [
                "Provide VIP or loyalty-program benefits.",
                "Offer early access to new products or collections.",
                "Use personalized recommendations based on purchase history.",
                "Encourage cross-selling and premium product adoption.",
                "Monitor changes in purchase frequency as an early warning signal."
            ],
            "kpi": [
                "Repeat purchase rate",
                "Average order value",
                "Purchase frequency",
                "Customer revenue"
            ]
        },

        "At-Risk Valuable Customers": {
            "objective": "Re-engage customers before further inactivity",
            "actions": [
                "Launch personalized win-back campaigns.",
                "Recommend products related to previous purchases.",
                "Use limited-time incentives selectively.",
                "Send reminders based on historical purchasing patterns.",
                "Prioritize customers with higher modeled CLTV for stronger interventions."
            ],
            "kpi": [
                "Reactivation rate",
                "Time to next purchase",
                "Repeat purchase rate",
                "Revenue recovered"
            ]
        },

        "Engaged Loyal Customers": {
            "objective": "Strengthen loyalty and encourage progression",
            "actions": [
                "Introduce loyalty rewards based on purchase frequency.",
                "Use personalized product recommendations.",
                "Encourage larger baskets through relevant cross-selling.",
                "Provide incentives for reaching higher loyalty tiers.",
                "Monitor movement toward the High-Value Champion segment."
            ],
            "kpi": [
                "Purchase frequency",
                "Average order value",
                "Customer revenue",
                "Loyalty-tier progression"
            ]
        },

        "Low-Value Inactive Customers": {
            "objective": "Reactivate selectively while controlling campaign cost",
            "actions": [
                "Use low-cost automated reactivation campaigns.",
                "Test simple product or discount recommendations.",
                "Avoid expensive one-to-one interventions unless justified by value.",
                "Measure campaign response before increasing investment.",
                "Consider suppressing persistently inactive customers from costly campaigns."
            ],
            "kpi": [
                "Reactivation rate",
                "Campaign cost per reactivated customer",
                "Incremental revenue",
                "Campaign ROI"
            ]
        }
    }

    strategy = strategies[selected_segment]

    st.markdown(
        f"### 🎯 Objective: {strategy['objective']}"
    )

    st.markdown("#### Recommended Actions")

    for action in strategy["actions"]:
        st.markdown(f"- {action}")

    st.markdown("#### Suggested KPIs")

    for kpi in strategy["kpi"]:
        st.markdown(f"- **{kpi}**")

    st.divider()

    # ---------------------------------------------------------
    # Modeled CLTV Contribution
    # ---------------------------------------------------------

    st.subheader("💰 Modeled CLTV Contribution")

    cltv_chart = (
        retail_strategy
        .set_index("BehavioralSegment")["TotalCLTV"]
        .sort_values(ascending=False)
    )

    st.bar_chart(cltv_chart)

    st.divider()

    # ---------------------------------------------------------
    # Telco Retention Strategy
    # ---------------------------------------------------------

    st.subheader("📡 Telco Churn Retention Strategy")

    st.markdown(
        """
        The Telco analysis uses churn probability to identify customers
        who may require retention attention. The model should be treated
        as a prioritization tool rather than proof that a customer will
        churn.
        """
    )

    # ---------------------------------------------------------
    # Contract Churn Profile
    # ---------------------------------------------------------

    contract_churn = (
        telco.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    st.subheader("Churn Rate by Contract Type")

    st.bar_chart(contract_churn)

    st.caption(
        "Historical churn rates shown here describe the Telco dataset "
        "and do not establish that contract type causes churn."
    )

    st.divider()

    # ---------------------------------------------------------
    # Telco Retention Actions
    # ---------------------------------------------------------

    st.subheader("🔧 Recommended Retention Actions")

    telco_actions = pd.DataFrame(
        {
            "Customer Signal": [
                "High churn probability",
                "Month-to-month contract",
                "High monthly charges",
                "Low tenure",
                "Limited support/security services"
            ],
            "Potential Action": [
                "Prioritize for retention outreach.",
                "Consider contract-focused retention offers.",
                "Review value proposition and available plans.",
                "Use onboarding and early-life engagement programs.",
                "Promote relevant support or security services."
            ],
            "Measurement": [
                "Retention rate / churn reduction",
                "Contract conversion rate",
                "Plan migration / retention rate",
                "Early-life retention",
                "Service adoption and retention"
            ]
        }
    )

    st.dataframe(
        telco_actions,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ---------------------------------------------------------
    # Retention Workflow
    # ---------------------------------------------------------

    st.subheader("🔄 Recommended Retention Workflow")

    workflow = pd.DataFrame(
        {
            "Step": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6"
            ],
            "Process": [
                "Identify",
                "Prioritize",
                "Intervene",
                "Measure",
                "Experiment",
                "Monitor"
            ],
            "Description": [
                "Identify customers showing inactivity, high CLTV, or elevated churn probability.",
                "Prioritize customers using historical value and predicted risk.",
                "Apply a retention action appropriate to the customer segment.",
                "Measure response, retention, revenue, and campaign cost.",
                "Run controlled A/B tests where possible.",
                "Monitor model performance, customer behavior, and drift over time."
            ]
        }
    )

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ---------------------------------------------------------
    # Important Limitations
    # ---------------------------------------------------------

    st.subheader("⚠️ Important Limitations")

    st.info(
        """
        **1. Dataset separation:** The Online Retail and Telco datasets
        represent different business contexts and do not contain the
        same customers.

        **2. CLTV:** The modeled CLTV is a scenario-based 12-month
        revenue proxy rather than a statistically estimated future
        profit value.

        **3. Segmentation:** Customer segments describe historical
        behavior and should not be interpreted as causal categories.

        **4. Churn prediction:** The XGBoost model estimates churn
        probability from historical Telco data. It does not establish
        why a customer will churn.

        **5. Retention actions:** Recommended interventions should be
        validated using controlled experiments and actual business
        outcomes.

        **6. Business cost:** The 20% churn-screening threshold is based
        on illustrative evaluation assumptions and should be recalibrated
        using real retention costs and customer value.
        """
    )

    st.success(
        """
        The overall framework is designed to connect **customer value,
        behavioral segmentation, churn risk, and retention actions**
        into a practical customer analytics workflow.
        """
    )
