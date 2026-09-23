import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit, minimize
import networkx as nx

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Biodiversity Loss Analysis",
    page_icon="🌿",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

STATUS_ORDER = ["LC", "NT", "VU", "EN", "CR", "EX"]

STATUS_NAMES = {
    "LC": "Least Concern",
    "NT": "Near Threatened",
    "VU": "Vulnerable",
    "EN": "Endangered",
    "CR": "Critically Endangered",
    "EX": "Extinct",
}

YEARS_ALL = [
    1996, 2000, 2002, 2003, 2004, 2006, 2007, 2008,
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
    2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024,
    2025,
]

TOTAL_AS_ALL = [
    np.nan, 16507, 16697, 22424, 38046, 40174, 41415,
    44838, 47677, 55926, 61914, 65518, 71576, 76199,
    79837, 85604, 91523, 96951, 112432, 128918,
    142577, 150388, 157190, 166061, 172620,
]

TOTAL_THR_ALL = [
    10533, 11046, 11167, 12259, 15503, 16116, 16306,
    16928, 17291, 18351, 19570, 20219, 21286, 22413,
    23250, 24307, 25821, 26840, 30178, 35765,
    40084, 42108, 44016, 46337, 48646,
]


# ============================================================
# SIDEBAR / FILE UPLOADS
# ============================================================

st.sidebar.title("🌿 Biodiversity Analysis")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "📊 Regression Analysis",
        "📈 Distribution Fitting",
        "🔄 Markov Chain",
        "🌲 Random Forest",
        "🔮 IUCN Prediction",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Files")

distribution_file = st.sidebar.file_uploader(
    "Distribution CSV",
    type=["csv"],
    help="Upload Distribution_Data.csv",
)

markov_file = st.sidebar.file_uploader(
    "Markov / Random Forest Excel",
    type=["xlsx", "xls"],
    help="Upload Markov_Random_Forest (1).xlsx",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_distribution_file(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data
def load_excel_file(file_bytes):
    from io import BytesIO
    return pd.read_excel(BytesIO(file_bytes))


def historical_dataframe():
    return pd.DataFrame(
        {
            "Year": YEARS_ALL,
            "Total_Assessed": TOTAL_AS_ALL,
            "Total_Threatened": TOTAL_THR_ALL,
        }
    )


def cubic_model(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3


def fit_cubic(x, y):
    return curve_fit(
        cubic_model,
        x,
        y,
        p0=[100, 100, 10, 10],
        maxfev=100000,
    )


def regression_summary(params, cov, x, y):
    residuals = y - cubic_model(x, *params)
    dof = len(x) - len(params)

    mse = np.sum(residuals**2) / dof
    perr = np.sqrt(np.diag(cov))
    t_values = params / perr

    p_values = [
        2 * (1 - stats.t.cdf(abs(t), dof))
        for t in t_values
    ]

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    summary = pd.DataFrame(
        {
            "Coefficient": [
                "Intercept",
                "Linear",
                "Quadratic",
                "Cubic",
            ],
            "Estimate": params,
            "Std_Error": perr,
            "t_value": t_values,
            "p_value": p_values,
        }
    )

    return summary, r_squared, np.sqrt(mse)


def calculate_cagr(data, column):
    first = data.iloc[0]
    last = data.iloc[-1]
    years = last["Year"] - first["Year"]

    if first[column] <= 0 or years <= 0:
        return np.nan

    return (last[column] / first[column]) ** (1 / years) - 1


@st.cache_data
def load_status_change_data(file_bytes):
    dataset = load_excel_file(file_bytes)

    dataset.columns = dataset.columns.astype(str).str.strip()

    required = {"Common name", "Scientific name"}
    missing = required - set(dataset.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    is_header = (
        dataset["Common name"].isna()
        & dataset["Scientific name"].notna()
    )

    dataset["Taxonomy"] = np.where(
        is_header,
        dataset["Scientific name"],
        np.nan,
    )

    dataset["Taxonomy"] = dataset["Taxonomy"].ffill()

    species = dataset[
        ~is_header
        & dataset["Scientific name"].notna()
    ].copy()

    return species


def prepare_rf_data(species):
    ds2 = species.dropna(
        subset=[
            "IUCN Red List (2021)",
            "IUCN Red List (2022)",
        ]
    ).copy()

    ds2 = ds2[
        ds2["IUCN Red List (2021)"].isin(STATUS_ORDER)
    ]

    ds2 = ds2.drop(
        columns=["Scientific name", "Common name"],
        errors="ignore",
    )

    encoder = OrdinalEncoder(
        categories=[STATUS_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    ds2["IUCN Red List (2021)"] = encoder.fit_transform(
        ds2[["IUCN Red List (2021)"]]
    )

    ds3 = pd.get_dummies(ds2, columns=["Taxonomy"])

    y = ds3["IUCN Red List (2022)"]

    X = ds3.drop(
        columns=[
            "IUCN Red List (2022)",
            "Unnamed: 4",
            "Reason for change",
            "Red List version",
        ],
        errors="ignore",
    )

    return X, y, encoder


def train_random_forest(species):
    X, y, encoder = prepare_rf_data(species)

    if y.nunique() < 2:
        raise ValueError(
            "At least two target IUCN classes are required."
        )

    class_counts = y.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Each target class needs at least 2 records "
            "for the stratified train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
        stratify=y,
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        random_state=0,
        class_weight="balanced",
    )

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    majority_class = y_train.value_counts().idxmax()

    baseline_accuracy = (
        y_test == majority_class
    ).mean()

    labels_sorted = sorted(y.unique())

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels_sorted,
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=labels_sorted,
        zero_division=0,
        output_dict=True,
    )

    importances = (
        pd.Series(
            classifier.feature_importances_,
            index=X.columns,
        )
        .sort_values(ascending=False)
    )

    return {
        "model": classifier,
        "encoder": encoder,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "labels": labels_sorted,
        "confusion_matrix": cm,
        "report": report,
        "importances": importances,
    }


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🌿 Biodiversity Loss Analysis")

    st.subheader(
        "Statistical Analysis and Machine Learning "
        "for IUCN Conservation Status"
    )

    st.markdown(
        """
        This application integrates four analytical modules:

        **1. Regression Analysis**  
        Historical IUCN assessment and threatened-species trends,
        cubic regression, CAGR and illustrative future extrapolation.

        **2. Distribution Fitting**  
        Descriptive statistics and comparison of count distributions.

        **3. Markov Chain Analysis**  
        Analysis of observed 2021 → 2022 IUCN status transitions.

        **4. Random Forest Classification**  
        Classification of 2022 IUCN status using 2021 status and
        taxonomy/group information.
        """
    )

    st.markdown("---")

    df = historical_dataframe()

    col1, col2, col3, col4 = st.columns(4)

    latest = df.iloc[-1]

    with col1:
        st.metric(
            "Latest Year",
            int(latest["Year"]),
        )

    with col2:
        st.metric(
            "Species Assessed",
            f"{int(latest['Total_Assessed']):,}",
        )

    with col3:
        st.metric(
            "Species Threatened",
            f"{int(latest['Total_Threatened']):,}",
        )

    with col4:
        ratio = (
            latest["Total_Threatened"]
            / latest["Total_Assessed"]
        )

        st.metric(
            "Threatened / Assessed",
            f"{ratio:.2%}",
        )

    st.markdown("---")

    st.info(
        "Use the sidebar to open an individual analysis module. "
        "Upload the Distribution CSV and Markov/Random Forest Excel "
        "file when using those modules."
    )

    st.warning(
        "The regression forecasts are illustrative extrapolations, "
        "not authoritative ecological forecasts. The Markov analysis "
        "uses available 2021→2022 change records and should not be "
        "interpreted as a complete population-level forecast."
    )


# ============================================================
# SECTION 1 — REGRESSION
# ============================================================

elif page == "📊 Regression Analysis":

    st.title("📊 Regression Analysis")

    df = historical_dataframe()

    st.subheader("Historical IUCN Red List Data")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Historical Trends")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["Year"],
        df["Total_Assessed"],
        linewidth=2,
        marker="o",
        label="Total Assessed",
    )

    ax.plot(
        df["Year"],
        df["Total_Threatened"],
        linewidth=2,
        marker="o",
        label="Total Threatened",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Species")
    ax.set_title(
        "IUCN Red List Assessment Trends (1996–2025)"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    st.pyplot(fig)

    st.subheader("Threatened / Assessed Ratio")

    df["Ratio"] = (
        df["Total_Threatened"]
        / df["Total_Assessed"]
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        df["Year"],
        df["Ratio"],
        linewidth=2,
        marker="o",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Threatened / Assessed")

    ax.set_title(
        "Share of Assessed Species Classified as Threatened"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    st.pyplot(fig)

    st.subheader("Cubic Regression Models")

    d_assessed = df.dropna(
        subset=["Total_Assessed"]
    ).copy()

    d_assessed["t"] = (
        d_assessed["Year"]
        - d_assessed["Year"].min()
    )

    d_threatened = df.copy()

    d_threatened["t"] = (
        d_threatened["Year"]
        - d_threatened["Year"].min()
    )

    params_assessed, cov_assessed = fit_cubic(
        d_assessed["t"].to_numpy(),
        d_assessed["Total_Assessed"].to_numpy(),
    )

    params_threatened, cov_threatened = fit_cubic(
        d_threatened["t"].to_numpy(),
        d_threatened["Total_Threatened"].to_numpy(),
    )

    summary_a, r2_a, se_a = regression_summary(
        params_assessed,
        cov_assessed,
        d_assessed["t"].to_numpy(),
        d_assessed["Total_Assessed"].to_numpy(),
    )

    summary_t, r2_t, se_t = regression_summary(
        params_threatened,
        cov_threatened,
        d_threatened["t"].to_numpy(),
        d_threatened["Total_Threatened"].to_numpy(),
    )

    tab1, tab2 = st.tabs(
        ["Total Assessed Model", "Total Threatened Model"]
    )

    with tab1:

        st.dataframe(
            summary_a,
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "R-squared",
            f"{r2_a:.5f}",
        )

        c2.metric(
            "Residual Std. Error",
            f"{se_a:.4f}",
        )

    with tab2:

        st.dataframe(
            summary_t,
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "R-squared",
            f"{r2_t:.5f}",
        )

        c2.metric(
            "Residual Std. Error",
            f"{se_t:.4f}",
        )

    st.subheader("CAGR Analysis")

    full = df.dropna(
        subset=["Total_Assessed"]
    )

    recent = full[
        full["Year"] >= 2015
    ]

    cagr_table = pd.DataFrame(
        {
            "Measure": [
                "Full-period assessed",
                "Full-period threatened",
                "2015–2025 assessed",
                "2015–2025 threatened",
            ],
            "CAGR": [
                calculate_cagr(
                    full,
                    "Total_Assessed",
                ),
                calculate_cagr(
                    full,
                    "Total_Threatened",
                ),
                calculate_cagr(
                    recent,
                    "Total_Assessed",
                ),
                calculate_cagr(
                    recent,
                    "Total_Threatened",
                ),
            ],
        }
    )

    cagr_table["CAGR"] = (
        cagr_table["CAGR"] * 100
    )

    st.dataframe(
        cagr_table.style.format(
            {"CAGR": "{:.4f}%"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Illustrative 2026–2031 Regression Extrapolation"
    )

    future_years = np.arange(
        2026,
        2032,
    )

    pred_assessed = cubic_model(
        future_years
        - d_assessed["Year"].min(),
        *params_assessed,
    )

    pred_threatened = cubic_model(
        future_years
        - d_threatened["Year"].min(),
        *params_threatened,
    )

    prediction_table = pd.DataFrame(
        {
            "Year": future_years,
            "Predicted_Assessed": np.maximum(
                0,
                np.round(
                    pred_assessed
                ).astype(int),
            ),
            "Predicted_Threatened": np.maximum(
                0,
                np.round(
                    pred_threatened
                ).astype(int),
            ),
        }
    )

    st.dataframe(
        prediction_table,
        use_container_width=True,
        hide_index=True,
    )

    csv = prediction_table.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Predictions CSV",
        csv,
        "species_predictions_2026_2031.csv",
        "text/csv",
    )

    st.info(
        "These predictions are mathematical extrapolations from "
        "historical data. They should not be interpreted as "
        "authoritative ecological forecasts."
    )


# ============================================================
# SECTION 2 — DISTRIBUTION FITTING
# ============================================================

elif page == "📈 Distribution Fitting":

    st.title("📈 Distribution Fitting")

    if distribution_file is None:

        st.warning(
            "Please upload Distribution_Data.csv using the "
            "sidebar to use this module."
        )

        st.stop()

    try:

        data = load_distribution_file(
            distribution_file.getvalue()
        )

    except Exception as e:

        st.error(
            f"Could not read the CSV file: {e}"
        )

        st.stop()

    if "Total" not in data.columns:

        st.error(
            "The uploaded CSV must contain a 'Total' column."
        )

        st.stop()

    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        f"Number of countries/territories: "
        f"**{len(data)}**"
    )

    try:

        total_sp = (
            data["Total"]
            .astype(str)
            .str.replace(
                ",",
                "",
                regex=False,
            )
            .astype(float)
            .dropna()
            .to_numpy()
        )

    except Exception as e:

        st.error(
            f"Could not convert the 'Total' column to numeric values: {e}"
        )

        st.stop()

    if len(total_sp) < 2:

        st.error(
            "At least two observations are required."
        )

        st.stop()

    mean_x = np.mean(total_sp)

    variance_x = np.var(
        total_sp,
        ddof=1,
    )

    sd_x = np.std(
        total_sp,
        ddof=1,
    )

    skewness = stats.skew(total_sp)

    kurtosis = stats.kurtosis(
        total_sp,
        fisher=False,
    )

    st.subheader("Descriptive Statistics")

    cols = st.columns(4)

    cols[0].metric(
        "Minimum",
        f"{np.min(total_sp):,.2f}",
    )

    cols[1].metric(
        "Maximum",
        f"{np.max(total_sp):,.2f}",
    )

    cols[2].metric(
        "Median",
        f"{np.median(total_sp):,.2f}",
    )

    cols[3].metric(
        "Mean",
        f"{mean_x:,.2f}",
    )

    stats_table = pd.DataFrame(
        {
            "Statistic": [
                "Standard Deviation",
                "Variance",
                "Variance / Mean",
                "Skewness",
                "Kurtosis",
            ],
            "Value": [
                sd_x,
                variance_x,
                variance_x / mean_x,
                skewness,
                kurtosis,
            ],
        }
    )

    st.dataframe(
        stats_table.style.format(
            {"Value": "{:,.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Distribution Histogram")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        total_sp,
        bins=32,
        edgecolor="black",
    )

    ax.set_xlabel(
        "Number of Threatened Species"
    )

    ax.set_ylabel("Frequency")

    ax.set_title(
        "Distribution of Threatened Species"
    )

    fig.tight_layout()

    st.pyplot(fig)

    st.subheader("Top Countries / Territories")

    if "Name" in data.columns:

        top_countries = (
            data.assign(
                Total=total_sp
            )
            .sort_values(
                "Total",
                ascending=False,
            )
            .head(15)
        )

        st.dataframe(
            top_countries[
                ["Name", "Total"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Distribution Fit")

    # Poisson
    lam_hat = np.mean(total_sp)

    ll_poisson = np.sum(
        stats.poisson.logpmf(
            total_sp,
            lam_hat,
        )
    )

    poisson_aic = (
        -2 * ll_poisson + 2
    )

    # Geometric
    def geom_nll(params, x):

        p = params[0]

        if not 0 < p < 1:
            return np.inf

        return -np.sum(
            x * np.log(1 - p)
            + np.log(p)
        )

    geom_res = minimize(
        geom_nll,
        x0=[0.01],
        args=(total_sp,),
        method="Nelder-Mead",
    )

    p_hat_geom = geom_res.x[0]

    # Negative Binomial
    def nbinom_nll(params, x):

        r, p = params

        if r <= 0 or not 0 < p < 1:
            return np.inf

        return -np.sum(
            stats.nbinom.logpmf(
                x,
                r,
                p,
            )
        )

    nbin_res = minimize(
        nbinom_nll,
        x0=[1.0, 0.5],
        args=(total_sp,),
        method="Nelder-Mead",
    )

    r_hat, p_hat_nb = nbin_res.x

    mu_hat_nb = (
        r_hat
        * (1 - p_hat_nb)
        / p_hat_nb
    )

    ll_nbinom = -nbinom_nll(
        [r_hat, p_hat_nb],
        total_sp,
    )

    nbinom_aic = (
        -2 * ll_nbinom + 4
    )

    distribution_table = pd.DataFrame(
        {
            "Distribution": [
                "Poisson",
                "Geometric",
                "Negative Binomial",
            ],
            "Parameter 1": [
                lam_hat,
                p_hat_geom,
                r_hat,
            ],
            "Parameter 2": [
                np.nan,
                np.nan,
                p_hat_nb,
            ],
            "AIC": [
                poisson_aic,
                np.nan,
                nbinom_aic,
            ],
        }
    )

    st.dataframe(
        distribution_table.style.format(
            {
                "Parameter 1": "{:.4f}",
                "Parameter 2": "{:.4f}",
                "AIC": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Poisson λ",
        f"{lam_hat:.4f}",
    )

    c2.metric(
        "Geometric p",
        f"{p_hat_geom:.4f}",
    )

    c3.metric(
        "Negative Binomial Mean",
        f"{mu_hat_nb:.4f}",
    )

    if nbinom_aic < poisson_aic:

        st.info(
            "For the Poisson vs Negative Binomial comparison, "
            "the Negative Binomial has the lower AIC."
        )

    else:

        st.info(
            "For the Poisson vs Negative Binomial comparison, "
            "the Poisson has the lower AIC."
        )

    st.subheader("Box Plot")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.boxplot(total_sp)

    ax.set_ylabel(
        "Threatened Species Count"
    )

    ax.set_title(
        "Box Plot of Threatened Species"
    )

    fig.tight_layout()

    st.pyplot(fig)


# ============================================================
# SECTION 3 — MARKOV CHAIN
# ============================================================

elif page == "🔄 Markov Chain":

    st.title("🔄 Markov Chain Analysis")

    if markov_file is None:

        st.warning(
            "Please upload Markov_Random_Forest (1).xlsx "
            "using the sidebar."
        )

        st.stop()

    try:

        species = load_status_change_data(
            markov_file.getvalue()
        )

    except Exception as e:

        st.error(
            f"Could not read the Excel file: {e}"
        )

        st.stop()

    required = {
        "IUCN Red List (2021)",
        "IUCN Red List (2022)",
    }

    missing = required - set(species.columns)

    if missing:

        st.error(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

        st.stop()

    species = species.dropna(
        subset=[
            "IUCN Red List (2021)",
            "IUCN Red List (2022)",
        ]
    )

    species = species[
        species["IUCN Red List (2021)"].isin(
            STATUS_ORDER
        )
        & species["IUCN Red List (2022)"].isin(
            STATUS_ORDER
        )
    ].copy()

    st.metric(
        "Species change records",
        len(species),
    )

    if "Reason for change" in species.columns:

        st.subheader("Reason for Change")

        reason_counts = (
            species["Reason for change"]
            .value_counts(dropna=False)
            .rename_axis("Reason")
            .reset_index(name="Count")
        )

        st.dataframe(
            reason_counts,
            use_container_width=True,
            hide_index=True,
        )

    rank = {
        "LC": 0,
        "NT": 1,
        "VU": 2,
        "EN": 3,
        "CR": 4,
        "EX": 5,
    }

    species["direction"] = np.where(
        species[
            "IUCN Red List (2022)"
        ].map(rank)
        > species[
            "IUCN Red List (2021)"
        ].map(rank),

        "Worsened",

        np.where(
            species[
                "IUCN Red List (2022)"
            ].map(rank)
            < species[
                "IUCN Red List (2021)"
            ].map(rank),

            "Improved",

            "Same",
        ),
    )

    st.subheader("Direction of Change")

    direction_counts = (
        species["direction"]
        .value_counts()
        .rename_axis("Direction")
        .reset_index(name="Count")
    )

    st.dataframe(
        direction_counts,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # TRANSITION COUNTS
    # ========================================================

    trans_counts = pd.crosstab(
        species["IUCN Red List (2021)"],
        species["IUCN Red List (2022)"],
    )

    trans_counts = trans_counts.reindex(
        index=STATUS_ORDER,
        columns=STATUS_ORDER,
        fill_value=0,
    )

    st.subheader("Transition Counts")

    st.dataframe(
        trans_counts,
        use_container_width=True,
    )

    # ========================================================
    # CALCULATE TRANSITION PROBABILITY MATRIX
    # ========================================================

    row_totals = trans_counts.sum(axis=1)

    transition_matrix = (
        trans_counts
        .div(
            row_totals.replace(0, np.nan),
            axis=0,
        )
        .fillna(0)
        .to_numpy(dtype=float)
        .copy()
    )

    # If a state has no observed outgoing transitions,
    # treat it as remaining in the same state.
    for i in range(len(STATUS_ORDER)):

        if transition_matrix[i].sum() == 0:

            transition_matrix[i, i] = 1.0

    transition_df = pd.DataFrame(
        transition_matrix,
        index=STATUS_ORDER,
        columns=STATUS_ORDER,
    )

    # ========================================================
    # TRANSITION PROBABILITY MATRIX DISPLAY
    # ========================================================

    st.subheader(
        "Transition Probability Matrix"
    )

    st.dataframe(
        transition_df.style.format(
            "{:.4f}"
        ),
        use_container_width=True,
    )

    # ========================================================
    # MARKOV TRANSITION GRAPH
    # ========================================================

    st.subheader("Markov Transition Graph")

    G = nx.DiGraph()

    G.add_nodes_from(
        STATUS_ORDER
    )

    for i, from_state in enumerate(
        STATUS_ORDER
    ):

        for j, to_state in enumerate(
            STATUS_ORDER
        ):

            probability = (
                transition_matrix[i, j]
            )

            if probability > 0:

                G.add_edge(
                    from_state,
                    to_state,
                    weight=probability,
                )

    pos = nx.circular_layout(G)

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1200,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_weight="bold",
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        connectionstyle="arc3,rad=0.15",
        arrowstyle="-|>",
        arrowsize=15,
        ax=ax,
    )

    edge_labels = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in G.edges(
            data=True
        )
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        ax=ax,
    )

    ax.set_title(
        "IUCN Conservation Status Transition Chain"
    )

    ax.axis("off")

    fig.tight_layout()

    st.pyplot(fig)

    # ========================================================
    # STEADY-STATE DISTRIBUTION
    # ========================================================

    st.subheader(
        "Steady-State Distribution"
    )

    def steady_state(P):

        eigenvalues, eigenvectors = np.linalg.eig(
            P.T
        )

        index = np.argmin(
            np.abs(
                eigenvalues - 1
            )
        )

        vector = np.real(
            eigenvectors[:, index]
        )

        vector = vector / vector.sum()

        return vector

    ss = steady_state(
        transition_matrix
    )

    ss_series = pd.Series(
        ss,
        index=STATUS_ORDER,
        name="Probability",
    )

    st.dataframe(
        ss_series
        .reset_index()
        .rename(
            columns={
                "index": "State"
            }
        )
        .style.format(
            {
                "Probability": "{:.6f}"
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        STATUS_ORDER,
        ss,
    )

    ax.set_xlabel(
        "IUCN Status"
    )

    ax.set_ylabel(
        "Long-run Probability"
    )

    ax.set_title(
        "Steady-State Conservation Status Distribution"
    )

    fig.tight_layout()

    st.pyplot(fig)

    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    transition_csv = (
        transition_df
        .to_csv()
        .encode("utf-8")
    )

    st.download_button(
        "📥 Download Transition Matrix",
        transition_csv,
        "status_transition_matrix.csv",
        "text/csv",
    )

    steady_csv = (
        ss_series
        .reset_index()
        .rename(
            columns={
                "index": "State"
            }
        )
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "📥 Download Steady-State Results",
        steady_csv,
        "status_steady_states.csv",
        "text/csv",
    )

    st.warning(
        "Limitation: the input file contains species with "
        "recorded status changes between 2021 and 2022. "
        "Therefore, this transition matrix represents the "
        "available change records rather than a complete "
        "population-level forecast."
    )


# ============================================================
# SECTION 4 — RANDOM FOREST
# ============================================================

elif page == "🌲 Random Forest":

    st.title("🌲 Random Forest Classification")

    if markov_file is None:

        st.warning(
            "Please upload Markov_Random_Forest (1).xlsx "
            "using the sidebar."
        )

        st.stop()

    try:

        species = load_status_change_data(
            markov_file.getvalue()
        )

    except Exception as e:

        st.error(
            f"Could not read the Excel file: {e}"
        )

        st.stop()

    try:

        rf = train_random_forest(
            species
        )

    except Exception as e:

        st.error(
            f"Could not train the Random Forest model: {e}"
        )

        st.stop()

    st.subheader("Model Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Training Records",
        len(rf["X_train"]),
    )

    c2.metric(
        "Testing Records",
        len(rf["X_test"]),
    )

    c3.metric(
        "Number of Features",
        rf["X"].shape[1],
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Test Accuracy",
        f"{rf['accuracy']:.2%}",
    )

    c2.metric(
        "Baseline Accuracy",
        f"{rf['baseline_accuracy']:.2%}",
    )

    st.subheader("Features Used")

    st.write(
        rf["X"].columns.tolist()
    )

    st.subheader("Confusion Matrix")

    cm = rf["confusion_matrix"]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    image = ax.imshow(cm)

    ax.set_xticks(
        range(len(rf["labels"]))
    )

    ax.set_yticks(
        range(len(rf["labels"]))
    )

    ax.set_xticklabels(
        rf["labels"]
    )

    ax.set_yticklabels(
        rf["labels"]
    )

    ax.set_xlabel(
        "Predicted Status"
    )

    ax.set_ylabel(
        "Actual Status"
    )

    ax.set_title(
        "Random Forest Confusion Matrix"
    )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    st.pyplot(fig)

    st.subheader(
        "Classification Report"
    )

    report_df = pd.DataFrame(
        rf["report"]
    ).transpose()

    st.dataframe(
        report_df,
        use_container_width=True,
    )

    st.subheader(
        "Feature Importance"
    )

    importances = rf[
        "importances"
    ]

    st.dataframe(
        importances
        .rename("Importance")
        .reset_index()
        .rename(
            columns={
                "index": "Feature"
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    importances.plot(
        kind="barh",
        ax=ax,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    fig.tight_layout()

    st.pyplot(fig)

    st.subheader(
        "Save Trained Model"
    )

    model_package = {
        "model": rf["model"],
        "encoder": rf["encoder"],
        "feature_columns": rf[
            "X"
        ].columns.tolist(),
        "status_order": STATUS_ORDER,
    }

    import io

    model_buffer = io.BytesIO()

    joblib.dump(
        model_package,
        model_buffer,
    )

    st.download_button(
        "📥 Download Random Forest Model",
        model_buffer.getvalue(),
        "iucn_random_forest_model.pkl",
        "application/octet-stream",
    )


# ============================================================
# SECTION 5 — INTERACTIVE PREDICTION
# ============================================================

elif page == "🔮 IUCN Prediction":

    st.title(
        "🔮 Interactive IUCN Status Prediction"
    )

    if markov_file is None:

        st.warning(
            "Please upload Markov_Random_Forest (1).xlsx "
            "using the sidebar."
        )

        st.stop()

    try:

        species = load_status_change_data(
            markov_file.getvalue()
        )

    except Exception as e:

        st.error(
            f"Could not read the Excel file: {e}"
        )

        st.stop()

    try:

        rf = train_random_forest(
            species
        )

    except Exception as e:

        st.error(
            f"Could not train the Random Forest model: {e}"
        )

        st.stop()

    X = rf["X"]

    encoder = rf["encoder"]

    classifier = rf["model"]

    taxonomy_columns = [
        column
        for column in X.columns
        if column.startswith(
            "Taxonomy_"
        )
    ]

    taxonomy_names = [
        column.replace(
            "Taxonomy_",
            "",
        )
        for column in taxonomy_columns
    ]

    if not taxonomy_names:

        st.error(
            "No taxonomy columns were found in the dataset."
        )

        st.stop()

    st.markdown(
        """
        Enter the species' taxonomy/group and its 2021 IUCN
        status. The model then predicts the 2022 IUCN status
        and displays the probability assigned to each class.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        user_taxonomy = st.selectbox(
            "Taxonomy / Group",
            taxonomy_names,
        )

    with col2:

        user_status = st.selectbox(
            "IUCN Status in 2021",
            STATUS_ORDER,
            format_func=lambda x:
            f"{x} — {STATUS_NAMES[x]}",
        )

    st.markdown("---")

    if st.button(
        "🔮 Predict IUCN Status",
        type="primary",
        use_container_width=True,
    ):

        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=X.columns,
        )

        encoded_status = encoder.transform(
            [[user_status]]
        )[0][0]

        input_data[
            "IUCN Red List (2021)"
        ] = encoded_status

        taxonomy_column = (
            "Taxonomy_"
            + user_taxonomy
        )

        taxonomy_found = (
            taxonomy_column
            in input_data.columns
        )

        if taxonomy_found:

            input_data[
                taxonomy_column
            ] = 1

        prediction = classifier.predict(
            input_data
        )[0]

        probabilities = (
            classifier.predict_proba(
                input_data
            )[0]
        )

        max_probability = (
            probabilities.max()
        )

        st.success(
            f"Predicted 2022 IUCN Status: "
            f"**{prediction} — "
            f"{STATUS_NAMES.get(prediction, prediction)}**"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Taxonomy",
            user_taxonomy,
        )

        c2.metric(
            "2021 Status",
            user_status,
        )

        c3.metric(
            "Prediction Probability",
            f"{max_probability:.2%}",
        )

        probability_table = pd.DataFrame(
            {
                "IUCN Status":
                    classifier.classes_,

                "Status Name": [
                    STATUS_NAMES.get(
                        x,
                        str(x),
                    )
                    for x
                    in classifier.classes_
                ],

                "Probability (%)":
                    probabilities * 100,
            }
        ).sort_values(
            "Probability (%)",
            ascending=False,
        )

        st.subheader(
            "Probability for Each IUCN Class"
        )

        st.dataframe(
            probability_table.style.format(
                {
                    "Probability (%)":
                        "{:.2f}%"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Prediction Probability Chart"
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.bar(
            probability_table[
                "IUCN Status"
            ],
            probability_table[
                "Probability (%)"
            ],
        )

        ax.set_xlabel(
            "IUCN Status"
        )

        ax.set_ylabel(
            "Probability (%)"
        )

        ax.set_title(
            "Random Forest IUCN Status Probabilities"
        )

        fig.tight_layout()

        st.pyplot(fig)

        prediction_result = pd.DataFrame(
            {
                "Taxonomy": [
                    user_taxonomy
                ],

                "IUCN_Status_2021": [
                    user_status
                ],

                "Predicted_IUCN_Status_2022": [
                    prediction
                ],

                "Prediction_Probability": [
                    max_probability
                ],

                "Taxonomy_Found": [
                    taxonomy_found
                ],
            }
        )

        result_csv = (
            prediction_result
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "📥 Download Prediction Result",
            result_csv,
            "species_prediction_result.csv",
            "text/csv",
        )

        st.info(
            "The prediction is a machine-learning classification "
            "based on the supplied dataset. It should not be treated "
            "as an authoritative ecological forecast."
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Biodiversity Loss Analysis • Statistical + ML Dashboard"
)
