import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import requests
from CONSTANT import *
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

# Sidebar navigation
st.sidebar.title("What to view?")
menu = st.sidebar.radio("Go to", [
    "Data Overview",
    "Statistics for Numerical Variables",
    "Statistics for Categorical Variables",
    "Univariate Analysis",
    "Bivariate Analysis",
    "Inference"
])

# Sample data (replace with your actual data loading)
@st.cache_data
def load_data():
    response = requests.get("http://localhost:8000/data/")
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error("Failed to load data from API.")
        return pd.DataFrame()

df_raw = load_data()

reverse_maps = {
    'sex': {v: k for k, v in sex_mapping.items()},
    'cp': {v: k for k, v in chest_pain_type_mapping.items()},
    'fbs': {v: k for k, v in fasting_blood_sugar_mapping.items()},
    'restecg': {v: k for k, v in resting_ecg_mapping.items()},
    'exang': {v: k for k, v in exercise_induced_angina_mapping.items()},
    'slope': {v: k for k, v in st_slope_mapping.items()},
    'thal': {v: k for k, v in thal_mapping.items()},
    # 'target': {v: k for k, v in heart_disease_mapping.items()}
}

df = df_raw.copy()
for col, mapping in reverse_maps.items():
    df[col] = df_raw[col].map(mapping)

continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = df.columns.difference(continuous_features)

if menu == "Data Overview":
    st.title("Data Overview")
    data_description = {
        "Variable": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ],
        "Description": [
            "Age of the patient in years",
            "Gender (0 = male, 1 = female)",
            "Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic)",
            "Resting blood pressure (mm Hg)",
            "Serum cholesterol (mg/dl)",
            "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
            "Resting ECG (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)",
            "Maximum heart rate achieved",
            "Exercise-induced angina (1 = yes, 0 = no)",
            "ST depression induced by exercise",
            "Slope of ST segment (0: Upsloping, 1: Flat, 2: Downsloping)",
            "Number of major vessels (0–4) by fluoroscopy",
            "Thalassemia (0: Normal, 1: Fixed defect, 2: Reversible defect, 3: Not described)",
            "Heart disease status (0 = no disease, 1 = presence of disease)"
        ]
    }
    desc_df = pd.DataFrame(data_description)

    st.subheader("🔎 Variable Descriptions")
    st.dataframe(desc_df, use_container_width=True)

    st.subheader("📊 Dataset Summary")
    st.markdown(f"""
    - **Number of Entries**: The dataset contains **{df.shape[0]}** patient records.
    - **Number of Features**: There are **{df.shape[1]}** columns (features) describing patient attributes and test results.
    - **Missing Values**: There are **no missing values** — all columns have complete data.
    """)
 
elif menu == "Statistics for Numerical Variables":
    st.title("Statistics for Numerical Variables")

    st.subheader("Summary Table")
    st.dataframe(df.describe().T)

    st.subheader("Insights")
    st.markdown("""
    - **Age**: The average age of the patients is approximately **54.4 years**, with the youngest being **29** and the oldest **77 years**.
    - **Resting Blood Pressure (trestbps)**: The average is about **131.62 mm Hg**, ranging from **94** to **200 mm Hg**.
    - **Cholesterol (chol)**: The average cholesterol level is approximately **246.26 mg/dl**, with a minimum of **126** and a maximum of **564 mg/dl**.
    - **Maximum Heart Rate Achieved (thalach)**: The average is around **149.65**, with a range from **71** to **202**.
    - **Oldpeak**: The average ST depression is about **1.04**, with values ranging from **0** to **6.2**.
    """)
    
elif menu == "Statistics for Categorical Variables":
    st.title("Statistics for Categorical Variables")
    st.subheader("Categorical Variables Summary")
    st.dataframe(df.describe(include='object'))

    st.markdown("""
    - **Sex**: There are two unique values. The most frequent category is **Male**, occurring **207 times** out of 303 entries.
    - **Chest Pain Type (cp)**: Four types are present. The most common is **Typical angina**, occurring **143 times**.
    - **Fasting Blood Sugar (fbs)**: Two categories exist. The most frequent is **False (<= 120 mg/dl)**, appearing **258 times**.
    - **Resting ECG (restecg)**: Three unique ECG results are present. The most common is **Having ST-T wave abnormality**, appearing **152 times**.
    - **Exercise-Induced Angina (exang)**: Two values observed. The most frequent is **No**, seen in **204 cases**.
    - **Slope of ST Segment (slope)**: Three slope types are present. The most frequent is **Downsloping**, occurring **142 times**.
    - **Number of Major Vessels (ca)**: Five unique values exist. The most frequent is **0**, appearing **175 times**.
    - **Thalassemia (thal)**: Four results are available. The most common is **Reversible defect**, seen **166 times**.
    - **Heart Disease (target)**: Two categories represent disease presence. The most frequent is **Presence of disease**, occurring **165 times**.
    """)


elif menu == "Univariate Analysis":
    st.title("Univariate Analysis")

    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df_continuous = df[continuous_features]

    st.subheader("Numerical Variables")
    st.markdown("Analyze the distribution of numerical variables.")

    col = st.selectbox("Select a column for Univariate Analysis", continuous_features)

    fig, ax = plt.subplots(figsize=(10, 5))
    values, bin_edges = np.histogram(
        df_continuous[col], 
        bins='auto', 
        range=(np.floor(df_continuous[col].min()), np.ceil(df_continuous[col].max()))
    )
    sns.histplot(data=df_continuous, x=col, bins=bin_edges, kde=True, ax=ax,
                edgecolor='none', alpha=0.6, line_kws={'lw': 3})

    ax.set_xlabel(col.capitalize(), fontsize=15)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticks(np.round(bin_edges, 1))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='x', rotation=45)
    ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)

    # Summary stats box
    textstr = '\n'.join((
        r'$\mu=%.2f$' % df_continuous[col].mean(),
        r'$\sigma=%.2f$' % df_continuous[col].std()
    ))
    ax.text(0.75, 0.9, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', 
                                            edgecolor='white', pad=0.5), color='white')

    st.pyplot(fig)


    if col == 'age':
        st.markdown(
            "The distribution is somewhat uniform, but there's a peak around the late 50s. "
            "The mean age is approximately 54.37 years with a standard deviation of 9.08 years."
        )
    elif col == 'trestbps':
        st.markdown(
            "The resting blood pressure for most individuals is concentrated around 120-140 mm Hg, "
            "with a mean of approximately 131.62 mm Hg and a standard deviation of 17.54 mm Hg."
        )
    elif col == 'chol':
        st.markdown(
            "Most individuals have cholesterol levels between 200 and 300 mg/dl. "
            "The mean cholesterol level is around 246.26 mg/dl with a standard deviation of 51.83 mg/dl."
        )
    elif col == 'thalach':
        st.markdown(
            "The majority of the individuals achieve a heart rate between 140 and 170 bpm during a stress test. "
            "The mean heart rate achieved is approximately 149.65 bpm with a standard deviation of 22.91 bpm."
        )
    elif col == 'oldpeak':
        st.markdown(
            "Most of the values are concentrated towards 0, indicating that many individuals did not experience "
            "significant ST depression during exercise. The mean ST depression value is 1.04 with a standard deviation of 1.16."
        )


    st.subheader("Categorical Variables")
    st.markdown("Analyze the distribution of categorical variables.")

    df_categorical = df[categorical_features]
    col = st.selectbox("Select a column for Univariate Analysis", categorical_features)

    # Calculate value counts as percentages
    value_counts = df[col].value_counts(normalize=True).mul(100).sort_values()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot bar chart
    value_counts.plot(kind='barh', ax=ax, width=0.7)

    # Annotate bars with percentages
    for index, value in enumerate(value_counts):
        ax.text(value + 1, index, f'{value:.1f}%', fontsize=12, va='center')

    ax.set_xlim([0, 100])
    ax.set_xlabel('Frequency Percentage', fontsize=12)
    ax.set_title(f'{col} Distribution', fontsize=16)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(color='lightgrey', linestyle='--', linewidth=0.5, axis='x')

    st.pyplot(fig)
    if col == 'sex':
        st.markdown("The dataset is predominantly female, constituting a significant majority.")
    elif col == 'cp':
        st.markdown("The dataset shows varied chest pain types among patients. Type 0 (Typical angina) seems to be the most prevalent, but an exact distribution among the types can be inferred from the bar plot.")
    elif col == 'fbs':
        st.markdown("A significant majority of the patients have their fasting blood sugar level below 120 mg/dl, indicating that high blood sugar is not a common condition in this dataset.")
    elif col == 'restecg':
        st.markdown("The results show varied resting electrocardiographic outcomes, with certain types being more common than others. The exact distribution can be gauged from the plots.")
    elif col == 'exang':
        st.markdown("A majority of the patients do not experience exercise-induced angina, suggesting that it might not be a common symptom among the patients in this dataset.")
    elif col == 'slope':
        st.markdown("The dataset shows different slopes of the peak exercise ST segment. A specific type might be more common, and its distribution can be inferred from the bar plot.")
    elif col == 'ca':
        st.markdown("Most patients have fewer major vessels colored by fluoroscopy, with '0' being the most frequent.")
    elif col == 'thal':
        st.markdown("The dataset displays a variety of thalium stress test results. One particular type seems to be more prevalent, but the exact distribution can be seen in the plot.")
    elif col == 'target':
        st.markdown("The dataset is nearly balanced in terms of heart disease presence, with about 54.5% having it and 45.5% not having it.")

elif menu == "Bivariate Analysis":
    st.title("Bivariate Analysis")
    st.subheader("Bivariate Analysis - Numerical Features vs Target")
    st.markdown("Explore how numerical variables vary with the target.")
    selected_num_col = st.selectbox("Select a numerical feature", continuous_features)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 2]})

    # Define a color palette to use for both plots
    palette = ["#1f77b4", "#ff7f0e"]  # Ensures two colors are used

    # --- Left Plot: Barplot of Mean Value for Each Target Class ---
    bar_ax = sns.barplot(data=df, x="target", y=selected_num_col, ax=ax[0], palette=palette)
    bar_ax.set_title(f"Mean of {selected_num_col} by Target")
    bar_ax.set_xlabel("Target (0 = No Disease, 1 = Disease)")
    bar_ax.set_ylabel(f"Mean {selected_num_col}")

    # Add mean value labels
    for container in bar_ax.containers:
        bar_ax.bar_label(container, fmt='%.1f', label_type='edge')

    # --- Right Plot: KDE Distribution for Each Target Class ---
    sns.kdeplot(data=df[df["target"] == 0], x=selected_num_col, fill=True, linewidth=2, ax=ax[1], label='No Disease (0)', color=palette[0])
    sns.kdeplot(data=df[df["target"] == 1], x=selected_num_col, fill=True, linewidth=2, ax=ax[1], label='Disease (1)', color=palette[1])

    ax[1].set_title(f"Distribution of {selected_num_col} by Target")
    ax[1].set_xlabel(selected_num_col)
    ax[1].set_yticks([])
    ax[1].legend(title="Target")

    # Layout adjustments
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if selected_num_col == "age":
        st.markdown("""
        The distributions show a slight shift with patients having heart disease being a bit younger on average than those without. 
        The mean age for patients without heart disease is higher.
        """)
    elif selected_num_col == "trestbps":
        st.markdown("""
        Both categories display overlapping distributions in the KDE plot, with nearly identical mean values, indicating limited differentiating power for this feature.
        """)
    elif selected_num_col == "chol":
        st.markdown("""
        The distributions of cholesterol levels for both categories are quite close, but the mean cholesterol level for patients with heart disease is slightly lower.
        """)
    elif selected_num_col == "thalach":
        st.markdown("""
        There's a noticeable difference in distributions. Patients with heart disease tend to achieve a higher maximum heart rate during stress tests compared to those without.
        """)
    elif selected_num_col == "oldpeak":
        st.markdown("""
        The ST depression induced by exercise relative to rest is notably lower for patients with heart disease. Their distribution peaks near zero, whereas the non-disease category has a wider spread.
        """)

    st.subheader("Bivariate Analysis - Categorical Features vs Target")
    st.markdown("Explore how the target variable is distributed across different categorical feature values.")

    selected_cat_col = st.selectbox("Select a categorical feature", categorical_features.drop("target"))

    # Create a cross tabulation showing the proportion of target classes per category
    cross_tab = pd.crosstab(index=df[selected_cat_col], columns=df['target'])
    cross_tab_prop = pd.crosstab(index=df[selected_cat_col], columns=df['target'], normalize='index')

    # Define colormap
    cmp = ListedColormap(["#1f77b4", "#ff7f0e"])  # Light red and red


    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot stacked bar chart
    cross_tab_prop.plot(kind='bar', ax=ax, stacked=True, width=0.7, colormap=cmp, legend=False)

    # Add proportion + count labels
    for idx, val in enumerate(cross_tab.index.values):
        for (proportion, count, y_location) in zip(
            cross_tab_prop.loc[val],
            cross_tab.loc[val],
            cross_tab_prop.loc[val].cumsum()
        ):
            ax.text(x=idx, y=(y_location - proportion) + (proportion / 2) - 0.03,
                    s=f'{count}\n({np.round(proportion * 100, 1)}%)',
                    color='black', fontsize=9, fontweight='bold', ha='center')

    # Customize plot
    ax.set_ylabel("Proportion")
    ax.set_title(f"{selected_cat_col} vs Target", fontsize=16)
    ax.set_ylim([0, 1.12])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Target', loc='upper right', fontsize=10)

    # Display in Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if selected_cat_col == "ca":
        st.markdown("""
        The majority of patients with heart disease have fewer major vessels colored by fluoroscopy. As the number of colored vessels increases, the proportion of patients with heart disease tends to decrease. Especially, patients with 0 vessels colored have a higher proportion of heart disease presence.
        """)
    elif selected_cat_col == "cp":
        st.markdown("""
        Different types of chest pain present varied proportions of heart disease. Notably, types 1, 2, and 3 have a higher proportion of heart disease presence compared to type 0. This suggests the type of chest pain can be influential in predicting the disease.
        """)
    elif selected_cat_col == "exang":
        st.markdown("""
        Patients who did not experience exercise-induced angina (0) show a higher proportion of heart disease presence compared to those who did (1). This feature seems to have a significant impact on the target.
        """)
    elif selected_cat_col == "fbs":
        st.markdown("""
        The distribution between those with fasting blood sugar > 120 mg/dl (1) and those without (0) is relatively similar, suggesting fbs might have limited impact on heart disease prediction.
        """)
    elif selected_cat_col == "restecg":
        st.markdown("""
        Type 1 displays a higher proportion of heart disease presence, indicating that this feature might have some influence on the outcome.
        """)
    elif selected_cat_col == "sex":
        st.markdown("""
        Females (1) exhibit a lower proportion of heart disease presence compared to males (0). This indicates gender as an influential factor in predicting heart disease.
        """)
    elif selected_cat_col == "slope":
        st.markdown("""
        The slope type 2 has a notably higher proportion of heart disease presence, indicating its potential as a significant predictor.
        """)
    elif selected_cat_col == "thal":
        st.markdown("""
        The reversible defect category (2) has a higher proportion of heart disease presence compared to the other categories, emphasizing its importance in prediction.
        """)

elif menu == "Inference":
    st.title("Inference")

    age = st.slider("Age", min_value=1, max_value=120, value=50)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.radio("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure in mm Hg", min_value=80, max_value=200)
    chol = st.number_input("Serum cholesterol in mg/dl", min_value=100, max_value=400)
    fbs = st.radio("Fasting Blood Sugar:", options=["True (> 120 mg/dl)", "False (<= 120 mg/dl)"], index=1)
    restecg = st.radio("Resting ECG", ["Normal", "Having ST-T wave abnormality", "Showing probable or definite left ventricular hypertrophy"])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)
    exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.radio("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.slider("Number of Major Vessels (0–4)", min_value=0, max_value=4, value=0)
    thal = st.radio("Thalassemia", ["Normal", "Fixed defect", "Reversible defect", "Not described"])

    if st.button("Predict"):
        payload = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        try:
            response = requests.post("http://localhost:8000/predict/", json=payload)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']}")
            else:
                st.error(f"Prediction failed: {response.text}")

        except Exception as e:
            st.error(f"Error calling prediction API: {e}")