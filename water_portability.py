import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier


# Load Data
data = pd.read_csv('./water_potability.csv')
data['Potability'] = data['Potability'].astype('category')

# Main Container
with st.container():
    st.title("Water Potability Dataset Analysis")

    # Display Raw Data
    st.header("Raw Data")
    st.dataframe(data)
    
    st.write(f"Data Shape: {data.shape}")

    # Display Data Info
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    st.subheader("Data Information")
    st.text(info_str)

    # Display Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    st.subheader("Unique Values per Feature")
    st.write(data.nunique())

    # Display Potability Statistics
    st.subheader("Potability Statistics")
    st.write("### Portable for Drinking")
    st.write(data[data['Potability'] == 1].describe())
    
    st.write("### Not Portable for Drinking")
    st.write(data[data['Potability'] == 0].describe())

    # Plot Missing Values
    st.subheader("Missing Values per Feature")
    nans = data.isna().sum().sort_values(ascending=False).to_frame()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(nans, annot=True, fmt='d', cmap='vlag', cbar=True, ax=ax)
    ax.set_title('Missing Values Per Feature')
    ax.set_xlabel('Features')
    ax.set_ylabel('Missing Values')
    st.pyplot(fig)

    st.markdown("""
    ### Missing Data Counts

    The following counts represent the number of missing values for each feature in the dataset. 
    These features have missing data, which we will need to address through imputation or other methods:
    """, unsafe_allow_html=True)

    missing_counts = {
        'Sulfate': data['Sulfate'].isnull().sum(),
        'pH': data['ph'].isnull().sum(),
        'Trihalomethanes': data['Trihalomethanes'].isnull().sum()
    }

    for feature, count in missing_counts.items():
        st.markdown(f"<p style='color: #ffffff;'>{feature}: {count} missing values</p>", unsafe_allow_html=True)

    # Impute Missing Values
    st.subheader("Data Imputation")
    
    # Impute 'ph'
    phMean_0 = data[data['Potability'] == 0]['ph'].mean(skipna=True)
    data.loc[(data['Potability'] == 0) & (data['ph'].isna()), 'ph'] = phMean_0
    phMean_1 = data[data['Potability'] == 1]['ph'].mean(skipna=True)
    data.loc[(data['Potability'] == 1) & (data['ph'].isna()), 'ph'] = phMean_1

    # Impute 'Sulfate'
    SulfateMean_0 = data[data['Potability'] == 0]['Sulfate'].mean(skipna=True)
    data.loc[(data['Potability'] == 0) & (data['Sulfate'].isna()), 'Sulfate'] = SulfateMean_0
    SulfateMean_1 = data[data['Potability'] == 1]['Sulfate'].mean(skipna=True)
    data.loc[(data['Potability'] == 1) & (data['Sulfate'].isna()), 'Sulfate'] = SulfateMean_1

    # Impute 'Trihalomethanes'
    TrihalomethanesMean_0 = data[data['Potability'] == 0]['Trihalomethanes'].mean(skipna=True)
    data.loc[(data['Potability'] == 0) & (data['Trihalomethanes'].isna()), 'Trihalomethanes'] = TrihalomethanesMean_0
    TrihalomethanesMean_1 = data[data['Potability'] == 1]['Trihalomethanes'].mean(skipna=True)
    data.loc[(data['Potability'] == 1) & (data['Trihalomethanes'].isna()), 'Trihalomethanes'] = TrihalomethanesMean_1

    st.write('#### Checking for Remaining Missing Data')
    st.write(data.isna().sum())

    # Display Processed Data
    st.subheader("Data After Preprocessing")
    st.dataframe(data)

    st.write(f"Data Shape: {data.shape}")

    # Display Data Info After Processing
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    st.subheader("Data Information After Processing")
    st.text(info_str)

    # Display Updated Descriptive Statistics
    st.subheader("Updated Descriptive Statistics")
    st.write(data.describe())

    st.subheader("Updated Unique Values per Feature")
    st.write(data.nunique())

    # Plot Correlations
    st.subheader("Feature Correlations")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), cmap="YlGnBu", square=True, annot=True, fmt='.2f', ax=ax)
    st.pyplot(fig)

# Prepare Data
X = data.drop('Potability', axis=1)
y = data['Potability']

# Initialize Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "LightGBM": lgb.LGBMClassifier(),
    "Neural Network (MLP)": MLPClassifier(max_iter=1000)
}

# Initialize variables to track the best model
best_model_name = None
best_avg_accuracy = 0
best_conf_matrix = None
best_class_report = None
best_y_pred = None

# DataFrame to store results
results_df = pd.DataFrame(columns=["Model", "Average Accuracy"])


kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and Evaluate Models
for name, model in models.items():
    # st.write(f"### Training {name} with {splits} splits")
    
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
        # Only compute report and confusion matrix if not XGBoost
        if name != "XGBoost":
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)

    # Average Accuracy
    avg_accuracy = np.mean(accuracies)
    # st.write(f"#### {name} Average Accuracy with {splits} splits: {avg_accuracy:.2f}")

    # Store the result in the DataFrame
    results_df = results_df._append({"Model": name, "Average Accuracy": avg_accuracy}, ignore_index=True)

    # Update the best model
    if avg_accuracy > best_avg_accuracy:
        best_avg_accuracy = avg_accuracy
        best_model_name = name
        best_conf_matrix = conf_matrix
        best_class_report = report
        best_y_pred = y_pred

# Display the results table
st.write("### Model Performance Summary")
st.dataframe(results_df)

# Display best model results
st.write(f"### Best Model: {best_model_name} with Average Accuracy: {best_avg_accuracy:.2f}")

if best_model_name:
    # Classification Report
    st.write(f"#### {best_model_name} Classification Report")
    st.text(classification_report(y_test, best_y_pred))
    
    # Confusion Matrix
    st.write(f"#### {best_model_name} Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(f'{best_model_name} Confusion Matrix')
    st.pyplot(fig)

# Visualize Model Performance
fig = px.bar(results_df, x='Model', y='Average Accuracy', title='Model Accuracy Comparison')
st.plotly_chart(fig)