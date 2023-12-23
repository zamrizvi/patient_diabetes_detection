#Patient Diabetes Prediction Model
#Author: Zamir Rizvi

#This project focuses on developing a machine learning-based solution to predict if a patient has diabetes based on metrics.


#Import libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Import Model and Evaluators
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

#NOTE-Unfortunately, streamlit does not ignore block comments so I have to use '#'
# Function to load data and preprocess data. The features we will be using are: 
#'age', 'hypertension', 
#'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level'
#
@st.cache
def load__data_preprocess():
    columns_to_keep = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
    data = pd.read_csv('../diabetes_prediction_dataset.csv')
    data = data[columns_to_keep]
    data.dropna()
    return data

# Load data
df = load__data_preprocess()

#Exploratory Data Evaluation (EDA):
#A crticial step in the data analysis process, where you examine the 
#dataset and discover patterns, spot anomalies, check hypotheses, 
#and check assumptions with the help of summary statistics and visualizations

#Key aspects of EDA:
#1. Understanding the data: familiarize yourself with the dataset's structure
#2. Visualizing Data: Using visual tools like plots and graphs, we can understand the distribution and relationship between variables. 
#3. Cleaning Data: Identify and handle missing values.
#4. Feature Engineering and Selection: Deriving new variables that may be more informative that the existing ones. Select the relevant features. 
#5. Preparing for Model: Understanding the data's characteristics allows us to choose the appropriate model (in this case a RandomForestClassifier)

#Scatter plot
#Displays individual data points on a 2-d graph:
#-used for exporing relationships or coorelations between two variables
#-identify patterns, cluster, or outliers in data
#-visualize the distribution of data points across two dimensions

def scatter_plot(data, x_column, y_column, color='red'):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(data[x_column], data[y_column], c=color)
    ax.set_title(f"{y_column} vs {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.grid(True)
    plt.close(fig)
    return fig

#Heatmap

#Visual representation of data where values are depicted by color
#specifically designed to show the coorelation matrix
#Uses:
#-identifying relatiolnships between different variables in a dataset
#-quickly spotting highly correlated variables
#-how variables related to each other

def correlation_heatmap(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.close(fig)
    return fig

#Histogram

#Graphical representation of the distribution of numerical data
#shows the frequency of the data intervals(bins)
#Uses: 
#-understanding the distribution of a dataset
#-detecting outliers and anomalies in data
#-analyzing the central tendency and spread of data

def histogram(data, column):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data[column], bins=30, edgecolor='black')
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    plt.close(fig)
    return fig

# Model Training Function:
#This function takes the DataFrame and splits the data into a training set and test set.
#The diabetes column is dropped and used for testing.
#Transforms numerical data using StandardScaler to standardize the features.
def train_model(data):
    # Split data into features and labels
    X = data.drop('diabetes', axis=1)  
    y = data['diabetes']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Training the RandomForest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    

    return clf, scaler, X_test, y_test

# Train the model on the DataFrame we loaded the csv in
clf, scaler, X_test, y_test = train_model(df)

#Make a prediction based on test data. 
y_pred = clf.predict(X_test)

#Model Evaluation
#After the model makes a prediction, we will evaluate it using metrics. 
#This function creates a classification report out of the predictions and the actual test data.
#Accuracy: Proportion of correct predictions over the total predictions.
#Precision: Measures the accuracy of positive predictions. 
#Recall: The ability of the model to find all relevant cases.
#F1 Score: Mean of Precision adn Recall.
#ROC-AUC: Area Under the Receiver Operating Characteristic Curve. Evaluates classifier output quality.

#Classification Report: Provides a breakdown of Precision, Recall, F1 Score, and Support for each class.
#Confusion Matrix: A table that compares the actual versus predicted values. 
def classification_report_df(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    return df[:-3]

#Create a classification report and confusion matrix 
class_report_df = classification_report_df(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

def metrics_evaluation(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = metrics_evaluation(y_test, y_pred)

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}
# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

def display_metrics(metrics):
    metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [accuracy, precision, recall, f1]})
    return metrics_df


#Model Prediction: Now using the model, we can make predictions based on criteria. 
#It is important to make sure transform the input data the same way the training data was encoded.

# Predict Function: takes in the model and features inputted by the user
def predict_diabetes(clf, input_data):
    # Make prediction
    prediction = clf.predict(input_data)
    return prediction

#Transforms data used the same scaler as the training data
def preprocess_and_scale_input(input_data, scaler):
    st.write(input_data.head())
    # Standardize numerical features
    numerical_cols = input_data.select_dtypes(include=['int64', 'float64']).columns
    input_data[numerical_cols] = scaler.transform(input_data)
    return input_data


#Confusion matrix
#Visual representation of the performance of a classification model
#shows the actual vs predicted classifications
#uses: 
#-assessing the accuracy of a classification model
#-identifying the types of errors made(eg false positive, false negatives, etc)
#-useful in training the model and understanding and interpreting complex data

def plot_confusion_matrix(conf_matrix, classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    plt.close(fig)
    return fig

#Class report
#can display a classification report that includes precision, recall, f1score

def plot_classification_report(df):
    fig , ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.astype(float), annot=True, cmap='Blues', fmt='.2f', linewidths=.5)
    ax.set_title('Classification Report')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # To keep class names horizontal
    plt.close(fig)
    return fig


#Feature Importance Visualization
def plot_feature_importance(clf):
    feat_importances = pd.Series(clf.feature_importances_, index=X_test.columns)
    fig, ax = plt.subplots()
    feat_importances.nlargest(5).plot(kind='barh', ax=ax)  # Adjust number of features as needed
    plt.close(fig)
    return fig

# Advanced Model Evaluation - ROC Curve
#Plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.close(fig)
    return fig

#Sidebar for navigation 
page = st.sidebar.selectbox("Choose a page", ["Home", "Predict Diabetes", "Model Evaluation"])

if page == "Home": 
    # Streamlit title
    st.title('Diabetes Prediction Dashboard')

    # Show dataset overview using Streamlit
    st.write("Dataset Overview:")
    st.write(df.head())
    st.write(df.describe())
    st.write(df.info())


     
    st.write("Scatter Plot")
    fig2 = scatter_plot(df, 'age', 'bmi')
    st.pyplot(fig2)
    st.write("Function Heatmap")
    fig3 = correlation_heatmap(df)
    st.pyplot(fig3)
    st.write("Histogram")
    fig4 = histogram(df, 'age')
    st.pyplot(fig4)
    

elif page == "Predict Diabetes":
    st.title("Diabetes Prediction")

    #Input collecion (e.g. slider, text inputs)
    with st.form(key='prediction_form'):
        #Creating sliders for input features
        age = st.slider('Age', min_value=0, max_value=100, value=30)
        bmi = st.slider('BMI', min_value=0.0, max_value=100.0, value=25.0)
        HbA1c_level = st.slider('HbA1c', min_value=0.0, max_value=10.0, value=5.5)
        blood_glucose_level = st.slider('Blood Glucose', min_value=0, max_value=300, value=150)
        hypertension = st.select_slider('Hypertension', options=[1,0])
        heart_disease = st.select_slider('Heart Disease', options=[1,0])
        submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            # Predicting from user inputs
            input_data = {'age': [age], 'hypertension': [hypertension],
                          'heart_disease': [heart_disease],
                          'bmi': [bmi], 'HbA1c_level': [HbA1c_level], 'blood_glucose_level': [blood_glucose_level]}
            input_df = pd.DataFrame(input_data)

            input_df_scale = preprocess_and_scale_input(input_df, scaler)
            prediction = predict_diabetes(clf, input_df_scale)

            if prediction[0] == 1:
                st.write("The model predicts: Diabetes")
            else:
                st.write("The model predicts: No Diabetes")

elif page == "Model Evaluation":
    st.title("Model Evaluation")

    col1, col2, col3 = st.columns(3)
    with col3:
        st.write("Confusion Matrix")
        st.dataframe(conf_matrix)
    with col2:
        st.write("Classification Report")
        st.dataframe(class_report_df)
    with col1:
        st.write("Metrics")
        st.write(display_metrics(metrics))

    st.write("Feature Importance in RandomForest Model")
    fig7= plot_feature_importance(clf)
    st.pyplot(fig7)
    st.write("ROC Curve")
    fig8 = plot_roc_curve(fpr, tpr, roc_auc)
    st.pyplot(fig8)

#elif page == ""
