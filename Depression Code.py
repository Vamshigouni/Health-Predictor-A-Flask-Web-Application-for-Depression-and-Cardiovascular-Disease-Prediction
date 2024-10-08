
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE


# # Data Loading

# In[21]:


# Loading the Dataset
data=pd.read_csv("b_depressed.csv")


# In[22]:


data


# # Data Cleaning

# In[23]:


data.isnull().sum()


# In[24]:


# Fill Missing Values with Mean
data['no_lasting_investmen'].fillna(data['no_lasting_investmen'].mean(), inplace=True)


# In[25]:


data.isnull().sum()


# # EDA

# In[26]:


data.info()


# In[27]:


# Descriptive statistics of the dataset
print(data.describe())


# In[28]:


# Histograms for numeric columns
numeric_cols = ['Age', 'Number_children', 'total_members', 'gained_asset', 'durable_asset',
                'save_asset', 'living_expenses', 'other_expenses', 'incoming_salary',
                'incoming_own_farm', 'incoming_business', 'incoming_no_business', 'incoming_agricultural',
                'farm_expenses', 'lasting_investment', 'no_lasting_investmen']

for col in numeric_cols:
    sns.histplot(data, x=col, kde=True, hue='depressed')
    plt.title(f'Distribution of {col}')
    plt.show()


# In[29]:


# Countplot for categorical columns
categorical_cols = ['sex', 'Married', 'education_level', 'labor_primary']

for col in categorical_cols:
    sns.countplot(data=data, x=col, hue='depressed')
    plt.title(f'{col} vs. depression')
    plt.xticks(rotation=45)
    plt.show()


# # Data Preprocessing

# In[30]:


# Split the data into features (X) and the target variable (y)
X = data.drop(['depressed', 'Survey_id', 'Ville_id'], axis=1)
y = data['depressed']


# In[31]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Model Training

# In[33]:


# Create a list of classifiers
classifiers = [
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("Naive Bayes", GaussianNB()),
    ("SVM", SVC(random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("KNN", KNeighborsClassifier())
]

# Initialize variables to store results
results = {"Classifier": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

# Iterate through each classifier and evaluate its performance
for clf_name, classifier in classifiers:
    classifier.fit(X_train, y_train)  # Train the classifier
    y_pred = classifier.predict(X_test)  # Make predictions

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Store results
    results["Classifier"].append(clf_name)
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1 Score"].append(f1)

# Create a DataFrame to display results
import pandas as pd
results_df = pd.DataFrame(results)

# Display the results
print(results_df)


# # Hyper Parameter Tuning For KNN and Random Forest

# In[34]:


# Create a pipeline with StandardScaler and K-Nearest Neighbors
scaler = StandardScaler()
knn_model = KNeighborsClassifier()
pipe = Pipeline([
    ('scaler', scaler),
    ('knn', knn_model)
])

# Define the range of k values to tune
k_values = list(range(1, 30))

# Define hyperparameters to tune
param_grid = {
    'knn__n_neighbors': k_values
}

# Use GridSearchCV for hyperparameter tuning
full_cv_classifier = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='accuracy')
full_cv_classifier.fit(X_train, y_train)

# Get the best hyperparameters
best_params = full_cv_classifier.best_params_
print("Best Hyperparameters:", best_params)

# Make predictions
y_pred = full_cv_classifier.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

# Display classification report
print(classification_report(y_test, y_pred))

# Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)


# In[35]:


# Create a pipeline with StandardScaler and Random Forest
rf_model = RandomForestClassifier(random_state=42)
pipe_rf = Pipeline([
    ('rf', rf_model)
])

# Define hyperparameters to tune
param_grid_rf = {
    'rf__n_estimators': [100, 150],
    'rf__max_depth': [20, 30],
    'rf__min_samples_leaf': [1, 2]
}

# Use GridSearchCV for hyperparameter tuning
full_cv_classifier_rf = GridSearchCV(estimator=pipe_rf, param_grid=param_grid_rf, cv=10, scoring='accuracy')
full_cv_classifier_rf.fit(X_train, y_train)

# Get the best hyperparameters
best_params_rf = full_cv_classifier_rf.best_params_
print("Best Hyperparameters for Random Forest:", best_params_rf)

# Make predictions
y_pred_rf = full_cv_classifier_rf.predict(X_test)

# Evaluate the model
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", cbar=False)

# Display classification report
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Calculate and display accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print('Accuracy for Random Forest: ', accuracy_rf)


# # Model Ensembling (Hypertuned KNN + Hyper Tuned Random Forest)

# In[36]:


from sklearn.ensemble import VotingClassifier

# Use the best hyperparameters for KNN and Random Forest
best_params_knn = full_cv_classifier.best_params_
best_params_rf = full_cv_classifier_rf.best_params_

# Create individual models with the best hyperparameters
knn_model = KNeighborsClassifier(n_neighbors=best_params_knn['knn__n_neighbors'])
rf_model = RandomForestClassifier(
    n_estimators=best_params_rf['rf__n_estimators'],
    max_depth=best_params_rf['rf__max_depth'],
    min_samples_leaf=best_params_rf['rf__min_samples_leaf'],
    random_state=42
)

# Create a Voting Classifier ensemble
ensemble_model = VotingClassifier(estimators=[
    ('knn', knn_model),
    ('rf', rf_model)
], voting='hard')

# Fit the ensemble model on the training data
ensemble_model.fit(X_train, y_train)

# Make predictions
y_pred_ensemble = ensemble_model.predict(X_test)

# Evaluate the ensemble model
cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
sns.heatmap(cm_ensemble, annot=True, fmt="d", cmap="Blues", cbar=False)

# Display classification report
print("Classification Report for Ensemble (KNN + Random Forest):")
print(classification_report(y_test, y_pred_ensemble))

# Calculate and display accuracy
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print('Accuracy for Ensemble (KNN + Random Forest): ', accuracy_ensemble)


# # Over Sampling Minority Class Using SMOTE

# In[37]:


# Use SMOTE to oversample the minority class in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create a list of classifiers
classifiers = [
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("Naive Bayes", GaussianNB()),
    ("SVM", SVC(random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("KNN", KNeighborsClassifier())
]

# Initialize variables to store results
results = {"Classifier": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

# Iterate through each classifier and evaluate its performance
for clf_name, classifier in classifiers:
    classifier.fit(X_train_resampled, y_train_resampled)  # Train the classifier on the resampled data
    y_pred = classifier.predict(X_test)  # Make predictions

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Store results
    results["Classifier"].append(clf_name)
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1 Score"].append(f1)

# Create a DataFrame to display results
results_df = pd.DataFrame(results)

# Display the results
print(results_df)


# In[38]:


# Create a pipeline with StandardScaler and K-Nearest Neighbors
scaler = StandardScaler()
knn_model = KNeighborsClassifier()
pipe = Pipeline([
    ('scaler', scaler),
    ('knn', knn_model)
])

# Define the range of k values to tune
k_values = list(range(1, 30))

# Define hyperparameters to tune
param_grid = {
    'knn__n_neighbors': k_values
}

# Use GridSearchCV for hyperparameter tuning
full_cv_classifier = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='accuracy')
full_cv_classifier.fit(X_train_resampled, y_train_resampled)

# Get the best hyperparameters
best_params = full_cv_classifier.best_params_
print("Best Hyperparameters:", best_params)

# Make predictions
y_pred = full_cv_classifier.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

# Display classification report
print(classification_report(y_test, y_pred))

# Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)


# In[39]:


# Create a pipeline with StandardScaler and Random Forest
rf_model = RandomForestClassifier(random_state=42)
pipe_rf = Pipeline([
    ('rf', rf_model)
])

# Define hyperparameters to tune
param_grid_rf = {
    'rf__n_estimators': [100, 150],
    'rf__max_depth': [20, 30],
    'rf__min_samples_leaf': [1, 2]
}

# Use GridSearchCV for hyperparameter tuning
full_cv_classifier_rf = GridSearchCV(estimator=pipe_rf, param_grid=param_grid_rf, cv=10, scoring='accuracy')
full_cv_classifier_rf.fit(X_train_resampled, y_train_resampled)

# Get the best hyperparameters
best_params_rf = full_cv_classifier_rf.best_params_
print("Best Hyperparameters for Random Forest:", best_params_rf)

# Make predictions
y_pred_rf = full_cv_classifier_rf.predict(X_test)

# Evaluate the model
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", cbar=False)

# Display classification report
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Calculate and display accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print('Accuracy for Random Forest: ', accuracy_rf)


# In[40]:


from sklearn.ensemble import VotingClassifier

# Use the best hyperparameters for KNN and Random Forest
best_params_knn = full_cv_classifier.best_params_
best_params_rf = full_cv_classifier_rf.best_params_

# Create individual models with the best hyperparameters
knn_model = KNeighborsClassifier(n_neighbors=best_params_knn['knn__n_neighbors'])
rf_model = RandomForestClassifier(
    n_estimators=best_params_rf['rf__n_estimators'],
    max_depth=best_params_rf['rf__max_depth'],
    min_samples_leaf=best_params_rf['rf__min_samples_leaf'],
    random_state=42
)

# Create a Voting Classifier ensemble
ensemble_model = VotingClassifier(estimators=[
    ('knn', knn_model),
    ('rf', rf_model)
], voting='hard')

# Fit the ensemble model on the training data
ensemble_model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred_ensemble = ensemble_model.predict(X_test)

# Evaluate the ensemble model
cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
sns.heatmap(cm_ensemble, annot=True, fmt="d", cmap="Blues", cbar=False)

# Display classification report
print("Classification Report for Ensemble (KNN + Random Forest):")
print(classification_report(y_test, y_pred_ensemble))

# Calculate and display accuracy
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print('Accuracy for Ensemble (KNN + Random Forest): ', accuracy_ensemble)

with open("depressionensemble_model.pkl", "wb") as ensemble_model_file:
    pickle.dump(ensemble_model, ensemble_model_file)

with open("standard_scalerdep.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Best Model is Over sampled Ensemble Model for KNN and Random Forest
