#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# In[2]:


# Load dataset
data = pd.read_csv('cardio_train.csv', delimiter=';')


# In[3]:


data.info()


# # Data cleaning

# In[4]:


# Check for missing values
print("Missing Values:\n", data.isnull().sum())


# In[5]:


# Check for duplicates
print("Duplicate Rows:", data.duplicated().sum())


# In[6]:


# Drop irrelevant columns (assuming 'id' is not useful for prediction)
data.drop('id', axis=1, inplace=True)


# In[7]:


# Columns to apply IQR-based outlier removal
columns_to_clean = ['ap_hi', 'ap_lo', 'height', 'weight']


# In[8]:


# Check for outliers using box plots
plt.figure(figsize=(15, 8))
for i, column in enumerate(columns_to_clean, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()


# In[9]:


# Function to remove outliers based on IQR
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]


# In[10]:


# Remove outliers based on IQR
for column in columns_to_clean:
    data = remove_outliers_iqr(data, column)

# Display boxplots after outlier removal
plt.figure(figsize=(15, 8))
for i, column in enumerate(columns_to_clean, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()


# In[11]:


data.info()


# # Exploratory Data Analysis (EDA)

# In[12]:


# Plot Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[13]:


# Plot Histograms
data.hist(figsize=(15, 12), bins=20)
plt.suptitle("Histograms of Numerical Features", y=0.95)
plt.show()


# # Data Pre Processing

# In[14]:


# Split the data into features (X) and target variable (y)
X = data.drop('cardio', axis=1)
y = data['cardio']


# In[15]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Model Training

# In[17]:


# Initialize classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(kernel='linear', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'k-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Voting Classifier': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(random_state=42)),
        ('svm', SVC(kernel='linear', probability=True, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('dt', DecisionTreeClassifier(random_state=42))
    ], voting='soft')
}


# In[18]:


# Train and evaluate each classifier
results = {'Classifier': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results['Classifier'].append(name)
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1 Score'].append(f1)

    # Display classification report and confusion matrix for each classifier
    print(f"\nClassifier: {name}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[19]:


# Plot the accuracy of each classifier
plt.figure(figsize=(10, 6))
sns.barplot(x='Classifier', y='Accuracy', data=pd.DataFrame(results))
plt.title("Accuracy of Different Classifiers")
plt.ylim(0, 1)
plt.show()

with open("cardioensemble_model.pkl", "wb") as ensemble_model_file:
    pickle.dump(cardioensemble_model, ensemble_model_file)

with open("cardioscaler.pkl", "wb") as scaler_file:
    pickle.dump(cardioscaler, scaler_file)