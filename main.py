import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

#Preferences 
# Suppress warnings
warnings.filterwarnings('ignore')

# Set Seaborn style
sns.set_theme(style='whitegrid', font='serif')

#(Helper Functions) Function to split the dataset into training and testing sets
def split_data(data, features, target, test_size=0.2, random_state=42):
    X = data[features]
    y = data[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Load the dataset and check for emptiness
data = pd.read_csv('C:/Users/mark jhoven/Downloads/Liquor_Brands.csv')
if data.empty:
    raise ValueError("The loaded dataset is empty. Please check the file path or contents.")

# Print some info to check data
print("Data shape:", data.shape)
print("Data columns:", data.columns)
print("Data types:", data.dtypes)

# Display basic statistics for the dataset
data.describe()

# Check for missing values
missing_values = data.isnull().sum()
missing_values[missing_values > 0]

# Visualize the distribution of 'STATUS' (example visualization)
sns.countplot(x='STATUS', data=data)
plt.title("Distribution of Liquor Brand Status")
plt.show()

#III. Data Cleaning and Preprocessing
# Define numerical and categorical features
numerical_features = ['CT-REGISTRATION-NUMBER', 'WHOLESALERS']
categorical_features = ['BRAND-NAME', 'STATUS', 'OUT-OF-STATE-SHIPPER', 'SUPERVISOR-CREDENTIAL-']
# Feature and target selection
features = numerical_features + categorical_features
target = 'STATUS'

#Handle non-numeric values in numerical features BEFORE applying ColumnTransformer
for col in numerical_features:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, invalid parsing will be set as NaN
    # Impute NaNs with 0 for numerical features
    data[col] = data[col].fillna(0)

# Convert mixed types in categorical columns to string
for col in categorical_features:
    data[col] = data[col].astype(str)

# Split data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dynamically determine numerical and categorical features based on X_train data types
numerical_features_updated = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features_updated = X_train.select_dtypes(include=['object']).columns.tolist()

# Create pipelines for numerical and categorical features (Updated to handle missing after numeric conversion)
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# Create ColumnTransformer using updated feature lists
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features_updated),
        ('cat', categorical_pipeline, categorical_features_updated)
    ])

# Split data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing before SMOTE (Important!)
X_train_preprocessed = preprocessor.fit_transform(X_train)  # Preprocess training data
X_test_preprocessed = preprocessor.transform(X_test)      # Preprocess testing data


# Check class distribution and apply SMOTE if needed
print("Class distribution:", data[target].value_counts())
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# --- Countplot for 'STATUS' by 'OUT-OF-STATE-SHIPPER' ---
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
sns.countplot(x='STATUS', hue='OUT-OF-STATE-SHIPPER', data=data)
plt.title("Distribution of Liquor Brand Status by Out-of-State Shipper", fontsize=16)
plt.xlabel("Status", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(title="Out-of-State Shipper", fontsize=10)
plt.tight_layout()
plt.show()

# --- Histogram for 'WHOLESALERS' ---
plt.figure(figsize=(8, 6))
sns.histplot(data['WHOLESALERS'], bins=20, kde=True)  # Added kde for smoother distribution
plt.title("Distribution of Wholesalers", fontsize=16)
plt.xlabel("Number of Wholesalers", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# --- Boxplot for 'CT-REGISTRATION-NUMBER' by 'STATUS' ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='STATUS', y='CT-REGISTRATION-NUMBER', data=data)
plt.title("Distribution of CT Registration Number by Status", fontsize=16)
plt.xlabel("Status", fontsize=12)
plt.ylabel("CT Registration Number", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

#comment : last second application to vscode, it was working in google colab