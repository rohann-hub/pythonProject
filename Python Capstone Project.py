# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set style for plots
sns.set_style('whitegrid')

# Load the dataset (assuming you've downloaded it to your project folder)
try:
    df = pd.read_csv('titanic.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Please download the dataset from Kaggle and place it in your project folder.")
    # You can also download directly with:
    import urllib.request
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url) # Define df if FileNotFoundError occurs by downloading the dataset.

  # Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# ... (rest of your code)

# Handle missing values
# Fill missing age with median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode (most common value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin as too many missing values
df.drop('Cabin', axis=1, inplace=True)

# Verify missing values are handled
print("\nAfter cleaning - missing values per column:")
print(df.isnull().sum())

# Display basic info
print(df.info())

# Show statistical summary
print(df.describe())

# Show first few rows
print(df.head())

# Set up the figure with subplots
plt.figure(figsize=(15, 10))

# 1. Survival count (pie chart)
plt.subplot(2, 2, 1)
df['Survived'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
plt.title('Survival Rate')

# 2. Age distribution (histogram)
plt.subplot(2, 2, 2)
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution')

# 3. Survival by passenger class (bar chart)
plt.subplot(2, 2, 3)
sns.countplot(x='Pclass', hue='Survived', data=df, palette=['lightcoral', 'lightgreen'])
plt.title('Survival by Passenger Class')

# 4. Fare distribution (box plot)
plt.subplot(2, 2, 4)
sns.boxplot(x='Pclass', y='Fare', data=df, palette='pastel')
plt.title('Fare Distribution by Class')

plt.tight_layout()
plt.show()

# Additional visualizations
plt.figure(figsize=(12, 6))
sns.countplot(x='Sex', hue='Survived', data=df, palette=['lightcoral', 'lightgreen'])
plt.title('Survival by Gender')
plt.show()

# Age distribution by survival
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', palette=['lightcoral', 'lightgreen'])
plt.title('Age Distribution by Survival')
plt.show()

# Calculate some key statistics
survival_rate = df['Survived'].mean() * 100
women_survival = df[df['Sex'] == 'female']['Survived'].mean() * 100
men_survival = df[df['Sex'] == 'male']['Survived'].mean() * 100
first_class_survival = df[df['Pclass'] == 1]['Survived'].mean() * 100

print("\nKey Findings:")
print(f"Overall survival rate: {survival_rate:.1f}%")
print(f"Female survival rate: {women_survival:.1f}%")
print(f"Male survival rate: {men_survival:.1f}%")
print(f"First class survival rate: {first_class_survival:.1f}%")

# Correlation analysis
correlation = df.corr(numeric_only=True)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

print("\nConclusions:")
print("1. Females had significantly higher survival rates than males.")
print("2. Higher passenger classes (1st class) had better survival rates.")
print("3. Age shows moderate correlation with survival, with children having better chances.")
print("4. Fare price (correlated with class) was an important factor in survival.")

# Prepare data for machine learning
# Convert categorical variables to numerical
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Feature importance
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.coef_[0]
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance)

# Future improvements suggestion
print("\nFuture Scope:")
print("1. Try more advanced ML algorithms (Random Forest, Gradient Boosting).")
print("2. Perform more feature engineering (create family size from SibSp + Parch).")
print("3. Use more sophisticated techniques for handling missing values.")
print("4. Implement hyperparameter tuning for better model performance.")
