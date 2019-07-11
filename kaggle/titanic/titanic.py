import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

# The classifiers we'll be experimenting with
lg = LogisticRegression(random_state = 0, solver='lbfgs')
rf = RandomForestClassifier(random_state=0, n_estimators=100)
gb = GradientBoostingClassifier(random_state=0, n_estimators=1000)
ab = AdaBoostClassifier(random_state=0, n_estimators=1000)
sv = LinearSVC(random_state=0, max_iter = 3000)

model = RandomForestClassifier(random_state=0, n_estimators=100)
all_models = [[lg, 'lg'], [rf, 'rf'], [gb, 'gb'], [ab, 'ab'], [sv, 'sv']] # List of models with identifiers
secondary_model = RandomForestClassifier(random_state=0, n_estimators=100) # Model to fit over the predictions of the primary models

# Add and remove features prior to feeding through the pipeline
def process_features(df):
    # Create new features from existing ones
    family_size = df.apply(lambda row: row.Parch + row.SibSp, axis=1)
    has_family = df.apply(lambda row: 0 if row.Parch + row.SibSp == 0 else 1, axis=1)
    has_cabin = df.apply(lambda row: 0 if pd.isnull(row.Cabin) else 1, axis=1)
    fare_class = df.apply(lambda row: 0 if row.Fare < 8 else 
                                    (1 if row.Fare < 15 else
                                    (2 if row.Fare < 32 else 3)), axis=1)
    title = df.apply(lambda row: get_title(row.Name), axis=1)
    new_features = pd.concat([family_size, has_family, has_cabin, title], axis=1)
    new_features.index = df.index
    new_features.columns = ['FamSize', 'HasFam', 'HasCabin', 'Title']
    #print(new_features.head())
    
    # Add engineered features and remove redundant ones
    return pd.concat([df, new_features], axis=1).drop(['Name', 'Ticket'], axis=1)

# Extract title from a passenger's name as listed in the dataset
def get_title(name):
    raw_title = re.search(' ([A-Za-z]+)\.', name).group(1)
    # Edge case handling
    if raw_title in ['Mr', 'Miss', 'Mrs', 'Master']: return raw_title
    elif raw_title in ['Mme', 'Ms', 'Mlle']: return 'Miss'
    else: return 'Rare'

# Preprocessing pipeline consists of missing data handling and categorical variable encoding
num_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

cat_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('one_hot', OneHotEncoder(handle_unknown = 'ignore'))])

ord_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'constant', fill_value='Missing')),
    ('labeler', OneHotEncoder(drop='first'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamSize', 'HasFam', 'HasCabin']),
    ('ord', ord_transformer, ['Sex']),
    ('cat', cat_transformer, ['Embarked', 'Title'])])

classifier = Pipeline(steps = [
    ('preprocessor', preprocessor),
     ('classifier', model)])

# Read data and prepare it for training
data_train = pd.read_csv('./train.csv', index_col='PassengerId')
data_test = pd.read_csv('./test.csv', index_col='PassengerId')

# Separate output from input in training data
X_train = data_train.drop('Survived', axis=1)
y_train = data_train['Survived']

X_test = data_test # Test data has no ground truth

#print(X_train.Fare.describe())
X_train = process_features(X_train)
X_test = process_features(X_test)
#print(X_train.head(20))

#X_train, X_valid, y_train, y_valid = train_test_split(X, y)

num_folds = 10
#cv_scores = cross_val_score(classifier, X_train, y_train, cv=num_folds, scoring='f1_macro')
#print(sum(cv_scores) / num_folds)

# Calculate feature importance
#print(X_train.head())
X_train_transform = pd.DataFrame(preprocessor.fit_transform(X=X_train), columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamSize', 'HasFam', 'HasCabin', 'Sex', 'C', 'Q', 'S', 'Master', 'Miss', 'Mr', 'Mrs', 'Rare'])
#X_train = pd.DataFrame(preprocessor.fit_transform(X=X_train))
#print(X_train.head())
#model.fit(X_train, y_train)
#sns.barplot(model.feature_importances_, X_train.columns)
#plt.show()
#plt.figure(figsize=(14, 12))
#sns.heatmap(X_train.corr(), cmap = plt.cm.RdBu, annot = True)
#plt.show()

# Create ensemble of classification models
model_predictions = pd.DataFrame()

for model, model_name in all_models:
    print('Training', model_name)
    y_train_preds = cross_val_predict(model, X_train_transform, y_train, cv=num_folds)
    model_predictions[model_name] = y_train_preds
print(model_predictions.head())
#sns.heatmap(model_predictions.corr(), cmap = plt.cm.RdBu, annot = True)
#plt.show()
#cv_scores = cross_val_score(secondary_model, model_predictions, y_train, cv=num_folds, scoring='f1_macro')
#print(sum(cv_scores) / num_folds) # .820, .815, .813

# Generate predictions
secondary_model.fit(model_predictions, y_train)
print(secondary_model.n_features_)

X_test_transform = pd.DataFrame(preprocessor.fit_transform(X=X_test), columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamSize', 'HasFam', 'HasCabin', 'Sex', 'C', 'Q', 'S', 'Master', 'Miss', 'Mr', 'Mrs', 'Rare'])

model_predictions = pd.DataFrame()
for model, model_name in all_models:
    print('Predicting with', model_name)
    model.fit(X_train_transform, y_train)
    y_test_preds = model.predict(X_test_transform)
    model_predictions[model_name] = y_test_preds
    
print(model_predictions.head())
print(secondary_model.n_features_)
y_test = pd.DataFrame(secondary_model.predict(model_predictions), columns=['Survived'])
y_test.index = X_test.index

#print(X_test.shape, y_test.shape)
X_test['Survived'] = y_test
print(X_test.head())

y_test.to_csv('./submission.csv')

#sns.barplot(x='Embarked', y='Survived', hue='Pclass', data=X_train)
#sns.swarmplot(x='Survived', y='Age', data=X_train)
#plt.show()