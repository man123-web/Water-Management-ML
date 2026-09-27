import pandas as pd, glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

csv_file = glob.glob("*.csv")[0]
print(f"Using: {csv_file}")
df = pd.read_csv(csv_file)
print(f"Shape: {df.shape}")

df = df.replace({'Banglore':'Bangalore','Mysuru':'Mysore','yes':'Yes','no':'No','YES':'Yes','NO':'No'})
target = 'Would you be interested in receiving training on water conservation methods?'
df = df[df[target].astype(str).str.strip().isin(['Yes','No','Maybe','yes'])]

X = df.drop(columns=[target, 'Timestamp'], errors='ignore')
y = df[target].astype(str).str.strip().replace({'yes':'Yes'})

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# FIXED: sparse_output=False so all models work
preprocess = ColumnTransformer([
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), X.columns.tolist())
])

models = {
    "1. Logistic Regression (Baseline)": LogisticRegression(max_iter=1000),
    "2. Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "3. XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
}

for name, clf in models.items():
    pipe = Pipeline([('pre', preprocess), ('clf', clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"\n{name} => Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

print("\nAll models done!")