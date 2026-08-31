import os
import json
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import joblib

def main():
    print("--- 1. Fetching Dataset from UCI Repository ---")
    autism_repo = fetch_ucirepo(id=426)
    X_raw = autism_repo.data.features.copy()
    y_raw = autism_repo.data.targets.copy()

    df = pd.concat([X_raw, y_raw], axis=1)
    print(f"Original dataset shape: {df.shape}")

    target_col = 'Class/ASD' if 'Class/ASD' in df.columns else df.columns[-1]
    df.rename(columns={target_col: 'class_asd', 'family_pdd': 'austim'}, inplace=True)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)
    
    # Clean age column
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age'].fillna(df['age'].median(), inplace=True)
    df = df[df['age'] <= 100]

    # Clean categorical missing values
    cat_cols_to_fill = ['ethnicity', 'country_of_res', 'relation']
    for col in cat_cols_to_fill:
        if col in df.columns:
            df[col].fillna('Unknown', inplace=True)
            df[col] = df[col].astype(str).str.title().str.strip()

    # Encode binary columns
    binary_map = {'f': 0, 'm': 1, 'no': 0, 'yes': 1, 'NO': 0, 'YES': 1, 'No': 0, 'Yes': 1}
    binary_cols = ['gender', 'jaundice', 'austim', 'used_app_before', 'class_asd']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: binary_map.get(str(x).strip().lower(), x) if pd.notnull(x) else x)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Ensure AQ-10 scores are integers
    aq_cols = [f'A{i}_Score' for i in range(1, 11)]
    for col in aq_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Recalculate result
    df['result'] = df[aq_cols].sum(axis=1)

    if 'age_desc' in df.columns:
        df.drop(columns=['age_desc'], inplace=True)

    print(f"Cleaned dataset shape: {df.shape}")
    print("Class distribution:\n", df['class_asd'].value_counts())

    # Save cleaned dataset
    os.makedirs('/Users/shuprov630/autism_screening_ml/data', exist_ok=True)
    cleaned_path = '/Users/shuprov630/autism_screening_ml/data/cleaned_autism_data.csv'
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset to {cleaned_path}")

    # Prepare X and y
    X = df.drop(columns=['class_asd'])
    y = df['class_asd']

    numeric_features = ['age', 'result'] + aq_cols
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # Preprocessing ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Train/Test Split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Extra Trees': ExtraTreesClassifier(random_state=42, n_estimators=100),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42)
    }

    results = []
    best_model_name = None
    best_score = -1
    best_pipeline = None

    print("\n--- 2. Benchmarking Machine Learning Models ---")
    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        if f1 > best_score:
            best_score = f1
            best_model_name = name
            best_pipeline = pipeline

    results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
    print("\n--- Model Benchmark Results ---")
    print(results_df.to_string(index=False))

    print(f"\n--- 3. Fine-tuning Best Model ({best_model_name}) ---")
    if 'Random Forest' in best_model_name or 'Extra Trees' in best_model_name or 'Gradient Boosting' in best_model_name:
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 10, None],
        }
        grid = GridSearchCV(best_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_
        print("Best Params:", grid.best_params_)

    final_pred = best_pipeline.predict(X_test)
    final_acc = accuracy_score(y_test, final_pred)
    final_f1 = f1_score(y_test, final_pred)
    print(f"Final Tuned {best_model_name} -> Accuracy: {final_acc:.4f}, F1-Score: {final_f1:.4f}")

    os.makedirs('/Users/shuprov630/autism_screening_ml/models', exist_ok=True)
    model_path = '/Users/shuprov630/autism_screening_ml/models/best_autism_model.joblib'
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best model pipeline to {model_path}")

    meta_path = '/Users/shuprov630/autism_screening_ml/models/model_metadata.json'
    metadata = {
        'model_name': best_model_name,
        'accuracy': float(final_acc),
        'f1_score': float(final_f1),
        'feature_names': X.columns.tolist(),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'benchmark': results
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Pipeline completed successfully!")

if __name__ == '__main__':
    main()
