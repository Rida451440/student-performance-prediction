import os
import joblib
import pandas as pd
import matplotlib

# True = graph file mein save hoga AUR screen par bhi dikhega
# False = sirf file mein save hoga
SHOW_GRAPHS = True
if not SHOW_GRAPHS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def finish():
    """Graph screen par dikhao (agar SHOW_GRAPHS=True) aur band karo."""
    if SHOW_GRAPHS:
        plt.show()
    plt.close()


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

os.makedirs("model", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

FEATURES = ["study_hours", "attendance", "previous_score", "assignments_completed"]
TARGET = "performance"

# ==========================================
# 1. Dataset load
# ==========================================
data = pd.read_csv("dataset/student_data.csv")
data = data.drop_duplicates().dropna()

X = data[FEATURES]
y = data[TARGET]

# ==========================================
# 2. Train / Test split (stratify = har class barabar)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training data:", len(X_train), "| Testing data:", len(X_test))

# ==========================================
# 3. Models + Hyperparameter tuning (GridSearchCV)
# ==========================================
candidates = {
    "Decision Tree": GridSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"max_depth": [3, 4, 5, 7, None], "min_samples_split": [2, 5, 10]},
        cv=5,
    ),
    "Random Forest": GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
        cv=5,
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    ),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}

results = []
fitted = {}

for name, clf in candidates.items():
    clf.fit(X_train, y_train)
    # GridSearchCV ho to best model nikaalo
    best = clf.best_estimator_ if isinstance(clf, GridSearchCV) else clf
    if isinstance(clf, GridSearchCV):
        print(f"\n{name} best params:", clf.best_params_)

    cv_scores = cross_val_score(best, X_train, y_train, cv=5)
    test_acc = accuracy_score(y_test, best.predict(X_test))

    results.append({
        "Model": name,
        "CV Mean Accuracy (%)": round(cv_scores.mean() * 100, 2),
        "CV Std (%)": round(cv_scores.std() * 100, 2),
        "Test Accuracy (%)": round(test_acc * 100, 2),
    })
    fitted[name] = best

# ==========================================
# 4. Model comparison
# ==========================================
results_df = pd.DataFrame(results).sort_values("CV Mean Accuracy (%)", ascending=False)
print("\n==============================")
print("Model Comparison")
print("==============================")
print(results_df.to_string(index=False))
results_df.to_csv("model/model_comparison.csv", index=False)

plt.figure(figsize=(9, 5))
plt.barh(results_df["Model"], results_df["Test Accuracy (%)"])
plt.xlabel("Test Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("graphs/model_comparison.png", dpi=150)
finish()

# ==========================================
# 5. Best model (CV accuracy ke hisab se) ki detail evaluation
# ==========================================
best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]
print("\nBest model:", best_name)

y_pred = best_model.predict(X_test)
classes = list(best_model.classes_)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=classes)
print("Confusion Matrix:")
print(cm)

ConfusionMatrixDisplay(cm, display_labels=classes).plot()
plt.title(f"{best_name} Confusion Matrix")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png", dpi=150)
finish()

# ==========================================
# 6. Feature importance (har model par kaam karta hai)
# ==========================================
perm = permutation_importance(
    best_model, X_test, y_test, n_repeats=20, random_state=42
)
importance = pd.Series(perm.importances_mean, index=FEATURES).clip(lower=0)
if importance.sum() > 0:
    importance = importance / importance.sum()  # percentage ki tarah normalize
importance = importance.sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

importance.sort_values().plot(kind="barh", title="Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("graphs/feature_importance.png", dpi=150)
finish()

# ==========================================
# 6b. Decision Tree Visualization
# (best model koi bhi ho, Decision Tree ka graph hamesha banega)
# ==========================================
dt_model = fitted["Decision Tree"]

plt.figure(figsize=(25, 12))
plot_tree(
    dt_model,
    feature_names=FEATURES,
    class_names=[str(c) for c in dt_model.classes_],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3,  # graph saaf rahe; poora tree dekhna ho to is line ko hata do
)
plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.savefig("graphs/decision_tree.png", dpi=150)
finish()

# ==========================================
# 7. Model save (app.py isi file ko load karegi)
# ==========================================
joblib.dump(
    {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURES,
        "classes": classes,
        "importance": importance.to_dict(),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
    },
    "model/best_model.pkl",
)
print("\nModel saved: model/best_model.pkl")
