import os
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

import seaborn as sns

os.makedirs("graphs", exist_ok=True)

FEATURES = ["study_hours", "attendance", "previous_score", "assignments_completed"]

# 1. Dataset load karna
data = pd.read_csv("dataset/student_data.csv")

# 2. Basic information
print("First 5 rows:")
print(data.head())

print("\nShape (rows, columns):", data.shape)

print("\nDataset Information:")
data.info()

print("\nStatistical Summary:")
print(data.describe())

# 3. Data quality checks
print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:", data.duplicated().sum())

# 4. Class distribution (imbalance check)
print("\nPerformance Count:")
print(data["performance"].value_counts())
print("\nPerformance Percentage:")
print((data["performance"].value_counts(normalize=True) * 100).round(2))

# 5. Performance countplot
plt.figure(figsize=(6, 4))
sns.countplot(x="performance", data=data)
plt.title("Student Performance Distribution")
plt.xlabel("Performance")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("graphs/performance_distribution.png", dpi=150)
finish()

# 6. Har feature ka histogram
data[FEATURES].hist(figsize=(10, 7), bins=15, edgecolor="black")
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("graphs/feature_histograms.png", dpi=150)
finish()

# 7. Performance ke hisab se boxplots
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feature in zip(axes.ravel(), FEATURES):
    sns.boxplot(x="performance", y=feature, data=data, ax=ax)
    ax.set_title(f"{feature} vs performance")
plt.tight_layout()
plt.savefig("graphs/feature_boxplots.png", dpi=150)
finish()

# 8. Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(data[FEATURES].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png", dpi=150)
finish()

print("\nAnalysis complete! Graphs 'graphs' folder mein save ho gaye.")
