
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')

print("# of samples: ", df.shape[0])
print("# of attributes: ", df.shape[1])
df.info()

true_numerical = ["BMI", "MentHlth", "PhysHlth"]
ordinal = ["GenHlth", "Age", "Education", "Income"]
binary = [col for col in df.columns if col not in true_numerical + ordinal] #hmmm

five_num_summary = df[true_numerical].describe().loc[["min", "25%", "50%", "75%", "max"]]
print(five_num_summary)

sns.set(style="whitegrid")

# Histograms
df[true_numerical].hist(figsize=(10, 4), bins=30, edgecolor='black')
plt.tight_layout()
plt.show()

# true num vis
for col in true_numerical:
    plt.figure()
    sns.boxplot(x='HeartDiseaseorAttack', y=col, data=df, palette="Set2")
    plt.title(f'Boxplot of {col} by Heart Disease')
    plt.show()

#ordinal feature vis
for col in ordinal:
    plt.figure()
    sns.countplot(x=col, data=df, hue='HeartDiseaseorAttack', palette="Set1")
    plt.title(f'Distribution of {col} by Heart Disease')
    plt.show()

#binary vis
binary.remove("HeartDiseaseorAttack")
n_cols = 4
n_rows = -(-len(binary) // n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(binary):
    sns.countplot(x=col, data=df, hue='HeartDiseaseorAttack', ax=axes[i], palette="Set1")
    axes[i].set_title(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#heatmap
plt.figure(figsize=(14, 12))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

#check for missing data
missing_counts = df.isnull().sum()
print("Missing values per column:\n", missing_counts)


#noisy on BMI
plt.figure()
sns.boxplot(y=df["BMI"])
plt.title("Boxplot of BMI (Outliers Detection)")
plt.show()

#over 60 ignored
df["BMI_capped"] = df["BMI"].apply(lambda x: min(x, 60))

