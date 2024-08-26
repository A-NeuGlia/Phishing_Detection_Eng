# Library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from scipy.stats import yeojohnson
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import optuna
import joblib
import warnings
# We also filter warnings emitted by the 'seaborn_oldcore' library to improve the readability of our outputs.
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn._oldcore')

# Reading the data.
df = pd.read_csv('Phishing_Legitimate_full - Copy.csv')

# Exploring our data.
print(f"{df.dtypes}\n") 
# Data dimensions.
print(f"Dimensions: {df.shape[0]} x {df.shape[1]}\n")
# Classification of data types.
datatype_counts = df.dtypes.value_counts()
for dtype, count in datatype_counts.items():
    print(f"{dtype}: {count} columns")

# Dropping the 'id' column.
df = df.drop("id", axis=1)

# Checking for missing values.
null = df.isnull().sum()
for i in range(len(df.columns)):
    print(f"{df.columns[i]}: {null[i]} ({(null[i]/len(df))*100}%)")
total_cells = np.prod(df.shape)
total_missing = null.sum()
print(f"\nTotal missing values: {total_missing} ({(total_missing/total_cells) * 100}%)\n")


def is_continuous(series):
    return series.nunique() > 10

continuous_columns = [col for col in df.columns if is_continuous(df[col])]

sns.pairplot(df[continuous_columns], height= 2.5)
plt.show()


# Displaying correlations.
corr = df.corr()
cols = corr.nlargest(50, 'CLASS_LABEL')['CLASS_LABEL'].index
cm = np.corrcoef(df[cols].values.T)
sns.set_theme(font_scale=0.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 4}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# Checking the spread of our data.
ordinal_columns = [col for col in df.columns if col not in continuous_columns]
sns.set_theme(font_scale=1)
for col in ordinal_columns:
    plt.hist(df[col], bins=10)  
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'{col}')
    plt.show()
    def normal(mean, std, color="black"):
        x = np.linspace(mean-4*std, mean+4*std, 200)
        p = stats.norm.pdf(x, mean, std)
        z = plt.plot(x, p, color, linewidth=2)

for column_name in continuous_columns:
    fig1, ax1 = plt.subplots()
    sns.histplot(x=df[column_name], stat="density", ax=ax1)
    normal(df[column_name].mean(), df[column_name].std())
    
    fig2, ax2 = plt.subplots()
    stats.probplot(df[column_name], plot=ax2)
    
    plt.show()


# Correcting outliers.
df = df[df['NumDots'] < 20]
df = df[df['NumDash'] < 40]
plt.scatter(x=df['NumDots'], y=df['NumDash'])
plt.xlabel('NumDots')
plt.ylabel('NumDash')
plt.show()

for col in continuous_columns:
    df[col], _ = yeojohnson(df[col]) 

for col_name in continuous_columns:
    fig1, ax1 = plt.subplots()
    sns.histplot(x=df[col_name], stat="density", ax=ax1)
    normal(df[col_name].mean(), df[col_name].std())
    
    fig2, ax2 = plt.subplots()
    stats.probplot(df[col_name], plot=ax2)
    
    plt.show()

# Extracting the "Class_label" column.
col = df.columns.to_list()
# Removing it from the dataset.
col.remove('CLASS_LABEL')

# Defining a new dataset.
X = df[col]
y = df["CLASS_LABEL"]

# Splitting our sample.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Defining the parameters for our algorithm.
def objective(trial):
    n_estimators = trial.suggest_int('n_estimations', 10, 300)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Creating a study to maximize accuracy by running our function 100 times.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
# Extracting the best trial to automate parameter selection.
best_trial = study.best_trial
result = best_trial.params
model = RandomForestClassifier(n_estimators=result['n_estimations'], max_depth=result['max_depth'], min_samples_split=result['min_samples_split'], min_samples_leaf=result['min_samples_leaf'], random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
