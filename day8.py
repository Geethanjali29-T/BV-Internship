import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
# Create DataFrame
data = {
    "PassengerId": [1, 2, 3, 4, 5,6,7,8,9,10],
    "Pclass": [3,1,3,1,3,3,1,3,2,3],
    "Sex": ["male", "female", "female", "female", "male","male","male","male","female","female"],
    "Age": [22, 38, 26, 35, 35,28,54,2,27,14],
    "Fare": [7.25,71.28,7.92,53.1,8.05,8.46,51.86,21.07,11.13,7.85],
    "Year":[1912,1912,1912,1912,1912,1912,1912,1912,1912,1912],
    "Survived": [0, 1, 1, 1,0,0,0,1,1,1]
}
df = pd.DataFrame(data)
# Gini Index function
def gini_index(groups, classes):
    total_samples = sum([len(group) for group in groups])
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = sum(group['Survived'] == class_val) / size
            score += proportion ** 2
        gini += (1 - score) * (size / total_samples)
    return gini
# Entropy function
def entropy(group):
    size = len(group)
    if size == 0:
        return 0
    ent = 0
    for val in [0, 1]:
        p = sum(group['Survived'] == val) / size
        if p > 0:
            ent -= p * math.log2(p)
    return ent
# Information Gain function
def information_gain(df, attribute):
    total_entropy = entropy(df)
    values = df[attribute].unique()
    weighted_entropy = 0
    for value in values:
        subset = df[df[attribute] == value]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset)
    info_gain = total_entropy - weighted_entropy
    return info_gain
# Split the data by 'Sex'
male_group = df[df['Sex'] == 'male']
female_group = df[df['Sex'] == 'female']
groups = [male_group, female_group]
classes = [0, 1]
# Gini Index for split on 'Sex'
gini = gini_index(groups, classes)
# Information Gain for split on 'Sex'
info_gain = information_gain(df, 'Sex')
print("Gini Index for 'Sex':", round(gini, 4))
print("Information Gain for 'Sex':", round(info_gain,4))
# Encode categorical column 'Sex'
df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)
# Split features and target
features = ['Sex', 'Age', 'Pclass']
target = 'Survived'
X = df[features]
y = df[target]
# Initialize and train the decision tree classifier
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X, y)
# Display decision rules in text format
print("\nDecision Tree Rules:\n")
tree_rules = export_text(model, feature_names=features)
print(tree_rules)
# Plot the decision tree
fig, ax = plt.subplots(figsize=(8, 6))
plot_tree(model, feature_names=features, class_names=['Not Survived', 'Survived'], filled=True, ax=ax)
plt.title("Decision Tree for Titanic Mini Dataset")
plt.tight_layout()
plt.show()
