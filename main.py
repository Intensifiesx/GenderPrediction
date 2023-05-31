# %%
# ======Libraries======
import pandas as pd  # Data manipulation i.e., {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}} -> {1, 2, 3, 4, 5, 6, 7, 8, 9}
import numpy as np  # High level math operations
import seaborn as sns  # Statistical data visualization
import sklearn  # Machine learning library (Tier 1)
from sklearn.linear_model import LogisticRegression  # Decision making function
import matplotlib.pyplot as plt  # Plots data into a graph
import warnings
warnings.filterwarnings('ignore')
# Types of machine learning (increase in difficulty)
# 1) Supervised learning
# To predict the label of instances based on a learned relationship.
# Data is typically provided beforehand, hence the supervised part.
# Examples:
# Determine gender based off height and weight,
# Mark email as "spam" or "not spam" based off subject text
# 2) Unsupervised learning
# To find hidden patterns or intrinsic structures in data.
# Examples:
# Book classification; given a book, give it the top 3 genres it represents
# Customer segmentation; demographics such as race, gender, socioeconomics -> given
# that a customer buys [list of products]
# 3) Reinforcement learning
# To train an agent to interact with an environment and maximize its reward.
# Examples:
# A racecar bot on a 2D track,
# Autonomous driving; Tesla,
# Recommendation algorithms for social media sites
# 4) Neutral Networking Learning (Deep Learning)
# To adapt to changing input. Can be supervised, semi-supervised, or unsupervised.
# Examples:
# Image recognition,
# Text generation

# Purpose: Can we predict the gender of an
# individual based off their height and weight?

# Four parts to the machine learning process:
# 1: Gather and read data
# 2: Cleaning the data (80%)
# Optional: Interpret the data i.e., graph it, find the meaning
# 3: Creating a machine learning model (20%) (Training)
# 4: Evaluation (testing the model)

# %%
# ======PART 1======
df = pd.read_csv("heights_weights_genders.csv")  # Data frame

# %%
# ======PART 2======
df = sklearn.utils.shuffle(df)  # Shuffles the data frame
print("Random sample of 10:")
print(df.head(10))

# %%
# ======PART 3======
# Classify the inputs and the outputs
# Inputs: Height, weight
# Outputs: Gender

x = df[["Height", "Weight"]]  # Inputs
y = df["Gender"]  # Outputs

# Create a model using logistic regression
# Logistic regression: A type of model for supervised learning
# Examples of supervised learning models:
# Linear regression
# Logistic regression
# Decision tree
# Random forest
# Kmeans cluster
# ...

# Logistic regression is used for classification problems
# If the model predicts an output within the range of some
# certain numbers -> classifies it as [whatever]
# It is a sigmoid function with the linear regression
# algorithm inside of it

model = LogisticRegression()  # Create a logistic regression model
model = model.fit(x, y)  # Train the model

theta_zero = model.intercept_[0]
theta_one = model.coef_[0][0]
theta_two = model.coef_[0][1]

# Decision formula
# theta_zero + theta_one * height + theta_two * weight = 0
# weight = -(theta_zero + theta_one * height) / theta_two
height = np.linspace(50, 85, 100)  # Create a list of heights
weight = -(theta_zero + theta_one * height) / theta_two  # Equation for weights

# Creates a graph
gender_palette = {
    "Male": "#88CCFF",
    "Female": "#FF88DD",
}  # Color palette for graphing gender
sns.scatterplot(
    data=df, x="Height", y="Weight", hue="Gender", s=10, palette=gender_palette
)  # Graphs the data
plt.scatter(x=height, y=weight, s=10, color="black")
plt.show()

# %%
# ======PART 4======
while True:
    my_height = int(input("Your height (inches): "))
    my_weight = int(input("Your weight (pounds): "))
    predicted_probabilities = model.predict_proba([[my_height, my_weight]])
    chance_female = round(float(np.format_float_positional(predicted_probabilities[0][0] * 100, 5)), 4)
    chance_male = round(float(np.format_float_positional(predicted_probabilities[0][1] * 100, 5)), 4)
    predicted_gender = model.predict([[my_height, my_weight]])[0]
    print(f"Your height: \t\t{my_height} inches")
    print(f"Your weight: \t\t{my_weight} pounds")
    print(f"% Chance of Female: \t{chance_female:07.4f}%")
    print(f"% Chance of Male: \t{chance_male:07.4f}%")
    print(f"Predicted Gender: \t{predicted_gender}\n")
# %%
