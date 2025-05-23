# Naive Bayes Classifier from Scratch for Weather Dataset

# Training data
data = [
    ['Sunny', 'Hot', 'No'],
    ['Sunny', 'Hot', 'No'],
    ['Overcast', 'Hot', 'Yes'],
    ['Rainy', 'Mild', 'Yes'],
    ['Rainy', 'Cool', 'Yes'],
    ['Rainy', 'Cool', 'No'],
    ['Overcast', 'Cool', 'Yes'],
    ['Sunny', 'Mild', 'No'],
    ['Sunny', 'Cool', 'Yes'],
    ['Rainy', 'Mild', 'Yes'],
    ['Sunny', 'Mild', 'Yes'],
    ['Overcast', 'Mild', 'Yes'],
    ['Overcast', 'Hot', 'Yes'],
    ['Rainy', 'Mild', 'No']
]

# Separate by class
def separate_by_class(dataset):
    separated = {}
    for row in dataset:
        class_value = row[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(row[:-1])
    return separated

# Count frequency of each feature value per class
def count_feature_values(separated):
    feature_counts = {}
    for class_value, rows in separated.items():
        feature_counts[class_value] = [{} for _ in range(len(rows[0]))]
        for row in rows:
            for i in range(len(row)):
                feature_value = row[i]
                if feature_value not in feature_counts[class_value][i]:
                    feature_counts[class_value][i][feature_value] = 0
                feature_counts[class_value][i][feature_value] += 1
    return feature_counts

# Calculate class probabilities
def calculate_class_probs(separated):
    total_rows = sum(len(rows) for rows in separated.values())
    class_probs = {}
    for class_value, rows in separated.items():
        class_probs[class_value] = len(rows) / total_rows
    return class_probs

# Predict
def predict(row):
    separated = separate_by_class(data)
    feature_counts = count_feature_values(separated)
    class_probs = calculate_class_probs(separated)

    probs = {}
    for class_value in separated:
        prob = class_probs[class_value]
        for i in range(len(row)):
            value = row[i]
            value_count = feature_counts[class_value][i].get(value, 0)
            total = sum(feature_counts[class_value][i].values())
            prob *= (value_count + 1) / (total + len(feature_counts[class_value][i]))  # Laplace smoothing
        probs[class_value] = prob

    return max(probs, key=probs.get)

# Get input from the user
user_outlook = input("Enter outlook (Sunny/Overcast/Rainy): ").capitalize()
user_temperature = input("Enter temperature (Hot/Mild/Cool): ").capitalize()

test_sample = [user_outlook, user_temperature]
prediction = predict(test_sample)
print(f"Prediction for {test_sample}: {prediction}")
