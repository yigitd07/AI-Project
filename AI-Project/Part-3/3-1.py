# Simple Decision Tree Classifier - With and Without Splits

# Sample dataset: Features: [Weather: Sunny=0, Rainy=1], Target: Play(0-No, 1-Yes)
data = [
    {'weather': 0, 'play': 0},
    {'weather': 0, 'play': 0},
    {'weather': 1, 'play': 1},
    {'weather': 1, 'play': 1},
]

# Without split: always predict the majority class
def decision_tree_no_split(data):
    # Count target frequencies
    counts = {}
    for row in data:
        counts[row['play']] = counts.get(row['play'], 0) + 1
    # Majority class
    majority_class = max(counts, key=counts.get)
    return majority_class

# With split on 'weather' feature
def decision_tree_with_split(data, feature):
    # Separate data by feature value
    subset0 = [row for row in data if row[feature] == 0]
    subset1 = [row for row in data if row[feature] == 1]

    # Majority class in each subset
    def majority_class(subset):
        counts = {}
        for row in subset:
            counts[row['play']] = counts.get(row['play'], 0) + 1
        return max(counts, key=counts.get) if counts else None

    majority0 = majority_class(subset0)
    majority1 = majority_class(subset1)
    return {0: majority0, 1: majority1}

# Predictions
print("Predictions without split (always majority class):")
majority = decision_tree_no_split(data)
for row in data:
    print(f"Input: {row['weather']} -> Predicted: {majority}")

print("\nPredictions with split on 'weather':")
split_model = decision_tree_with_split(data, 'weather')
for row in data:
    pred = split_model[row['weather']]
    print(f"Input: {row['weather']} -> Predicted: {pred}")
