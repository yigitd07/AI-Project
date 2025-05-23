#Implement a Single Hidden Layer Perceptron with Sigmoid activation function, you may
#choose any classification or regression task you prefer. Provide comments on the effect
#of changing the number of hidden neurons in your implementation.

import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Perceptron class
class SingleHiddenLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.1):
        self.lr = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weight initialization
        self.W1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        self.b2 = np.zeros((1, self.output_size))
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.a1)
        
        # Update weights and biases
        self.W2 += self.a1.T.dot(output_delta) * self.lr
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * self.lr
        self.W1 += X.T.dot(hidden_delta) * self.lr
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.lr
        
        # Mean squared error loss
        loss = np.mean(output_error ** 2)
        return loss
    
    def train(self, X, y, epochs=10000):
        for epoch in range(epochs+1):
            output = self.forward(X)
            loss = self.backward(X, y, output)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch} - Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

# XOR problem dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],[1],[1],[0]])  # XOR labels

hidden_neurons_list = [1, 2, 4, 8]  

for hidden_neurons in hidden_neurons_list:
    print(f"\nTraining with {hidden_neurons} hidden neurons:")
    model = SingleHiddenLayerPerceptron(input_size=2, hidden_size=hidden_neurons, learning_rate=0.5)
    model.train(X, y, epochs=10000)
    
    predictions = model.predict(X)
    print("Predictions:")
    for i, x in enumerate(X):
        print(f"Input: {x} -> Predicted: {predictions[i][0]} Actual: {y[i][0]}")
