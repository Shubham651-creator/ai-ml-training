# Deep Learning

### Why do neutral network need activation function?
- Activation function introduce non-linearity into the network. Without them, multiple layers collapse into a single linear transformation, making neutral network equivalent to linear regression.
- Activation function allow network to learn complex patterns, feature hierechies, and non-linear relationship with real data.
- single neuron = Linear Regression + Activation function

---
### Forward Propagation
> How does information travel from INPUT layer to HIDDEN layer to OUTPUT layer?

- Forward propagation calculates two main components
    - Linear transformation
    - Activation function

---

### Loss function & Back Propogation
- Loss function tell us about errors or difference between actual and predicated value.

    > Prediction = 0.05, Actual = 1, Loss = High

- Backpropogation traces backward and determines which weights contributed most to the error.

---
#### Deep Learning flow
1. Forward Propagation - Make Prediction
2. Loss Function - Measure Error
3. Backpropagation - Find Which Weights Caused Error
4. Gradient Descent - Optimizer - Update Weights
5. Repeat Thousands of Times - Epoch 1000 - Network Learns