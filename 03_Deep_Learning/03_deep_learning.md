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
    - ### Activation function
        - ReLU
            - for Hidden layer
            - range: 0 to infinite
            - Dying ReLU, Leaky ReLU

        - Sigmoid 
            - for binary & multi-label classfication
            - range: 0 to 1

        - Tanh
            - for older RNNs, hidden layer
            - range: -1 to 1

        - Softmax 
            - for multi-class classfication
            - range: Probabilities sum to 1

---

### Loss function & Back Propogation
- Loss function tell us about errors or difference between actual and predicated value.

    > Prediction = 0.05, Actual = 1, Loss = High

- Backpropogation traces backward and determines which weights contributed most to the error.

---
### Overfitting, Validation Set, and Early stopping
- Good training
> Train Loss      ↓
> Validation Loss ↓

- Overfitting
> Train Loss      ↓
> Validation Loss ↑

- Early stopping
> To avoid overfitting issue

---
#### Deep Learning flow
1. Forward Propagation - Make Prediction
2. Loss Function - Measure Error
3. Backpropagation - Find Which Weights Caused Error
4. Gradient Descent - Optimizer(Learning rate) - Update Weights
5. Repeat Thousands of Times - Epoch 1000 - Network Learns

-----
-----
## Convolutional Neural Network CNN
- CNN cares more about PATTERNS than exact pixel location.

- Feature Map & Sliding Window
- Pooling
- Flatten Layer

---
#### CNN work flow
- Image
- Filter (Kernel)
- Sliding Window
- Feature Map
- Detect Edges
- Detect Shapes
- Detect Objects

---
#### Use Cases
1. Face Recognition
2. Image Classification
3. Medical Imaging
4. Self Driving Car

----
### Pre-trained Model & Transfer Learning
- Transfer learning is take experienced model instead of train everything from zero

1. Feature extraction
    - Freeze some layers

2. Fine-Tuning
    - Train some deeper layers

