# Handwritten-Digit-Recognition-using-Neural-Networks-from-Scratch

## **Introduction**
<div style="text-align: justify">In this project, the Neural network is implemented from scratch without using any external machine learning libraries. We will create and train a simple neural network in python language. Neural networks form the base of deep learning. The algorithm is inspired by the human brain. Neural networks mimic the behaviour of the human brain to solve complex data-driven problems. Neural networks take input data, train themselves to recognize patterns within the data, and hence predict the output for newly provided data.
<br>


We will use the popular MNIST data set of handwritten digits to train and then test the network’s performance when introduced to real world problems. At each step results are visualized using several plots and graphs using matplotlib library. Different activation functions are used, and the results are compared to obtain the best activation function for the given dataset.</div>

## **Flow Chart**
<p align="center">
 <img src="https://user-images.githubusercontent.com/50751235/129025939-83eb5fb6-9526-4791-88da-8b07c5edc923.png">
 </p>

## **Methodology**
<div style="text-align: justify">
In this project “Handwritten Digit Recognition using Neural Networks” we shall try to build a neural network from scratch using several algorithms and mathematical inductions to classify handwritten digits. In order to build a neural network, the first thing that we need is data. We will be using the dataset provided by MNIST
(Modified National Institute of Standards and Technology). MNIST is a widely used dataset for the handwritten digit classification task. Process of building a neural network involves mainly three steps:


1. Data Preparation: For data to be able to help us with this classification, we need to pre-process the data. Data pre-processing includes the conversion of categorical attributes to numerical attributes, handling null values and missing values, standardization and normalization.

2. Training of Neural Networks: The process of fine-tuning the weights and biases from the input data is known as Training of Neural Network. It includes an input layer, output layer, choosing an arbitrary number of hidden layers and choice of activation function to get better results. The network has been trained using different activation functions such as Sigmoid function, tanh function and ReLUfunction.

3. Testing and Visualization: Here, the error sum for various activation functions on training and testing data sets has been calculated. The results have been compared to obtain the best activation function for the given dataset. Hyperparameter tuning is performed to obtain better results. Graphs and loss metrics are used to visualize outcomes.
</div>
### **Initialization**:

Here the number of hidden units are 15 so the W1 weight matrix is initialized using random normal distribution with shape (15, number of input units) and bias matrix of shape (15, 1).
Since, the output layer has 10 units so, W2 matrix is initialized with shape (10, 15) and b2 matrix with (10,1) shape.

Forward Pass

*	The Neural summation takes input and weight matrix as input and calculate the weighted sum of input.

		Z =W*(output from previous layer) +b

*	Neural Activation function applies the desired activation function the linear sum.
*	A=g(Z) where g is the activation function A is the output from the neuron and z is the linear sum of weighted inputs

### **Forward Pass**

*	The Neural summation takes input and weight matrix as input and calculate the weighted sum of input.

		Z =W*(output from previous layer) +b

*	Neural Activation function applies the desired activation function the linear sum.
*	A=g(Z) where g is the activation function A is the output from the neuron and z is the linear sum of weighted inputs

<p align="center">
<br>
 <img src="https://user-images.githubusercontent.com/50751235/129049024-a5bc7eb2-5032-4358-8b26-d9c3cdd1a60d.png">
</p>


<p align="center">
 <img src="https://user-images.githubusercontent.com/50751235/129049535-4660f381-3806-4d07-9e0c-1907241c8de1.png">
</p>

### **Cost Function**

<p align="center">
<br>
 <img src="https://user-images.githubusercontent.com/50751235/129049850-c2c13242-a27e-4443-ab82-52c7ea538608.png">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/50751235/129065507-f25f14c5-d567-4bac-b2ce-9d75ee5e95fc.png">
</p>
<p align="center"> a[L](i) = 0</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/50751235/129067369-1d854eef-f451-45ea-85cd-f8e070fdf05d.png">
</p>
<p align="center"> a[L](i) = 1<p>

### **Backward Pass**

<p align="center">
<br>
<img src="https://user-images.githubusercontent.com/50751235/129067683-2c65fff1-1dc7-45ca-9c63-b16c1415bd51.png">
</p>

*	Backpropagation algorithm is a fast way of computing gradients of cost function.
 
*	In backpropagation module, we iterate through all the hidden layers backward.
*	We take the output of forward pass (Z [L]) and then apply activation backward function g’(.) to it.
<p align="center">
<br>
<img src="https://user-images.githubusercontent.com/50751235/129067778-1e1117d9-2640-4d90-af96-1dc8efe887e9.png">
</p>
 
*	Further, compute derivatives of W, b and A using dZ[L]:
*	Later use these gradients to update parameters.


<p align="center">
<br>
<img src="https://user-images.githubusercontent.com/50751235/129067953-4ebf5919-49ca-4382-b3d5-395907ba0f91.png">
</p>
 
### **Update Parameters**

Parameters of the model are updated using gradient descent:
<p align="center">
<br>
<img src="https://user-images.githubusercontent.com/50751235/129068240-2b0e8e9d-a567-4dbb-8492-a1a3f3748459.png">
</p>
where α : learning rate

Updated parameters are stored to later train the network with new parameters.

