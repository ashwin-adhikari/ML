# Vector in python ###

> ``x = np.array([1,2,3,4])``
> here x is a vector.

### Addition and Subtraction
> ``x1 = np.array([3,4])``
> ``x2 = np.array([1,2])``
> sum = x1+x2 
> or sum = np.add(x1,x2)
> diff = x1-x2

### Multiplication
Scalar multiplication plays a pivotal role in adjusting the learning rate. For instance, when training a neural network, scaling the learning rate dynamically based on the progress of training helps avoid overshooting or slow convergence, ensuring effective model training.

> scaler_mul = x1 * 2
> Output : [6,8]

#### Calculating magnitude and direction
>``magnitude = np.linalg.norm(x1)``
> ``linalg`` is linear algebra module and ``norm`` is normal is used to calculate magnitude of x1 i.e $\sqrt{(3^2 + 4^2)}$
>``direction = np.arctan2(x1[1], x1[0])``<br>
> ``arctan`` is inverse of tan and above expression is ${tan^-1(x1[1]/x1[0])}$ 


> Magnitude of x1: 5.0
> Direction of x1: 0.9272952180016122 radians

### Dot product
Dot products are fundamental for text classification in natural language processing (NLP). By calculating dot products between word vectors and predefined sentiment vectors, you can determine the sentiment of a text. This enables automated sentiment analysis for large volumes of text data.

>dot_product = np.dot(x1, x2)
> x1_3D = np.append(x1, 0)
> x2_3D = np.append(x2, 0)
>``append`` is used to add element at end of the array or vector. We can add **only one element** or a variable 
> ``x1_3D = np.append(x1,x2)``
>Output = [3,4,1,2]

### Cross Product
Cross products are crucial in 3D graphics and physics, particularly for calculating normals to surfaces, which are essential for lighting and rendering. They are also used in physics for finding torques and rotational vectors, helping to understand the dynamics of rotating bodies.

>cross_product = np.cross(x1_3D, x2_3D)<br>

> **TO PERFORM CROSS PRODUCT VECTOR LENGTH MUST BE EQUAL**



**Vector norms** are used in machine learning for regularization and similarity calculations.


To calculate the L2 norm of a vector in Python, you can use the numpy library:

>vector_v = np.array([3, 4])
>l2_norm = np.linalg.norm(vector_v)

## Real-world Application: Vector in Computer Graphics

- Understanding vector representation and manipulation in digital space is fundamental in computer graphics and game development.
- Predicting object movements under various conditions is crucial in mechanical and aerospace engineering.

# Matrix

>Created using: `` x1 = np.array([1,2,3],[4,5,6])``
>``y1 = np.array([[7, 8, 9], [10, 11, 12]])``
> print(x1.shape,'\n') 
The *.shape after the variable, allows you to verify the shape of the matrix.

#### output:

```
[[1 2 3]
 [4 5 6]]
(2, 3) 
```

> Addition is same as of vector
```
addition = np.add(x1,y1) or x1+y1
sub = np.subtract(x1,y1) or x1-y1
```
### Multiplication
```
m_sca = 5 * x1 #Scalar multiplication
m_mul = x1@z1
@ is used to for matrix multiplication
```

### Transpose
```
X = np.array([[1, 3, 5], [7, 9, 11], [13, 15, 17]])
X_T = X.transpose()

Output:
[[ 1  7 13]
 [ 3  9 15]
 [ 5 11 17]]
```

### Inverse
```
X = np.array([[1, 3, 5], [7, 9, 11], [13, 15, 17]])
X_inv = np.linalg.inv(X)

Output:
[[-2.81474977e+14  5.62949953e+14 -2.81474977e+14]
 [ 5.62949953e+14 -1.12589991e+15  5.62949953e+14]
 [-2.81474977e+14  5.62949953e+14 -2.81474977e+14]]

```

# Derivatives

Derivatives are the fundamental tools of Calculus. It is very useful for optimizing a loss function with **_gradient descent_** in Machine Learning is possible only because of derivatives.

Python ``SymPy`` library is created for symbolic mathematics. The SymPy project aims to become a full-featured computer algebra system (CAS) while keeping the code simple to understand. Let’s see how to calculate derivatives in Python using SymPy.

```
import sympy as sp

# create a "symbol" called x
x = Symbol('x')
 
#Define function
f = x**2
 
#Calculating Derivative
derivative_f = f.diff(x)
```

### Applications of Derivatives

In Machine Learning and AI:
- Optimization: <br>
 Derivatives are fundamental in optimizing cost functions in algorithms like Gradient Descent.
- Training Neural Networks: <br>
 The backpropagation algorithm, used in training neural networks, utilizes derivatives to compute gradients for weight updates.

- Sensitivity Analysis: <br>Derivatives help in understanding how changes in input affect the output, aiding in feature selection and model understanding.

- Model Behavior Analysis:<br> In algorithms like SVM, derivatives help formulate the optimization problem, and in decision trees, they can assist in node splitting and feature importance analysis.

