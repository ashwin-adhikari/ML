**<h2>Exporting Notebooks</h2>**

When you are working with Jupyter Notebooks, you will find that you need to share your results with non-technical people. When that happens, you can use the nbconvert tool which comes with Jupyter Notebook to convert or export your Notebook into one of the following formats:

    - HTML
    - LaTeX
    - PDF
    - RevealJS
    - Markdown
    - ReStructured Text
    - Executable script

The ``nbconvert`` tool uses _Jinja_ templates under the covers to convert your Notebook files (.ipynb) into these other formats.

_Jinja_ is a template engine that was made for Python. Also note that nbconvert also depends on **_Pandoc_** and **_TeX_** to be able to export to all the formats above. If you don’t have one or more of these, some of the export types may not work. For more information, you should check out the documentation.

**<h3>How to Use nbconvert</h3>**

The ``nbconvert`` command does not take very many parameters, which makes learning how to use it easier. Open up a terminal and navigate to the folder that contains the Notebook you wish to convert. The basic conversion command looks like this:

>``$ jupyter nbconvert <input notebook> --to <output format>``

>``$ jupyter nbconvert py_examples.ipynb --to pdf``


**<h2>Day 3: Loops</h2>**
>``for`` ``else`` loop <br>
    for number in range(1, 5):
    if number == 6:
        break
    else:
    print("Number 6 not found in range.")
    # The else clause will execute here, as the break was not triggered.

**<h2>Day 4: Functions</h2>**

_Positional vs. Keyword Arguments_: Positional arguments are arguments that need to be included in the proper position or order. Keyword arguments are arguments accompanied by an identifier (e.g., ``name='John'``) and can be listed in any order.

_Default Parameter Values_: The value provided to a keyword argument in the function’s definition is the **default value**, and it’s not mandatory to provide a value for that argument to call the function. The value can be overridden by whatever value is provided when the function is called, though.

_Using Keywords to Reorder Arguments_: With keyword arguments, the order of arguments can be shuffled, allowing more flexibility. Arguments passed in with no keyword have to match up with a position. Positional arguments can’t follow keyword arguments.

_Local vs. Global Variables_: Global variables are defined outside a function and can be accessed throughout the program, while local variables are confined within the function they are declared in.

_Variable Shadowing_: ``Don’t Do This.`` Avoid using the same name for local and global variables as it can lead to confusing “shadowing” effects, where the local variable “shadows” the global variable. The local variable will be used inside the function without modifying the global. This can be especially confusing for beginners, who may struggle with the same name being used for two separate variables in different scopes.


**<h3>NumPy:</h3>**
- ``import numpy as np``
- Define Coefficients and Data
    - Suppose we have a linear model with coefficients a and b (for a simple linear equation y = ax + b). We’ll represent these as a NumPy array:
        >``coefficients = np.array([a, b])``

    - Let’s also define our data points. In a simple linear regression, we often have input (x) and output (y) pairs. For this example, x and y are both arrays. Our x data needs an additional column of ones to account for the intercept b:

       >``x_data = np.array([[x1, 1], [x2, 1], ..., [xn, 1]])``
        >``y_data = np.array([y1, y2, ..., yn])``

- Perform Matrix Multiplication for Prediction
    - We compute the predicted y values (``y_pred``) using matrix multiplication between our data (``x_data``) and coefficients. In NumPy, matrix multiplication is done using the ``@ operator`` or ``np.dot()`` function:

        >``y_pred = x_data @ coefficients``

- Compute the Loss
    - The loss function quantifies how far our predictions are from the actual values. A common loss function in linear regression is **_Mean Squared Error (MSE)_**, calculated as the average of the squares of the differences between actual (``y_data``) and predicted (``y_pred``) values:

        >``loss = np.mean((y_data - y_pred) ** 2)``

