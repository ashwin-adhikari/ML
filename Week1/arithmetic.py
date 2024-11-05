# Define initial variables
x = 4
y = 3.14
t = "hi"

# Dictionary to store results
integer_results = {}
float_results = {}
string_results = {}
assignment_results = {}


# Function to attempt an operation and store the result or error
def test_operation(var1, var2, operation):
    try:
        result = eval(f"{var1} {operation} {var2}")
        return f"{result} (type: {type(result).__name__})"
    except Exception as e:
        return f"Error: {e}"


# List of operations to perform
operations = ["+", "-", "*", "/", "//", "**", "%", "=="]
assignment_operations = ["+=", "-="]

# Integer tests with x

for op in operations:
    integer_results[f"x {op} int"] = test_operation("x", "4", op)
    integer_results[f"x {op} float"] = test_operation("x", "y", op)
    integer_results[f"x {op} str"] = test_operation("x", "t", op)

# Floating point tests with y

for op in operations:
    float_results[f"y {op} int"] = test_operation("y", "4", op)
    float_results[f"y {op} float"] = test_operation("y", "y", op)
    float_results[f"y {op} str"] = test_operation("y", "t", op)

# String tests with t

for op in operations:
    string_results[f"t {op} int"] = test_operation("t", "4", op)
    string_results[f"t {op} float"] = test_operation("t", "y", op)
    string_results[f"t {op} str"] = test_operation("t", "'hello'", op)

# Assignment operations for x, y, t

x_test, y_test, t_test = 4, 3.14, "hi"  # Re-initialize for assignment operations

for op in assignment_operations:
    # x+=int
    try:
        exec(f"x_test {op} {x}")
        assignment_results[f"x {op} {x}"] = f"{x_test} (type: {type(x_test).__name__})"
    except Exception as e:
        assignment_results[f"x {op} {x}"] = f"Error: {e}"

    #x+=float
    try:
        exec(f"x_test {op} {y}")
        assignment_results[f"x {op} {y}"] = f"{x_test} (type: {type(x_test).__name__})"
    except Exception as e:
        assignment_results[f"x {op} {y}"] = f"Error: {e}"
    
    #x+=str
    try:
        exec(f"x_test {op} {t}")
        assignment_results[f"x {op} {t}"] = f"{x_test} (type: {type(x_test).__name__})"
    except Exception as e:
        assignment_results[f"x {op} {t}"] = f"Error: {e}"
    
    #y+=int
    try:
        exec(f"y_test {op} {x}")
        assignment_results[f"y {op} {x}"] = f"{y_test} (type: {type(y_test).__name__})"
    except Exception as e:
        assignment_results[f"y {op} {x}"] = f"Error: {e}"

    #y+=float
    try:
        exec(f"y_test {op} {y}")
        assignment_results[f"y {op} {y}"] = f"{y_test} (type: {type(y_test).__name__})"
    except Exception as e:
        assignment_results[f"y {op} {y}"] = f"Error: {e}"

    #y+=str
    try:
        exec(f"y_test {op} {t}")
        assignment_results[f"y {op} {t}"] = f"{y_test} (type: {type(y_test).__name__})"
    except Exception as e:
        assignment_results[f"y {op} {t}"] = f"Error: {e}"


    #t+=int
    try:
        exec(f"t_test {op} {x}")
        assignment_results[f"t {op} {x}"] = f"{t_test} (type: {type(t_test).__name__})"
    except Exception as e:
        assignment_results[f"t {op} x"] = f"Error: {e}"
    
    #t+=float
    try:
        exec(f"t_test {op} {y}")
        assignment_results[f"t {op} {y}"] = f"{t_test} (type: {type(t_test).__name__})"
    except Exception as e:
        assignment_results[f"t {op} {y}"] = f"Error: {e}"

    #t+=str
    
    try:
        exec(f"t_test {op} 'hello'")
        assignment_results[f"t {op} 'hello'"] = f"{t_test} (type: {type(t_test).__name__})"
    except Exception as e:
        assignment_results[f"t {op} 'hello'"] = f"Error: {e}"

# Display all results
print("\n--- Results ---")
print("\n--- Integer Results ---")

for operation, result in integer_results.items():
    print(f"{operation}: {result}")

print("\n--- Float Results ---")

for operation, result in float_results.items():
    print(f"{operation}: {result}")

print("\n--- String Results ---")

for operation, result in string_results.items():
    print(f"{operation}: {result}")

print("\n--- Assignment Results ---")

for operation, result in assignment_results.items():
    print(f"{operation}: {result}")