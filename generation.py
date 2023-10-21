import types

def sample_function(a, b):
    return (a + b)

# Get the code object from an existing function
code_object = sample_function.__code__
print([x for x in code_object.co_varnames])

# Create a new function from the code object
new_function = types.FunctionType(code_object, globals(), "new_function")

# Now, you can call the new_function with arguments
result = new_function(3, 5)  # Pass any two arguments here
print(result)  # Output will be 80
