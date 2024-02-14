import ast
import importlib
import testclass
import generation
import inspect

# Define the function to rewrite module.py
def rewrite_method_with_ast(file_path, class_name, method_name, new_method_ast):
    # Read the existing code from module.py
    with open(file_path, "r") as file:
        existing_code = file.read()

    # Function to replace the method with a new AST
    def replace_method_with_ast(code, class_name, method_name, new_method_ast):
        tree = ast.parse(code)

        # Find the class node
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Find the existing method node
                for body_item in node.body:
                    if isinstance(body_item, ast.FunctionDef) and body_item.name == method_name:
                        # Replace the method node with the new AST
                        method_index = node.body.index(body_item)
                        node.body[method_index] = new_method_ast
                        break

        # Convert the modified AST back to code
        modified_code = ast.unparse(tree)

        return modified_code

    # Modify the code
    modified_code = replace_method_with_ast(existing_code, class_name, method_name, new_method_ast)

    # Write the modified code back to module.py
    with open(file_path, "w") as file:
        file.write(modified_code)

def main():
    module_source_path = testclass.__file__
    class_name = testclass.MyClass.__name__
    method_name = testclass.MyClass.my_method.__name__

    new_method_code = """
def my_method(self):
    print("Modified method:", self.value + 10)
    """

    new_method_ast = ast.parse(new_method_code).body[0]

    # Rewrite the method in module.py with the new AST
    rewrite_method_with_ast(module_source_path, class_name, method_name, new_method_ast)

    # Reload the module to apply the changes
    importlib.reload(testclass)

# Example usage:
if __name__ == "__main__":
    main()
