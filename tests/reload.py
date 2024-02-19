import model
import tempfile
import torch
import sys
import importlib
import textwrap
import inspect
import ast


class ReloadableModel:
    def __init__(self, model_class, *args):
        self.model = model_class(*args)
        self.model_class = model_class
        self.model_args = args

    def rewrite(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(self.model.forward)))
        ast_list = tree.body[0].body
        node = ast.Module(
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="print", ctx=ast.Load()),
                        args=[
                            ast.JoinedStr(
                                values=[
                                    ast.FormattedValue(
                                        value=ast.Name(id="3", ctx=ast.Load()),
                                        conversion=-1,
                                    )
                                ]
                            )
                        ],
                        keywords=[],
                    )
                )
            ],
            type_ignores=[],
        )

        ast_list.insert(2, node)

        def saveForwardFunction(model, new_tree):
            filePath = inspect.getmodule(model).__file__
            class_name = model.__class__.__name__
            method_name = model.forward.__name__

            # Read the existing code from module.py
            with open(filePath, "r") as file:
                existing_code = file.read()

            # Parse the existing code into an AST
            tree = ast.parse(existing_code)

            # Find the class node
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # Find the existing method node
                    for body_item in node.body:
                        if (
                            isinstance(body_item, ast.FunctionDef)
                            and body_item.name == method_name
                        ):
                            # Replace the method node with the new AST
                            method_index = node.body.index(body_item)
                            node.body[method_index].body = new_tree
                            break

            # Convert the modified AST back to code
            modified_code = ast.unparse(tree)

            # Write the modified code back to module.py
            with open(filePath, "w") as file:
                file.write(modified_code)

        saveForwardFunction(self.model, ast_list)

    def reload(self):
        with tempfile.TemporaryFile() as file:
            torch.save(self.model.state_dict(), file)
            file.seek(0)
            module_name = self.model_class.__module__
            importlib.reload(sys.modules[module_name])
            reloaded_module = sys.modules[module_name]
            self.model_class = getattr(reloaded_module, self.model_class.__name__)
            self.model = self.model_class(*(self.model_args))
            self.model.load_state_dict(torch.load(file))

    def getModel(self):
        return self.model


rModel = ReloadableModel(model.Net)
rModel.rewrite()

input = torch.randn(1, 1, 32, 32)
out = rModel.getModel()(input)
print(out)

rModel.reload()
out = rModel.getModel()(input)
print(out)
