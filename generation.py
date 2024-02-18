import ast
from ast import Assign, FunctionDef, Name, Store, Call, Attribute, Load, Constant, UnaryOp, USub, keyword, BinOp, Mult, Return, Tuple, Lt, Compare, If, Add, Module
import inspect
import textwrap
import torch # Ignore LSP error "not accessed", needed to compile the ast code
import torch.nn as nn # Ignore LSP error "not accessed", needed to compile the ast code

exitAst = [
    Assign(
        targets=[Name(id='y', ctx=Store())],
        value=Call(
            func=Attribute(
                value=Name(id='self', ctx=Load()),
                attr='avgpool',
                ctx=Load()
            ),
            args=[Name(id='x', ctx=Load())],
            keywords=[]
        )
    ),

    Assign(
        targets=[Name(id='y', ctx=Store())],
        value=Call(
            func=Attribute(
                value=Name(id='y', ctx=Load()),
                attr='view',
                ctx=Load()
            ),
            args=[
                Call(
                    func=Attribute(
                        value=Name(id='x', ctx=Load()),
                        attr='size',
                        ctx=Load()
                    ),
                    args=[Constant(value=0)],
                    keywords=[]
                ),
                UnaryOp(
                    op=USub(),
                    operand=Constant(value=1)
                )
            ], 
            keywords=[]
        )
    ),

    Assign(
        targets=[Name(id='y', ctx=Store())],
        value=Call(
            func=Call(
                func=Attribute(
                    value=Attribute(
                        value=Name(id='torch', ctx=Load()),
                        attr='nn',
                        ctx=Load()
                    ),
                    attr='Linear',
                    ctx=Load()
                ),
                args=[
                    Call(
                        func=Attribute(
                            value=Name(id='y', ctx=Load()),
                            attr='size',
                            ctx=Load()
                        ),
                        args=[Constant(value=1)],
                        keywords=[]
                    ),
                    Attribute(
                        value=Name(id='self', ctx=Load()),
                        attr='num_classes',
                        ctx=Load()
                    )
                ],
                keywords=[]
            ),
            args=[Name(id='y', ctx=Load())],
            keywords=[]
        )
    ),

    Assign(
        targets=[Name(id='y', ctx=Store())],
        value=Call(
            func=Attribute(
                value=Attribute(
                    value=Attribute(
                        value=Name(id='torch', ctx=Load()),
                        attr='nn',
                        ctx=Load()
                    ),
                    attr='functional',
                    ctx=Load()
                ),
                attr='softmax',
                ctx=Load()
            ),
            args=[Name(id='y', ctx=Load())],
            keywords=[keyword(arg='dim', value=Constant(value=1))]
        )
    ),

    Assign(
        targets=[Name(id='entropy', ctx=Store())],
        value=UnaryOp(
            op=USub(),
            operand=Call(
                func=Attribute(
                    value=Name(id='torch', ctx=Load()),
                    attr='sum',
                    ctx=Load()
                ),
                args=[
                    BinOp(
                        left=Name(id='y', ctx=Load()),
                        op=Mult(),
                        right=Call(
                            func=Attribute(
                                value=Name(id='torch', ctx=Load()),
                                attr='log2',
                                ctx=Load()
                            ),
                            args=[
                                BinOp(
                                    left=Name(id='y', ctx=Load()),
                                    op=Add(),
                                    right=Constant(value=1e-20)
                                )
                            ],
                            keywords=[]
                        )
                    )
                ],
                keywords=[keyword(arg='dim', value=Constant(value=1))]
            )
        )
    ),

    If(
        test=Call(
            func=Attribute(
                value=Name(id='torch', ctx=Load()),
                attr='all',
                ctx=Load()
            ),
            args=[
                Compare(
                    left=Name(id='entropy', ctx=Load()),
                    ops=[Lt()],
                    comparators=[Constant(value=0)]
                )
            ],
            keywords=[]
        ),
        body=[
            Return(
                value=Tuple(
                    elts=[
                        Constant(value=1),
                        Name(id='y', ctx=Load())
                    ],
                    ctx=Load()
                )
            )
        ],
        orelse=[]
    ),
]

class AddExitTransformer(ast.NodeTransformer):
    def __init__(self, astlist):
        self.exitAst = astlist

    def visit_Assign(self, node):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "layer1":
            return [node] + self.exitAst
        return node

class EditExitThreshold(ast.NodeTransformer):
    def __init__(self, threshold):
        self.threshold = threshold

    def visit_If(self, node):
        if isinstance(node, If) and isinstance(node.test, Call) and isinstance(node.test.args[0], Compare) and isinstance(node.test.args[0].comparators[0], Constant):
            node.test.args[0].comparators[0] = Constant(value=self.threshold)
            return node
        return node

class EditReturnId(ast.NodeTransformer):
    def __init__(self, id):
        self.id = id

    def visit_If(self, node):
        if isinstance(node, If) and isinstance(node.body[0], Return) and isinstance(node.body[0].value, Tuple) and isinstance(node.body[0].value.elts[0], Constant):
            node.body[0].value.elts[0] = Constant(value=self.id)
            return node
        return node

class EarlyExit():
    def __init__(self, node, threshold, id):
        self.node = node
        self.setThreshold(threshold)
        self.setId(id)

    def getAst(self):
        return self.node

    def updateThreshold(self, threshold=None):
        if threshold is None:
            threshold = self.threshold

        def edit_exit_threshold(nodes, threshold):
            edited_nodes = []
            for node in nodes:
                if isinstance(node, ast.If) and isinstance(node.test, ast.Call) and isinstance(node.test.args[0], ast.Compare) and isinstance(node.test.args[0].comparators[0], ast.Constant):
                    node.test.args[0].comparators[0] = ast.Constant(value=threshold)
                edited_nodes.append(node)
            return edited_nodes

        self.node = edit_exit_threshold(self.node, threshold)

    def setThreshold(self, threshold):
        self.threshold = threshold
        self.updateThreshold()

    def resetThreshold(self):
        self.updateThreshold(self.threshold)

    def updateReturn(self):
        def edit_return_id(nodes, new_id):
            edited_nodes = []
            for node in nodes:
                if isinstance(node, ast.If) and isinstance(node.body[0], ast.Return) and isinstance(node.body[0].value, ast.Tuple) and isinstance(node.body[0].value.elts[0], ast.Constant):
                    node.body[0].value.elts[0] = ast.Constant(value=new_id)
                edited_nodes.append(node)
            return edited_nodes

        self.node = edit_return_id(self.node, self.id)

    def setId(self, id):
        self.id = id
        self.updateReturn()

getAstFromSource = lambda x: ast.parse(textwrap.dedent(inspect.getsource(x)))
getAstDump = lambda x: ast.dump(x, indent=4)

class ExitTracker:
    # TODO: Make the model taken in here a ReloadableModel
    def __init__(self, model, accuracy):
        self.targetAccuracy = accuracy
        self.first_transform_complete = False
        self.model = model
        self.original_ast = getAstFromSource(self.model.forward)
        self.prev_ast = getAstFromSource(self.model.forward)
        self.current_ast = getAstFromSource(self.model.forward)
        assert(isinstance(self.current_ast.body[0], FunctionDef))
        self.current_list = [x for x in self.current_ast.body[0].body]

    # TODO: Function to test exit and set entropy threshold correctly

    # TODO: Function to label all exit return values correctly

    # TODO: Function to add a new exit and disable other exits

    def transformFunction(self):
        self.prev_ast = self.current_ast
        self.current_list.insert(3, EarlyExit(exitAst, 37, 6))

        b = []
        for x in self.current_list:
            if (isinstance(x, EarlyExit)):
                b.extend(x.node)
            else:
                b.append(x)

        assert(isinstance(self.current_ast.body[0], FunctionDef))
        self.current_ast.body[0].body = b
        ast.fix_missing_locations(self.current_ast)

        # TODO: Instead of recompileForward, write the updated module and reload
        # self.recompileForward()
        self.printCurrentAstAsSource()

    def printCurrentAstAsSource(self):
        filePath = inspect.getmodule(self.model).__file__ # type: ignore
        class_name = self.model.__class__.__name__
        method_name = self.model.forward.__name__

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

            # # Write the modified code back to module.py
            # with open(file_path, "w") as file:
            #     file.write(modified_code)

            with open("test-output.py", "w") as file:
                file.write(modified_code)

        rewrite_method_with_ast(filePath, class_name, method_name, self.current_ast)

    # TODO: Put this as the other ast stuff into a separate appendix, not applicable anymore
    def recompileForward(self):
        code_object = compile(self.current_ast, '', 'exec')
        exec(code_object, globals())
        bound_method = globals()["forward"].__get__(self.model, self.model.__class__)
        setattr(self.model, 'forward', bound_method)

    def dumpCurrentAst(self):
        return getAstDump(self.current_ast)
