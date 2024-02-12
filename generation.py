import ast
from ast import Assign, Name, Store, Call, Attribute, Load, Constant, UnaryOp, USub, keyword, BinOp, Mult, Return, Tuple, Lt, Compare, If, Add
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
                    comparators=[Constant(value=300000000000)]
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
    def visit_Assign(self, node):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "layer1":
            return [node] + exitAst
        return node

class EditExitThreshold(ast.NodeTransformer):
    def __init__(self, threshold):
        self.threshold = threshold

    def visit_If(self, node):
        if isinstance(node, If) and isinstance(node.test, Call) and isinstance(node.test.args[0], Compare) and isinstance(node.test.args[0].comparators[0], Constant):
            node.test.args[0].comparators[0] = Constant(value=self.threshold)
            return node
        return node

class EarlyExit():
    # AST here is a list of the AST nodes specific to the exit, not the entire forward function.
    # Threshold is the actual threshold to keep the same accuracy as the full model
    # ID is a number to keep track of which exit this is TODO: This needs to be updated if an exit is added before this one
    def __init__(self, ast, threshold, id):
        self.ast = ast
        self.threshold = threshold
        self.id = id

    def updateThreshold(self, threshold=None):
        if threshold is None:
            threshold = self.threshold

        exitEditor = EditExitThreshold(threshold)
        exitEditor.visit(self.ast)

    def getAst(self):
        return self.ast

    def setThreshold(self, threshold):
        self.threshold = threshold
        self.updateThreshold()

getAstFromSource = lambda x: ast.parse(textwrap.dedent(inspect.getsource(x)))
getAstDump = lambda x: ast.dump(x, indent=4)

class ExitTracker:
    def __init__(self, model, accuracy):
        self.targetAccuracy = accuracy
        self.first_transform_complete = False
        self.model = model
        self.original_ast = getAstFromSource(self.model.forward)
        self.prev_ast = getAstFromSource(self.model.forward)
        self.current_ast = getAstFromSource(self.model.forward)

    def transformFunction(self):
        exitTransformer = AddExitTransformer()
        self.prev_ast = self.current_ast
        self.current_ast = exitTransformer.visit(self.current_ast if self.first_transform_complete else self.original_ast)
        ast.fix_missing_locations(self.current_ast)
        self.recompileForward()

    def recompileForward(self):
        code_object = compile(self.current_ast, '', 'exec')
        exec(code_object, globals())

        # https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/11
        # ^^ WHY??
        bound_method = globals()["forward"].__get__(self.model, self.model.__class__)
        setattr(self.model, 'forward', bound_method)

    def dumpCurrentAst(self):
        return getAstDump(self.current_ast)
