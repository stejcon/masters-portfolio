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
                    comparators=[Constant(value=300)]
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

# TODO: Check no exit already exists
# TODO: Make this go in after a specified layer, not after "layer1"
# TODO: Make ast_code be a parameter, not a random global
# TODO: Create ast_code from a function where entropy and the dimensions are passed in or calculated
# TODO: Create an EditEntropyTransformer to update the entropy for a specific exit
class AddExitTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "layer1":
            return [node] + exitAst
        return node

getAstFromSource = lambda x: ast.parse(textwrap.dedent(inspect.getsource(x)))
getAstDump = lambda x: ast.dump(x, indent=4)

class ExitTracker:
    def __init__(self, model):
        self.first_transform_complete = False
        self.model = model
        self.original_ast = getAstFromSource(self.model.forward)
        self.prev_ast = getAstFromSource(self.model.forward)
        self.current_ast = getAstFromSource(self.model.forward)

    def transformFunction(self):
        exitTransformer = AddExitTransformer()
        self.current_ast = exitTransformer.visit(self.current_ast if self.first_transform_complete else self.original_ast)
        ast.fix_missing_locations(self.current_ast)
        code_object = compile(self.current_ast, '', 'exec')
        exec(code_object, globals())

        # https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/11
        # ^^ WHY??
        bound_method = globals()["forward"].__get__(self.model, self.model.__class__)
        setattr(self.model, 'forward', bound_method)

    def dumpCurrentAst(self):
        return getAstDump(self.current_ast)
