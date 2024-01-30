import ast
import inspect
import textwrap

ast_code = [
    ast.Assign(
        targets=[ast.Name(id='y', ctx=ast.Store())],
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
                attr='avgpool',
                ctx=ast.Load()
            ),
            args=[ast.Name(id='x', ctx=ast.Load())],
            keywords=[]
        )
    ),
    ast.Assign(
        targets=[ast.Name(id='y', ctx=ast.Store())],
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='y', ctx=ast.Load()),
                attr='view',
                ctx=ast.Load()
            ),
            args=[
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='x', ctx=ast.Load()),
                        attr='size',
                        ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=0)],
                    keywords=[]
                ),
                ast.UnaryOp(
                    op=ast.USub(),
                    operand=ast.Constant(value=1)
                )
            ],
            keywords=[]
        )
    ),
    ast.Assign(
        targets=[ast.Name(id='y', ctx=ast.Store())],
        value=ast.Call(
            func=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='nn', ctx=ast.Load()),
                    attr='Linear',
                    ctx=ast.Load()
                ),
                args=[
                    ast.Constant(value=61952),
                    ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()),
                        attr='num_classes',
                        ctx=ast.Load()
                    )
                ],
                keywords=[]
            ),
            args=[ast.Name(id='y', ctx=ast.Load())],
            keywords=[]
        )
    ),
    ast.Assign(
        targets=[ast.Name(id='y', ctx=ast.Store())],
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id='torch', ctx=ast.Load()),
                        attr='nn',
                        ctx=ast.Load()
                    ),
                    attr='functional',
                    ctx=ast.Load()
                ),
                attr='softmax',
                ctx=ast.Load()
            ),
            args=[ast.Name(id='y', ctx=ast.Load())],
            keywords=[
                ast.keyword(
                    arg='dim',
                    value=ast.Constant(value=1)
                )
            ]
        )
    ),
    ast.Assign(
        targets=[ast.Name(id='entropy', ctx=ast.Store())],
        value=ast.UnaryOp(
            op=ast.USub(),
            operand=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='sum',
                    ctx=ast.Load()
                ),
                args=[
                    ast.BinOp(
                        left=ast.Name(id='y', ctx=ast.Load()),
                        op=ast.Mult(),
                        right=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id='torch', ctx=ast.Load()),
                                attr='log2',
                                ctx=ast.Load()
                            ),
                            args=[
                                ast.BinOp(
                                    left=ast.Name(id='y', ctx=ast.Load()),
                                    op=ast.Add(),
                                    right=ast.Constant(value=1e-20)
                                )
                            ],
                            keywords=[]
                        )
                    )
                ],
                keywords=[
                    ast.keyword(
                        arg='dim',
                        value=ast.Constant(value=1)
                    )
                ]
            )
        )
    ),
    ast.If(
        test=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='torch', ctx=ast.Load()),
                attr='all',
                ctx=ast.Load()
            ),
            args=[
                ast.Compare(
                    left=ast.Name(id='entropy', ctx=ast.Load()),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(value=0.5)]
                )
            ],
            keywords=[]
        ),
        body=[
            ast.Return(
                value=ast.Name(id='y', ctx=ast.Load())
            )
        ],
        orelse=[]
    )
]

# TODO: Check no exit already exists
# TODO: Make this go in after a specified layer, not after "layer1"
# TODO: Make ast_code be a parameter, not a random global
# TODO: Create ast_code from a function where entropy and the dimensions are passed in or calculated
# TODO: Create an EditEntropyTransformer to update the entropy for a specific exit
class AddExitTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "layer1":
            return [node] + ast_code
        return node

getAstDump = lambda x: ast.dump(ast.parse(textwrap.dedent(inspect.getsource(x))), indent=4)
getAst = lambda x: ast.parse(textwrap.dedent(inspect.getsource(x)))

def transformFunction(forward_function):
    visitor = AddExitTransformer()
    ast_x = getAst(forward_function)
    ast_x = visitor.visit(ast_x)
    ast.fix_missing_locations(ast_x)
    code_object = compile(ast_x, '<generated-exit>', 'exec')
    exec(code_object, globals())
    # TODO: This string should be known beforehand and made to not clash with any other possible function
    forward_function = globals()["forward"]
    print(f"{getAstDump(forward_function)}")
