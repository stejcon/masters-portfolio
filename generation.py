import ast
from copy import deepcopy
from ast import (
    Assign,
    FunctionDef,
    Name,
    Store,
    Call,
    Attribute,
    Load,
    Constant,
    UnaryOp,
    USub,
    keyword,
    BinOp,
    Mult,
    Return,
    Tuple,
    Lt,
    Compare,
    If,
    Add,
    Expr,
)
import inspect
import textwrap
import helpers


class EarlyExit:
    def __init__(self, node, threshold, id):
        self.node = deepcopy(node)
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
                if (
                    isinstance(node, ast.If)
                    and isinstance(node.test, ast.Call)
                    and isinstance(node.test.args[0], ast.Compare)
                    and isinstance(node.test.args[0].comparators[0], ast.Constant)
                ):
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
                if (
                    isinstance(node, ast.If)
                    and isinstance(node.body[0], ast.Return)
                    and isinstance(node.body[0].value, ast.Tuple)
                    and isinstance(node.body[0].value.elts[0], ast.Constant)
                ):
                    node.body[0].value.elts[0] = ast.Constant(value=new_id)
                edited_nodes.append(node)
            return edited_nodes

        self.node = edit_return_id(self.node, self.id)

    def setId(self, id):
        self.id = id
        self.updateReturn()


def getAstFromSource(x):
    return ast.parse(textwrap.dedent(inspect.getsource(x)))


def getAstDump(x):
    return ast.dump(x, indent=4)


class ExitTracker:
    # TODO: Make the model taken in here a ReloadableModel
    # TODO: Need to edit the __init__ function of the model to contain the layer of each exit
    def __init__(self, model, accuracy):
        self.targetAccuracy = accuracy
        self.reloadable_model = model
        self.ff_body_ast = getAstFromSource(self.reloadable_model.getModel().forward)
        assert isinstance(self.ff_body_ast.body[0], FunctionDef)
        self.ff_node_list = [x for x in self.ff_body_ast.body[0].body]

        self.exits = []

        # for i in range(len(self.ff_node_list) - 1):
        #     # Disable all exits at beginning
        #     self.exits.append(EarlyExit(exitAst, 0, i + 1))

        # self.ff_new_node_list = []
        # for i in range(len(self.exits)):
        #     self.ff_new_node_list.append(self.ff_node_list[i])
        #     self.ff_new_node_list.append(self.exits[i])

        # self.ff_new_node_list.append(self.ff_node_list[-1])

        self.ff_new_node_list = deepcopy(self.ff_node_list)
        self.ff_new_node_list.insert(4, EarlyExit(exitAst, 0, 1))

    def transformFunction(self):
        b = []
        for x in self.ff_new_node_list:
            if isinstance(x, EarlyExit):
                b.extend(x.node)
            else:
                b.append(x)

        self.ff_body_ast.body[0].body = b
        ast.fix_missing_locations(self.ff_body_ast)

        self.saveForwardFunction()
        self.reloadable_model.reload()

    def enableMiddleExit(self):
        self.ff_new_node_list[4].updateThreshold(300)
        self.saveForwardFunction()
        self.reloadable_model.reload()

    def setMiddleExitCorrectly(self):
        _, _, testLoader = helpers.Cifar10Splits()
        self.ff_new_node_list[4].setThreshold(
            helpers.getEntropyForAccuracy(
                self.reloadable_model.getModel(), testLoader, self.targetAccuracy
            )
        )
        self.saveForwardFunction()
        self.reloadable_model.reload()

    def saveForwardFunction(self):
        filePath = inspect.getmodule(self.reloadable_model.getModel()).__file__
        class_name = self.reloadable_model.getModel().__class__.__name__
        method_name = self.reloadable_model.getModel().forward.__name__

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
                        node.body[method_index] = self.ff_body_ast
                        break

        # Convert the modified AST back to code
        modified_code = ast.unparse(tree)

        # Write the modified code back to module.py
        with open(filePath, "w") as file:
            file.write(modified_code)

    # TODO: Put this as the other runtime ast stuff into appendix
    def recompileForward(self):
        code_object = compile(self.ff_body_ast, "", "exec")
        exec(code_object, globals())
        bound_method = globals()["forward"].__get__(self.model, self.model.__class__)
        setattr(self.model, "forward", bound_method)


class AddExitTransformer(ast.NodeTransformer):
    def __init__(self, astlist):
        self.exitAst = astlist

    def visit_Assign(self, node):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "layer1"
        ):
            return [node] + self.exitAst
        return node


class EditExitThreshold(ast.NodeTransformer):
    def __init__(self, threshold):
        self.threshold = threshold

    def visit_If(self, node):
        if (
            isinstance(node, If)
            and isinstance(node.test, Call)
            and isinstance(node.test.args[0], Compare)
            and isinstance(node.test.args[0].comparators[0], Constant)
        ):
            node.test.args[0].comparators[0] = Constant(value=self.threshold)
            return node
        return node


class EditReturnId(ast.NodeTransformer):
    def __init__(self, id):
        self.id = id

    def visit_If(self, node):
        if (
            isinstance(node, If)
            and isinstance(node.body[0], Return)
            and isinstance(node.body[0].value, Tuple)
            and isinstance(node.body[0].value.elts[0], Constant)
        ):
            node.body[0].value.elts[0] = Constant(value=self.id)
            return node
        return node


exitAst = [
    Assign(
        targets=[Name(id="y", ctx=Store())],
        value=Call(
            func=Attribute(
                value=Name(id="self", ctx=Load()), attr="avgpool", ctx=Load()
            ),
            args=[Name(id="x", ctx=Load())],
            keywords=[],
        ),
    ),
    Assign(
        targets=[Name(id="y", ctx=Store())],
        value=Call(
            func=Attribute(value=Name(id="y", ctx=Load()), attr="view", ctx=Load()),
            args=[
                Call(
                    func=Attribute(
                        value=Name(id="x", ctx=Load()), attr="size", ctx=Load()
                    ),
                    args=[Constant(value=0)],
                    keywords=[],
                ),
                UnaryOp(op=USub(), operand=Constant(value=1)),
            ],
            keywords=[],
        ),
    ),
    Assign(
        targets=[Name(id="exitlayer", ctx=Store())],
        value=Call(
            func=Attribute(
                value=Attribute(
                    value=Name(id="torch", ctx=Load()), attr="nn", ctx=Load()
                ),
                attr="Linear",
                ctx=Load(),
            ),
            args=[
                Call(
                    func=Attribute(
                        value=Name(id="y", ctx=Load()), attr="size", ctx=Load()
                    ),
                    args=[Constant(value=1)],
                    keywords=[],
                ),
                Attribute(
                    value=Name(id="self", ctx=Load()), attr="num_classes", ctx=Load()
                ),
            ],
            keywords=[],
        ),
    ),
    Expr(
        value=Call(
            func=Attribute(
                value=Name(id="self", ctx=Load()), attr="register_module", ctx=Load()
            ),
            args=[Constant(value="exitlayer"), Name(id="exitlayer", ctx=Load())],
            keywords=[],
        )
    ),
    Assign(
        targets=[Name(id="y", ctx=Store())],
        value=Call(
            func=Name(id="exitlayer", ctx=Load()),
            args=[Name(id="y", ctx=Load())],
            keywords=[],
        ),
    ),
    Assign(
        targets=[Name(id="y", ctx=Store())],
        value=Call(
            func=Attribute(
                value=Attribute(
                    value=Attribute(
                        value=Name(id="torch", ctx=Load()),
                        attr="nn",
                        ctx=Load(),
                    ),
                    attr="functional",
                    ctx=Load(),
                ),
                attr="softmax",
                ctx=Load(),
            ),
            args=[Name(id="y", ctx=Load())],
            keywords=[keyword(arg="dim", value=Constant(value=1))],
        ),
    ),
    Assign(
        targets=[Name(id="entropy", ctx=Store())],
        value=UnaryOp(
            op=USub(),
            operand=Call(
                func=Attribute(
                    value=Name(id="torch", ctx=Load()), attr="sum", ctx=Load()
                ),
                args=[
                    BinOp(
                        left=Name(id="y", ctx=Load()),
                        op=Mult(),
                        right=Call(
                            func=Attribute(
                                value=Name(id="torch", ctx=Load()),
                                attr="log2",
                                ctx=Load(),
                            ),
                            args=[
                                BinOp(
                                    left=Name(id="y", ctx=Load()),
                                    op=Add(),
                                    right=Constant(value=1e-20),
                                )
                            ],
                            keywords=[],
                        ),
                    )
                ],
                keywords=[keyword(arg="dim", value=Constant(value=1))],
            ),
        ),
    ),
    If(
        test=Call(
            func=Attribute(value=Name(id="torch", ctx=Load()), attr="all", ctx=Load()),
            args=[
                Compare(
                    left=Name(id="entropy", ctx=Load()),
                    ops=[Lt()],
                    comparators=[Constant(value=0)],
                )
            ],
            keywords=[],
        ),
        body=[
            Return(
                value=Tuple(
                    elts=[Constant(value=1), Name(id="y", ctx=Load())],
                    ctx=Load(),
                )
            )
        ],
        orelse=[],
    ),
]
