from rply.token import BaseBox
from compiler.JSONparsedTree import Node
from compiler.errors import *
from llvmlite import ir


global_fmt = 1
fnctns = {}


class Program(BaseBox):
    def __init__(self, statement, program, state, builder, module):
        self.state = state
        self.builder = builder
        self.module = module
        if type(program) is Program:
            self.statements = program.get_statements()
            self.statements.insert(0, statement)
        else:
            self.statements = [statement]

    def add_statement(self, statement):
        self.statements.insert(0, statement)

    def get_statements(self):
        return self.statements

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        result = None
        for i, statement in enumerate(self.statements):
            left = Node('statement_full')
            right = Node('program')
            if i == len(self.statements) - 1:
                node.children.extend([left])
            else:
                node.children.extend([left, right])
            node = right
            result = statement.eval(left, builder=builder)
        return result


class Block(BaseBox):
    def __init__(self, statement, block, state, builder, module):
        self.state = state
        self.builder = builder
        self.module = module
        if type(block) is Block:
            self.statements = block.get_statements()
            self.statements.insert(0, statement)
        else:
            self.statements = [statement]

    def add_statement(self, statement):
        self.statements.insert(0, statement)

    def get_statements(self):
        return self.statements

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        result = None
        for i, statement in enumerate(self.statements):
            left = Node('statement_full')
            right = Node('block')
            if i == len(self.statements) - 1:
                node.children.extend([left])
            else:
                node.children.extend([left, right])
            node = right

            result = statement.eval(left, builder=builder)
        return result


class Arg(BaseBox):
    def __init__(self, typ, name, state, builder, module):
        self.state = state
        self.builder = builder
        self.module = module
        self.typ = typ
        self.name = name

    def eval(self, node):
        node.children.extend([Node('type', self.typ), Node('IDENTIFIER', self.name)])
        return self


class Args(BaseBox):
    def __init__(self, arg, args, state, builder, module):
        self.state = state
        self.builder = builder
        self.module = module
        self.arg = arg

        if type(args) is Args:
            self.args = args.get_args()
            self.args.insert(0, arg)
        else:
            self.args = [self.arg]

    def add_arg(self, arg):
        self.args.insert(0, arg)

    def get_args(self):
        return self.args

    def eval(self, node):
        for i, statement in enumerate(self.args):
            left = Node('arg')
            right = Node('args')
            if i == len(self.args) - 1:
                node.children.extend([left])
            else:
                node.children.extend([left, right])
            node = right

        return self.args


class ArgsCall(BaseBox):
    def __init__(self, arg, args, state, builder, module):
        self.state = state
        self.builder = builder
        self.module = module
        self.arg = arg

        if type(args) is ArgsCall:
            self.args = args.get_args()
            self.args.insert(0, arg)
        else:
            self.args = [self.arg]

    def add_arg(self, arg):
        self.args.insert(0, arg)

    def get_args(self):
        return self.args

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder
        # print(self.args)
        args = []
        for i, statement in enumerate(self.args):
            left = Node('expression')
            right = Node('args_call')
            if i == len(self.args) - 1:
                node.children.extend([left])
            else:
                node.children.extend([left, right])
            node = right
            args.append(statement.eval(left, builder=builder))

        return args


class If(BaseBox):
    def __init__(self, condition, body, builder, module, else_body=None, state=None):
        self.condition = condition
        self.body = body
        self.else_body = else_body
        self.state = state
        self.builder = builder
        self.module = module

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        expression = Node("expression")
        node.children.extend([Node("IF"), Node("("), expression, Node(")")])
        condition = self.condition.eval(expression, builder=builder)
        block = Node("block")
        node.children.extend([Node("{"), block, Node("}")])
        else_block = Node("block")

        if self.else_body is not None:
            with builder.if_else(condition) as (then, otherwise):
                with then:
                    self.body.eval(block, builder=builder)
                with otherwise:
                    self.else_body.eval(else_block, builder=builder)
        else:
            with builder.if_else(condition) as (then, otherwise):
                with then:
                    self.body.eval(block, builder=builder)
                with otherwise:
                    pass
        if self.else_body is not None:
            node.children.extend([Node("else"), Node("{"), else_block, Node("}")])
        # if bool(condition) is True:
        #     return self.body.eval(block)
        # else:
        #     if self.else_body is not None:
        #         return self.else_body.eval(else_block)
        return None


class While(BaseBox):
    def __init__(self, condition, body, builder, module, state=None):
        self.condition = condition
        self.body = body
        self.state = state
        self.builder = builder
        self.module = module

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        expression = Node("expression")
        node.children.extend([Node("WHILE"), Node("("), expression, Node(")")])
        condition = self.condition.eval(expression, builder=builder)
        block = Node("block")
        node.children.extend([Node("{"), block, Node("}")])
        tmp = None

        while_block = builder.append_basic_block('while')
        self.state.while_body.append(while_block)
        while_block_end = builder.append_basic_block('while_end')
        self.state.while_end.append(while_block_end)

        builder.cbranch(condition, while_block, while_block_end)

        builder.position_at_end(while_block)
        self.body.eval(block, builder=builder)
        condition = self.condition.eval(expression, builder=builder)
        builder.cbranch(condition, while_block, while_block_end)

        builder.position_at_end(while_block_end)
        self.state.while_body.pop()
        self.state.while_end.pop()

        if bool(condition) is True:
            return tmp
        else:
            pass
        return None


class Break(BaseBox):
    def __init__(self, builder, module, state=None):
        self.state = state
        self.builder = builder
        self.module = module

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        node.children.extend([Node("BREAK")])
        builder.branch(self.state.while_end[-1])


class Continue(BaseBox):
    def __init__(self, builder, module, state=None):
        self.state = state
        self.builder = builder
        self.module = module

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        node.children.extend([Node("BREAK")])
        builder.branch(self.state.while_body[-1])


class Variable(BaseBox):
    def __init__(self, name, state, builder, module):
        self.name = str(name)
        self.value = None
        self.state = state
        self.builder = builder
        self.module = module

    def get_name(self):
        return str(self.name)

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        identifier = Node("IDENTIFIER")
        # node.children.extend([identifier])
        v_name = self.name

        if v_name not in self.state.variables[builder.function].keys():
            raise LogicError("Unknown name: <%s> is not defined in function <%s>" % (str(self.name), builder.function._name))

        if self.state.variables[builder.function][v_name] is not None:
            self.value = self.state.variables[builder.function][v_name]['value']
            identifier.children.extend([Node(self.name, [Node(self.value)])])
            i = builder.load(self.state.variables[builder.function][v_name]['ptr'], v_name)
            return i
        identifier.children.extend([Node("Unknown name: <%s> is not defined" % str(self.name))])
        raise LogicError("Unknown name: <%s> is not defined in function <%s>" % (str(self.name), builder.function._name))

    def to_string(self):
        return str(self.name)


class FunctionDeclaration(BaseBox):
    def __init__(self, name, typ, args, block, state, builder, module):
        self.name = name
        self.args = args
        self.block = block
        self.typ = typ
        self.builder = builder
        self.module = module
        self.state = state
        state.functions[self.name] = self

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        identifier = Node(self.name)
        node.children.extend([Node("FUNCTION"), Node('type', self.typ), identifier, Node("{"), Node("block"), Node("}")])

        int_ = ir.IntType(32)
        flt_ = ir.FloatType()
        f_typ = {'int': int_, 'float': flt_}

        f_args = [f_typ[i.typ] for i in self.args.get_args()]

        fnty = ir.FunctionType(f_typ[self.typ], f_args)
        func = ir.Function(self.module, fnty, name=self.name)
        fnctns[self.name] = func
        block = func.append_basic_block(name="entry")
        f_builder = ir.IRBuilder(block)
        self.state.variables[f_builder.function] = {}
        self.state.variables[f_builder.function]['args'] = {}
        types_dict = {ir.IntType(32): 'INT', ir.FloatType(): 'FLT', str: 'STR'}
        for arg_ in range(len(func.args)):
            var_name = self.args.get_args()[arg_].name.getstr()
            var_type = func.args[arg_].type
            alloc = f_builder.alloca(var_type, size=None, name=var_name)
            if f_builder.function not in self.state.variables.keys():
                self.state.variables[f_builder.function] = {}
            self.state.variables[f_builder.function][var_name] = {'value': func.args[arg_], 'type': types_dict[var_type], 'ptr': alloc}
            f_builder.store(func.args[arg_], alloc)

        # print(self.state.variables)

        self.block.eval(Node('block'), builder=f_builder)
        # a, b = func.args
        # result = f_builder.sub(a, b, name="res")
        # f_builder.ret(ir.Constant(ir.IntType(8), 1))
        # print(f_builder.function)

        # print(self.args.get_args()[0].typ)
        # return self


class CallFunction(BaseBox):
    def __init__(self, name, args, state, builder, module):
        self.name = name
        self.args = args
        self.state = state
        self.builder = builder
        self.module = module

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        identifier = Node(self.name + " ( )")
        node.children.extend([identifier])

        args_eval = self.args.eval(Node('args_call'), builder=builder)
        res = builder.call(fnctns[self.name], args_eval)
        return res

        # return self.state.functions[self.name].block.eval(identifier, builder=builder)

    def to_string(self):
        return "<call '%s'>" % self.name


class Return(BaseBox):
    def __init__(self, builder, module, statement, state=None):
        self.state = state
        self.builder = builder
        self.module = module
        self.statement = statement

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        statement = Node('expression', self.statement)
        node.children.extend([Node("RETURN"), Node('('), statement, Node(')')])
        if self.statement is None:
            builder.ret_void()
        else:
            builder.ret(self.statement.eval(statement, builder=builder))


class BaseFunction(BaseBox):
    def __init__(self, expression, state):
        self.expression = expression
        self.value = None
        self.state = state
        self.roundOffDigits = 10

    def eval(self, node):
        raise NotImplementedError("This is abstract method from abstract class BaseFunction(BaseBox){...} !")

    def to_string(self):
        return str(self.value)


class Constant(BaseBox):
    def __init__(self, state, builder, module):
        self.value = None
        self.state = state
        self.builder = builder
        self.module = module

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        value = Node(self.value)
        typed = Node(self.__class__.__name__.upper(), [value])
        constant = Node("const", [typed])
        # node.children.extend([constant])
        if self.__class__.__name__.upper() == 'INTEGER':
            i = ir.Constant(ir.IntType(32), int(self.value))
            return i
        elif self.__class__.__name__.upper() == 'FLOAT':
            i = ir.Constant(ir.FloatType(), float(self.value))
            return i
        return self.value

    def to_string(self):
        return str(self.value)


class Integer(Constant):
    def __init__(self, value, state, builder, module):
        super().__init__(state, builder, module)
        self.value = int(value)

    def to_string(self):
        return str(self.value)


class Float(Constant):
    def __init__(self, value, state, builder, module):
        super().__init__(state, builder, module)
        self.value = float(value)

    def to_string(self):
        return str(self.value)


class String(Constant):
    def __init__(self, value, state, builder, module):
        super().__init__(state, builder, module)
        self.value = str(value)

    def to_string(self):
        return '"%s"' % str(self.value)


class BinaryOp(BaseBox):
    def __init__(self, left, right, state, builder, module):
        self.left = left
        self.right = right
        self.state = state
        self.module = module
        self.builder = builder


class Assignment(BinaryOp):
    def __init__(self, left, right, state, builder, module, new=True, type_='INT'):
        super().__init__(left, right, state, builder, module)
        self.new = new
        self.type_ = type_

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        if isinstance(self.left, Variable):
            var_name = self.left.get_name()
            types_dict = {ir.IntType(32): 'INT', ir.FloatType(): 'FLT', str: 'STR'}
            if self.new:
                if builder.function not in self.state.variables.keys():
                    self.state.variables[builder.function] = {}
                if var_name not in self.state.variables[builder.function].keys():
                    identifier = Node("IDENTIFIER", [Node(var_name)])
                    expression = Node("expression")
                    node.children.extend([Node("LET"), identifier, Node("="), expression])
                    tmp_eval = self.right.eval(expression, builder=builder)
                    if types_dict[tmp_eval.type] != self.type_:
                        raise LogicError('Cannot assign <%s> to <%s>-type variable' %
                                         (types_dict[type(tmp_eval)], self.type_))
                    alloc = builder.alloca(tmp_eval.type, size=None, name=var_name)
                    if builder.function not in self.state.variables.keys():
                        self.state.variables[builder.function] = {}
                    self.state.variables[builder.function][var_name] = {'value': tmp_eval, 'type': self.type_, 'ptr': alloc}
                    builder.store(tmp_eval, alloc)
                    return self.state.variables
                else:
                    raise ImmutableError(var_name)
            else:
                if var_name in self.state.variables[builder.function].keys():
                    identifier = Node("IDENTIFIER", [Node(var_name)])
                    expression = Node("expression")
                    node.children.extend([identifier, Node("="), expression])
                    tmp_eval = self.right.eval(expression, builder=builder)
                    # print(type(tmp_eval))
                    # if types_dict[type(tmp_eval)] != self.state.variables[var_name]['type']:
                    #     raise LogicError('Cannot assign <%s> to <%s>-type variable' %
                    #                      (types_dict[type(tmp_eval)], self.type_))
                    self.state.variables[builder.function][var_name]['value'] = tmp_eval
                    alloc = self.state.variables[builder.function][var_name]['ptr']
                    builder.store(tmp_eval, alloc)
                    return self.state.variables
                else:
                    raise LogicError("Variable <%s> is not defined" % var_name)

        else:
            raise LogicError("Cannot assign to <%s>" % self)


class Sum(BinaryOp):
    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        left = Node("expression")
        right = Node("expression")
        # node.children.extend([left, Node("+"), right])
        eval_left = self.left.eval(left, builder=builder)
        eval_right = self.right.eval(right, builder=builder)
        if eval_left.type == ir.FloatType():
            i = builder.fadd(eval_left, eval_right)
        else:
            i = builder.add(eval_left, eval_right)
        return i


class Sub(BinaryOp):
    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        left = Node("expression")
        right = Node("expression")
        node.children.extend([left, Node("-"), right])
        eval_left = self.left.eval(left, builder=builder)
        eval_right = self.right.eval(right, builder=builder)
        if eval_left.type == ir.FloatType():
            i = builder.fsub(eval_left, eval_right)
        else:
            i = builder.sub(eval_left, eval_right)
        return i


class Mul(BinaryOp):
    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        left = Node("expression")
        right = Node("expression")
        node.children.extend([left, Node("*"), right])
        eval_left = self.left.eval(left, builder=builder)
        eval_right = self.right.eval(right, builder=builder)
        if eval_left.type == ir.FloatType():
            i = builder.fmul(eval_left, eval_right)
        else:
            i = builder.mul(eval_left, eval_right)
        return i


class Div(BinaryOp):
    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        left = Node("expression")
        right = Node("expression")
        node.children.extend([left, Node("/"), right])
        eval_left = self.left.eval(left, builder=builder)
        eval_right = self.right.eval(right, builder=builder)
        if eval_left.type == ir.FloatType():
            i = builder.fdiv(eval_left, eval_right)
        else:
            i = builder.sdiv(eval_left, eval_right)
        return i


class Additive(BaseBox):
    def __init__(self, expression, state, builder, module):
        self.value = expression
        self.state = state
        self.builder = builder
        self.module = module

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        right = Node("expression")
        node.children.extend([Node("-"), right])
        eval_right = self.value.eval(right, builder=builder)
        if eval_right.type == ir.FloatType():
            i = builder.fsub(ir.Constant(ir.FloatType(), 0.0), eval_right)
        else:
            i = builder.mul(ir.Constant(ir.IntType(32), -1), eval_right)
        return i


class Equal(BinaryOp):
    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        left = Node("expression")
        right = Node("expression")
        node.children.extend([left, Node("=="), right])
        eval_left = self.left.eval(left, builder=builder)
        eval_right = self.right.eval(right, builder=builder)
        if eval_left.type == ir.FloatType():
            i = builder.fcmp_ordered('==', eval_left, eval_right)
        else:
            i = builder.icmp_signed('==', eval_left, eval_right)
        return i


class NotEqual(BinaryOp):
    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        left = Node("expression")
        right = Node("expression")
        node.children.extend([left, Node("!="), right])
        eval_left = self.left.eval(left, builder=builder)
        eval_right = self.right.eval(right, builder=builder)
        if eval_left.type == ir.FloatType():
            i = builder.fcmp_ordered('!=', eval_left, eval_right)
        else:
            i = builder.icmp_signed('!=', eval_left, eval_right)
        return i


class GreaterThan(BinaryOp):
    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        left = Node("expression")
        right = Node("expression")
        node.children.extend([left, Node(">"), right])
        eval_left = self.left.eval(left, builder=builder)
        eval_right = self.right.eval(right, builder=builder)
        if eval_left.type == ir.FloatType():
            i = builder.fcmp_ordered('>', eval_left, eval_right)
        else:
            i = builder.icmp_signed('>', eval_left, eval_right)
        return i


class LessThan(BinaryOp):
    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        left = Node("expression")
        right = Node("expression")
        node.children.extend([left, Node("<"), right])
        eval_left = self.left.eval(left, builder=builder)
        eval_right = self.right.eval(right, builder=builder)
        if eval_left.type == ir.FloatType():
            i = builder.fcmp_ordered('<', eval_left, eval_right)
        else:
            i = builder.icmp_signed('<', eval_left, eval_right)
        return i


class GreaterThanEqual(BinaryOp):
    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        left = Node("expression")
        right = Node("expression")
        node.children.extend([left, Node(">="), right])
        eval_left = self.left.eval(left, builder=builder)
        eval_right = self.right.eval(right, builder=builder)
        if eval_left.type == ir.FloatType():
            i = builder.fcmp_ordered('>=', eval_left, eval_right)
        else:
            i = builder.icmp_signed('>=', eval_left, eval_right)
        return i


class LessThanEqual(BinaryOp):
    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        left = Node("expression")
        right = Node("expression")
        node.children.extend([left, Node("<="), right])
        eval_left = self.left.eval(left, builder=builder)
        eval_right = self.right.eval(right, builder=builder)
        if eval_left.type == ir.FloatType():
            i = builder.fcmp_ordered('<=', eval_left, eval_right)
        else:
            i = builder.icmp_signed('<=', eval_left, eval_right)
        return i


class And(BinaryOp):
    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        left = Node("expression")
        right = Node("expression")
        node.children.extend([left, Node("and"), right])
        left_eval = self.left.eval(left, builder=builder)
        right_eval = self.right.eval(right, builder=builder)
        i = builder.and_(left_eval, right_eval)
        return i


class Or(BinaryOp):
    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        left = Node("expression")
        right = Node("expression")
        node.children.extend([left, Node("or"), right])
        left_eval = self.left.eval(left, builder=builder)
        right_eval = self.right.eval(right, builder=builder)
        i = builder.or_(left_eval, right_eval)
        return i


class Not(BaseBox):
    def __init__(self, expression, state, builder, module):
        self.value = expression
        self.state = state
        self.builder = builder
        self.module = module

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        expression = Node("expression")
        node.children.extend([Node("Not"), expression])
        self.value = self.value.eval(expression, builder=builder)
        i = builder.not_(self.value)
        return i


class Print(BaseBox):
    def __init__(self, builder, module, printf, expression=None, state=None):
        self.builder = builder
        self.module = module
        self.printf = printf
        self.value = expression
        self.state = state

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        node.children.extend([Node("PRINT"), Node("(")])
        if self.value is None:
            print()
        else:
            expression = Node("expression")
            node.children.extend([expression])
            value = self.value.eval(expression, builder=builder)

            voidptr_ty = ir.IntType(32).as_pointer()
            fmt_arg = builder.bitcast(global_fmt, voidptr_ty)

            builder.call(self.printf, [fmt_arg, value])
        node.children.extend([Node(")")])


class Input(BaseBox):
    def __init__(self, expression=None, state=None):
        self.value = expression
        self.state = state

    def eval(self, node):
        node.children.extend([Node("CONSOLE_INPUT"), Node("(")])
        if self.value is None:
            result = input()
        else:
            expression = Node("expression")
            node.children.extend([expression])
            result = input(self.value.eval(expression))
        node.children.extend([Node(")")])
        import re as regex
        if regex.search('^-?\d+(\.\d+)?$', str(result)):
            return float(result)
        else:
            return str(result)


class Main(BaseBox):
    def __init__(self, program, builder, module):
        global global_fmt
        self.program = program
        self.builder = builder
        self.module = module

        # import printf func
        fmt = "%i \n\0"
        c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)),
                            bytearray(fmt.encode("utf8")))
        global_fmt = ir.GlobalVariable(self.module, c_fmt.type, name="fstr")
        global_fmt.linkage = 'internal'
        global_fmt.global_constant = True
        global_fmt.initializer = c_fmt

        flt = ir.FloatType()
        int_ = ir.IntType(32)

        fnty = ir.FunctionType(int_, (int_, int_))
        func = ir.Function(self.module, fnty, name="sum")
        fnctns['sum'] = func
        block = func.append_basic_block(name="entry")
        f_builder = ir.IRBuilder(block)
        a, b = func.args
        result = f_builder.add(a, b, name="res")
        f_builder.ret(result)

        fnty = ir.FunctionType(flt, (flt, flt))
        func = ir.Function(self.module, fnty, name="fsum")
        fnctns['sumf'] = func
        block = func.append_basic_block(name="entry")
        f_builder = ir.IRBuilder(block)
        a, b = func.args
        result = f_builder.fadd(a, b, name="res")
        f_builder.ret(result)

        fnty = ir.FunctionType(int_, (int_, int_))
        func = ir.Function(self.module, fnty, name="sub")
        fnctns['sub'] = func
        block = func.append_basic_block(name="entry")
        f_builder = ir.IRBuilder(block)
        a, b = func.args
        result = f_builder.sub(a, b, name="res")
        f_builder.ret(result)

        fnty = ir.FunctionType(flt, (flt, flt))
        func = ir.Function(self.module, fnty, name="fsub")
        fnctns['subf'] = func
        block = func.append_basic_block(name="entry")
        f_builder = ir.IRBuilder(block)
        a, b = func.args
        result = f_builder.fsub(a, b, name="res")
        f_builder.ret(result)

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        program = Node("program")
        node.children.extend([program])
        return self.program.eval(program, builder=builder)


class ExpressParenthesis(BaseBox):
    def __init__(self, expression, builder, module):
        self.expression = expression
        self.builder = builder
        self.module = module

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        expression = Node("expression")
        node.children.extend([Node("("), expression, Node(")")])
        return self.expression.eval(expression, builder=builder)


class StatementFull(BaseBox):
    def __init__(self, statement, builder, module):
        self.statement = statement
        self.builder = builder
        self.module = module

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        statement = Node("statement")
        node.children.extend([statement, Node(";")])
        return self.statement.eval(statement, builder=builder)


class Statement(BaseBox):
    def __init__(self, expression, builder, module):
        self.expression = expression
        self.builder = builder
        self.module = module

    def eval(self, node, builder=None):
        if builder is None:
            builder = self.builder

        expression = Node("expression")
        node.children.extend([expression])
        return self.expression.eval(expression, builder=builder)
