from rply import ParserGenerator
from compiler.JSONparsedTree import Node
from compiler.AbstractSyntaxTree import *
from compiler.errors import *


class ParserState(object):
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.while_body = []
        self.while_end = []


class Parser:
    def __init__(self, module, builder, printf, syntax=False):
        self.pg = ParserGenerator(
            ['INTEGER', 'FLOAT',
             'OPEN_BRACE', 'CLOSE_BRACE', 'COMMA', 'SEMI_COLON', 'OPEN_PAREN', 'CLOSE_PAREN',
             'AND', 'OR', 'NOT',
             'IF', 'ELSE',
             'WHILE', 'BREAK', 'CONTINUE',
             'INT', 'FLT', 'STR',
             'ASSIGNMENT', 'EQUAL', 'NEQUAL', 'GTEQUAL', 'GREATER', 'LESS', 'LTEQUAL',
             'SUM', 'SUB', 'MUL', 'DIV', 'IDENTIFIER', 'PRINT',
             'FUNC', 'RETURN',
             ],
            precedence=(
                ('left', ['FUNC']),
                ('left', ['LET']),
                ('left', ['ASSIGNMENT']),
                ('left', ['IF', 'ELSE', 'SEMI_COLON']),
                ('left', ['AND', 'OR']),
                ('left', ['NOT']),
                ('left', ['EQUAL', 'NEQUAL', 'GTEQUAL', 'GREATER', 'LESS', 'LTEQUAL']),
                ('left', ['SUM', 'SUB']),
                ('left', ['MUL', 'DIV']),
                ('left', ['STRING', 'INTEGER', 'FLOAT', 'BOOLEAN'])
            )
        )
        self.builder = builder
        self.module = module
        self.printf = printf
        self.syntax = syntax
        self.parse()

    def parse(self):

        @self.pg.production("global : program")
        def main_program(state, p):
            if self.syntax is True:
                return [Node("program", p[0])]
            return Main(p[0], self.builder, self.module)

        @self.pg.production('program : statement_full')
        def program_statement(state, p):
            if self.syntax is True:
                return [Node("statement_full", p[0])]
            return Program(p[0], None, state, builder=self.builder, module=self.module)

        @self.pg.production('program : statement_full program')
        def program_statement_program(state, p):
            if self.syntax is True:
                return [Node("statement_full", p[0]), Node("program", p[1])]
            return Program(p[0], p[1], state, builder=self.builder, module=self.module)

        @self.pg.production('expression : OPEN_BRACE expression CLOSE_BRACE')
        def expression_parenthesis(state, p):
            if self.syntax is True:
                return [Node("("), Node("expression", p[1]), Node(")")]
            return ExpressParenthesis(p[1], self.builder, self.module)

        @self.pg.production('statement_full : IF OPEN_BRACE expression CLOSE_BRACE OPEN_PAREN block CLOSE_PAREN')
        def expression_if(state, p):
            if self.syntax is True:
                return [Node("IF"), Node("("), Node("expression", p[2]), Node(")"), Node("{"), Node("block", p[5]), Node("}")]
            return If(p[2], p[5], self.builder, self.module, state=state)

        @self.pg.production('statement_full : IF OPEN_BRACE expression CLOSE_BRACE OPEN_PAREN block CLOSE_PAREN ELSE OPEN_PAREN block CLOSE_PAREN')
        def expression_if_else(state, p):
            if self.syntax is True:
                return [Node("IF"), Node("("), Node("expression", p[2]), Node(")"), Node("{"), Node("block", p[5]), Node("}"), Node("ELSE"), Node("{"),
                        Node("block", p[9]), Node("}")]
            return If(p[2], p[5], self.builder, self.module, else_body=p[9], state=state)

        @self.pg.production('block : statement_full')
        def block_expr(state, p):
            if self.syntax is True:
                return [Node("statement_full", p[0])]
            return Block(p[0], None, state, self.builder, self.module)

        @self.pg.production('block : statement_full block')
        def block_expr_block(state, p):
            if self.syntax is True:
                return [Node("statement_full", p[0]), Node("block", p[1])]
            return Block(p[0], p[1], state, self.builder, self.module)

        @self.pg.production('statement_full : WHILE OPEN_BRACE expression CLOSE_BRACE OPEN_PAREN block CLOSE_PAREN')
        def expression_while(state, p):
            if self.syntax is True:
                return [Node("WHILE"), Node("("), Node("expression", p[2]), Node(")"), Node("{"), Node("block", p[5]),
                        Node("}")]
            return While(p[2], p[5], self.builder, self.module, state=state)

        @self.pg.production('statement_full : statement SEMI_COLON')
        def statement_full(state, p):
            if self.syntax is True:
                return [Node("statement", p[0]), Node(";")]
            return StatementFull(p[0], self.builder, self.module)

        @self.pg.production('statement : expression')
        def statement_expr(state, p):
            if self.syntax is True:
                return [Node("expression", p[0])]
            return Statement(p[0], builder=self.builder, module=self.module)

        @self.pg.production('statement : BREAK')
        def statement_break(state, p):
            if self.syntax is True:
                return [Node("BREAK", p[0])]
            return Break(self.builder, self.module, state=state)

        @self.pg.production('statement : CONTINUE')
        def statement_continue(state, p):
            if self.syntax is True:
                return [Node("CONTINUE", p[0])]
            return Continue(self.builder, self.module, state=state)

        @self.pg.production('statement : RETURN')
        def statement_return_void(state, p):
            if self.syntax is True:
                return [Node("RETURN", p[0])]
            return Return(self.builder, self.module, statement=None, state=state)

        @self.pg.production('statement : RETURN OPEN_BRACE expression CLOSE_BRACE')
        def statement_return(state, p):
            if self.syntax is True:
                return [Node("RETURN", p[0]), Node('('), Node('expression', p[2]), Node(')')]
            return Return(self.builder, self.module, statement=p[2], state=state)

        @self.pg.production('statement : INT IDENTIFIER ASSIGNMENT expression')
        @self.pg.production('statement : FLT IDENTIFIER ASSIGNMENT expression')
        def statement_assignment_type(state, p):
            if p[0].gettokentype() == 'INT':
                if self.syntax is True:
                    return [Node("INT"), Node("IDENTIFIER", p[1]), Node("="), Node("expression", p[3])]
            elif p[0].gettokentype() == 'FLT':
                if self.syntax is True:
                    return [Node("FLT"), Node("IDENTIFIER", p[1]), Node("="), Node("expression", p[3])]
            return Assignment(Variable(p[1].getstr(), state, self.builder, self.module), p[3], state, self.builder, self.module, type_=p[0].gettokentype())

        @self.pg.production('statement : IDENTIFIER ASSIGNMENT expression')
        def statement_assignment(state, p):
            if self.syntax is True:
                return [Node("IDENTIFIER", p[0]), Node("="), Node("expression", p[2])]
            return Assignment(Variable(p[0].getstr(), state, self.builder, self.module), p[2], state, self.builder, self.module, new=False)

        @self.pg.production('statement_full : FUNC INT IDENTIFIER OPEN_BRACE args CLOSE_BRACE OPEN_PAREN block CLOSE_PAREN')
        @self.pg.production('statement_full : FUNC FLT IDENTIFIER OPEN_BRACE args CLOSE_BRACE OPEN_PAREN block CLOSE_PAREN')
        def statement_func(state, p):
            if p[1].gettokentype() == 'INT':
                if self.syntax is True:
                    return [Node("FUNCTION"), Node('INT'), Node("IDENTIFIER", p[2]), Node("("), Node("args", p[4]), Node(")"), Node("{"), Node("block", p[7]), Node("}")]
                return FunctionDeclaration(name=p[2].getstr(), typ=p[1].getstr(), args=p[4], block=p[7], state=state, builder=self.builder, module=self.module)
            else:
                if self.syntax is True:
                    return [Node("FUNCTION"), Node('FLT'), Node("IDENTIFIER", p[2]), Node("("), Node("args", p[4]), Node(")"), Node("{"), Node("block", p[7]), Node("}")]
                return FunctionDeclaration(name=p[2].getstr(), typ=p[1].getstr(), args=p[4], block=p[7], state=state, builder=self.builder, module=self.module)

        @self.pg.production('arg : INT IDENTIFIER')
        @self.pg.production('arg : FLT IDENTIFIER')
        def args_expr(state, p):
            if p[0].gettokentype() == 'INT':
                if self.syntax is True:
                    return [Node('INT'), Node("arg", p[1])]
            else:
                if self.syntax is True:
                    return [Node('FLT'), Node("arg", p[1])]
            return Arg(p[0].getstr(), p[1], state, self.builder, self.module)

        @self.pg.production('args : arg')
        def args_expr_args(state, p):
            if self.syntax is True:
                return [Node("arg", p[0])]
            return Args(p[0], None, state, self.builder, self.module)

        @self.pg.production('args : arg COMMA args')
        def args_expr_args(state, p):
            if self.syntax is True:
                return [Node("arg", p[0]), Node(','), Node("args", p[2])]
            return Args(p[0], p[2], state, self.builder, self.module)

        @self.pg.production('expression : NOT expression')
        def expression_not(state, p):
            if self.syntax is True:
                return [Node("NOT"), Node("expression", p[1])]
            return Not(p[1], state, self.builder, self.module)

        @self.pg.production('expression : expression SUM expression')
        @self.pg.production('expression : expression SUB expression')
        @self.pg.production('expression : expression MUL expression')
        @self.pg.production('expression : expression DIV expression')
        def expression_binary_operator(state, p):
            if p[1].gettokentype() == 'SUM':
                if self.syntax is True:
                    return [Node("expression", p[0]), Node("+"), Node("expression", p[2])]
                return Sum(p[0], p[2], state, self.builder, self.module)
            elif p[1].gettokentype() == 'SUB':
                if self.syntax is True:
                    return [Node("expression", p[0]), Node("-"), Node("expression", p[2])]
                return Sub(p[0], p[2], state, self.builder, self.module)
            elif p[1].gettokentype() == 'MUL':
                if self.syntax is True:
                    return [Node("expression", p[0]), Node("*"), Node("expression", p[2])]
                return Mul(p[0], p[2], state, self.builder, self.module)
            elif p[1].gettokentype() == 'DIV':
                if self.syntax is True:
                    return [Node("expression", p[0]), Node("/"), Node("expression", p[2])]
                return Div(p[0], p[2], state, self.builder, self.module)
            else:
                raise LogicError('Unknown operator: %s' % p[1].gettokentype())

        @self.pg.production('expression : SUB expression')
        def expression_minus(state, p):
            if self.syntax is True:
                return [Node("-"), Node("expression", p[0])]
            return Additive(p[1], state, self.builder, self.module)

        @self.pg.production('expression : expression NEQUAL expression')
        @self.pg.production('expression : expression EQUAL expression')
        @self.pg.production('expression : expression GTEQUAL expression')
        @self.pg.production('expression : expression LTEQUAL expression')
        @self.pg.production('expression : expression GREATER expression')
        @self.pg.production('expression : expression LESS expression')
        @self.pg.production('expression : expression AND expression')
        @self.pg.production('expression : expression OR expression')
        def expression_equality(state, p):
            if p[1].gettokentype() == 'EQUAL':
                if self.syntax is True:
                    return [Node("expression", p[0]), Node("=="), Node("expression", p[2])]
                return Equal(p[0], p[2], state, self.builder, self.module)
            elif p[1].gettokentype() == 'NEQUAL':
                if self.syntax is True:
                    return [Node("expression", p[0]), Node("!="), Node("expression", p[2])]
                return NotEqual(p[0], p[2], state, self.builder, self.module)
            elif p[1].gettokentype() == 'GTEQUAL':
                if self.syntax is True:
                    return [Node("expression", p[0]), Node(">="), Node("expression", p[2])]
                return GreaterThanEqual(p[0], p[2], state, self.builder, self.module)
            elif p[1].gettokentype() == 'LTEQUAL':
                if self.syntax is True:
                    return [Node("expression", p[0]), Node("<="), Node("expression", p[2])]
                return LessThanEqual(p[0], p[2], state, self.builder, self.module)
            elif p[1].gettokentype() == 'GREATER':
                if self.syntax is True:
                    return [Node("expression", p[0]), Node(">"), Node("expression", p[2])]
                return GreaterThan(p[0], p[2], state, self.builder, self.module)
            elif p[1].gettokentype() == 'LESS':
                if self.syntax is True:
                    return [Node("expression", p[0]), Node("<"), Node("expression", p[2])]
                return LessThan(p[0], p[2], state, self.builder, self.module)
            elif p[1].gettokentype() == 'AND':
                if self.syntax is True:
                    return [Node("expression", p[0]), Node("AND"), Node("expression", p[2])]
                return And(p[0], p[2], state, self.builder, self.module)
            elif p[1].gettokentype() == 'OR':
                if self.syntax is True:
                    return [Node("expression", p[0]), Node("OR"), Node("expression", p[2])]
                return Or(p[0], p[2], state, self.builder, self.module)
            else:
                raise LogicError("Unknown operator: %s" % p[1].gettokentype())

        @self.pg.production('statement : PRINT OPEN_BRACE CLOSE_BRACE')
        def program(state, p):
            if self.syntax is True:
                return [Node("PRINT"), Node("("), Node(")")]
            return Print(self.builder, self.module, self.printf)

        @self.pg.production('statement : PRINT OPEN_BRACE expression CLOSE_BRACE')
        def program(state, p):
            if self.syntax is True:
                return [Node("PRINT"), Node("("), Node("expression", p[2]), Node(")")]
            return Print(self.builder, self.module, self.printf, expression=p[2], state=state)

        @self.pg.production('expression : IDENTIFIER')
        def expression_variable(state, p):
            if self.syntax is True:
                return [Node("IDENTIFIER", p[0].getstr())]
            return Variable(p[0].getstr(), state, self.builder, self.module)

        @self.pg.production('expression : IDENTIFIER OPEN_BRACE CLOSE_BRACE')
        @self.pg.production('expression : IDENTIFIER OPEN_BRACE args_call CLOSE_BRACE')
        def expression_call(state, p):
            if p[2] == 'CLOSE_BRACE':
                if self.syntax is True:
                    return [Node("IDENTIFIER", p[0]), Node("("), Node("args_call", p[2]), Node(")")]
                return CallFunction(name=p[0].getstr(), args=None, state=state, builder=self.builder, module=self.module)
            else:
                if self.syntax is True:
                    return [Node("IDENTIFIER", p[0]), Node("("), Node("args_call", p[2]), Node(")")]
                return CallFunction(name=p[0].getstr(), args=p[2], state=state, builder=self.builder, module=self.module)

        @self.pg.production('args_call : expression')
        def args_expr_args(state, p):
            if self.syntax is True:
                return [Node("arg_call", p[0])]
            return ArgsCall(p[0], None, state, self.builder, self.module)

        @self.pg.production('args_call : expression COMMA args_call')
        def args_expr_args(state, p):
            if self.syntax is True:
                return [Node("arg_call", p[0]), Node(','), Node("args_call", p[2])]
            return ArgsCall(p[0], p[2], state, self.builder, self.module)

        @self.pg.production('expression : const')
        def expression_const(state, p):
            if self.syntax is True:
                return [Node("const", p[0])]
            return p[0]

        @self.pg.production('const : FLOAT')
        def constant_float(state, p):
            if self.syntax is True:
                return [Node("FLOAT", p[0])]
            return Float(p[0].getstr(), state, self.builder, self.module)

        @self.pg.production('const : INTEGER')
        def constant_integer(state, p):
            if self.syntax is True:
                return [Node("INTEGER", p[0])]
            return Integer(p[0].getstr(), state, self.builder, self.module)

        @self.pg.error
        def error_handle(state, token):
            raise ValueError(token)

    def build(self):
        return self.pg.build()
