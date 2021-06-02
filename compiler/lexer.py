from rply import LexerGenerator


class Lexer:
    def __init__(self):
        self.lexer = LexerGenerator()
        self.__add_tokens()

    def __add_tokens(self):
        # Types
        self.lexer.add('FLOAT', r'-?\d{1,}\.\d{1,}')
        self.lexer.add('INTEGER', r'-?\d{1,}')
        self.lexer.add('STRING', r'("[^"]*")|(\'[^\']*\')')
        # Mathematical Operators
        self.lexer.add('SUM', r'\+')
        self.lexer.add('SUB', r'\-')
        self.lexer.add('MUL', r'\*')
        self.lexer.add('DIV', r'\/')
        # Binary Operator
        self.lexer.add('AND', r'and(?!\w)')
        self.lexer.add('OR', r'or(?!\w)')
        self.lexer.add('EQUAL', r'\=\=')
        self.lexer.add('NEQUAL', r'\!\=')
        self.lexer.add('GTEQUAL', r'\>\=')
        self.lexer.add('LTEQUAL', r'\<\=')
        self.lexer.add('GREATER', r'\>')
        self.lexer.add('LESS', r'\<')
        self.lexer.add('ASSIGNMENT', r'\=')
        # Statement
        self.lexer.add('IF', r'if(?!\w)')
        self.lexer.add('ELSE', r'else(?!\w)')
        self.lexer.add('NOT', r'not(?!\w)')
        self.lexer.add('WHILE', r'while(?!\w)')
        self.lexer.add('BREAK', r'break(?!\w)')
        self.lexer.add('CONTINUE', r'continue(?!\w)')
        # Semi Colon
        self.lexer.add('SEMI_COLON', r'\;')
        self.lexer.add('COMMA', r'\,')
        # Parenthesis
        self.lexer.add('OPEN_BRACE', r'\(')
        self.lexer.add('CLOSE_BRACE', r'\)')
        self.lexer.add('OPEN_PAREN', r'\{')
        self.lexer.add('CLOSE_PAREN', r'\}')
        # Function
        self.lexer.add('PRINT', r'print')
        self.lexer.add('FUNC', r'func(?!\w)')
        self.lexer.add('RETURN', r'return(?!\w)')
        # Assignment
        self.lexer.add('STR', r'str(?!\w)')
        self.lexer.add('INT', r'int(?!\w)')
        self.lexer.add('FLT', r'float(?!\w)')
        self.lexer.add('IDENTIFIER', "[a-zA-Z_][a-zA-Z0-9_]*")
        # Ignore spaces
        self.lexer.ignore('\/\/.*')
        self.lexer.ignore('\/[*](.|\n)*[*]\/')
        self.lexer.ignore('\s+')

    def build(self):
        return self.lexer.build()
