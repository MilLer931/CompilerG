from compiler.lexer import Lexer
from compiler.parser import Parser, ParserState
from compiler.JSONparsedTree import Node, write
from compiler.codegen import CodeGen
from rply.lexer import LexerStream
from copy import copy
from pprint import pprint
import traceback
import json

input_file = open('start.code').read()

lexer = Lexer().build()
tokens: LexerStream
try:
    tokens = lexer.lex(input_file)
    tokenType = map(lambda x: x.gettokentype(), copy(tokens))
    tokenName = map(lambda x: x.getstr(), copy(tokens))
    pprint(list(copy(tokens)))
except (BaseException, Exception):
    traceback.print_exc()
finally:
    print("\n\nCompile log:")

codegen = CodeGen()
module = codegen.module
builder = codegen.builder
printf = codegen.printf

SymbolTable = ParserState()
syntaxRoot: Node
semanticRoot = Node("global")
has_errors = False
try:
    Parser(module, builder, printf).build().parse(copy(tokens), state=SymbolTable).eval(semanticRoot)
except (BaseException, Exception) as e:
    # traceback.print_exc()
    print('Error occurred: %s' % e)
    has_errors = True
finally:
    write(semanticRoot, "SemanticAnalyzer")

    codegen.create_ir()
    codegen.save_ir("finalcode.ll")

    if not has_errors:
        print('Compile complete without errors')
    else:
        print('Compile complete with errors!')
    print("\n\nТаблица символов:\nName\t|\tType\t|\tFeature")
    for m in SymbolTable.variables.keys():
        for v in SymbolTable.variables[m].keys():
            if v != 'args':
                print('%s\t|\t%s\t|\t%s' % (v, SymbolTable.variables[m][v]['type'], m._name))
    for v in SymbolTable.functions.keys():
        print('%s\t|\t%s\t|\t-' % (v, SymbolTable.functions[v].typ))

    # with open('treant-js-master/SemanticAnalyzer.json', 'r') as file:
    #     print(json.dumps(json.loads(file.read()), sort_keys=False, indent=4))
