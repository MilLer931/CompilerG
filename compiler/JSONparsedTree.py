import json


class Node:
    def __init__(self, arg_name, arg_children=None):
        self.text = arg_name
        if arg_children is None:
            self.children = []
        else:
            self.children = arg_children


# class Node:
#     def __init__(self, arg_name, arg_children=None):
#         self.text = {"name": arg_name}
#         if arg_children is None:
#             self.children = []
#         else:
#             self.children = arg_children


class ParsedTree:
    def __init__(self, root: Node):
        self.nodeStructure = root


class Wrapper:
    def __init__(self, arg):
        self.JSONParsedTree = arg


def serialize(obj):
    """JSON serializer for objects not serializable by default json code"""
    try:
        # return obj.__dict__
        if type(obj) == ParsedTree:
            return {'parsedTree': obj.nodeStructure}
        return {str(obj.text): obj.children}
    except AttributeError:
        return None


def write(root: Node, filename: str):
    data = json.dumps(ParsedTree(root), default=serialize)
    with open('treant-js-master/%s.json' % filename, 'w') as f:
        f.write(data)
