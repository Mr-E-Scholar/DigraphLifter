import ast
import astor

file_path = 'digraphLifter.py'
with open(file_path, 'r') as file:
    source_code = file.read()
tree = ast.parse(source_code)
pretty_ast = astor.dump_tree(tree)
print(pretty_ast)
# source_from_ast = astor.to_source(tree)
# print(source_from_ast)
