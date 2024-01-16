import multiprocessing
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

class DigraphLifter:
    def __init__(self, k, M=2, a=1, b=1, method='numpy'):
        self.k = k
        self.M = M
        self.a = a
        self.b = b
        self.method = method
        self.base_digraph = self.create_base_digraph()
        self.lifted_graph = self.lift_digraph()

    def create_base_digraph(self):
        G = nx.DiGraph()
        if self.method == 'numpy':
            edges = np.array([[i % self.k, (i + 1) % self.k] for i in range(self.k)])
            for edge in edges:
                G.add_edge(edge[0], edge[1])
        elif self.method == 'pandas':
            edges = pd.DataFrame({'from': [i % self.k for i in range(self.k)], 
                                  'to': [(i + 1) % self.k for i in range(self.k)]})
            for index, row in edges.iterrows():
                G.add_edge(row['from'], row['to'])
        return G

    def lift_digraph(self):
        new_graph = nx.DiGraph()
        for node in self.base_digraph.nodes():
            for i in range(self.M):
                new_graph.add_node((node, i))

        for u, v in self.base_digraph.edges():
            for i in range(self.M):
                new_graph.add_edge((u, i), (v, (i + self.a) % self.M))
                new_graph.add_edge((u, i), (v, (i + self.b) % self.M))
        return new_graph
    
    def find_cycles(self, graph):
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                smallest_cycle = min(cycles, key=len)
                largest_cycle = max(cycles, key=len)
                return smallest_cycle, len(smallest_cycle), largest_cycle, len(largest_cycle)
            else:
                return "No cycle", 0, "No cycle", 0
        except Exception as e:
            return f"Error in finding cycles: {e}", -1, f"Error in finding cycles: {e}", -1

    def draw_digraphs(self):
        fig = plt.figure(figsize=(12, 6))
        # Base Digraph
        plt.subplot(1, 2, 1)
        nx.draw(self.base_digraph, with_labels=True, node_color='lightblue', node_size=2000, arrowstyle='->', arrowsize=20)
        smallest_cycle_base, length_smallest_base, largest_cycle_base, length_largest_base = self.find_cycles(self.base_digraph)
        plt.title(f"Base Digraph\nSmallest cycle: {length_smallest_base}\nLargest cycle: {length_largest_base}")
        # Lifted Digraph
        plt.subplot(1, 2, 2)
        nx.draw(self.lifted_graph, with_labels=True, node_color='lightgreen', node_size=2000, arrowstyle='->', arrowsize=20)
        smallest_cycle_lifted, length_cycle_lifted, largest_cycle_lifted, length_largest_lifted = self.find_cycles(self.lifted_graph)
        plt.title(f"Lifted Digraph\nSmallest cycle: {length_cycle_lifted}\nLargest cycle: {length_largest_lifted}")
        return fig
        
def numpy_graph(k, M, a, b):
    start = datetime.now()
    for i in range(50):
        # graph_transformer = DigraphLifter(k, M, a, b)
        graph_transformer = DigraphLifter(k, M, a, b, method='numpy')
        fig = graph_transformer.draw_digraphs()
        fig.suptitle(f"Numpy Approach {i+1}", fontsize=16)
    end = datetime.now()
    print(f"NumPy Execution Time: {end - start}")
    plt.show()

def pandas_graph(k, M, a, b):
    start = datetime.now()
    for i in range(50):
        # graph_transformer = DigraphLifter(k, M, a, b)
        graph_transformer = DigraphLifter(k, M, a, b, method='pandas')
        fig = graph_transformer.draw_digraphs()
        fig.suptitle(f"Pandas Approach {i+1}", fontsize=16)
    end = datetime.now()
    print(f"Pandas Execution Time: {end - start}")
    plt.show()

if __name__ == '__main__':
    k = 5  # Size of Zk
    M = 2  # Size of ZM
    a = 0  # Voltage a (from ZM)
    b = 3  # Voltage b (from ZM)

    process1 = multiprocessing.Process(target=numpy_graph, args=(k, M, a, b))
    process2 = multiprocessing.Process(target=pandas_graph, args=(k, M, a, b))

    process1.start()
    process2.start()

    process1.join()
    process2.join()


# Module(
#     body=[Import(names=[alias(name='multiprocessing', asname=None)]),
#         Import(names=[alias(name='numpy', asname='np')]),
#         Import(names=[alias(name='pandas', asname='pd')]),
#         Import(names=[alias(name='networkx', asname='nx')]),
#         Import(names=[alias(name='matplotlib.pyplot', asname='plt')]),
#         ImportFrom(module='datetime', names=[alias(name='datetime', asname=None)], level=0),
#         ClassDef(name='DigraphLifter',
#             bases=[],
#             keywords=[],
#             body=[
#                 FunctionDef(name='__init__',
#                     args=arguments(posonlyargs=[],
#                         args=[arg(arg='self', annotation=None, type_comment=None),
#                             arg(arg='k', annotation=None, type_comment=None),
#                             arg(arg='M', annotation=None, type_comment=None),
#                             arg(arg='a', annotation=None, type_comment=None),
#                             arg(arg='b', annotation=None, type_comment=None),
#                             arg(arg='method', annotation=None, type_comment=None)],
#                         vararg=None,
#                         kwonlyargs=[],
#                         kw_defaults=[],
#                         kwarg=None,
#                         defaults=[Constant(value=2, kind=None),
#                             Constant(value=1, kind=None),
#                             Constant(value=1, kind=None),
#                             Constant(value='numpy', kind=None)]),
#                     body=[
#                         Assign(targets=[Attribute(value=Name(id='self'), attr='k')],
#                             value=Name(id='k'),
#                             type_comment=None),
#                         Assign(targets=[Attribute(value=Name(id='self'), attr='M')],
#                             value=Name(id='M'),
#                             type_comment=None),
#                         Assign(targets=[Attribute(value=Name(id='self'), attr='a')],
#                             value=Name(id='a'),
#                             type_comment=None),
#                         Assign(targets=[Attribute(value=Name(id='self'), attr='b')],
#                             value=Name(id='b'),
#                             type_comment=None),
#                         Assign(targets=[Attribute(value=Name(id='self'), attr='method')],
#                             value=Name(id='method'),
#                             type_comment=None),
#                         Assign(targets=[Attribute(value=Name(id='self'), attr='base_digraph')],
#                             value=Call(func=Attribute(value=Name(id='self'), attr='create_base_digraph'),
#                                 args=[],
#                                 keywords=[]),
#                             type_comment=None),
#                         Assign(targets=[Attribute(value=Name(id='self'), attr='lifted_graph')],
#                             value=Call(func=Attribute(value=Name(id='self'), attr='lift_digraph'),
#                                 args=[],
#                                 keywords=[]),
#                             type_comment=None)],
#                     decorator_list=[],
#                     returns=None,
#                     type_comment=None),
#                 FunctionDef(name='create_base_digraph',
#                     args=arguments(posonlyargs=[],
#                         args=[arg(arg='self', annotation=None, type_comment=None)],
#                         vararg=None,
#                         kwonlyargs=[],
#                         kw_defaults=[],
#                         kwarg=None,
#                         defaults=[]),
#                     body=[
#                         Assign(targets=[Name(id='G')],
#                             value=Call(func=Attribute(value=Name(id='nx'), attr='DiGraph'), args=[], keywords=[]),
#                             type_comment=None),
#                         If(
#                             test=Compare(left=Attribute(value=Name(id='self'), attr='method'),
#                                 ops=[Eq],
#                                 comparators=[Constant(value='numpy', kind=None)]),
#                             body=[
#                                 Assign(targets=[Name(id='edges')],
#                                     value=Call(func=Attribute(value=Name(id='np'), attr='array'),
#                                         args=[
#                                             ListComp(
#                                                 elt=List(
#                                                     elts=[
#                                                         BinOp(left=Name(id='i'),
#                                                             op=Mod,
#                                                             right=Attribute(value=Name(id='self'), attr='k')),
#                                                         BinOp(
#                                                             left=BinOp(left=Name(id='i'),
#                                                                 op=Add,
#                                                                 right=Constant(value=1, kind=None)),
#                                                             op=Mod,
#                                                             right=Attribute(value=Name(id='self'), attr='k'))]),
#                                                 generators=[
#                                                     comprehension(target=Name(id='i'),
#                                                         iter=Call(func=Name(id='range'),
#                                                             args=[Attribute(value=Name(id='self'), attr='k')],
#                                                             keywords=[]),
#                                                         ifs=[],
#                                                         is_async=0)])],
#                                         keywords=[]),
#                                     type_comment=None),
#                                 For(target=Name(id='edge'),
#                                     iter=Name(id='edges'),
#                                     body=[
#                                         Expr(
#                                             value=Call(func=Attribute(value=Name(id='G'), attr='add_edge'),
#                                                 args=[
#                                                     Subscript(value=Name(id='edge'),
#                                                         slice=Constant(value=0, kind=None)),
#                                                     Subscript(value=Name(id='edge'),
#                                                         slice=Constant(value=1, kind=None))],
#                                                 keywords=[]))],
#                                     orelse=[],
#                                     type_comment=None)],
#                             orelse=[
#                                 If(
#                                     test=Compare(left=Attribute(value=Name(id='self'), attr='method'),
#                                         ops=[Eq],
#                                         comparators=[Constant(value='pandas', kind=None)]),
#                                     body=[
#                                         Assign(targets=[Name(id='edges')],
#                                             value=Call(func=Attribute(value=Name(id='pd'), attr='DataFrame'),
#                                                 args=[
#                                                     Dict(
#                                                         keys=[Constant(value='from', kind=None),
#                                                             Constant(value='to', kind=None)],
#                                                         values=[
#                                                             ListComp(
#                                                                 elt=BinOp(left=Name(id='i'),
#                                                                     op=Mod,
#                                                                     right=Attribute(value=Name(id='self'), attr='k')),
#                                                                 generators=[
#                                                                     comprehension(target=Name(id='i'),
#                                                                         iter=Call(func=Name(id='range'),
#                                                                             args=[
#                                                                                 Attribute(value=Name(id='self'),
#                                                                                     attr='k')],
#                                                                             keywords=[]),
#                                                                         ifs=[],
#                                                                         is_async=0)]),
#                                                             ListComp(
#                                                                 elt=BinOp(
#                                                                     left=BinOp(left=Name(id='i'),
#                                                                         op=Add,
#                                                                         right=Constant(value=1, kind=None)),
#                                                                     op=Mod,
#                                                                     right=Attribute(value=Name(id='self'), attr='k')),
#                                                                 generators=[
#                                                                     comprehension(target=Name(id='i'),
#                                                                         iter=Call(func=Name(id='range'),
#                                                                             args=[
#                                                                                 Attribute(value=Name(id='self'),
#                                                                                     attr='k')],
#                                                                             keywords=[]),
#                                                                         ifs=[],
#                                                                         is_async=0)])])],
#                                                 keywords=[]),
#                                             type_comment=None),
#                                         For(target=Tuple(elts=[Name(id='index'), Name(id='row')]),
#                                             iter=Call(func=Attribute(value=Name(id='edges'), attr='iterrows'),
#                                                 args=[],
#                                                 keywords=[]),
#                                             body=[
#                                                 Expr(
#                                                     value=Call(func=Attribute(value=Name(id='G'), attr='add_edge'),
#                                                         args=[
#                                                             Subscript(value=Name(id='row'),
#                                                                 slice=Constant(value='from', kind=None)),
#                                                             Subscript(value=Name(id='row'),
#                                                                 slice=Constant(value='to', kind=None))],
#                                                         keywords=[]))],
#                                             orelse=[],
#                                             type_comment=None)],
#                                     orelse=[])]),
#                         Return(value=Name(id='G'))],
#                     decorator_list=[],
#                     returns=None,
#                     type_comment=None),
#                 FunctionDef(name='lift_digraph',
#                     args=arguments(posonlyargs=[],
#                         args=[arg(arg='self', annotation=None, type_comment=None)],
#                         vararg=None,
#                         kwonlyargs=[],
#                         kw_defaults=[],
#                         kwarg=None,
#                         defaults=[]),
#                     body=[
#                         Assign(targets=[Name(id='new_graph')],
#                             value=Call(func=Attribute(value=Name(id='nx'), attr='DiGraph'), args=[], keywords=[]),
#                             type_comment=None),
#                         For(target=Name(id='node'),
#                             iter=Call(
#                                 func=Attribute(value=Attribute(value=Name(id='self'), attr='base_digraph'),
#                                     attr='nodes'),
#                                 args=[],
#                                 keywords=[]),
#                             body=[
#                                 For(target=Name(id='i'),
#                                     iter=Call(func=Name(id='range'),
#                                         args=[Attribute(value=Name(id='self'), attr='M')],
#                                         keywords=[]),
#                                     body=[
#                                         Expr(
#                                             value=Call(func=Attribute(value=Name(id='new_graph'), attr='add_node'),
#                                                 args=[Tuple(elts=[Name(id='node'), Name(id='i')])],
#                                                 keywords=[]))],
#                                     orelse=[],
#                                     type_comment=None)],
#                             orelse=[],
#                             type_comment=None),
#                         For(target=Tuple(elts=[Name(id='u'), Name(id='v')]),
#                             iter=Call(
#                                 func=Attribute(value=Attribute(value=Name(id='self'), attr='base_digraph'),
#                                     attr='edges'),
#                                 args=[],
#                                 keywords=[]),
#                             body=[
#                                 For(target=Name(id='i'),
#                                     iter=Call(func=Name(id='range'),
#                                         args=[Attribute(value=Name(id='self'), attr='M')],
#                                         keywords=[]),
#                                     body=[
#                                         Expr(
#                                             value=Call(func=Attribute(value=Name(id='new_graph'), attr='add_edge'),
#                                                 args=[Tuple(elts=[Name(id='u'), Name(id='i')]),
#                                                     Tuple(
#                                                         elts=[Name(id='v'),
#                                                             BinOp(
#                                                                 left=BinOp(left=Name(id='i'),
#                                                                     op=Add,
#                                                                     right=Attribute(value=Name(id='self'), attr='a')),
#                                                                 op=Mod,
#                                                                 right=Attribute(value=Name(id='self'), attr='M'))])],
#                                                 keywords=[])),
#                                         Expr(
#                                             value=Call(func=Attribute(value=Name(id='new_graph'), attr='add_edge'),
#                                                 args=[Tuple(elts=[Name(id='u'), Name(id='i')]),
#                                                     Tuple(
#                                                         elts=[Name(id='v'),
#                                                             BinOp(
#                                                                 left=BinOp(left=Name(id='i'),
#                                                                     op=Add,
#                                                                     right=Attribute(value=Name(id='self'), attr='b')),
#                                                                 op=Mod,
#                                                                 right=Attribute(value=Name(id='self'), attr='M'))])],
#                                                 keywords=[]))],
#                                     orelse=[],
#                                     type_comment=None)],
#                             orelse=[],
#                             type_comment=None),
#                         Return(value=Name(id='new_graph'))],
#                     decorator_list=[],
#                     returns=None,
#                     type_comment=None),
#                 FunctionDef(name='find_cycles',
#                     args=arguments(posonlyargs=[],
#                         args=[arg(arg='self', annotation=None, type_comment=None),
#                             arg(arg='graph', annotation=None, type_comment=None)],
#                         vararg=None,
#                         kwonlyargs=[],
#                         kw_defaults=[],
#                         kwarg=None,
#                         defaults=[]),
#                     body=[
#                         Try(
#                             body=[
#                                 Assign(targets=[Name(id='cycles')],
#                                     value=Call(func=Name(id='list'),
#                                         args=[
#                                             Call(func=Attribute(value=Name(id='nx'), attr='simple_cycles'),
#                                                 args=[Name(id='graph')],
#                                                 keywords=[])],
#                                         keywords=[]),
#                                     type_comment=None),
#                                 If(test=Name(id='cycles'),
#                                     body=[
#                                         Assign(targets=[Name(id='smallest_cycle')],
#                                             value=Call(func=Name(id='min'),
#                                                 args=[Name(id='cycles')],
#                                                 keywords=[keyword(arg='key', value=Name(id='len'))]),
#                                             type_comment=None),
#                                         Assign(targets=[Name(id='largest_cycle')],
#                                             value=Call(func=Name(id='max'),
#                                                 args=[Name(id='cycles')],
#                                                 keywords=[keyword(arg='key', value=Name(id='len'))]),
#                                             type_comment=None),
#                                         Return(
#                                             value=Tuple(
#                                                 elts=[Name(id='smallest_cycle'),
#                                                     Call(func=Name(id='len'),
#                                                         args=[Name(id='smallest_cycle')],
#                                                         keywords=[]),
#                                                     Name(id='largest_cycle'),
#                                                     Call(func=Name(id='len'),
#                                                         args=[Name(id='largest_cycle')],
#                                                         keywords=[])]))],
#                                     orelse=[
#                                         Return(
#                                             value=Tuple(
#                                                 elts=[Constant(value='No cycle', kind=None),
#                                                     Constant(value=0, kind=None),
#                                                     Constant(value='No cycle', kind=None),
#                                                     Constant(value=0, kind=None)]))])],
#                             handlers=[
#                                 ExceptHandler(type=Name(id='Exception'),
#                                     name='e',
#                                     body=[
#                                         Return(
#                                             value=Tuple(
#                                                 elts=[
#                                                     JoinedStr(
#                                                         values=[Constant(value='Error in finding cycles: ', kind=None),
#                                                             FormattedValue(value=Name(id='e'),
#                                                                 conversion=-1,
#                                                                 format_spec=None)]),
#                                                     UnaryOp(op=USub, operand=Constant(value=1, kind=None)),
#                                                     JoinedStr(
#                                                         values=[Constant(value='Error in finding cycles: ', kind=None),
#                                                             FormattedValue(value=Name(id='e'),
#                                                                 conversion=-1,
#                                                                 format_spec=None)]),
#                                                     UnaryOp(op=USub, operand=Constant(value=1, kind=None))]))])],
#                             orelse=[],
#                             finalbody=[])],
#                     decorator_list=[],
#                     returns=None,
#                     type_comment=None),
#                 FunctionDef(name='draw_digraphs',
#                     args=arguments(posonlyargs=[],
#                         args=[arg(arg='self', annotation=None, type_comment=None)],
#                         vararg=None,
#                         kwonlyargs=[],
#                         kw_defaults=[],
#                         kwarg=None,
#                         defaults=[]),
#                     body=[
#                         Assign(targets=[Name(id='fig')],
#                             value=Call(func=Attribute(value=Name(id='plt'), attr='figure'),
#                                 args=[],
#                                 keywords=[
#                                     keyword(arg='figsize',
#                                         value=Tuple(elts=[Constant(value=12, kind=None), Constant(value=6, kind=None)]))]),
#                             type_comment=None),
#                         Expr(
#                             value=Call(func=Attribute(value=Name(id='plt'), attr='subplot'),
#                                 args=[Constant(value=1, kind=None),
#                                     Constant(value=2, kind=None),
#                                     Constant(value=1, kind=None)],
#                                 keywords=[])),
#                         Expr(
#                             value=Call(func=Attribute(value=Name(id='nx'), attr='draw'),
#                                 args=[Attribute(value=Name(id='self'), attr='base_digraph')],
#                                 keywords=[keyword(arg='with_labels', value=Constant(value=True, kind=None)),
#                                     keyword(arg='node_color', value=Constant(value='lightblue', kind=None)),
#                                     keyword(arg='node_size', value=Constant(value=2000, kind=None)),
#                                     keyword(arg='arrowstyle', value=Constant(value='->', kind=None)),
#                                     keyword(arg='arrowsize', value=Constant(value=20, kind=None))])),
#                         Assign(
#                             targets=[
#                                 Tuple(
#                                     elts=[Name(id='smallest_cycle_base'),
#                                         Name(id='length_smallest_base'),
#                                         Name(id='largest_cycle_base'),
#                                         Name(id='length_largest_base')])],
#                             value=Call(func=Attribute(value=Name(id='self'), attr='find_cycles'),
#                                 args=[Attribute(value=Name(id='self'), attr='base_digraph')],
#                                 keywords=[]),
#                             type_comment=None),
#                         Expr(
#                             value=Call(func=Attribute(value=Name(id='plt'), attr='title'),
#                                 args=[
#                                     JoinedStr(
#                                         values=[Constant(value='Base Digraph\nSmallest cycle: ', kind=None),
#                                             FormattedValue(value=Name(id='length_smallest_base'),
#                                                 conversion=-1,
#                                                 format_spec=None),
#                                             Constant(value='\nLargest cycle: ', kind=None),
#                                             FormattedValue(value=Name(id='length_largest_base'),
#                                                 conversion=-1,
#                                                 format_spec=None)])],
#                                 keywords=[])),
#                         Expr(
#                             value=Call(func=Attribute(value=Name(id='plt'), attr='subplot'),
#                                 args=[Constant(value=1, kind=None),
#                                     Constant(value=2, kind=None),
#                                     Constant(value=2, kind=None)],
#                                 keywords=[])),
#                         Expr(
#                             value=Call(func=Attribute(value=Name(id='nx'), attr='draw'),
#                                 args=[Attribute(value=Name(id='self'), attr='lifted_graph')],
#                                 keywords=[keyword(arg='with_labels', value=Constant(value=True, kind=None)),
#                                     keyword(arg='node_color', value=Constant(value='lightgreen', kind=None)),
#                                     keyword(arg='node_size', value=Constant(value=2000, kind=None)),
#                                     keyword(arg='arrowstyle', value=Constant(value='->', kind=None)),
#                                     keyword(arg='arrowsize', value=Constant(value=20, kind=None))])),
#                         Assign(
#                             targets=[
#                                 Tuple(
#                                     elts=[Name(id='smallest_cycle_lifted'),
#                                         Name(id='length_cycle_lifted'),
#                                         Name(id='largest_cycle_lifted'),
#                                         Name(id='length_largest_lifted')])],
#                             value=Call(func=Attribute(value=Name(id='self'), attr='find_cycles'),
#                                 args=[Attribute(value=Name(id='self'), attr='lifted_graph')],
#                                 keywords=[]),
#                             type_comment=None),
#                         Expr(
#                             value=Call(func=Attribute(value=Name(id='plt'), attr='title'),
#                                 args=[
#                                     JoinedStr(
#                                         values=[Constant(value='Lifted Digraph\nSmallest cycle: ', kind=None),
#                                             FormattedValue(value=Name(id='length_cycle_lifted'),
#                                                 conversion=-1,
#                                                 format_spec=None),
#                                             Constant(value='\nLargest cycle: ', kind=None),
#                                             FormattedValue(value=Name(id='length_largest_lifted'),
#                                                 conversion=-1,
#                                                 format_spec=None)])],
#                                 keywords=[])),
#                         Return(value=Name(id='fig'))],
#                     decorator_list=[],
#                     returns=None,
#                     type_comment=None)],
#             decorator_list=[]),
#         FunctionDef(name='numpy_graph',
#             args=arguments(posonlyargs=[],
#                 args=[arg(arg='k', annotation=None, type_comment=None),
#                     arg(arg='M', annotation=None, type_comment=None),
#                     arg(arg='a', annotation=None, type_comment=None),
#                     arg(arg='b', annotation=None, type_comment=None)],
#                 vararg=None,
#                 kwonlyargs=[],
#                 kw_defaults=[],
#                 kwarg=None,
#                 defaults=[]),
#             body=[
#                 Assign(targets=[Name(id='start')],
#                     value=Call(func=Attribute(value=Name(id='datetime'), attr='now'), args=[], keywords=[]),
#                     type_comment=None),
#                 For(target=Name(id='i'),
#                     iter=Call(func=Name(id='range'), args=[Constant(value=50, kind=None)], keywords=[]),
#                     body=[
#                         Assign(targets=[Name(id='graph_transformer')],
#                             value=Call(func=Name(id='DigraphLifter'),
#                                 args=[Name(id='k'), Name(id='M'), Name(id='a'), Name(id='b')],
#                                 keywords=[keyword(arg='method', value=Constant(value='numpy', kind=None))]),
#                             type_comment=None),
#                         Assign(targets=[Name(id='fig')],
#                             value=Call(
#                                 func=Attribute(value=Name(id='graph_transformer'), attr='draw_digraphs'),
#                                 args=[],
#                                 keywords=[]),
#                             type_comment=None),
#                         Expr(
#                             value=Call(func=Attribute(value=Name(id='fig'), attr='suptitle'),
#                                 args=[
#                                     JoinedStr(
#                                         values=[Constant(value='Numpy Approach ', kind=None),
#                                             FormattedValue(
#                                                 value=BinOp(left=Name(id='i'),
#                                                     op=Add,
#                                                     right=Constant(value=1, kind=None)),
#                                                 conversion=-1,
#                                                 format_spec=None)])],
#                                 keywords=[keyword(arg='fontsize', value=Constant(value=16, kind=None))]))],
#                     orelse=[],
#                     type_comment=None),
#                 Assign(targets=[Name(id='end')],
#                     value=Call(func=Attribute(value=Name(id='datetime'), attr='now'), args=[], keywords=[]),
#                     type_comment=None),
#                 Expr(
#                     value=Call(func=Name(id='print'),
#                         args=[
#                             JoinedStr(
#                                 values=[Constant(value='NumPy Execution Time: ', kind=None),
#                                     FormattedValue(value=BinOp(left=Name(id='end'), op=Sub, right=Name(id='start')),
#                                         conversion=-1,
#                                         format_spec=None)])],
#                         keywords=[])),
#                 Expr(value=Call(func=Attribute(value=Name(id='plt'), attr='show'), args=[], keywords=[]))],
#             decorator_list=[],
#             returns=None,
#             type_comment=None),
#         FunctionDef(name='pandas_graph',
#             args=arguments(posonlyargs=[],
#                 args=[arg(arg='k', annotation=None, type_comment=None),
#                     arg(arg='M', annotation=None, type_comment=None),
#                     arg(arg='a', annotation=None, type_comment=None),
#                     arg(arg='b', annotation=None, type_comment=None)],
#                 vararg=None,
#                 kwonlyargs=[],
#                 kw_defaults=[],
#                 kwarg=None,
#                 defaults=[]),
#             body=[
#                 Assign(targets=[Name(id='start')],
#                     value=Call(func=Attribute(value=Name(id='datetime'), attr='now'), args=[], keywords=[]),
#                     type_comment=None),
#                 For(target=Name(id='i'),
#                     iter=Call(func=Name(id='range'), args=[Constant(value=50, kind=None)], keywords=[]),
#                     body=[
#                         Assign(targets=[Name(id='graph_transformer')],
#                             value=Call(func=Name(id='DigraphLifter'),
#                                 args=[Name(id='k'), Name(id='M'), Name(id='a'), Name(id='b')],
#                                 keywords=[keyword(arg='method', value=Constant(value='pandas', kind=None))]),
#                             type_comment=None),
#                         Assign(targets=[Name(id='fig')],
#                             value=Call(
#                                 func=Attribute(value=Name(id='graph_transformer'), attr='draw_digraphs'),
#                                 args=[],
#                                 keywords=[]),
#                             type_comment=None),
#                         Expr(
#                             value=Call(func=Attribute(value=Name(id='fig'), attr='suptitle'),
#                                 args=[
#                                     JoinedStr(
#                                         values=[Constant(value='Pandas Approach ', kind=None),
#                                             FormattedValue(
#                                                 value=BinOp(left=Name(id='i'),
#                                                     op=Add,
#                                                     right=Constant(value=1, kind=None)),
#                                                 conversion=-1,
#                                                 format_spec=None)])],
#                                 keywords=[keyword(arg='fontsize', value=Constant(value=16, kind=None))]))],
#                     orelse=[],
#                     type_comment=None),
#                 Assign(targets=[Name(id='end')],
#                     value=Call(func=Attribute(value=Name(id='datetime'), attr='now'), args=[], keywords=[]),
#                     type_comment=None),
#                 Expr(
#                     value=Call(func=Name(id='print'),
#                         args=[
#                             JoinedStr(
#                                 values=[Constant(value='Pandas Execution Time: ', kind=None),
#                                     FormattedValue(value=BinOp(left=Name(id='end'), op=Sub, right=Name(id='start')),
#                                         conversion=-1,
#                                         format_spec=None)])],
#                         keywords=[])),
#                 Expr(value=Call(func=Attribute(value=Name(id='plt'), attr='show'), args=[], keywords=[]))],
#             decorator_list=[],
#             returns=None,
#             type_comment=None),
#         If(
#             test=Compare(left=Name(id='__name__'), ops=[Eq], comparators=[Constant(value='__main__', kind=None)]),
#             body=[
#                 Assign(targets=[Name(id='k')], value=Constant(value=5, kind=None), type_comment=None),
#                 Assign(targets=[Name(id='M')], value=Constant(value=2, kind=None), type_comment=None),
#                 Assign(targets=[Name(id='a')], value=Constant(value=0, kind=None), type_comment=None),
#                 Assign(targets=[Name(id='b')], value=Constant(value=3, kind=None), type_comment=None),
#                 Assign(targets=[Name(id='process1')],
#                     value=Call(func=Attribute(value=Name(id='multiprocessing'), attr='Process'),
#                         args=[],
#                         keywords=[keyword(arg='target', value=Name(id='numpy_graph')),
#                             keyword(arg='args',
#                                 value=Tuple(elts=[Name(id='k'), Name(id='M'), Name(id='a'), Name(id='b')]))]),
#                     type_comment=None),
#                 Assign(targets=[Name(id='process2')],
#                     value=Call(func=Attribute(value=Name(id='multiprocessing'), attr='Process'),
#                         args=[],
#                         keywords=[keyword(arg='target', value=Name(id='pandas_graph')),
#                             keyword(arg='args',
#                                 value=Tuple(elts=[Name(id='k'), Name(id='M'), Name(id='a'), Name(id='b')]))]),
#                     type_comment=None),
#                 Expr(value=Call(func=Attribute(value=Name(id='process1'), attr='start'), args=[], keywords=[])),
#                 Expr(value=Call(func=Attribute(value=Name(id='process2'), attr='start'), args=[], keywords=[])),
#                 Expr(value=Call(func=Attribute(value=Name(id='process1'), attr='join'), args=[], keywords=[])),
#                 Expr(value=Call(func=Attribute(value=Name(id='process2'), attr='join'), args=[], keywords=[]))],
#             orelse=[])],
#     type_ignores=[])
