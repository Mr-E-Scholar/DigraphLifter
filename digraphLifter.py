import multiprocessing
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

class DigraphLifter:
    def __init__(self, k, M=2, a=1, b=1):
        self.k = k
        self.M = M
        self.a = a
        self.b = b
        self.base_digraph = self.create_base_digraph()
        self.lifted_graph = self.lift_digraph()

    def create_base_digraph(self):
        G = nx.DiGraph()
        for i in range(self.k):
            G.add_edge(i % self.k, (i + 1) % self.k)
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
    for i in range(10):
        graph_transformer = DigraphLifter(k, M, a, b)
        fig = graph_transformer.draw_digraphs()
        fig.suptitle(f"Numpy Approach {i+1}", fontsize=16)
    end = datetime.now()
    print(f"NumPy Execution Time: {end - start}")
    plt.show()

def pandas_graph(k, M, a, b):
    start = datetime.now()
    for i in range(10):
        graph_transformer = DigraphLifter(k, M, a, b)
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
