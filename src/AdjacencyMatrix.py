import numpy as np
from MinCut import Graph
import sys


class AdjacencyMatrix:
    def __init__(self, height, width,pos_matrix):
        self.pos_matrix = pos_matrix
        self.node_count = height * width + 2
        self.adjacency_matrix  = np.zeros([self.node_count,self.node_count],dtype=float)

    def add_terminal_edge_capacities(self,overlap_pos,cut):

        if cut == "vertical":
            height_adj = self.adjacency_matrix.shape[0]
            width_adj = self.adjacency_matrix.shape[1]
            # height_adj = len(adjacency_matrix)
            # width_adj = len(adjacency_matrix)

            width = overlap_pos.shape[1]
            left_col = overlap_pos[:, 0]
            right_col = overlap_pos[:, width - 1]
            for i, j in enumerate(left_col):
                self.adjacency_matrix[0, j] = sys.maxsize
                # adjacency_matrix[0][j] = sys.maxsize

            for i, j in enumerate(right_col):
                self.adjacency_matrix[j, width_adj - 1] = sys.maxsize
                # adjacency_matrix[j][width_adj-1] = sys.maxsize
        else:
            height_adj = self.adjacency_matrix.shape[0]
            width_adj = self.adjacency_matrix.shape[1]
            # height_adj = len(adjacency_matrix)
            # width_adj = len(adjacency_matrix)

            height = overlap_pos.shape[0]
            top_row = overlap_pos[0, :]
            bottom_row  = overlap_pos[height-1,:]
            for i, j in enumerate(top_row):
                self.adjacency_matrix[0, j] = sys.maxsize

            for i, j in enumerate(bottom_row):
                self.adjacency_matrix[j, width_adj - 1] = sys.maxsize
        pass

    def right_weights(self, col, matrix1, matrix2, height, overlap_pos):

        edge_weights = np.zeros([height, 1])
        for i in range(0, height):
            # value = np.linalg.norm(matrix1[i, col] - matrix2[i, col]) + np.linalg.norm(matrix1[i, col + 1] - matrix2[i, col + 1])
            value = edge_weights[i] = np.sum(np.square(matrix1[i, col] - matrix2[i, col], dtype=np.float) + np.square(matrix1[i, col + 1] - matrix2[i, col + 1]), dtype=np.float) + 0.000000001
            edge_weights[i] = value

        col1 = overlap_pos[:, col]
        col2 = overlap_pos[:, col + 1]

        for i in range(0, len(col1)):
            leftPos = col1[i]
            rightPos = col2[i]

            self.adjacency_matrix[leftPos][rightPos] = edge_weights[i]
        pass

    def bottom_weights(self, row, matrix1, matrix2, overlap_size, overlap_pos):
        edge_weights = np.zeros([overlap_size, 1])
        for i in range(0, overlap_size):
            # edge_weights[i] = np.linalg.norm(matrix1[row, i] - matrix2[row, i]) + np.linalg.norm(matrix1[row + 1, i] - matrix2[row + 1, i])
            edge_weights[i] = np.sum(np.square(matrix1[row, i] - matrix2[row, i], dtype=np.float) + np.square(matrix1[row + 1, i] - matrix2[row + 1, i]), dtype=np.float) + 0.000000001
        # adjacency_matrix[currentPos][rightPos] = edge_weight

        row1 = overlap_pos[row, :]
        row2 = overlap_pos[row + 1, :]

        for i in range(0, len(row1)):
            topPos = row1[i]
            bottomPos = row2[i]

            self.adjacency_matrix[topPos, bottomPos] = edge_weights[i]

        pass

    def calculateGraph(self,left_overlap,right_overlap,cut):

        if cut == "vertical":
            height = left_overlap.shape[0]
            width = left_overlap.shape[1]
            self.add_terminal_edge_capacities(self.pos_matrix,cut)
            overlap_size = left_overlap.shape[1]

            for col in range(width - 1):
                self.right_weights(col, left_overlap, right_overlap, height, self.pos_matrix)

            for row in range(height - 1):
                self.bottom_weights(row, left_overlap, right_overlap, overlap_size,self.pos_matrix)
        else:
            self.add_terminal_edge_capacities(self.pos_matrix,cut)
            top_overlap = left_overlap
            bottom_overlap = right_overlap
            height = top_overlap.shape[0]
            width = top_overlap.shape[1]
            for col in range(width - 1):
                self.right_weights(col, top_overlap, bottom_overlap, height, self.pos_matrix)

            for row in range(height - 1):
                self.bottom_weights(row, top_overlap, bottom_overlap, width,self.pos_matrix)

        graph = self.adjacency_matrix.tolist()

        graph = Graph(graph)

        source = 0
        sink = self.adjacency_matrix.shape[1] - 1
        # sink = len(adjacency_matrix)
        nodes_to_cut = graph.minCut(source, sink)

        return nodes_to_cut