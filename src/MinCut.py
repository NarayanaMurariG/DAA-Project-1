def breadthFirstSearch(graph, s, t, parent, no_of_rows):
    # We need to flag all the nodes which we have not visited and in the queue we have to add the visited nodes
    visited_nodes = [False] * (no_of_rows)
    breadthFirstSearch_queue = []
    breadthFirstSearch_queue.append(s)
    visited_nodes[s] = True

    while breadthFirstSearch_queue:

        node = breadthFirstSearch_queue.pop(0)

        for index, value in enumerate(graph[node]):
            if visited_nodes[index] == False and value > 0:
                breadthFirstSearch_queue.append(index)
                visited_nodes[index] = True
                parent[index] = node

    # Returing true if a path has been found from source to sink, else false
    return True if visited_nodes[t] else False


def minCut(graph, source, sink):
    original_graph = [x[:] for x in graph]
    no_of_rows = len(graph)
    no_of_columns = len(graph[0])

    parent_nodes = [-1] * no_of_rows

    max_flow = 0  # This is set to zero as there is no initial flow

    # Traversing through path from source to sink
    while breadthFirstSearch(graph, source, sink, parent_nodes, no_of_rows):

        # We need to find the min residual cap of the edge paths in the path found by BFS
        path_flow = float("Inf")
        s = sink
        while (s != source):
            path_flow = min(path_flow, graph[parent_nodes[s]][s])
            s = parent_nodes[s]

        max_flow += path_flow  # Add path flow to overall flow

        # Along the path, we are updating the residual cap of edges in both directions
        v = sink
        while (v != source):
            u = parent_nodes[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent_nodes[v]

    visited_nodes = len(graph) * [False]
    depthFirstSearch(graph, s, visited_nodes)

    # Compare the original graph and graph on which computations have been done, the nodes
    # should have zero weight and also they should be in the visited nodes group
    # If all the above are conditions are satisfied we add them to the final cut list
    finalCut = []
    for i in range(no_of_rows):
        for j in range(no_of_columns):
            if graph[i][j] == 0 and original_graph[i][j] > 0 and visited_nodes[i]:
                # Adding the final cut values to a list
                finalCut.append(str(i) + "-" + str(j))

    return finalCut


def depthFirstSearch(graph, s, visited_nodes):
    visited_nodes[s] = True
    for i in range(len(graph)):
        if graph[s][i] > 0 and not visited_nodes[i]:
            depthFirstSearch(graph, i, visited_nodes)
