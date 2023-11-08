from mip import *
import sys

def binary_model(graph, relaxation):
    """Returns the minimum number of colours using the linear relaxation of the binary model"""
    
    # Create a model
    model = Model()

    # Do not output solver statistics
    model.verbose = 1

    ####################################
    # Getting the data from the graph. #
    ####################################

    # Number of vertices & number of edges
    list_graph = graph.split()
    n, m = int(list_graph[0]), int(list_graph[1])

    # Getting all edges
    edges = []
    for i in range(m):
        from_vertex, to_vertex = int(list_graph[2+i*2]), int(list_graph[3+i*2])
        edges.append((from_vertex, to_vertex))

    #############################
    # Adding decision variables #
    #############################

    if relaxation:
        # Continous variables indicating if vertex v has color i
        X = [[model.add_var(lb=0, ub=1) for v in range(n)] for i in range(n)]

    else:
        # Binary variables indicating if vertex v has color i
        X = [[model.add_var(var_type=BINARY) for v in range(n)] for i in range(n)]

    # Binary variables indicating if color i is present in the Graph
    C = [model.add_var(var_type=BINARY) for i in range(n)]
        
    #########################################################################
    # Objective function: Minimize the number of colors used in the graph. #
    #########################################################################

    model.objective = minimize(xsum(C[i] for i in range(n)))

    ######################################
    # Function subject to (CONSTRAINTS): #
    ######################################

    # Every node has a single color
    for v in range(n):
        model += xsum(X[v][i] for i in range(n)) == 1

    # Color conflict constraint: 2 adjacent nodes cannot have the same color
    for edge in edges:
        v = edge[0]
        w = edge[1]
        for i in range(n):
            model += X[v][i] + X[w][i] <= 1

    # Handling C variable: A node v cannot have color i if variable C[i] == 0 
    for v in range(n):
        for i in range(n):
            model += X[v][i] <= C[i]

    # All colors assigned to vertices must be used colors
    for i in range(n):
        model += xsum(X[v][i] for v in range(n)) >= C[i]

    # Solve the model and return the objective value
    model.optimize()
    return model.objective_value

def integer_model(graph, relaxation):
    """Returns the minimum number of colours using the linear relaxation of the integer model"""
    
    # Create a model
    model = Model()

    # Do not output solver statistics
    model.verbose = 1

    ####################################
    # Getting the data from the graph. #
    ####################################

    # Number of vertices & number of edges
    list_graph = graph.split()
    n, m = int(list_graph[0]), int(list_graph[1])

    # Big M
    M = n

    # Getting all edges
    edges = []
    for i in range(m):
        from_vertex, to_vertex = int(list_graph[2+i*2]), int(list_graph[3+i*2])
        edges.append((from_vertex, to_vertex))

    #############################
    # Adding decision variables #
    #############################

    if relaxation:
        # Continous variables indicating if vertex v has color i
        X = [model.add_var(lb=1, ub=n) for _ in range(n)]

    else:
        # Integer variables indicating the color of vertex v
        X = [model.add_var(var_type=INTEGER, lb=1, ub=n) for v in range(n)]

    # Binary variables modeling OR restrictions for adjacent vertices to have different colors
    z = [model.add_var(var_type=BINARY) for i in range(m)]

    # Integer variable indicating the number of colors used
    c  = model.add_var(var_type=INTEGER, lb=1, ub=n)

    #########################################################################
    # Objective function: Minimize the number of colors used in the graph. #
    #########################################################################

    # model.objective = minimize(xsum(X[v] for v in range(n)))
    model.objective = minimize(c)

    ###########################
    # Constraints subject to: #
    ###########################

    # X[v] != X[w] if v and w are adjacent
    for i in range(m):
        v = edges[i][0]
        w = edges[i][1]
        model += X[v] - X[w] >= 1 - M*z[i], "OR constraint 1"
        model += X[w] - X[v] >= 1 - M*(1-z[i]), "OR constraint 2"
    
    # All colors must be less than the number of colors used
    for v in range(n):
        model += X[v] <= c, "all colors less than C"

    # Solve the model and return the objective value
    model.optimize()
    return model.objective_value

archive_name = "n10.txt"
with open(archive_name, "r") as archivo:
    graph = archivo.read()


# print("BINARY MODEL NO RELAXATION")
# print(binary_model(graph, False))
# print('-'*50)
# print("BINARY MODEL WITH RELAXATION")
# print(binary_model(graph, True))
print("INTEGER MODEL NO RELAXATION")
print(integer_model(graph, False))
print('-'*50)
print("INTEGER MODEL WITH RELAXATION")
print(integer_model(graph, True))