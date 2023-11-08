from mip import *

def exists_set_H_of_size_k(instance):
    # Create a model
    model = Model()

    # Do not output solver statistics
    model.verbose = 0

    ##################################
    # Getting data froma the subsets #
    ##################################

    # Number of tasks n, number of subsets m, and max size of H k
    list_variables = instance.split('\n')[0].split()
    n, m, k = int(list_variables[0]), int(list_variables[1]), int(list_variables[2])

    # Getting all subsets
    list_subsets = instance.split('\n')[1:]
    subsets = []
    for i in range(m):
        subsets.append(list(map(int, list_subsets[i].split())))

    #############################
    # Adding decision variables #
    #############################

    # Binary variable indicatin if task i is in H
    X = [model.add_var(var_type=BINARY) for i in range(n)]

    ##########################################################
    # Objective function: Minimize the number of tasks in H. #
    ##########################################################

    model.objective = minimize(xsum(X[i] for i in range(n)))

    ######################################
    # Function subject to (CONSTRAINTS): #
    ######################################

    # H has at most k tasks
    model += xsum(X[i] for i in range(n)) <= k

    # Every subset has at least one task in H
    for subset in subsets:
        model += xsum(X[i] for i in subset) >= 1

    # Solve the model and return the objective value
    status = model.optimize()

    # Return True if there exists a set H of size k, and False otherwise
    return status == OptimizationStatus.OPTIMAL

if '__main__' == __name__:

    archive_name = "mdb_example.txt"
    with open(archive_name, 'r') as archive:
        mdb_instance = archive.read()

    print(exists_set_H_of_size_k(mdb_instance))