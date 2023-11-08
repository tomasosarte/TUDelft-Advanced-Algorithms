import random
from mip import *
import time
import matplotlib.pyplot as plt


##EXERCICI 3 BOUNDED
def bounded_tree_search(subsets, k):
    """
    This function implements a tree bounded search by k.
    It receives a list of subsets and a number k, and returns
    True if there exists a set H of size k, and False otherwise.

    Args:
        subsets (list): List of subsets.
        k (int): Max size of H.
    
    Returns:
        bool: True if there exists a set H of size k, and False otherwise.
    """
    
    # Tree bounded search by k
    if k < 0: return False

    # All subsets are covered by H
    if len(subsets) == 0: return True

    # Get the first subset
    subset = subsets[0]

    for components in subset:

        # Remove all subsets from the subset list that contains the components
        new_subsets = [s for s in subsets if components not in s]

        # Recursive call to the bound tree search
        exists = bounded_tree_search(new_subsets, k - 1)

        # If exists, return True
        if exists: return True
    
    # If not exists, return False
    return False

def exists_set_H_of_size_k(instance):

    ##################################
    # Getting data froma the subsets #
    ##################################

    # Number of componentss n, number of subsets m, and max size of H k
    list_variables = instance.split('\n')[0].split()
    n, m, k = int(list_variables[0]), int(list_variables[1]), int(list_variables[2])

    # Getting all subsets
    list_subsets = instance.split('\n')[1:]
    subsets = []
    for i in range(m):
        subsets.append(list(map(int, list_subsets[i].split())))

    # Call the bounded tree search
    return bounded_tree_search(subsets, k)

##EXERCICI 1 MIP
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

def generate_mdb_instance(n_components, n_conflict_sets, k):
    components = list(range(0, n_components))
    conflict_sets = []
    for _ in range(n_conflict_sets):
        conflict_set_size = random.randint(1, n_components)
        conflict_set = random.sample(components,conflict_set_size)
        conflict_sets.append(conflict_set)

    res_str = f"{n_components} {n_conflict_sets} {k}\n"
    z = [" ".join(str(i) for i in conf_set) for conf_set in conflict_sets]
    res_str += "\n".join(z)
    return res_str


if '__main__' == __name__:
    #1. Increment the number of components, with same conflict sets and k
    #initialize vectors for mip and bounded tree times
    time_vector_mip = []
    time_vector_bounded_tree = []

    for i in range(1, 1000, 10):

        with open(f"mdb_example_{i}.txt", 'w') as archive:
            archive.write(generate_mdb_instance(i, 10, 5))
        
        #########################################################################################
        #1. measure the time of each iteration and save it in a vector to plot it later with mip#
        #########################################################################################
        
        #start counting time
        start = time.time()

        #set to archive_name the name of the archive in each iteration
        archive_name = f"mdb_example_{i}.txt"

        with open(archive_name, 'r') as archive:
            mdb_instance = archive.read()

        exists_set_H_of_size_k(mdb_instance)

        #stop counting time
        end = time.time()

        #save the time in a vector
        time_vector_mip.append(end-start)

        ##################################################################################################
        #2. measure the time of each iteration and save it in a vector to plot it later with bounded tree#
        ##################################################################################################
        
        #start counting time
        start = time.time()


        archive_name = f"mdb_example_{i}.txt"
        with open(archive_name, 'r') as archive:
            mdb_instance = archive.read()

        exists_set_H_of_size_k(mdb_instance)

        #stop counting time
        end = time.time()

        #save the time in a vector
        time_vector_bounded_tree.append(end-start)



    #store results in two files with one row for each iteration
    file_name = "mip_res.txt"
    
    # Open the file for writing
    with open(file_name, "w") as file:
        # Write each element to the file, one per line
        for element in time_vector_mip:
            number_str = str(element).replace(".", ",")
            file.write(str(number_str) + "\n")  

    file_name = "bounded_tree_res.txt"
    # Open the file for writing
    with open(file_name, "w") as file:
        # Write each element to the file, one per line
       for element in time_vector_bounded_tree:
            number_str = str(element).replace(".", ",")
            file.write(str(number_str) + "\n")  

    #plot the results
    plt.plot(time_vector_mip, label = "mip")
    plt.plot(time_vector_bounded_tree, label = "bounded tree")
    plt.legend()
    plt.show()


        



