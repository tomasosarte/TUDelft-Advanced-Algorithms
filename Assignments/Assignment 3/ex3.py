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

if '__main__' == __name__:

    archive_name = "mdb_example.txt"
    with open(archive_name, 'r') as archive:
        mdb_instance = archive.read()

    print(exists_set_H_of_size_k(mdb_instance))