import queue
import time
import os
from timeout import timeout
import random
from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt

@timeout(timeout=60)
def solve(r, p, d, v):

    priorityQueue = queue.PriorityQueue()
    # Note, X starts with {0, len(r)-1}, as these are dummy jobs.
    priorityQueue.put([0, {0, len(r)-1}, 0])
    
    valuesMap = {}
    valuesMap[(0, frozenset({0, len(r)-1}), 0)] = 0

    max_value = 0
    
    # We use a priority queue with start time of the last job as the key,
    # as to efficiently go over all states.
    while not priorityQueue.empty():
        t_i, X, i = priorityQueue.get()
        if not (t_i, frozenset(X), i) in valuesMap:
            continue

        # interm_v is the value we have 'before' doing job i
        interm_v = valuesMap[(t_i, frozenset(X), i)]
        del valuesMap[(t_i, frozenset(X), i)]
        C_i = t_i + p[i]

        # Note that F is the set of jobs that we can still do if we do job i ordered by their proximity to their deadline
        F = [j for j in range(len(p)) if not j in X and C_i <= d[j] - p[j]]

        # Eliminate all states that have the same job, same or bigger starting time, and X_K
        # is a subset of the frozenset on the valueMap that united with all elements in X_k
        # that theire deadline minus theire processing time is bigger or equal tu C_k
        # This is done to avoid having suboptimal states that lead to worse or equal solutions
        # from the currently evaluated state. 
        # keys_to_delete = []
        # for (t2, X2, i2), value_s2 in valuesMap.items():
        #     C_i2 = t2 + p[i2]
        #     if i == i2 and t_i <= t2 and interm_v >= value_s2 and X.issubset(X2.union(set(j for j in X if d[j] - p[j] >= C_i2))):
        #         keys_to_delete.append((t2, X2, i2))
        # for key in keys_to_delete:
        #     del valuesMap[key]

        # If there are no jobs that we cannot do anymore, we have found
        # 'a full schedule', thus we only need to check if this is better than all
        # the other full schedules we have found until now.
        if len(F) == 0:
            max_value = max(interm_v, max_value)
            continue

        # If there is only one job that we can still do because of our choice,
        # there is only one to consider if the current max value is better or 
        # not than the value of the subproblem X adding the value of this job.
        if len(F) == 1:
            max_value = max(interm_v + v[F[0]], max_value)
            continue
        
        # In case we only have left 2 jobs that we can still do because of our choice,
        # we can either do both of them or only one of them, depending on the compatibility
        # of their deadlines and processing times.
        if len(F) == 2:
            # If it is possible to do both jobs, the max value is the comparison between
            # the actual max value and the value of the subproblem X adding the value of both jobs.
            if C_i + p[F[0]] <= d[F[1]] or C_i + p[F[1]] <= d[F[0]]:
                max_value = max(interm_v + v[F[0]] + v[F[1]], max_value)
                continue
            # If it is not possible to do both jobs, we can only do one of them. Thus, the max value
            # is the comparison between the actual max value and the value of the subproblem X adding 
            # the value of the job that can be done.
            else:
                max_value = max(max(interm_v + v[F[0]], max_value), interm_v + v[F[1]])
                continue

        # If there are jobs that we can still do because of our choice,
        # We should consider all recursive subproblems of planning this job after job i.
        for k in F:
            t_k = max(C_i, r[k])
            X_prime = X.copy()
            X_prime.add(k)
            C_k = t_k + p[k]
            # We take out from X_prime all the jobs that the dead line is passed the
            # completion time of k + the processing time of the element.
            X_k = X_prime.difference(j for j in X_prime if (d[j] - p[j]) < C_k)

            interm_value = v[k] + interm_v
            priorityQueue.put([t_k, X_k, k])
            if (t_k, frozenset(X_k), k) in valuesMap:
                valuesMap[(t_k, frozenset(X_k), k)] = max(interm_value, valuesMap[(t_k, frozenset(X_k), k)])
            else:
                valuesMap[(t_k, frozenset(X_k), k)] = interm_value

    return max_value

@timeout(timeout=60)
def first_solve(r,p,d,v):

    priorityQueue = queue.PriorityQueue()
    # Note, X starts with {0, len(r)-1}, as these are dummy jobs.
    priorityQueue.put([0, {0, len(r)-1}, 0])
    
    valuesMap = {}
    valuesMap[(0, frozenset({0, len(r)-1}), 0)] = 0

    max_value = 0
    
    # We use a priority queue with start time of the last job as the key,
    # as to efficiently go over all states.
    while not priorityQueue.empty():
        t_i, X, i = priorityQueue.get()
        if not (t_i, frozenset(X), i) in valuesMap:
            continue

        # interm_v is the value we have 'before' doing job i
        interm_v = valuesMap[(t_i, frozenset(X), i)]
        del valuesMap[(t_i, frozenset(X), i)]
        C_i = t_i + p[i]

        # Note that F is the set of jobs that we can still do if we do job i ordered by their proximity to their deadline
        F = [j for j in range(len(p)) if not j in X and C_i <= d[j] - p[j]]

        # If there are no jobs that we cannot do anymore, we have found
        # 'a full schedule', thus we only need to check if this is better than all
        # the other full schedules we have found until now.
        if len(F) == 0:
            max_value = max(interm_v, max_value)
            continue

        # If there are jobs that we can still do because of our choice,
        # We should consider all recursive subproblems of planning this job after job i.
        for k in F:
            t_k = max(C_i, r[k])
            X_prime = X.copy()
            X_prime.add(k)
            C_k = t_k + p[k]
            X_k = X_prime
            interm_value = v[k] + interm_v
            priorityQueue.put([t_k, X_k, k])
            if (t_k, frozenset(X_k), k) in valuesMap:
                valuesMap[(t_k, frozenset(X_k), k)] = max(interm_value, valuesMap[(t_k, frozenset(X_k), k)])
            else:
                valuesMap[(t_k, frozenset(X_k), k)] = interm_value

    return max_value

def runFromFile(filename, first=False):
    with open(filename) as f_in:
        input = [[float(i) for i in l.split(',')] for l in f_in.readlines()]
        r = input[0][:]
        p = input[1][:]
        d = input[2][:]
        v = input[3][:]

        if first: result = first_solve(r, p, d, v)
        else: result = solve(r, p, d, v)

        return result
    
# R and t should be in the range of [0.1,0.9]
def generateOrderSattelike(n, R=0.9, t=0.2, q=0.4):
    p = [1]*n
    v = [1]*n
    p_T = round(sum(p) * q)
    range_r = np.random.randint(1,ceil(p_T * t)+1)
    r = np.random.randint(1, range_r+1, n)
    d_range = np.random.randint(floor(p_T * (1-t-R/2)), floor(p_T * (1-t+R/2)) + 1, n)
    d = r + np.maximum(d_range, p)
    return genInstanceString(r, p, d, v)

def genInstanceString(r, p, d, v):
    assert len(r) == len(p) == len(d) == len(v), "Non-equal length of r, p, d and v"
    return "\n".join([",".join([str(l) for l in el]) for el in [r,p,d,v]])

def all_tests():
    #Open examples folder and get the name of the files inside it.
    examples = os.listdir("examples")
    fails = 0
    total_time = 0
    # Run the algorithm for each file in the examples folder. If the algorithm spends more than 30 seconds in one file, it will stop and go to the next one.
    for file in examples:
        start = time.time()
        print("File: ", file)
        try: result = runFromFile("examples/" + file)
        except: 
            print("Time limit exceeded.") 
            fails += 1
        end = time.time()
        example_time = end - start
        total_time += example_time
        print("Result: ", result)
        print("Time taken: ", example_time)
        print("-"*50)

    print("Number of fails: ", fails)
    print("Average time: ", total_time / (len(examples)))

def average_time():
    total_time = 0
    for i in range(10): 
        start = time.time()
        runFromFile("examples/Dataslack_20orders_Tao7R5_1.csv")
        end = time.time()
        print("Time taken: ", end - start)
        total_time += end - start

    print("Average time: ", total_time / 10)

if __name__ == "__main__":

    result = runFromFile("examples/Dataslack_20orders_Tao7R5_1.csv")
    print("Result: ", result)

    # Generate instances from 10 to 100 jobs with 10 jobs increment
    # Create the file in generated folder.
    for i in range(1, 12):
        instance = generateOrderSattelike(i*2)
        with open("generated/instance_" + str(i*2) + ".csv", "w") as f_out:
            f_out.write(instance)
    
    improved_times = []
    first_times = []

    # Run the algorithm for each file in the generated folder. 
    # If the algorithm spends more than 10 seconds in one file, it will stop and go to the next one.
    for i in range(1, 12):
        start = time.time()
        print("File: ", "instance_" + str(2*i) + ".csv")

        print("Improved version:")
        # Trying with improves version
        try: result = runFromFile("generated/instance_" + str(2*i) + ".csv")
        except: print("Time limit exceeded.") 
        end = time.time()
        total_time = end - start
        improved_times.append(total_time)
        print("Improved Time taken: ", total_time)


        print("First version:")
        # Trying with first version
        start = time.time()
        try: result = runFromFile("generated/instance_" + str(2*i) + ".csv", True)
        except: print("Time limit exceeded.") 
        end = time.time()
        total_time = end - start
        first_times.append(total_time)
        print("First Time taken: ", total_time)


    # Plot time taken for each version and save plot in plots folder
    plt.plot(range(2, 24, 2), improved_times, label="Improved")
    plt.plot(range(2, 24, 2), first_times, label="First")
    plt.xlabel("Number of jobs")
    plt.ylabel("Time taken (s)")
    plt.legend()
    plt.savefig("plots/times.png")