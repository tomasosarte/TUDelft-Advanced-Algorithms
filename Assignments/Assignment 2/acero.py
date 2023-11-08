from mip import *
from enum import Enum, unique
import numpy as np
import math
from typing import List, Tuple

INFINITY = float('inf')
EPSILON = 1e-6

class ProblemType(Enum):
    MAXIMIZATION = 1
    MINIMIZATION = 2

class VariableSelectionStrategy(Enum):
    LECTURE = 1
    SELF = 2

class Solution:
        
    def __init__(self, problem_type: ProblemType, selection_strategy: VariableSelectionStrategy):
        """
        Constructor of the class. It receives the problem type and the variable selection strategy.
        Args:
            problem_type: Type of the problem (minimization or maximization).
            selection_strategy: Variable selection strategy (lecture or self).
        Returns:
            None
        """
        self.problem_type = problem_type
        self.selection_strategy = selection_strategy

    def is_ILP_solution(self, m):
        """
        Checks if the solution of the LP relaxation is an integer solution.
        Args:
            m: Model of the problem.
        Returns:
            True if the solution is integer, False otherwise.
        """
        for var in m.vars:
            if not self.is_integral(var.x):
                return False
        return True
    
    def is_integral(self, x):
        """
        Checks if a number is integer.
        Args:
            x: Number to check.
        Returns:
            True if the number is integer, False otherwise.
        """
        return (abs(x - round(x)) < EPSILON)

    def variable_selection_method_lecture(self, variables: List[mip.Var]) -> mip.Var:
        """
        Selects the variable to branch on. It selects the variable with the fractional part closest to 0.5.
        Args:
            variables: List of mip variables.
        Returns:
            The selected variable.
        """
        score = 1
        best = None
        for x in variables:
            result = abs((x.x % 1) - 0.5)
            if result < score:
                score = result
                best = x
        return best
    
    def variable_selection_method_self(self, variables: List[mip.Var]) -> mip.Var:
        """
        Selects the variable to branch on. It selects the first variable which is not integral.
        Args:
            variables: List of mip variables.
        Returns:
            The selected variable.
        """
        for x in variables:
            if not self.is_integral(x.x):
                return x
        return None

    def add_models_to_stack(self, models: List[Model], m: Model, x: mip.Var) -> List[Model]:
        """
        Adds the two models created by branching on the given variable to the stack.
        Args:
            models (List[Model]): The stack of models.
            m (Model): The model to branch on.
            x (mip.Var): The variable to branch on.
        Returns:
            List[Model]: The stack of models with the new models added.
        """
        m_left = m.copy()
        m_right = m.copy()
        m_left += x <= math.floor(x.x)
        m_right += x >= math.ceil(x.x)
        m_left.verbose = 0
        m_right.verbose = 0
        models.append(m_left)
        models.append(m_right)

        return models
    
    def branch_and_bound(self, m: Model) -> Tuple[List[int], int]:
        """
        Executes the branch and bound algorithm to solve the (M)ILP given problem.
        Args:
            m: Model of the problem.
        Returns:
            A tuple with the optimal solution and the optimal objective value.
        """
        #1. Set the set of current problems to be problem_stack = [m]
        m.verbose = 0
        problem_stack = [m]

        #2. Set the upper bound to be infinity and the lower bound to be -infinity
        if self.problem_type == ProblemType.MINIMIZATION:
            upper_bound = INFINITY
        if self.problem_type == ProblemType.MAXIMIZATION:
            lower_bound = -INFINITY

        #3. Set the optimal solution to be None
        optimal_solution = None
        
        #4. While the problem stack is not empty:
        while len(problem_stack) > 0:
            #4.1 Choose current problem from the stack
            current_problem = problem_stack.pop()

            #4.2 Solve the LP relaxation of the current problem
            status = current_problem.optimize(relax=True)

            #4.3 PRUNATION PHASE
            
            # 4.3.1 If the current problem has no feasible solution, PRUNE BY INFEASIBILITY:
            if status == OptimizationStatus.INFEASIBLE or status == OptimizationStatus.NO_SOLUTION_FOUND or current_problem.objective_value == None: continue  #objective value is None if the model was not optimized(i.e solution not found)
            else:
                #4.3.2 Set the lower/upper bound of the current problem
                if self.problem_type == ProblemType.MINIMIZATION: 
                    subproblem_lower_bound = math.ceil(current_problem.objective_value)
                else: subproblem_upper_bound = math.floor(current_problem.objective_value)

                #4.3.3 If the current solution is integral and bound constraints are satisfied, PRUNE BY OPTIMALITY:
                if (status == OptimizationStatus.OPTIMAL and self.is_ILP_solution(current_problem)):               
                    if self.problem_type == ProblemType.MINIMIZATION and subproblem_lower_bound < upper_bound :
                        upper_bound = subproblem_lower_bound
                        optimal_solution = current_problem.vars            
                        continue
                    if  self.problem_type == ProblemType.MAXIMIZATION and subproblem_upper_bound > lower_bound :
                        lower_bound = subproblem_upper_bound
                        optimal_solution = current_problem.vars             
                        continue
                
                #4.3.4 PRUNE BY BOUND:
                if self.problem_type == ProblemType.MINIMIZATION:
                    if subproblem_lower_bound >= upper_bound: continue
                if self.problem_type == ProblemType.MAXIMIZATION:
                    if subproblem_upper_bound <= lower_bound: continue

                #4.4 Branch the current problem adding two new subproblems to the stack
                
                #4.4.1 Select variable to branch on
                if self.selection_strategy == VariableSelectionStrategy.LECTURE: x = self.variable_selection_method_lecture(current_problem.vars)
                else: x = self.variable_selection_method_self(current_problem.vars)

                #4.4.2 Branch on the variable
                problem_stack = self.add_models_to_stack(problem_stack, current_problem, x)

        #5. Return the optimal solution and the optimal objective value
        if optimal_solution is None:
            if self.problem_type == ProblemType.MAXIMIZATION:
                return [], -INFINITY
            return [], INFINITY
        
        optimal_solution = [int(x.x) for x in optimal_solution]
        if self.problem_type == ProblemType.MAXIMIZATION:
            return (optimal_solution, lower_bound)
        return (optimal_solution, upper_bound)
    
if __name__ == '__main__':
    
    # Measure time
    import time
    start_time = time.time()

    for i in range(1):
       # Example 1
        m = Model(sense=MAXIMIZE)
        m.read("random.mps")
        sol = Solution(ProblemType.MAXIMIZATION, VariableSelectionStrategy.LECTURE)
        optimal_solution, optimal_objective_value = sol.branch_and_bound(m)

        print("Optimal solution: ", optimal_solution)
        print("Optimal objective value: ", optimal_objective_value)

        # Example 2
        m = Model(sense=MAXIMIZE)
        m.read("knapsack_students.mps")
        sol = Solution(ProblemType.MAXIMIZATION, VariableSelectionStrategy.LECTURE)
        optimal_solution, optimal_objective_value = sol.branch_and_bound(m)

        print("Optimal solution: ", optimal_solution) 
        print("Optimal objective value: ", optimal_objective_value)

        # Example 3
        m = Model(sense=MINIMIZE)
        m.read("g503inf.mps")
        sol = Solution(ProblemType.MINIMIZATION, VariableSelectionStrategy.LECTURE)
        optimal_solution, optimal_objective_value = sol.branch_and_bound(m)

        print("Optimal solution: ", optimal_solution)
        print("Optimal objective value: ", optimal_objective_value)

    # End time
    print("--- %s seconds ---" % ((time.time() - start_time)))