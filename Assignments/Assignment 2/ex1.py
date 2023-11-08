from mip import *
from enum import Enum, unique
import numpy as np
import math
from typing import List, Tuple

INFINITY = float('inf')
EPSILON = 10e-8

class ProblemType(Enum):
    MAXIMIZATION = 1
    MINIMIZATION = 2

class VariableSelectionStrategy(Enum):
    LECTURE = 1
    SELF = 2

class Solution:

    def __init__(self, problem_type: ProblemType, selection_strategy: VariableSelectionStrategy):
        """
        Constructor for the for solutions to (M)ILP problems using Branch & Bound.
        Args:
            problem_type (ProblemType): The type of the problem (maximization or minimization)
            selection_strategy (VariableSelectionStrategy): The strategy to use for selecting the next variable to branch on (lecture or self)
        """
        self.problem_type = problem_type
        self.selection_strategy = selection_strategy

    def is_ILP_solution(self, vars: [mip.Var]) -> bool:
        """
        Checks if the given solution is an integer solution.
        Args:
            vars ([mip.Var]): The variables to check.
        Returns:
            bool: True if the solution is an integer solution, False otherwise.
        """
        for x in vars:
            if abs(x.x - math.floor(x.x)) > EPSILON and abs(math.ceil(x.x) - x.x) > EPSILON:
                return False
        return True

    def selection_of_variable(self, vars: [mip.Var]) -> mip.Var:
        """
        Selects the next variable to branch on.
        Args:
            vars ([mip.Var]): The variables to choose from.
        Returns:
            mip.Var: The variable to branch on.
        """
        # LECTURE
        if self.selection_strategy == VariableSelectionStrategy.LECTURE:
            return self.variable_selection_method_lecture(vars)
        # SELF
        return self.variable_selection_method_self(vars)
    
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

        # Branch on the variable
        m_left = m.copy()
        m_right = m.copy()
        m_left += x <= math.floor(x.x)
        m_right += x >= math.ceil(x.x)
        m_left.verbose = 0
        m_right.verbose = 0
        m_left.optimize(relax=True)
        m_right.optimize(relax=True)

        # Append the new models if not infeasible
        # if m_left.num_solutions > 0: models.append(m_left)
        # if m_right.num_solutions > 0: models.append(m_right)

        models.append(m_left)
        models.append(m_right)

        return models
    
    def best_node_first(self, models: List[Model]) -> int:  
        """
        Returns the index of the best model in the stack.
        Args:
            models (List[Model]): The stack of models.
        Returns:
            Int: The index of the best model in the stack.
        """

        best = 0
        for i in range(1, len(models)):
            if models[i].objective_value > models[best].objective_value:
                best = i
        return best

    def branch_and_bound1(self, m: Model) -> Tuple[List[int], int]:
        """
        Solves the given Maximixation (M)ILP problem using Branch & Bound with lecture variable selection.
        Args:
            m (Model): The (M)ILP problem to solve.
        Returns:
            Tuple[List[int], int]: A tuple containing the optimal solution and the optimal objective value.
        """
        m.verbose = 0

        # Solve the model
        m.optimize(relax=True)

        # Return if the problem is infeasible
        if m.num_solutions == 0:
            if self.problem_type == ProblemType.MAXIMIZATION:
                return [], -INFINITY
            return [], INFINITY

        # Get the optimal solution
        optimal_solution = []

        # print("INITIAL SOLUTION: ", optimal_solution)
        # print("INITIAL OBJECTIVE VALUE: ", m.objective_value)   

        # Finish if the solution is already integer
        if self.is_ILP_solution(m.vars):
            return [int(x.x) for x in m.vars], m.objective_value
        
        # Bound objective value
        if self.problem_type == ProblemType.MAXIMIZATION:
            upper_bound = math.floor(m.objective_value)
            lower_bound = -INFINITY
        else: 
            upper_bound = INFINITY
            lower_bound = math.ceil(m.objective_value)

        # Select variable to branch on
        x = self.selection_of_variable(m.vars)

        models = []
        models = self.add_models_to_stack(models, m, x)

        while len(models) > 0:
            # Get next node (Best-node-first)
            # models.sort(key=lambda x: x.objective_value, reverse=True)
            # actual = models.pop(0)
            index = self.best_node_first(models)
            actual = models.pop(index)    
                  
            ####################
            # PRUNATION PHASE: #
            ####################

            # Prunation by Infeasibility (not necessary since nodes are not in stack if infeasible)
            if actual.num_solutions == 0: continue

            # Prunation by Bound
            if actual.objective_value <= lower_bound and self.problem_type == ProblemType.MAXIMIZATION: continue
            if actual.objective_value >= upper_bound and self.problem_type == ProblemType.MINIMIZATION: continue

            # Prunation by Optimality
            if self.is_ILP_solution(actual.vars):
                if self.problem_type == ProblemType.MAXIMIZATION and actual.objective_value > lower_bound:
                    lower_bound = actual.objective_value
                    optimal_solution = [int(x.x) for x in actual.vars]
                    continue
                if self.problem_type == ProblemType.MINIMIZATION and actual.objective_value < upper_bound:
                    upper_bound = actual.objective_value
                    optimal_solution = [int(x.x) for x in actual.vars]
                    continue

            # Select variable to branch on
            x = self.selection_of_variable(actual.vars)

            # Add new models to stack
            models = self.add_models_to_stack(models, actual, x)

        # MAXIMIZATION
        if self.problem_type == ProblemType.MAXIMIZATION:
            return optimal_solution, lower_bound
        # MINIMIZATION
        return optimal_solution, upper_bound

    def variable_selection_method_lecture(self, variables: List[mip.Var]) -> mip.Var:
        """
        Selects the next variable with fractional value closest to 1/2.
        Args:
            variables (List[mip.Var]): The variables to choose from.
        Returns:
            mip.Var: The variable to branch on.
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
        Selects the next variable to branch using a random variable.
        Args:
            variables (List[mip.Var]): The variables to choose from.
        Returns:
            mip.Var: The variable to branch on.
        """
        
        # Select a random variable
        return np.random.choice(variables)
    
    def minimization_branch_and_bound(self, m: Model) -> Tuple[List[int], int]:
        """
        Solves the given Maximixation (M)ILP problem using Branch & Bound.
        Args:
            m (Model): The (M)ILP problem to solve.
        Returns:
            Tuple[List[int], int]: A tuple containing the optimal solution and the optimal objective value.
        """
        
        optimal_solution = []
        C = [m]
        upper_bound = INFINITY

        while len(C) > 0:
            # print("STACK SIZE: ", len(C))

            # Get next node 
            current_problem = C[0]

            # Solve the model
            current_problem.verbose = 0
            status = current_problem.optimize(relax=True)

            # Prunation by Infeasibility
            if status == OptimizationStatus.INFEASIBLE or status == OptimizationStatus.NO_SOLUTION_FOUND or current_problem.objective_value == None: 
                # print("Prunation by Infeasibility")
                pass
            else: 

                current_objective_value = current_problem.objective_value
                current_lower_bound = math.ceil(current_objective_value)

                # Prunation by Optimality
                if status == OptimizationStatus.OPTIMAL and self.is_ILP_solution(current_problem.vars) and current_objective_value < upper_bound:
                    upper_bound = current_objective_value
                    optimal_solution = [int(x.x) for x in current_problem.vars]
                    # print("Prunation by Optimality")
                    # print("UPPER BOUND: ", current_objective_value)
                    
                # Prunation by bound
                elif current_lower_bound >= upper_bound: 
                    # print("Prunation by Bound")
                    pass

                # Branch current problem
                else:
                    # print("Branching")
                    x = self.selection_of_variable(current_problem.vars)
                    C = self.add_models_to_stack(C, current_problem, x)

            # Remove current problem from stack
            C.remove(current_problem)
            
        return optimal_solution, upper_bound

    def maximization_branch_and_bound(self, m: Model) -> Tuple[List[int], int]:
        """
        Solves the given Maximixation (M)ILP problem using Branch & Bound.
        Args:
            m (Model): The (M)ILP problem to solve.
        Returns:
            Tuple[List[int], int]: A tuple containing the optimal solution and the optimal objective value.
        """

        optimal_solution = []
        C = [m]
        lower_bound = -INFINITY

        while len(C) > 0:
            # print("STACK SIZE: ", len(C))

            # Get next node 
            current_problem = C[0]

            # Solve the model
            current_problem.verbose = 0
            status = current_problem.optimize(relax=True)

            # Prunation by Infeasibility
            if status == OptimizationStatus.INFEASIBLE or status == OptimizationStatus.NO_SOLUTION_FOUND or current_problem.objective_value == None: 
                # print("Prunation by Infeasibility")
                pass
            else: 

                current_objective_value = current_problem.objective_value
                current_upper_bound = math.floor(current_objective_value)

                # Prunation by Optimality
                if status == OptimizationStatus.OPTIMAL and self.is_ILP_solution(current_problem.vars) and current_objective_value > lower_bound:
                    lower_bound = current_objective_value
                    optimal_solution = [int(x.x) for x in current_problem.vars]
                    # print("Prunation by Optimality")
                    # print("LOWER BOUND: ", current_objective_value)
                    
                # Prunation by bound
                elif current_upper_bound <= lower_bound: 
                    # print("Prunation by Bound")
                    pass

                # Branch current problem
                else:
                    # print("Branching")
                    x = self.selection_of_variable(current_problem.vars)
                    C = self.add_models_to_stack(C, current_problem, x)

            # Remove current problem from stack
            C.remove(current_problem)

        else: return optimal_solution, lower_bound

    def branch_and_bound(self, m: Model) -> Tuple[List[int], int]:
        """
        Solves the given (M)ILP problem using Branch & Bound.
        Args:
            m (Model): The (M)ILP problem to solve.
        Returns:
            Tuple[List[int], int]: A tuple containing the optimal solution and the optimal objective value.
        """

        if self.problem_type == ProblemType.MAXIMIZATION:
            return self.maximization_branch_and_bound(m)
        return self.minimization_branch_and_bound(m)

if __name__ == '__main__':
    
    # Measure time
    import time
    start_time = time.time()

    # for i in range(10):
    # Example 1
    m = Model(sense=MAXIMIZE)
    m.read("random.mps")
    sol = Solution(ProblemType.MAXIMIZATION, VariableSelectionStrategy.LECTURE)
    optimal_solution, optimal_objective_value = sol.branch_and_bound(m)

    print("Optimal solution: ", optimal_solution)
    print("Optimal objective value: ", optimal_objective_value)
    print('-'*50)

    # Example 2
    m = Model(sense=MAXIMIZE)
    m.read("knapsack_students.mps")
    sol = Solution(ProblemType.MAXIMIZATION, VariableSelectionStrategy.LECTURE)
    optimal_solution, optimal_objective_value = sol.branch_and_bound(m)

    print("Optimal solution: ", optimal_solution) 
    print("Optimal objective value: ", optimal_objective_value)
    print('-'*50)

    # Example 3
    m = Model(sense=MINIMIZE)
    m.read("g503inf.mps")
    sol = Solution(ProblemType.MINIMIZATION, VariableSelectionStrategy.LECTURE)
    optimal_solution, optimal_objective_value = sol.branch_and_bound(m)

    print("Optimal solution: ", optimal_solution)
    print("Optimal objective value: ", optimal_objective_value)
    
    # End time
    print("--- %s seconds ---" % ((time.time() - start_time) / 10))
