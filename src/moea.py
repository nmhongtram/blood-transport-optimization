# Implementation of Hybrid Multi-Objective Evolutionary Algorithms
# Modified from Non-dominated Sorting Genetic Algorithm II (NSGA-II) (Deb et al., 2002)
# and Strength Pareto Evolutionary Algorithm 2 (SPEA2) (Zitzler et al., 2001)


import numpy as np
import copy
    

class NSGA2:
    def __init__(self, problem, population_size=50, num_generations=20, mutation_rate=0.1, crossover_rate=0.9, mutation_type="swap", crossover_type="OX"):
        self.problem = problem
        self.pop_size = population_size
        self.num_generations = num_generations 
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type


    def initialize_pop(self):
        # Initialize population using Random Initialization method
        # A population includes pop_size individuals
        pop = []
        while len(pop) < self.pop_size:
            indiv = self.problem.initialize_individual()
            if indiv == None:
                continue
            else:  
                indiv.ob1 = self.problem.calc_ob1(indiv)
                indiv.ob2 = self.problem.calc_ob2(indiv)
                pop.append(indiv)
        return pop
        

    def fnds(self, pop):
        # Fast Non-Dominated Sorting Algorithm
        front_dict = {}
        front_dict[1] = []

        # Stage 1
        for idx1, indiv1 in enumerate(pop):
            indiv1.S = []
            indiv1.n = 0
            for idx2, indiv2 in enumerate(pop):
                if idx2 != idx1:
                    dom = indiv1.check_dom(indiv2)
                    # If indiv1 dominates indiv2
                    if dom == "dominate":
                        # Add indiv2 to the set of solutions dominated by indiv1
                        indiv1.S.append(indiv2)
                    elif dom == "dominated":
                        # Increase the domination counter of indiv1
                        indiv1.n += 1

            if indiv1.n == 0:
                indiv1.rank = 1
                # indiv1 belongs to the first front
                front_dict[1].append(indiv1)

        # Stage 2
        i = 1
        while front_dict[i]:
            Q = [] # Used to store members of the next front
            for indiv1 in front_dict[i]:
                for indiv2 in indiv1.S:
                    indiv2.n -= 1
                    if indiv2.n == 0:
                        # indiv2 belongs to the next front
                        indiv2.rank = i + 1
                        Q.append(indiv2)
            
            i += 1
            front_dict[i] = Q

        del front_dict[i] # Remove the last front with 0 solution
        return front_dict

    
    def calc_cd(self, front):
        # If the front has 1 or 2 solutions, these solutions have cd = inf
        if len(front) <= 2:
            for i in range(len(front)):
                front[i].d = np.inf
            return

        r = len(front) - 1
        for indiv in front:
            indiv.d = 0

        # Objective 1
        F = sorted(front, key=lambda indiv:indiv.ob1)
        F[0].d = np.inf
        F[r].d = np.inf
        normalized_range = F[r].ob1 - F[0].ob1
        # Ensure that normalized_range is not zero to avoid division by zero
        if normalized_range != 0:
            for i in range(1, r-1):
                F[i].d = F[i].d + (np.abs(F[i+1].ob1 - F[i-1].ob1) / normalized_range)
        else:
            # Handle the case when normalized_range is zero
            for i in range(1, r-1):
                F[i].d = F[i].d  + (np.abs(F[i+1].ob1 - F[i-1].ob1) / 0.0000001)

        # Objective 2
        F = sorted(front, key=lambda indiv:indiv.ob2)
        F[0].d = np.inf
        F[r].d = np.inf
        normalized_range = F[r].ob2 - F[0].ob2
        # Ensure that normalized_range is not zero to avoid division by zero
        if normalized_range != 0:
            for i in range(1, r-1):
                F[i].d = F[i].d + (np.abs(F[i+1].ob2 - F[i-1].ob2) / normalized_range)
        else:
            # Handle the case when normalized_range is zero 
            for i in range(1, r-1):
                F[i].d = F[i].d  + (np.abs(F[i+1].ob2 - F[i-1].ob2) / 0.0000001)
        


    def select(self, pop):
        # Survival of the fittest
        # Crowded Binary Tournament Selection Operator (without replacement)
        # For solution i and j, one with better rank get selected
        # When rank of solution i and j is the same, choose one with larger crowding distance
        # In case of tie, select randomly
        selected_pop = []
        copy_pop = copy.deepcopy(pop)
        while len(selected_pop) < self.pop_size:
            # Ensure there are enough solutions for selection
            if len(copy_pop) < 2:
                copy_pop = copy.deepcopy(pop)

            i, j = np.random.choice(len(copy_pop), 2, replace=False)
            indiv1 = copy_pop.pop(i)
            indiv2 = copy_pop.pop(j - 1 if j > i else j)

            if indiv1.rank < indiv2.rank:
                selected_pop.append(indiv1)
            elif indiv1.rank > indiv2.rank:
                selected_pop.append(indiv2)
            else:
                if indiv1.d >= indiv2.d:
                    selected_pop.append(indiv1)
                else:
                    selected_pop.append(indiv2)

        return selected_pop
    
    

    def variate(self, pop):
        """
        Perform crossover and mutation to generate a new population while adhering to constraints.

        Args:
            pop (list): Current population of individuals.

        Returns:
            list: Variated population.
        """
        variated_pop = []
        copy_pop = copy.deepcopy(pop)

        while len(variated_pop) < self.pop_size:
            # Ensure there are enough solutions for selection
            if len(copy_pop) < 2:
                copy_pop = copy.deepcopy(pop)
            
            # Randomly select two parents and remove them from the copy population
            parent_indices = np.random.choice(len(copy_pop), size=2, replace=False)
            parent1 = copy_pop.pop(parent_indices[0])
            # When parent_indices[0] is deleted, indexes of the following individuals += 1
            parent2 = copy_pop.pop(parent_indices[1] - 1 if parent_indices[1] > parent_indices[0] else parent_indices[1])

            # Perform crossover
            if np.random.rand() <= self.crossover_rate:
                customers1, customers2 = self.problem.crossover(parent1.customers, parent2.customers, type=self.crossover_type)
                child1 = self.problem.repair_individual(customers1)
                child2 = self.problem.repair_individual(customers2)
                if child1 != None:
                    variated_pop.append(child1)
                if child2 != None and len(variated_pop) < self.pop_size:
                    variated_pop.append(child2)

            else:
                # Copy the parents directly
                variated_pop.append(parent1)
                if len(variated_pop) < self.pop_size:
                    variated_pop.append(parent2)

        # Perform mutation on the variated population
        for i in range(len(variated_pop)):
            if np.random.rand() <= self.mutation_rate:
                mutated = None
                mutated_customers = self.problem.mutate(variated_pop[i].customers, type=self.mutation_type)
                mutated = self.problem.repair_individual(mutated_customers)
                if mutated != None:
                    variated_pop[i] = mutated
            variated_pop[i].ob1 = self.problem.calc_ob1(variated_pop[i])
            variated_pop[i].ob2 = self.problem.calc_ob2(variated_pop[i])

        return variated_pop
    

    def local_search(self, pop):
        opt_pop = copy.deepcopy(pop)
        for indiv in opt_pop:
            new_sub_routes = []
            new_customers = []
            for route in indiv.sub_routes:
                opt_route = self.problem.optimize_3opt(route)
                result = self.problem.repair_sub_route(opt_route)
                if result == None:
                    break
                sub_routes, customers = result
                new_sub_routes.extend(sub_routes)
                new_customers.extend(customers)
            indiv.sub_routes = new_sub_routes
            indiv.customers = new_customers 
            indiv.ob1 = self.problem.calc_ob1(indiv)
            indiv.ob2 = self.problem.calc_ob2(indiv)
        return opt_pop



    def eliminate(self, pop):
        # Survival of the fittest
        # Combines parent and offspring populations and choose the best pop_size solutions
        # base on rank and crowding distance
        front_dict = self.fnds(pop)
        for rank, front in front_dict.items():
            self.calc_cd(front)

        next_pop = []
        next_front_dict = {}
        for rank, front in front_dict.items():
            length = len(front) + len(next_pop)
            if length < self.pop_size:
                next_pop.extend(front)
                next_front_dict[rank] = front
            elif length == self.pop_size:
                next_pop.extend(front)
                next_front_dict[rank] = front
                break
            else:
                F = sorted(front, key=lambda indiv:indiv.d, reverse=True)
                idx = self.pop_size - len(next_pop)
                next_pop.extend(F[:idx])
                next_front_dict[rank] = F[:idx]
                break
        
        return next_pop, next_front_dict


    def nsga2(self):
        gen = 0
        P = self.initialize_pop()
        # self.evaluate_obs(P)
        front_dict = self.fnds(P)
        for rank, front in front_dict.items():
            self.calc_cd(front)

        while gen < self.num_generations:
            gen += 1
            M = self.select(P)
            Q = self.variate(M)
            # self.evaluate_obs(Q)

            # Local Search in Q neighborhood
            LS = []
            if gen % 10 == 0:
                LS = self.local_search(Q)

            pop = P + Q + LS
            next_P, front_dict = self.eliminate(pop)
            P = next_P
            

        return P




class SPEA2:
    def __init__(self, problem, population_size=50, archive_size=None, num_generations=20, mutation_rate=0.1, crossover_rate=0.9, mutation_type="swap", crossover_type="OX"):
        self.problem = problem
        self.pop_size = population_size
        if archive_size is None:
            self.arc_size = self.pop_size
        else:
            self.arc_size = archive_size
        self.k = round(np.sqrt(self.pop_size + self.arc_size)) # Round up or round down ????
        self.num_generations = num_generations 
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type


    # General framework 
    def initialize_pop(self):
        # Initialize population using Random Initialization method
        # A population includes pop_size individuals
        pop = []
        while len(pop) < self.pop_size:
            indiv = self.problem.initialize_individual()
            if indiv == None:
                break
            else:  
                indiv.ob1 = self.problem.calc_ob1(indiv)
                indiv.ob2 = self.problem.calc_ob2(indiv)
                pop.append(indiv)
        return pop
    

    
    def calc_raw_fitness(self, pop):
        # Calculate strength + raw fitness and Return Non-dominated solutions
        nd_solutions = [] 
        other_solutions = []

        # Calculate strength - the number of solutions an indiv dominates
        for idx1, indiv1 in enumerate(pop):
            indiv1.strength = 0
            indiv1.is_dominated_by = []
            for idx2, indiv2 in enumerate(pop):
                if idx2 != idx1:
                    dom = indiv1.check_dom(indiv2)
                    if dom == "dominate":
                        indiv1.strength += 1
                    elif dom == "dominated":
                        indiv1.is_dominated_by.append(indiv2)

            # Calculate raw fitness - the summation of strength of solutions dominating an indiv
            strengths = [indiv.strength for indiv in indiv1.is_dominated_by]
            indiv1.R = np.sum(strengths)

            # Non-dominated solutions which have raw fitness = 0
            if indiv1.R == 0:
                nd_solutions.append(indiv1)
            else:
                other_solutions.append(indiv1)

        return nd_solutions, other_solutions
            
    
    def calc_density(self, pop):
        # Calculate kth distance + density + total fitness and return distance matrix
        # k-th Nearest Neighnor method for Diversity 
        # For each solution, the distances in the objective space to all solutions are calculated and sorted in ascending order
        # Extract objective values
        f1 = [indiv.ob1 for indiv in pop]
        f2 = [indiv.ob2 for indiv in pop]
        
        # Normalize objectives
        f1_range = np.max(f1) - np.min(f1)
        f1_range = f1_range if f1_range != 0 else 0.00001

        f2_range = np.max(f2) - np.min(f2)
        f2_range = f2_range if f2_range != 0 else 0.00001

        normalized_f1 = [(val - np.min(f1)) / f1_range for val in f1]
        normalized_f2 = [(val - np.min(f2)) / f2_range for val in f2]

        # Initialize a 2D array to store distances
        n = len(pop)
        distance_matrix = np.zeros((n, n))

        # Calculate pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(
                    (normalized_f1[i] - normalized_f1[j])**2 +
                    (normalized_f2[i] - normalized_f2[j])**2
                )
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

        # Find the k-th nearest neighbor for each individual
        for i in range(n):
            distances = distance_matrix[i]
            distances = np.sort(distances[distances > 0])  # Exclude self-distance (0)
            k_distance = distances[self.k-1] if len(distances) >= self.k else float('inf')
            pop[i].D = 1 / (k_distance + 2)
            # Calculate total fitness
            pop[i].F = pop[i].R + pop[i].D
            
        return distance_matrix
    
        
    
    def select(self, pop):
        # Survival of the fittest
        # Binary tournament selection with replacement 
        selected_pop = []
       
        while len(selected_pop) < self.pop_size:
            i, j = np.random.choice(len(pop), 2, replace=False)
            indiv1 = pop[i]
            indiv2 = pop[j]
            if indiv1.F <= indiv2.F:
                selected_pop.append(indiv1)
            else:
                selected_pop.append(indiv2)

        return selected_pop
    
    
    # Geeneral framework 
    def variate(self, pop):
        """
        Perform crossover and mutation to generate a new population while adhering to constraints.

        Args:
            pop (list): Current population of individuals.

        Returns:
            list: Variated population.
        """
        variated_pop = []
        copy_pop = copy.deepcopy(pop) 

        while len(variated_pop) < self.pop_size:
            # Ensure there are enough solutions for selection
            if len(copy_pop) < 2:
                copy_pop = copy.deepcopy(pop)

            # Randomly select two parents and remove them from the copy population
            parent_indices = np.random.choice(len(copy_pop), size=2, replace=False)
            parent1 = copy_pop.pop(parent_indices[0])
            # When parent_indices[0] is deleted, indexes of the following individuals += 1
            parent2 = copy_pop.pop(parent_indices[1] - 1 if parent_indices[1] > parent_indices[0] else parent_indices[1])

            # Perform crossover
            if np.random.rand() <= self.crossover_rate:
                customers1, customers2 = self.problem.crossover(parent1.customers, parent2.customers, type=self.crossover_type)
                child1 = self.problem.repair_individual(customers1)
                child2 = self.problem.repair_individual(customers2)
                if child1 != None:
                    variated_pop.append(child1)
                if child2 != None:
                    variated_pop.append(child2)

            else:
                # Copy the parents directly
                variated_pop.append(parent1)
                variated_pop.append(parent2)

        # Perform mutation on the variated population
        for i in range(len(variated_pop)):
            if np.random.rand() <= self.mutation_rate:
                mutated = None
                mutated_customers = self.problem.mutate(variated_pop[i].customers, type=self.mutation_type)
                mutated = self.problem.repair_individual(mutated_customers)
                if mutated != None:
                    variated_pop[i] = mutated
            variated_pop[i].ob1 = self.problem.calc_ob1(variated_pop[i])
            variated_pop[i].ob2 = self.problem.calc_ob2(variated_pop[i])

        return variated_pop
    

    def local_search(self, pop):
        opt_pop = copy.deepcopy(pop)
        for indiv in opt_pop:
            new_sub_routes = []
            new_customers = []
            for route in indiv.sub_routes:
                opt_route = self.problem.optimize_3opt(route)
                result = self.problem.repair_sub_route(opt_route)
                if result == None:
                    break
                sub_routes, customers = result
                new_sub_routes.extend(sub_routes)
                new_customers.extend(customers)
            indiv.sub_routes = new_sub_routes
            indiv.customers = new_customers 
            indiv.ob1 = self.problem.calc_ob1(indiv)
            indiv.ob2 = self.problem.calc_ob2(indiv)
        return opt_pop
    
    
    def eliminate(self, pop):
        # Survival of the fittest
        next_arc = []

        # Calculate total fitness
        nd_solutions, other_solutions = self.calc_raw_fitness(pop)
        self.calc_density(pop)

        num_nds = len(nd_solutions)
        # Case 1
        if num_nds == self.arc_size:
            return nd_solutions
        # Case 2
        elif num_nds < self.arc_size:
            next_arc.extend(nd_solutions)
            other_solutions.sort(key=lambda indiv: indiv.F)
            idx = self.arc_size - num_nds
            next_arc.extend(other_solutions[:idx])
        # Case 3
        else:
            return self.truncate(nd_solutions)
        
        return next_arc


    def truncate(self, nd_solutions):
        # Extract objective values
        f1 = [indiv.ob1 for indiv in nd_solutions]
        f2 = [indiv.ob2 for indiv in nd_solutions]
        
        # Normalize objectives
        f1_range = np.max(f1) - np.min(f1)
        f1_range = f1_range if f1_range != 0 else 0.00001

        f2_range = np.max(f2) - np.min(f2)
        f2_range = f2_range if f2_range != 0 else 0.00001

        normalized_f1 = [(val - np.min(f1)) / f1_range for val in f1]
        normalized_f2 = [(val - np.min(f2)) / f2_range for val in f2]

         # Initialize a 2D array to store distances
        n = len(nd_solutions)
        distance_matrix = np.zeros((n, n))

        # Calculate pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(
                    (normalized_f1[i] - normalized_f1[j])**2 +
                    (normalized_f2[i] - normalized_f2[j])**2
                )
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

        remove_count = 0
        while n - remove_count > self.arc_size:
            # Find the k-th nearest neighbor for each individual
            k_distances = {}
            for i in range(n):
                if nd_solutions[i] != None:
                    distances = distance_matrix[i]
                    distances = np.sort(distances[distances > 0])  # Exclude self-distance (0)
                    k_distance = distances[self.k-1] if len(distances) >= self.k else float('inf')
                    k_distances[i] = k_distance

            # remove a solution which has minimum k_distance from nd_solutions
            idx, indiv = min(k_distances.items(), key=lambda item: item[1])
            nd_solutions[idx] = None
            remove_count += 1

        return [indiv for indiv in nd_solutions if indiv != None]


    def spea2(self):
        gen = 0
        P = self.initialize_pop()
        # self.evaluate_obs(P)
        # Calculate total fitness
        self.calc_raw_fitness(P)
        self.calc_density(P)
        A = P

        while gen < self.num_generations:
            M = self.select(A)
            next_P = self.variate(M)
            # self.evaluate_obs(next_P)
            LS = []
            if gen % 10 == 0:
                LS = self.local_search(next_P)
            pop = next_P + A + LS
            next_A = self.eliminate(pop)
            A = next_A
            gen += 1

        return A


    