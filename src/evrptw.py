import numpy as np


class EVRPTW:
    def __init__(self, data):
        self.name = data["NAME"]
        self.dimension = data["DIMENSION"]  # Dimension = num_customers + 1
        self.num_customers = self.dimension - 1
        self.num_stations = data["STATIONS"]
        self.max_capacity = data["CAPACITY"]
        self.max_energy_capacity = data["ENERGY_CAPACITY"]
        self.energy_consumption = data["ENERGY_CONSUMPTION"]
        self.charging_time = data["CHARGING_TIME"]

        self.nodes = data["NODES"]
        # self.coordinates = data["COORDINATES"]

        # self.demands = data["DEMANDS"]
        # self.ready_time = data["READY_TIME"]
        # self.due_time = data["DUE_TIME"]
        # self.service_time = data["SERVICE_TIME"]

        self.coordinates = {int(k): v for k, v in data["COORDINATES"].items()}
        self.demands = {int(k): v for k, v in data["DEMANDS"].items()}
        self.ready_time = {int(k): v for k, v in data["READY_TIME"].items()}
        self.due_time = {int(k): v for k, v in data["DUE_TIME"].items()}
        self.service_time = {int(k): v for k, v in data["SERVICE_TIME"].items()}

        self.distances = np.array(data["DISTANCES"])
        self.times = np.array(data["TIMES"])

        self.depot = 1
        self.customer_ids = np.array([id for id in self.demands.keys() if id != self.depot])
        self.station_ids = np.array([node["id"] for node in self.nodes if node["demand"] == 0 and node["id"] != self.depot])

        self.w_distance = 1
        self.w_fleet_size = 1500
   


    def initialize_individual(self):
        def geometric_selection(savings_list, p=0.2):
            """Select a saving using a geometric distribution."""
            # Calculate weights based on a geometric distribution
            weights = [(1 - p) ** i for i in range(len(savings_list))]
            weights = np.array(weights)
            probabilities = weights / weights.sum()

            selected_index = np.random.choice(len(savings_list), p=probabilities)
            return savings_list[selected_index]


        # Step 1: Compute savings
        savings_list = []
        customers = list(self.customer_ids)  # Danh sách khách hàng
        np.random.shuffle(customers)            # Xáo trộn ngẫu nhiên
        for i in customers:
            for j in customers:
                if i != j:
                    saving = self.distances[self.depot-1][i-1] + self.distances[self.depot-1][j-1] - self.distances[i-1][j-1]
                    savings_list.append((i, j, saving))
        savings_list.sort(key=lambda x: x[2], reverse=True)

        # Step 2: Initialize routes
        routes = {i: [self.depot, i, self.depot] for i in customers}
        capacities = {i: self.demands[i] for i in customers}
        energies = {i: self.max_energy_capacity - self.distances[self.depot-1][i-1] * self.energy_consumption for i in customers}
        times = {i: max(self.ready_time[i], self.times[self.depot-1][i-1]) + self.service_time[i] for i in customers}

        # Step 3: Merge routes using probabilistic savings
        while savings_list:
            i, j, _ = geometric_selection(savings_list)
            savings_list.remove((i, j, _))  # Remove selected saving to avoid duplication

            if i in routes and j in routes and i != j:
                route_i = routes[i]
                route_j = routes[j]

                new_route = route_i[:-1] + route_j[1:]
                new_capacity = capacities[i] + capacities[j]
                new_energy = energies[i]
                new_time = times[i]

                feasible = True
                for k in range(len(new_route) - 1):
                    current, next_customer = new_route[k], new_route[k + 1]
                    travel_time = self.times[current-1][next_customer-1]

                    new_time += travel_time + (self.service_time[next_customer] if next_customer in self.customer_ids else 0)
                    if next_customer in self.customer_ids and new_time > self.due_time[next_customer]:
                        feasible = False
                        break


                    energy_required = self.distances[current-1][next_customer-1] * self.energy_consumption
                    if new_energy < energy_required:
                        station = self.find_optimal_station(current, next_customer, new_energy)
                        if station:
                            new_route.insert(k + 1, station)
                            new_energy = self.max_energy_capacity - self.distances[station-1][next_customer-1] * self.energy_consumption
                            new_time += self.charging_time + self.times[station-1][next_customer-1]
                        else:
                            feasible = False
                            break
                    else:
                        new_energy -= energy_required
                        new_time += self.times[current-1][next_customer-1]

                    if new_time > self.due_time[next_customer]:
                        feasible = False
                        break


                if feasible and new_capacity <= self.max_capacity and new_time <= self.due_time[self.depot]:
                    routes[j] = new_route
                    capacities[j] = new_capacity
                    energies[j] = new_energy
                    times[j] = new_time
                    del routes[i]


        # Step 4: Return merged routes
        final_routes = list(routes.values())
        # Clean up routes to remove consecutive duplicates
        final_routes = [self.remove_consecutive_duplicates(route) for route in final_routes]

        for route in final_routes:
            if not (self.check_tw_constraint(route)[0]
                and self.check_capa_constraint(route)
                and self.check_e_constraint(route)):
                return None
            
        customers = []
        for route in final_routes:
            for id in route:
                if id in self.customer_ids:
                    customers.append(id)

        return Individual(final_routes, np.array(customers))

    
    
    def find_optimal_station(self, current, next_customer, energy_remaining):
        """Find an optimal station between current and next_customer."""
        # Stations reachable from current
        S1 = [station for station in self.station_ids if 
            energy_remaining >= self.distances[current-1][station-1] * self.energy_consumption]
        # Stations from which next_customer is reachable
        S2 = [station for station in self.station_ids if 
            self.max_energy_capacity >= self.distances[station-1][next_customer-1] * self.energy_consumption]
        # Intersection of feasible stations
        feasible_stations = set(S1).intersection(S2)
        if not feasible_stations:
            return None  # No valid station found
        # Choose station minimizing travel distance and maximizing energy remaining
        best_station = min(
            feasible_stations,
            key=lambda station: (
                self.distances[current-1][station-1] + self.distances[station-1][next_customer-1],
                -self.max_energy_capacity + self.distances[station-1][next_customer-1] * self.energy_consumption
            )
        )
        return best_station
    


    def remove_consecutive_duplicates(self, route):
        """Remove consecutive duplicate stations from a route."""
        cleaned_route = [route[0]]  # Always keep the first node
        for node in route[1:]:
            if node != cleaned_route[-1]:  # Only add node if it's not the same as the last one
                cleaned_route.append(node)
        return cleaned_route


    

    def check_tw_constraint(self, route):
        cur_time = 0
        cus_wait_time = 0

        for idx, id in enumerate(route[:-1]):  # Prevent index out of bounds
            next_id = route[idx + 1]
            arrival_time = cur_time + self.times[id - 1][next_id - 1]
            cur_time += self.times[id - 1][next_id - 1]

            if next_id in self.customer_ids:
                if arrival_time <= self.ready_time[next_id]:
                    # Early arrival: wait until ready time
                    cur_time = self.ready_time[next_id]  # Adjust time to ready time
                else:
                    cus_wait_time = cus_wait_time + (arrival_time - self.ready_time[next_id])

                # Add service time
                done_time = cur_time + self.service_time[next_id]

                if done_time > self.due_time[next_id]:
                    # Exceeded due time
                    return False, None
                
            elif next_id == self.depot:
                if cur_time > self.due_time[next_id]:
                    # Exceeded due time
                    return False, None

            
            elif next_id in self.station_ids:
                # Add charging time for stations
                cur_time += self.charging_time

        return True, cus_wait_time

    
   
    def check_capa_constraint(self, route):
        cur_capa = self.max_capacity
        for id in route:
            if id in self.customer_ids:
                cur_capa -= self.demands[id]
            if cur_capa < 0:
                return False
        return True
    
  
    def check_e_constraint(self, route):
        cur_energy = self.max_energy_capacity
        for idx, id in enumerate(route[:-1]):
            next_id = route[idx+1]
            cur_energy -= self.distances[id-1, next_id-1] * self.energy_consumption
            if next_id in self.station_ids:
                if cur_energy >= 0:
                    cur_energy = self.max_energy_capacity
                else:
                    return False 
        return True
    

    def calc_ob1(self, indiv):
        # Minimizing cost = total distance + num of vehicles
        total_distance = 0
        for sub_route in indiv.sub_routes:
            current_distance = 0
            for i in range(len(sub_route) - 1):
                current_distance += self.distances[sub_route[i]-1][sub_route[i + 1]-1] 
        
            total_distance += current_distance
        return total_distance + ((self.w_fleet_size/self.w_distance) * len(indiv.sub_routes))
    

    def calc_ob2(self, indiv):
        # Minimizing Customers' waiting time
        total_time = 0
        for route in indiv.sub_routes:
            total_time += self.check_tw_constraint(route)[1]
        return total_time
    

    def crossover(self, parent1, parent2, type='OX'):

        def random_cut(length):
            # Chọn cut1 ngẫu nhiên trong khoảng từ 1 đến length-1
            cut1 = np.random.randint(1, length - 1)
            
            # Chọn cut2 sao cho cách cut1 ít nhất 2 đơn vị
            if cut1 + 2 < length - 1:
                cut2 = np.random.randint(cut1 + 2, length - 1)
            else:
                cut2 = np.random.randint(1, cut1 - 1)
            
            return sorted([cut1, cut2])

        def crossover_OX(parent1, parent2):
            parent1 = list(parent1)
            parent2 = list(parent2)

            length = len(parent1)

            cut1, cut2 = random_cut(length)

            child1 = [-1] * length
            child2 = [-1] * length

            index = cut2
            child1[cut1:cut2] = parent1[cut1:cut2]
            parent_index = index
            while -1 in child1:
                if parent2[parent_index] not in child1:
                    child1[index] = parent2[parent_index]
                    index += 1
                    if index == length:
                        index = 0
                else:
                    parent_index += 1
                    if parent_index == length:
                        parent_index = 0

            index = cut2
            child2[cut1:cut2] = parent2[cut1:cut2]
            parent_index = index
            while -1 in child2:
                if parent1[parent_index] not in child2:
                    child2[index] = parent1[parent_index]
                    index += 1
                    if index == length:
                        index = 0
                else:
                    parent_index += 1
                    if parent_index == length:
                        parent_index = 0

            return np.array(child1), np.array(child2)


        def crossover_PMX(parent1, parent2):
            # Generate two random cut points
            length = len(parent1)
            cut1, cut2 = random_cut(length)

            # Initialize children with -1
            child1 = np.full(len(parent1), -1)
            child2 = np.full(len(parent2), -1)

            # Copy the segment between cut points from the parents
            child1[cut1:cut2] = parent1[cut1:cut2]
            child2[cut1:cut2] = parent2[cut1:cut2]

            # Create mappings from the segments
            mapping1 = {parent1[i]: parent2[i] for i in range(cut1, cut2)}
            mapping2 = {parent2[i]: parent1[i] for i in range(cut1, cut2)}

            # Fill the remaining positions for child1
            for i in range(len(parent1)):
                if i >= cut1 and i < cut2:
                    continue  # Skip the segment already copied
                val = parent2[i]
                while val in mapping1 and val in child1:
                    val = mapping1[val]
                child1[i] = val

            # Fill the remaining positions for child2
            for i in range(len(parent2)):
                if i >= cut1 and i < cut2:
                    continue  # Skip the segment already copied
                val = parent1[i]
                while val in mapping2 and val in child2:
                    val = mapping2[val]
                child2[i] = val

            return child1, child2


        def crossover_CX(parent1, parent2):
            # Initialize children as arrays filled with -1
            child1 = np.full(len(parent1), -1)
            child2 = np.full(len(parent2), -1)
            
            # Cycle for child1
            visited = np.zeros(len(parent1), dtype=bool)
            index = 0
            while not visited[index]:
                visited[index] = True
                child1[index] = parent1[index]
                indices = np.where(parent2 == parent1[index])[0]
                if len(indices) == 0:
                    break
                index = indices[0]
            
            for i in range(len(parent1)):
                if child1[i] == -1:
                    child1[i] = parent2[i]
            
            # Cycle for child2
            visited = np.zeros(len(parent2), dtype=bool)
            index = 0
            while not visited[index]:
                visited[index] = True
                child2[index] = parent2[index]
                indices = np.where(parent1 == parent2[index])[0]
                if len(indices) == 0:
                    break
                index = indices[0]
            
            for i in range(len(parent2)):
                if child2[i] == -1:
                    child2[i] = parent1[i]
            
            return child1, child2

    
        if type == 'OX':
            return crossover_OX(parent1, parent2)
        elif type == 'PMX':
            return crossover_PMX(parent1, parent2)
        elif type == 'CX':
            return crossover_CX(parent1, parent2)
        


    def mutate(self, parent, type="swap"):
    
        def swap_mutation(route):
            route = route.copy()
            point1, point2 = np.random.choice(len(route), size=2, replace=False)
            route[point1], route[point2] = route[point2], route[point1]
            return route

        def inverse_mutation(route):
            route = route.copy()
            point1, point2 = np.sort(np.random.choice(len(route), size=2, replace=False))
            route[point1:point2] = route[point1:point2][::-1]  # Reverse the subset
            return route

        def insert_mutation(route):
            route = route.copy()
            point1, point2 = np.random.choice(len(route), size=2, replace=False)
            city = route[point1]
            route = np.delete(route, point1)  # Loại bỏ phần tử tại point1
            route = np.insert(route, point2, city)  # Thêm city vào vị trí point2
            return route
        
        if type == 'swap':
            return swap_mutation(parent)
        elif type == 'inverse':
            return inverse_mutation(parent)
        elif type == 'insert':
            return insert_mutation(parent)
        


    def repair_individual(self, customers):
        # Step 1: Initialize the route by adding depot
        routes = []
        current_route = [self.depot]
        current_capacity = 0
        current_energy = self.max_energy_capacity
        current_time = 0

        for customer in customers:
            # Step 2: Capacity check
            if current_capacity + self.demands[customer] > self.max_capacity:
                # Finalize current route and start a new one
                current_route.append(self.depot)
                routes.append(current_route)
                current_route = [self.depot, customer]
                current_capacity = self.demands[customer]
                current_energy = self.max_energy_capacity - self.distances[self.depot-1][customer-1] * self.energy_consumption
                current_time = max(self.ready_time[customer], self.times[self.depot-1][customer-1]) + self.service_time[customer]
                continue

            # Step 3: Energy check
            last_node = current_route[-1]
            energy_needed = self.distances[last_node-1][customer-1] * self.energy_consumption
            if current_energy < energy_needed:
                # Find and insert a charging station
                station = self.find_optimal_station(last_node, customer, current_energy)
                if station:
                    current_route.append(station)
                    current_energy = self.max_energy_capacity - self.distances[station-1][customer-1] * self.energy_consumption
                    current_time += self.charging_time + self.times[station-1][customer-1]
                else:
                    # Backtracking to find feasible station
                    i = len(current_route) - 2
                    while i > 0 and current_route[i] not in self.station_ids:
                        prev_node = current_route[i]
                        station = self.find_optimal_station(prev_node, current_route[i+1], current_energy)
                        if station:
                            # Insert the found station and exit the loop
                            current_route.insert(i+1, station)
                            current_energy = self.max_energy_capacity - self.distances[station-1][current_route[i+2]-1] * self.energy_consumption
                            break
                        i -= 1
            
                    # This `else` runs only if the `while` loop completes without finding a station
                    else:
                        # If no station is feasible, finalize current route
                        current_route.append(self.depot)
                        routes.append(current_route)
                        current_route = [self.depot, customer]
                        current_capacity = self.demands[customer]
                        current_energy = self.max_energy_capacity - self.distances[self.depot-1][customer-1] * self.energy_consumption
                        current_time = max(self.ready_time[customer], self.times[self.depot-1][customer-1]) + self.service_time[customer]
                        continue


            # Step 4: Time window check
            arrival_time = current_time + self.times[last_node-1][customer-1]
            if arrival_time > self.due_time[customer]:
                # Finalize current route if time window is violated
                current_route.append(self.depot)
                routes.append(current_route)
                current_route = [self.depot, customer]
                current_capacity = self.demands[customer]
                current_energy = self.max_energy_capacity - self.distances[self.depot-1][customer-1] * self.energy_consumption
                current_time = max(self.ready_time[customer], self.times[self.depot-1][customer-1]) + self.service_time[customer]
                continue

            # Update route and metrics
            current_route.append(customer)
            current_capacity += self.demands[customer]
            current_energy -= energy_needed
            current_time = max(self.ready_time[customer], arrival_time) + self.service_time[customer]

        # Finalize the last route
        if current_route[-1] != self.depot:
            current_route.append(self.depot)
        routes.append(current_route)

        # Step 5: Ensure all customers are served
        assigned_customers = set(c for route in routes for c in route if c != self.depot)
        unassigned_customers = set(customers) - assigned_customers
        for customer in unassigned_customers:
            # Start a new route for unassigned customers
            routes.append([self.depot, customer, self.depot])

        # Clean up routes to remove consecutive duplicates
        final_routes = [self.remove_consecutive_duplicates(route) for route in routes]
        for route in final_routes:
            if not (self.check_tw_constraint(route)[0]
                and self.check_capa_constraint(route)
                and self.check_e_constraint(route)):
                return None
            
        customers = []
        for route in final_routes:
            for id in route:
                if id in self.customer_ids:
                    customers.append(id)

        return Individual(final_routes, np.array(customers))
    

    # Vì chạy rất lâu đối với số node lớn nên chỉ áp dụng sau 10 generation 
    def optimize_3opt(self, route):
        """
        Perform 3-opt optimization on a given sub-route, and use 2-opt for routes with only 3 customers.
        Exclude charging stations and keep the depot fixed at the start and end.

        Args:
            route (list): The sub-route to optimize (includes depot and may include charging stations).

        Returns:
            list: The optimized sub-route.
        """
        def calculate_route_cost(route):
            """Calculate the total cost (distance) of a route."""
            cost = 0
            for i in range(len(route) - 1):
                cost += self.distances[route[i] - 1][route[i + 1] - 1]
            return cost

        def optimize_2opt(route):
            """Perform 2-opt optimization for a given route."""
            best_route = route[:]
            best_cost = calculate_route_cost(best_route)
            n = len(best_route)

            improved = True
            while improved:
                improved = False
                for i in range(1, n - 2):  # Start from 1 to avoid the depot
                    for j in range(i + 1, n - 1):
                        # Reverse the segment between i and j
                        new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                        new_cost = calculate_route_cost(new_route)
                        if new_cost < best_cost:
                            best_route = new_route
                            best_cost = new_cost
                            improved = True
            return best_route

        # Step 1: Remove charging stations from the route
        customers_only = [node for node in route if node != self.depot and node not in self.station_ids]

        # Reconstruct the route with only the depot and customers
        current_route = [self.depot] + customers_only + [self.depot]

        # If the route has only 3 customers, use 2-opt
        if len(customers_only) == 3:
            return optimize_2opt(current_route)

        # Step 2: 3-opt optimization for routes with more than 3 customers
        best_route = current_route[:]
        best_cost = calculate_route_cost(best_route)
        improved = True

        while improved:
            improved = False
            n = len(best_route)

            # Iterate over all combinations of three edges, excluding depot (start and end)
            for i in range(1, n - 3):  # Start from 1 to avoid the depot
                for j in range(i + 1, n - 2):
                    for k in range(j + 1, n - 1):
                        # Generate all 7 possible new routes by rearranging the three edges
                        new_routes = [
                            best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:k + 1][::-1] + best_route[k + 1:],  # Reverse two segments
                            best_route[:i] + best_route[j + 1:k + 1] + best_route[i:j + 1] + best_route[k + 1:],              # Swap two segments
                            best_route[:i] + best_route[j + 1:k + 1] + best_route[i:j + 1][::-1] + best_route[k + 1:],     # Swap with reverse
                            best_route[:i] + best_route[i:j + 1] + best_route[k:j:-1] + best_route[k + 1:],              # Reverse the middle segment
                            best_route[:i] + best_route[k:j:-1] + best_route[j + 1:k + 1] + best_route[k + 1:],          # Reverse the middle and last segments
                            best_route[:i] + best_route[j + 1:k + 1][::-1] + best_route[i:j + 1] + best_route[k + 1:],   # Swap and reverse the last segment
                            best_route[:i] + best_route[k:j:-1] + best_route[i:j + 1][::-1] + best_route[k + 1:],        # Reverse all segments
                        ]

                        # Evaluate all new routes
                        for new_route in new_routes:
                            # Ensure new route contains all nodes (except the depot) without duplication
                            if sorted(new_route[1:-1]) != sorted(customers_only):
                                continue  # Skip invalid routes

                            new_cost = calculate_route_cost(new_route)
                            if new_cost < best_cost:
                                best_route = new_route
                                best_cost = new_cost
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        # Step 3: Return the optimized route
        return best_route





    def repair_sub_route(self, route):
        """Repair sub route after optimizing by local search methods"""
        # return list of routes (consider the case when route is split)
        # Step 1: Initialize the route by adding depot
        routes = []
        current_route = [self.depot]
        current_capacity = 0
        current_energy = self.max_energy_capacity
        current_time = 0

        for customer in route[1:-1]:
            # Step 2: Energy check
            last_node = current_route[-1]
            energy_needed = self.distances[last_node-1][customer-1] * self.energy_consumption
            if current_energy < energy_needed:
                # Find and insert a charging station
                station = self.find_optimal_station(last_node, customer, current_energy)
                if station:
                    current_route.append(station)
                    current_energy = self.max_energy_capacity - self.distances[station-1][customer-1] * self.energy_consumption
                    current_time += self.charging_time + self.times[station-1][customer-1]
                else:
                    # Backtracking to find feasible station
                    i = len(current_route) - 2
                    while i > 0 and current_route[i] not in self.station_ids:
                        prev_node = current_route[i]
                        station = self.find_optimal_station(prev_node, current_route[i+1], current_energy)
                        if station:
                            # Insert the found station and exit the loop
                            current_route.insert(i+1, station)
                            current_energy = self.max_energy_capacity - self.distances[station-1][current_route[i+2]-1] * self.energy_consumption
                            break
                        i -= 1
            
                    # This `else` runs only if the `while` loop completes without finding a station
                    else:
                        # If no station is feasible, finalize current route
                        current_route.append(self.depot)
                        routes.append(current_route)
                        current_route = [self.depot, customer]
                        current_capacity = self.demands[customer]
                        current_energy = self.max_energy_capacity - self.distances[self.depot-1][customer-1] * self.energy_consumption
                        current_time = max(self.ready_time[customer], self.times[self.depot-1][customer-1]) + self.service_time[customer]
                        continue


            # Step 3: Time window check
            arrival_time = current_time + self.times[last_node-1][customer-1]
            if arrival_time > self.due_time[customer]:
                # Finalize current route if time window is violated
                current_route.append(self.depot)
                routes.append(current_route)
                current_route = [self.depot, customer]
                current_capacity = self.demands[customer]
                current_energy = self.max_energy_capacity - self.distances[self.depot-1][customer-1] * self.energy_consumption
                current_time = max(self.ready_time[customer], self.times[self.depot-1][customer-1]) + self.service_time[customer]
                continue

            # Update route and metrics
            current_route.append(customer)
            current_capacity += self.demands[customer]
            current_energy -= energy_needed
            current_time = max(self.ready_time[customer], arrival_time) + self.service_time[customer]

        # Finalize the last route
        if current_route[-1] != self.depot:
            current_route.append(self.depot)
        routes.append(current_route)

        # Step 4: Ensure all customers are served
        assigned_customers = set(c for route in routes for c in route if c != self.depot)
        unassigned_customers = set(route[1:-1]) - assigned_customers
        for customer in unassigned_customers:
            # Start a new route for unassigned customers
            routes.append([self.depot, customer, self.depot])

        # Clean up routes to remove consecutive stations
        final_routes = [self.remove_consecutive_duplicates(route) for route in routes]
        for route in final_routes:
            if not (self.check_tw_constraint(route)[0]
                and self.check_capa_constraint(route)
                and self.check_e_constraint(route)):
                return None
            
        customers = []
        for route in final_routes:
            for id in route:
                if id in self.customer_ids:
                    customers.append(id)

        return final_routes, customers
    

    def get_distinct_solutions(self, pop):
        # Pareto front
        # Fast Non-Dominated Sorting Algorithm
        pareto_front = []

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
                pareto_front.append(indiv1)

        # Distinct solutions
        solutions  = []
        for indiv in pareto_front:
            if indiv not in solutions:
                solutions.append(indiv)
        return solutions
    

    # def find_reference_point(self, solutions):
    #     ob1_values = [indiv.ob1 for indiv in solutions]
    #     ob2_values = [indiv.ob2 for indiv in solutions]

    #     return (max(ob1_values) + 0.1, max(ob2_values) + 0.1)


    # def calc_hypervolume(self, pareto_front, reference_point=None):
    #     if reference_point is None:
    #         reference_point = self.find_reference_point(pareto_front)

    #     # Sort solutions in Pareto front based on ob1 and ob2 (ascending)
    #     pareto_front = sorted(pareto_front, key=lambda x: (x.ob1, x.ob2))

    #     hypervolume = 0
    #     for i, solution in enumerate(pareto_front):
    #         if i == 0:
    #             # For the first point, we calculate the area from the reference point to the first solution
    #             volume = (reference_point[0] - solution.ob1) * (reference_point[1] - solution.ob2)
    #         else:
    #             # For the remaining points, calculate the area between consecutive solutions
    #             prev_solution = pareto_front[i - 1]
    #             volume = (prev_solution.ob1 - solution.ob1) * (prev_solution.ob2 - solution.ob2)
    #         hypervolume += volume

    #     return hypervolume





class Individual:
    def __init__(self, sub_routes, customers):
        self.sub_routes = sub_routes
        self.customers = customers
 
        self.ob1 = None
        self.ob2 = None

        # NSGA2
        # S is the set of solutions dominated by this individual
        self.S = []
        # n is the domination counter 
        self.n = np.inf
        # rank using dominance depth method
        self.rank = np.inf

        # d is the crowding distance of this solution with respect to its neighbors lying on the same front
        self.d = -1

        # SPEA2
        # Strength - the number of solutions it dominates
        self.strength = 0
        # R represents the raw fitness - the summation of strength of solutions dominating this solution
        self.R = np.inf
        # list of solutions dominating this solution
        self.is_dominated_by = []
        # Density of solution 
        self.D = np.inf
        # Total fitness
        self.F = np.inf


    def __iter__(self):
        return iter(self.sub_routes)
    
    def __str__(self):
        return f"Routes: {self.sub_routes}\tOb1: {self.ob1}\tOb2: {self.ob2}"

    def __eq__(self, other):
        return self.sub_routes == other.sub_routes if other != None else False

    
    def check_dom(self, other):
        """
        Check if the current solution dominates another solution.

        Args:
            other (Solution): The other solution to compare with.

        Returns:
            "dominate": if the current solution dominates, 
            "dominated": if it is dominated, 
            None: if neither dominates.
        """
        if ((other.ob1 > self.ob1 and other.ob2 > self.ob2) or
            (other.ob1 == self.ob1 and other.ob2 > self.ob2) or
            (other.ob1 > self.ob1 and other.ob2 == self.ob2)):
            return "dominate"
        elif ((other.ob1 < self.ob1 and other.ob2 < self.ob2) or
            (other.ob1 == self.ob1 and other.ob2 < self.ob2) or
            (other.ob1 < self.ob1 and other.ob2 == self.ob2)):
            return "dominated"
        else:
            return None