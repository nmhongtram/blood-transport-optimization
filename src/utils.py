# Utility functions for:
# - hyperparameter tuning
# - result extraction
# - hypervolume calculation
# - merging comparison results


import os
import numpy as np
import pandas as pd
import time
import json
from read_data import read_evrptw_data
from evrptw import EVRPTW
from moea import NSGA2, SPEA2


def tune_hyperparameters():
    data, json_data = read_evrptw_data("data\\tuning-data\\rc102C50.txt")
    problem = EVRPTW(data)

    pop_size = 80
    num_gen = 100

    param_space = {
        'crossover_rate': [0.7, 0.8, 0.9],
        'mutation_rate': [0.05, 0.1, 0.15],
        'crossover_type': ['OX', 'PMX', 'CX'],  # Categorical
        'mutation_type': ['swap', 'inverse', 'insert']  # Categorical
    }

    # Number of iterations
    iterations = 25

    results = []
    failed_count = 0

    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations}...")

        # Choose random hyperparameters
        params = {
            'population_size': pop_size,
            'num_generations': num_gen,
            'crossover_rate': np.random.choice(param_space['crossover_rate']),
            'mutation_rate': np.random.choice(param_space['mutation_rate']),
            'crossover_type': np.random.choice(param_space['crossover_type']),
            'mutation_type': np.random.choice(param_space['mutation_type'])
        }

        # Initialize the solver
        solver = NSGA2(problem, **params)
        # solver = SPEA2(problem, **params)

        try:
            start_time = time.time()
            pop = solver.spea2()
            # pop = solver.spea2()
            run_time = time.time() - start_time
            print(run_time)
        except:
            failed_count += 1
            continue

        # Extract final population and non-dominated solutions
        solutions = problem.get_distinct_solutions(pop)
        num_solution = len(solutions)
        front = [solution.sub_routes for solution in solutions]
        ob1 = [solution.ob1 for solution in solutions]
        ob2 = [solution.ob2 for solution in solutions]

        # save results
        results.append({
            'run_time': run_time,
            'pop_size': pop_size,
            'num_gen': num_gen,
            'crossover_rate': params['crossover_rate'],
            'mutation_rate': params['mutation_rate'],
            'crossover_type': params['crossover_type'],
            'mutation_type': params['mutation_type'],
            'num_solution': num_solution,
            'front': str(front),  # convert list to string for Excel
            'ob1': ob1,
            'ob2': ob2
        })

    df = pd.DataFrame(results)
    output_file = "spea2_results_25_C50.xlsx"
    df.to_excel(output_file, index=False)

    print(f"Results have been saved into {output_file}")
    print("Number of fail trials:", failed_count)



def get_results():
    # Read data
    # data, json_data = read_evrptw_data("EVRPTW_Data\\rc201C50.txt")
    # problem = EVRPTW(data)
    with open('dataBVfinal.json', 'r', encoding="UTF-8") as file:
        data = json.load(file)
    problem = EVRPTW(data)

    # Number of iterations
    iterations = 10

    results = []
    failed_count = 0

    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations}...")

        # Define hyperparameters
        params = {
            'population_size': 80,
            'num_generations': 50,
            'crossover_rate': 0.9,
            'mutation_rate': 0.15,
            'crossover_type': "OX",
            'mutation_type': "insert"
        }

        # Initialize the solver
        solver = NSGA2(problem, **params)
        # solver = SPEA2(problem, **params)

        try:
            start_time = time.time()
            pop = solver.nsga2()
            # pop = solver.spea2()
            run_time = time.time() - start_time
            print(run_time)
        except:
            failed_count += 1
            continue

        # Extract final population and non-dominated solutions
        solutions = problem.get_distinct_solutions(pop)
        num_solution = len(solutions)
        front = [solution.sub_routes for solution in solutions]
        ob1 = [solution.ob1 for solution in solutions]
        ob2 = [solution.ob2 for solution in solutions]

        # save results
        results.append({
            'run_time': run_time,
            'pop_size': params['population_size'],
            'num_gen': params['num_generations'],
            'crossover_rate': params['crossover_rate'],
            'mutation_rate': params['mutation_rate'],
            'crossover_type': params['crossover_type'],
            'mutation_type': params['mutation_type'],
            'num_solution': num_solution,
            'front': str(front),  # convert list to string for Excel
            'ob1': ob1,
            'ob2': ob2
        })

    df = pd.DataFrame(results)
    output_file = "nsga2_realdata_gen50.xlsx"
    df.to_excel(output_file, index=False)

    print(f"Results have been saved into {output_file}")
    print("Number of fail trials:", failed_count)



def calc_hv(file_name):
    df = pd.read_excel(file_name)

    # Convert string representation of lists back to actual lists
    df["ob1"] = df["ob1"].apply(eval)  
    df["ob2"] = df["ob2"].apply(eval)
    df["front"] = df["front"].apply(eval)

    # find max min ob1, ob2
    df["max_ob1"] = df["ob1"].apply(max)
    df["min_ob1"] = df["ob1"].apply(min)
    df["max_ob2"] = df["ob2"].apply(max)
    df["min_ob2"] = df["ob2"].apply(min)

    # find max, min of columns: max_ob1, min_ob1, max_ob2, min_ob2
    max_ob1 = df["max_ob1"].max()
    min_ob1 = df["min_ob1"].min()
    max_ob2 = df["max_ob2"].max()
    min_ob2 = df["min_ob2"].min()

    range_ob1 = max_ob1 - min_ob1
    range_ob2 = max_ob2 - min_ob2

    print("Range ob1:", range_ob1)
    print("Range ob2:", range_ob2)

    # nomalize ob1, ob2 to [0, 1]
    df["ob1_normalized"] = df["ob1"].apply(
        lambda ob1: [(val - min_ob1) / range_ob1 for val in ob1]
    )

    df["ob2_normalized"] = df["ob2"].apply(
        lambda ob2: [(val - min_ob2) / range_ob2 for val in ob2]
    )

    # Define reference point (1.1, 1.1) assuming normalized objectives are in [0, 1]
    ref_point = (1.1, 1.1)

    # calculate hypervolume normalized 
    def calculate_hypervolume(ob1_normalized, ob2_normalized, ref_point):
        hv = 0
        count = 0

        # merge ob1 and ob2 into list of tuples
        obs = list(zip(ob1_normalized, ob2_normalized))
        
        # Sort points by ob1 ascending, and by ob2 descending for same ob1
        obs_sorted = sorted(obs, key=lambda x: (-x[0], x[1]))

        prev = [ref_point[0], ref_point[1]]  # initialize previous point as reference point
       
        for ob1, ob2 in obs_sorted:
            if count == 0:
                # For the first point, we calculate the area from the reference point to the first solution
                volume = ((ref_point[0] - ob1) / ref_point[0]) * ((ref_point[1] - ob2) / ref_point[1])
            else:
                # For the remaining points, calculate the area between consecutive solutions
                volume = ((prev[0] - ob1) / ref_point[0]) * ((ref_point[1] - ob2) / ref_point[1])
            prev = [ob1, ob2]
            hv += volume
            count += 1

        return hv

    # apply calculate_hypervolume to each row
    df["hypervolume"] = df.apply(
        lambda row: calculate_hypervolume(row["ob1_normalized"], row["ob2_normalized"], ref_point), axis=1
    )

    # print(df["hypervolume"])
    df.to_excel("combined_comparison_hv.xlsx", index=False)
    print("File saved successfully.")



def merge_comparison_results(folder_path, output_file):
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            file_path = os.path.join(folder_path, file_name)
            # Đọc file Excel
            data = pd.read_excel(file_path)
            # Thêm cột để biết file này đến từ đâu (tuỳ chọn)
            data["Source_File"] = file_name
            all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)
    split_columns = combined_data["Source_File"].str.split("_", expand=True)
    split_columns.columns = [f"Part_{i+1}" for i in range(split_columns.shape[1])]

    combined_data = pd.concat([combined_data, split_columns], axis=1)
    combined_data = combined_data.rename(columns={"Part_1": "algorithm", "Part_2": "set_params","Part_3": "num_gen", "Part_4": "size"})

    combined_data.to_excel(output_file, index=False)