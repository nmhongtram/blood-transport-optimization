import numpy as np
import pandas as pd
import time
from read_data import read_evrptw_data
from evrptw import EVRPTW
from moea import NSGA2, SPEA2

if __name__ == "__main__":
    # Đọc dữ liệu
    data, json_data = read_evrptw_data("EVRPTW_Data\\tuning_data\\rc102C50.txt")
    problem = EVRPTW(data)

    pop_size = 80
    num_gen = 100

    # Không gian tham số
    param_space = {
        'crossover_rate': [0.7, 0.8, 0.9],
        'mutation_rate': [0.05, 0.1, 0.15],
        'crossover_type': ['OX', 'PMX', 'CX'],  # Categorical
        'mutation_type': ['swap', 'inverse', 'insert']  # Categorical
    }

    # Số lần lặp random search
    iterations = 25

    # Lưu kết quả vào một danh sách
    results = []

    failed_count = 0

    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations}...")

        # Random chọn tham số từ param_space
        params = {
            'population_size': pop_size,
            'num_generations': num_gen,
            'crossover_rate': np.random.choice(param_space['crossover_rate']),
            'mutation_rate': np.random.choice(param_space['mutation_rate']),
            'crossover_type': np.random.choice(param_space['crossover_type']),
            'mutation_type': np.random.choice(param_space['mutation_type'])
        }

        # Khởi tạo solver
        solver = SPEA2(problem, **params)
        # solver = SPEA2(problem, **params)

        try:
            # Đo thời gian chạy
            start_time = time.time()
            pop = solver.spea2()
            # pop = solver.spea2()
            run_time = time.time() - start_time
            print(run_time)
        except:
            failed_count += 1
            continue


        # Thu thập thông tin từ kết quả
        solutions = problem.get_distinct_solutions(pop)
        num_solution = len(solutions)
        front = [solution.sub_routes for solution in solutions]
        ob1 = [solution.ob1 for solution in solutions]
        ob2 = [solution.ob2 for solution in solutions]

        # Lưu kết quả lần chạy này vào danh sách
        results.append({
            'run_time': run_time,
            'pop_size': pop_size,
            'num_gen': num_gen,
            'crossover_rate': params['crossover_rate'],
            'mutation_rate': params['mutation_rate'],
            'crossover_type': params['crossover_type'],
            'mutation_type': params['mutation_type'],
            'num_solution': num_solution,
            'front': str(front),  # Chuyển thành chuỗi để lưu vào Excel
            'ob1': ob1,
            'ob2': ob2
        })

    # Chuyển kết quả thành DataFrame
    df = pd.DataFrame(results)

    # Ghi vào file Excel
    output_file = "spea2_results_25_C50.xlsx"
    df.to_excel(output_file, index=False)

    print(f"Kết quả đã được lưu vào {output_file}")
    print("Số lần gặp lỗi:", failed_count)
