import numpy as np
import json

def read_evrptw_data(file_path):
    data = {
        "NAME": None,
        "DIMENSION": None,
        "STATIONS": None,
        "CAPACITY": None,
        "ENERGY_CAPACITY": None,
        "ENERGY_CONSUMPTION": None,
        "CHARGING_TIME": None,
        "NODES": [],
        "COORDINATES": {},
        "DEMANDS": {},
        "READY_TIME": {},
        "DUE_TIME": {},
        "SERVICE_TIME": {},
        "DISTANCES": [],
        "TIMES": []
    }

    with open(file_path) as file:
        lines = file.readlines()

    node_start_idx = None
    node_end_idx = np.inf
    node_data = []
    data["NAME"] = file_path
    average_velocity = 0
    inverse_refueling_rate = 0      # Tỷ lệ nghịch với tốc độ tiếp nhiên liệu, thể hiện thời gian cần để nạp một đơn vị nhiên liệu.

    for i, line in enumerate(lines):
        parts = line.strip().split()

        if "StringID" in line:
            node_start_idx = i + 1
        elif line == "":
            node_end_idx = i
        elif node_start_idx and i < node_end_idx:
            node_data.append(parts)

        if line.startswith("Q Vehicle fuel tank capacity"):
            data["ENERGY_CAPACITY"] = float(parts[-1].strip("/"))
        if line.startswith("C Vehicle load capacity"):
            data["CAPACITY"] = float(parts[-1].strip("/"))
        if line.startswith("r fuel consumption rate"):
            data["ENERGY_CONSUMPTION"] = float(parts[-1].strip("/"))

        if line.startswith("g inverse refueling rate"):
            inverse_refueling_rate = float(parts[-1].strip("/"))
            data["CHARGING_TIME"] = inverse_refueling_rate * data["ENERGY_CAPACITY"] 

        if line.startswith("v average Velocity"):
            average_velocity = float(parts[-1].strip("/"))

        
    def calculate_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)



    depot_count = 1
    customer_count = 0
    station_count = 0
    id_counter = 2
    stations = []

    for entry in node_data:
        if len(entry) < 8:  # Skip empty or malformed lines
            continue
        string_id, node_type, x, y, demand, ready_time, due_date, service_time = entry
        x, y, demand, ready_time, due_date, service_time = float(x), float(y), float(demand), float(ready_time), float(due_date), float(service_time)

        if node_type == "d":
            node_id = 1  # Depot always gets ID 1
        elif node_type == "c":
            customer_count += 1
            node_id = id_counter
            id_counter += 1
        elif node_type == "f":
            station_count += 1
            stations.append(entry)

        if node_type != "f":
            node_id = node_id
            data["NODES"].append({
                "id": node_id,
                "x": x,
                "y": y,
                "demand": demand
            })

            data["COORDINATES"][node_id] = [x, y]
            data["DEMANDS"][node_id] = demand
            data["READY_TIME"][node_id] = ready_time
            data["DUE_TIME"][node_id] = due_date
            data["SERVICE_TIME"][node_id] = service_time
            # data["SERVICE_TIME"][node_id] = 18


    for entry in stations:
        string_id, node_type, x, y, demand, ready_time, due_date, service_time = entry
        x, y, demand, ready_time, due_date = float(x), float(y), float(demand), float(ready_time), float(due_date)
        node_id = id_counter
        id_counter += 1

        data["NODES"].append({
            "id": node_id,
            "x": x,
            "y": y,
            "demand": demand
        })

        data["COORDINATES"][node_id] = [x, y]
        data["DUE_TIME"][node_id] = due_date


    # Calculate distance and time matrices
    dimension = len(data["NODES"])
    distance_matrix = [[0] * dimension for _ in range(dimension)]
    time_matrix = [[0] * dimension for _ in range(dimension)]

    for i in range(dimension):
        for j in range(dimension):
            if i != j:
                x1, y1 = data["COORDINATES"][data["NODES"][i]["id"]]
                x2, y2 = data["COORDINATES"][data["NODES"][j]["id"]]
                dist = calculate_distance(x1, y1, x2, y2)
                distance_matrix[i][j] = dist
                time_matrix[i][j] = dist / average_velocity

    data["DISTANCES"] = distance_matrix
    data["TIMES"] = time_matrix

    data["DIMENSION"] = depot_count + customer_count
    data["STATIONS"] = station_count
    data["CHARGING_TIME"] = 5

    # Convert Python to JSON  
    json_object = json.dumps(data, indent = 4) 

    return data, json_object