import numpy as np
from scipy.stats import uniform, randint
from miplearn.problems.pmedian import PMedianGenerator, build_pmedian_model_gurobipy
import json

# setting random seed for purposes of reproducibility
np.random.seed(123456789)

# Generate random instances with ten customers located in a
# 100x100 square, with demands in [0,10], capacities in [0, 250].
data = PMedianGenerator(
    x=uniform(loc=0.0, scale=100.0),
    y=uniform(loc=0.0, scale=100.0),
    n=randint(low=100, high=101),
    p=randint(low=40, high=41),
    demands=uniform(loc=0, scale=50),
    capacities=uniform(loc=0, scale=1000),
    distances_jitter=uniform(loc=.9, scale=0),
    demands_jitter=uniform(loc=.9, scale=1),
    capacities_jitter=uniform(loc=.9, scale=0),
    fixed=True,
).generate(10)

facility_opening_costs = np.random.uniform(low=200, high=1000, size=100)

# print data for reference instance
print("p =", data[0].p)
print("demands =", data[0].demands)
print()

demands_array = np.array([data[i].demands for i in range(10)])
serialize_demands = demands_array.tolist()
demands_filename = "demands.json"
with open(demands_filename, 'w') as json_file:
    json.dump(serialize_demands, json_file, indent=4)

serialize_distances = data[0].distances.tolist()
distances_filename = "distances.json"
with open(distances_filename, 'w') as json_file:
    json.dump(serialize_distances, json_file, indent=4)

serialize_capacities = data[0].capacities.tolist()
capacities_filename = "capacities.json"
with open(capacities_filename, 'w') as json_file:
    json.dump(serialize_capacities, json_file, indent=4)

serialize_facility_opening_costs = facility_opening_costs.tolist()
facility_opening_costs_filename = "facility_opening_costs.json"
with open(facility_opening_costs_filename, 'w') as json_file:
    json.dump(serialize_facility_opening_costs, json_file, indent=4)

# Build and optimize first model
#model = build_pmedian_model_gurobipy(data[0])
#model.optimize()
