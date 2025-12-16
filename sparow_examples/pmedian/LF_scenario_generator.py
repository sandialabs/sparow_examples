import random
import json

random.seed(123456789)

def random_add_subtract(tmp_list):
    """
    Function for randomly adding/subtracting from each element in a python list.
    Used for generating LF scenarios.
    """
    new_list = []
    for list_item in tmp_list:
        # random choice of addition/subtraction
        add_subtract_choice = random.choice([-1, 1])
        # random value added/subtracted
        choice_amount = random.uniform(0, 6)
        new_list.append(list_item + (add_subtract_choice * choice_amount))
    return new_list

with open('demands.json', 'r') as demands_file:
    customer_demand_list = json.load(demands_file)

LF_demand_list = []
for d_list in customer_demand_list[:5]:
    LF_demand_list.append(random_add_subtract(d_list))

#serialize_demands = demands_array.tolist()
LF_demands_filename = "LF_demands.json"
with open(LF_demands_filename, 'w') as json_file:
    json.dump(LF_demand_list, json_file, indent=4)
