import pyomo.environ as pyo
import json
import math
import random
from sparow.sp import stochastic_program

"""
CAPACITATED P-MEDIAN
    - LF model is an approximation using continuous variables
        - LF scenarios are restricted to Low, Medium, and High
    - HF model is a BIP (binary integer program)
        - HF scenarios can be Low, Medium, and High for each customer (e.g., ['Low', 'High', 'Low', 'Medium'])
* Can specify number of HF scenarios by including "num_HF" key in app_data; otherwise, defaults to 10
* Problem data adapted from https://anl-ceeesa.github.io/MIPLearn/0.4/guide/problems/ 
"""

app_data = {"n": 100, "t": 100}  # number of facilities & customers

with open(
    "../sparow_examples/sparow_examples/facilityloc/distances.json", "r"
) as distances_file:
    app_data["w"] = json.load(distances_file)

with open(
    "../sparow_examples/sparow_examples/facilityloc/capacities.json", "r"
) as capacities_file:
    app_data["c"] = json.load(capacities_file)

app_data["p"] = 40  # p-value

with open(
    "../sparow_examples/sparow_examples/facilityloc/facility_opening_costs.json", "r"
) as fcosts_file:
    app_data["f"] = json.load(fcosts_file)

with open(
    "../sparow_examples/sparow_examples/facilityloc/demands.json", "r"
) as demands_file:
    customer_demand_list = json.load(demands_file)

HFscens_list = []  # list of HF scenarios
for scen_idx, scen in enumerate(customer_demand_list):
    HFscens_list.append(
        {
            "ID": f"HF_{scen_idx}",
            "Demand": scen,
            "Probability": 0.1,
        }
    )

with open(
    "../sparow_examples/sparow_examples/facilityloc/LF_demands.json", "r"
) as LF_demands_file:
    LF_demand_list = json.load(LF_demands_file)
LFscens_list = []  # list of LF scenarios
for d_list_idx, d_list in enumerate(LF_demand_list):
    LFscens_list.append(
        {
            "ID": f"LF_{d_list_idx}",
            "Demand": d_list,
            "Probability": 0.2,
        }
    )

model_data = {"LF": {"scenarios": LFscens_list}, "HF": {"scenarios": HFscens_list}}


def LF_builder(data, args):
    num_facilities = data["n"]
    num_customers = data["t"]
    servicing_costs = data["w"]
    facility_caps = data["c"]
    facility_costs = data["f"]
    p_value = data["p"]

    ### STOCHASTIC DATA
    d = data["Demand"]  # d[i] = demand of customer i

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.CUSTOMERS = pyo.Set(initialize=[i for i in range(num_customers)])
    model.FACILITIES = pyo.Set(initialize=[j for j in range(num_facilities)])

    ### VARIABLES
    model.y = pyo.Var(model.FACILITIES, bounds=[0, 1])  # y[j] = 1 if facility j is open
    model.x = pyo.Var(
        model.CUSTOMERS, model.FACILITIES, bounds=[0, 1]
    )  # x[i,j] = 1 if customer i serviced by facility j

    ### CONSTRAINTS
    def ServiceAllCustomers_rule(model, i):
        return sum(model.x[i, j] for j in range(num_facilities)) == 1

    model.MeetDemand = pyo.Constraint(model.CUSTOMERS, rule=ServiceAllCustomers_rule)

    def SufficientProduction_rule(model, j):
        return facility_caps[j] * model.y[j] >= sum(
            d[i] * model.x[i, j] for i in range(num_customers)
        )

    model.SufficientProduction = pyo.Constraint(
        model.FACILITIES, rule=SufficientProduction_rule
    )

    def PMedian_rule(model):
        return sum(model.y[j] for j in model.FACILITIES) == p_value

    model.Capacity = pyo.Constraint(rule=PMedian_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(
            sum(servicing_costs[i][j] * model.x[i, j] for j in model.FACILITIES)
            for i in model.CUSTOMERS
        )
        expr += sum(facility_costs[j] * model.y[j] for j in model.FACILITIES)
        return expr

    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model


def HF_builder(data, args):
    num_facilities = data["n"]
    num_customers = data["t"]
    servicing_costs = data["w"]
    facility_costs = data["f"]
    facility_caps = data["c"]
    p_value = data["p"]

    ### STOCHASTIC DATA
    d = data["Demand"]  # d[i] = demand of customer i

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.CUSTOMERS = pyo.Set(initialize=[i for i in range(num_customers)])
    model.FACILITIES = pyo.Set(initialize=[j for j in range(num_facilities)])

    ### VARIABLES
    model.y = pyo.Var(
        model.FACILITIES, within=pyo.Binary
    )  # y[j] = 1 if facility j is open
    model.x = pyo.Var(
        model.CUSTOMERS, model.FACILITIES, within=pyo.Binary
    )  # x[i,j] = 1 if customer i serviced by facility j

    ### CONSTRAINTS
    def ServiceAllCustomers_rule(model, i):
        return sum(model.x[i, j] for j in range(num_facilities)) == 1

    model.MeetDemand = pyo.Constraint(model.CUSTOMERS, rule=ServiceAllCustomers_rule)

    def SufficientProduction_rule(model, j):
        return facility_caps[j] * model.y[j] >= sum(
            d[i] * model.x[i, j] for i in range(num_customers)
        )

    model.SufficientProduction = pyo.Constraint(
        model.FACILITIES, rule=SufficientProduction_rule
    )

    def PMedian_rule(model):
        return sum(model.y[j] for j in model.FACILITIES) == p_value

    model.Capacity = pyo.Constraint(rule=PMedian_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(
            sum(servicing_costs[i][j] * model.x[i, j] for j in model.FACILITIES)
            for i in model.CUSTOMERS
        )
        expr += sum(facility_costs[j] * model.y[j] for j in model.FACILITIES)
        return expr

    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model


#
# options to solve, LF, HF, or MF random models:
#


def HF_pmedian():
    sp = stochastic_program(first_stage_variables=["y"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    return sp


def LF_pmedian():
    sp = stochastic_program(first_stage_variables=["y"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder
    )
    return sp


def MFrandom_pmedian():
    sp = stochastic_program(first_stage_variables=["y"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )
    sp.initialize_bundles(
        scheme="mf_random",
        LF=2,
        seed=1234567890,
    )
    return sp


def MFdissimilar_pmedian():
    sp = stochastic_program(first_stage_variables=["y"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )
    sp.initialize_bundles(
        scheme="mf_kmeans_dissimilar",
        LF=2,
        seed=1234567890,
    )
    return sp


def MFsimilar_pmedian():
    sp = stochastic_program(first_stage_variables=["y"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )
    sp.initialize_bundles(
        scheme="mf_kmeans_similar",
        LF=2,
        seed=1234567890,
    )
    return sp


def MFsimilar_HFweighted_pmedian():
    sp = stochastic_program(first_stage_variables=["y"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )
    sp.initialize_bundles(
        scheme="mf_kmeans_similar",
        LF=2,
        seed=1234567890,
        model_weight={"HF": 3.0, "LF": 1.0},
    )
    return sp


def MFsimilar_LFweighted_pmedian():
    sp = stochastic_program(first_stage_variables=["y"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )
    sp.initialize_bundles(
        scheme="mf_kmeans_similar",
        LF=2,
        seed=1234567890,
        model_weight={"HF": 1.0, "LF": 3.0},
    )
    return sp


def MFdissimilar_HFweighted_pmedian():
    sp = stochastic_program(first_stage_variables=["y"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )
    sp.initialize_bundles(
        scheme="mf_kmeans_dissimilar",
        LF=2,
        seed=1234567890,
        model_weight={"HF": 3.0, "LF": 1.0},
    )
    return sp


def MFdissimilar_LFweighted_pmedian():
    sp = stochastic_program(first_stage_variables=["y"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )
    sp.initialize_bundles(
        scheme="mf_kmeans_dissimilar",
        LF=2,
        seed=1234567890,
        model_weight={"HF": 1.0, "LF": 3.0},
    )
    return sp


def MFrandom_HFweighted_pmedian():
    sp = stochastic_program(first_stage_variables=["y"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )
    sp.initialize_bundles(
        scheme="mf_random",
        LF=2,
        seed=1234567890,
        model_weight={"HF": 3.0, "LF": 3.0},
    )
    return sp


def MFrandom_LFweighted_pmedian():
    sp = stochastic_program(first_stage_variables=["y"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )
    sp.initialize_bundles(
        scheme="mf_random",
        LF=2,
        seed=1234567890,
        model_weight={"HF": 1.0, "LF": 3.0},
    )
    return sp
