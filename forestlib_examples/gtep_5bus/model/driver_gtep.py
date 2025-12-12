import argparse

from forestlib.sp import stochastic_program
from forestlib.ef import ExtensiveFormSolver
from forestlib.ph import ProgressiveHedgingSolver

from gtep_model import ExpansionPlanningModel
from gtep_data import ExpansionPlanningData
#from gtep_solution import ExpansionPlanningSolution

import pyomo.environ as pyo
from pyomo.core import TransformationFactory

app_data = {"stages": 2, "num_reps": 2, "len_reps": 24, "num_commit": 24, "num_dispatch": 1}
model_data = {"scenarios": [{"ID": "scen_0", "Demand": 1.0, "Probability": 1.0}]}


def model_builder(data, args):
    data_path = "./data"
    data_object = ExpansionPlanningData()
    data_object.load_prescient(data_path)
    # data_object.load_storage_csv(data_path)

    num_stages = data["stages"]
    num_rep_days = data["num_reps"]
    len_rep_days = data["len_reps"]
    num_commit_p = data["num_commit"]
    num_disp = data["num_dispatch"]
    scenario_placeholder = data["Demand"]

    mod_object = ExpansionPlanningModel(
        stages=num_stages,
        data=data_object,
        num_reps=num_rep_days,  # num rep days
        len_reps=len_rep_days,  # len rep days
        num_commit=num_commit_p,  # num commitment periods
        num_dispatch=num_disp,  # num dispatch per commitment period
    )

    mod_object.config["include_commitment"] = True

    mod_object.config["flow_model"] = "CP"  # change this to "DC" to run DCOPF!

    mod_object.config["transmission"] = True  # TRANSMISSION INVESTMENT FLAG
    mod_object.config["thermal_generation"] = True  # THERMAL GENERATION INVESTMENT FLAG
    mod_object.config["renewable_generation"] = True  # RENEWABLE GENERATION INVESTMENT FLAG
    mod_object.config["scale_loads"] = False  # LEAVE AS FALSE
    mod_object.config["scale_texas_loads"] = False  # LEAVE AS FALSE

    mod_object.create_model()
    TransformationFactory("gdp.bound_pretransformation").apply_to(mod_object.model)
    TransformationFactory("gdp.bigm").apply_to(mod_object.model)

    return mod_object.model


#
# options to solve model with PH or EF:
#

def create_stochastic_program():
    sp = stochastic_program(
        first_stage_variables=[
            "investmentStage[*].renewableOperational[*]",
            "investmentStage[*].renewableInstalled[*]",
            "investmentStage[*].renewableRetired[*]",
            "investmentStage[*].renewableExtended[*]",
            "investmentStage[*].renewableDisabled[*]",
            "investmentStage[*].genOperational[*].binary_indicator_var",
            "investmentStage[*].genInstalled[*].binary_indicator_var",
            "investmentStage[*].genRetired[*].binary_indicator_var",
            "investmentStage[*].genDisabled[*].binary_indicator_var",
            "investmentStage[*].genExtended[*].binary_indicator_var",
            "investmentStage[*].branchOperational[*].binary_indicator_var",
            "investmentStage[*].branchInstalled[*].binary_indicator_var", 
            "investmentStage[*].branchRetired[*].binary_indicator_var", 
            "investmentStage[*].branchDisabled[*].binary_indicator_var",
            "investmentStage[*].branchExtended[*].binary_indicator_var"
        ]
    )
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="model", model_data=model_data, model_builder=model_builder
    )
    return sp
