import pytest

from sparow.ef import ExtensiveFormSolver

import pyomo.opt
from pyomo.common import unittest

solvers = set(pyomo.opt.check_available_solvers("gurobi"))

try:
    from sparow_examples.gtep_5bus.dummy import create_sp

    dummy_available = True
except:
    dummy_available = False


@unittest.pytest.mark.parametrize("mip_solver", solvers)
class Test_dummy:

    def test(self, mip_solver):
        sp = create_sp()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(474175.151, 0.01)
