import pytest
from sparow.ef import ExtensiveFormSolver
from sparow.ph import ProgressiveHedgingSolver
from sparow.benders import BendersSolver
from or_topas.solnpool import PyomoPoolManager, PoolPolicy
import pyomo.opt
from pyomo.common import unittest
from sparow.sp.util import relax_second_stage
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#from post_process_gtep_solution import post_process

solvers = set(pyomo.opt.check_available_solvers("gurobi"))
script_start = time.perf_counter()
try:
    from sparow_examples.bus_123_benders.worked_on_local_test import create_sp

    dummy_available = True
except:
    dummy_available = False


relaxations = {'scenario_A': {'relax_second_stage': True, 'unit_commitment': True}, 'scenario_B': {'relax_second_stage': True, 'unit_commitment': True}, 'scenario_C': {'relax_second_stage': True, 'unit_commitment': True}}
rd = {k: v["relax_second_stage"] for k, v in relaxations.items()}


sp = create_sp()
sp.add_transformation(relax_second_stage, relax_dict=rd)
solver = BendersSolver()

TIME_LIMIT = 1200.0
CONVERGENCE_TOL = 1e-3
REL_TOL = 0.01
MAX_ITERATIONS = 100
ETA_LOWER_BOUND_DEFAULT = -1e8
ITERATE_POOL_SIZE = 50

eta_bounds_map = {b: (ETA_LOWER_BOUND_DEFAULT, None) for b in sp.bundles}


def _on_iteration(data):
    n_cuts = len(data.cuts_added) if data.cuts_added is not None else 0
    best_lb = data.best_lb
    best_ub = data.best_ub
    if best_lb is not None and best_ub is not None and abs(best_lb) > 0:
        rel_gap = (best_ub - best_lb) / abs(best_lb)
        gap_str = f"{rel_gap}"
    else:
        gap_str = "n/a"
    print(f"  --- Benders iteration {data.iter_idx} ---", flush=True)
    print(f"      cuts_added={n_cuts}", flush=True)
    print(f"      L_k={data.L_k}  U_k={data.U_k}", flush=True)
    print(
        f"      best_lower_bound={best_lb}  best_upper_bound={best_ub}  "
        f"rel_gap={gap_str}",
        flush=True,
    )

iterate_pool = PyomoPoolManager()
iterate_pool.add_pool(
    name="feasible_benders_iterates",
    policy=PoolPolicy.keep_latest,
    max_pool_size=ITERATE_POOL_SIZE,
)

solver.set_options(
    solver="gurobi_persistent",
    subproblem_solver="gurobi_persistent",
    max_iterations=MAX_ITERATIONS,
    is_persistent_solver=True,
    allow_infeasible_subproblems=True,
    loglevel="INFO",
    rel_tol=REL_TOL,
    feasible_iterate_pool=iterate_pool,
)

print(
    f"--- Calling solve_and_return_model "
    f"(generate_cut_tol={CONVERGENCE_TOL}, "
    f"rel_tol={REL_TOL}, keep_latest={ITERATE_POOL_SIZE}) ---",
    flush=True,
)

solve_start = time.perf_counter()
res_munch = solver.solve_and_return_model(
    sp,
    eta_bounds_map,
    subproblem_transforms=[relax_second_stage],
    master_transforms=None,
    on_iteration=_on_iteration,
    convergence_tol=CONVERGENCE_TOL,
)
results = res_munch.solutions
solve_end = time.perf_counter()

results_dict = results.to_dict()

soln = next(iter(results_dict["solutions"].values()))

final_master_obj = soln["objectives"][0]["value"]
best_lb = res_munch.best_lb
best_ub = res_munch.best_ub
pool = res_munch.feasible_iterate_pool
n_iterates = 0 if pool is None else len(pool)
total_end = time.perf_counter()

print("=== Converged summary ===", flush=True)
print(f"termination: {results.metadata.termination_condition}", flush=True)
print(f"best solution objective (min U_k / best_ub): {best_ub}", flush=True)
print(f"best lower bound (max L_k / best_lb): {best_lb}", flush=True)
print(f"final master objective: {final_master_obj}", flush=True)
if best_lb is not None and best_ub is not None and abs(best_lb) > 0:
    print(
        f"relative gap: {(best_ub - best_lb) / abs(best_lb)}",
        flush=True,
    )
print(f"Solve time (seconds): {solve_end - solve_start:.4f}", flush=True)
print(f"Total runtime (seconds): {total_end - script_start:.4f}", flush=True)

print("=== Feasible iterate pool ===", flush=True)
print(f"n_feasible_iterates_kept: {n_iterates}", flush=True)
if pool is None:
    print("  (no iterate pool)", flush=True)
else:
    print("  id  lower_bound(L_k)  upper_bound(U_k)", flush=True)
    for stored in pool:
        print(
            f"  {stored.id}  {stored.objective(0).value}  "
            f"{stored.objective(1).value}",
            flush=True,
        )

#mod_object=res_munch.model.s['model','scenario_A']
#post_process(mod_object)
