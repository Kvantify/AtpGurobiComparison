import matplotlib.pyplot as plt
import argparse
from datetime import datetime

plt.style.use("seaborn-v0_8-whitegrid")

from icecream import ic
import time
import cvxpy as cp
import numpy as np
import atp_mosek
import atp_gurobi

""" Could be moved to benchmark tool - but as i am pressed for time i will do it here
"""


def time_decorate(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        return result, te - ts

    return timed


def get_data():
    """Extract 10 days of data"""
    pass


@time_decorate
def cvxpy_gurobi_convex_solve(problem, time_limit=600, predual=-1):
    ic("test QCPDUAL")
    return problem.solve(
        solver=cp.GUROBI,
        verbose=True,
        TimeLimit=time_limit,
        predual=predual,
        QCPDual=0,
    )


@time_decorate
def gurobipy_solve(model, time_limit=30, predual=-1):
    # model.setParam("NonConvex", 0)
    model.setParam("predual", predual)
    model.setParam("TimeLimit", time_limit)
    model.optimize()
    return model.objVal


@time_decorate
def cxvpy_mosek_convex_solve(problem):
    return problem.solve(solver=cp.MOSEK, verbose=False)


def cvxpy_gurobipy_test(
    data, add_trade_cost=True, add_market_cost=False, time_limit=35
):
    """Compare gurobi and cvxpy time and results
    Test will be without market cost, and with and without trade cost
    """
    cvx_problem = atp_mosek.create_problem(
        data, days=1, add_trade_cost=add_trade_cost, add_market_cost=add_market_cost
    )
    cvx_cost, cvx_time = cxvpy_mosek_convex_solve(cvx_problem)
    # cvx_cost, cvx_time = cvxpy_gurobi_convex_solve(cvx_problem, predual=1)

    gur_problem, _ = atp_gurobi.make_gurobi(
        data, add_trade_cost=add_trade_cost, add_market_cost=add_market_cost, K=-1
    )
    gur_problem.write("dum.lp")
    gur_problem.setParam("NonConvex", 0)
    # gur_problem.setParam("premiqcpform", 2)
    gpy_cost, gpy_time = gurobipy_solve(gur_problem, predual=1, time_limit=time_limit)
    print(f"Time Used Cvxpy Gurobi: {cvx_time}, Time Used gurobipy: {gpy_time}")
    print(
        f"Cost Achieved Cvxpy Gurobi: {-cvx_cost}, Cost Achieved gurobipy: {gpy_cost}, diff = {np.abs(-cvx_cost - gpy_cost)}"
    )
    # assert np.allclose(np.abs(-cvx_cost - gpy_cost), 0.0)


def cvxpy_gurobi_market_impact_cost_test(data, days=1):
    cvx_problem_false = atp_mosek.create_problem(
        data, days=days, add_trade_cost=False, add_market_cost=False
    )
    cvx_cost_false, cvx_time_false = cvxpy_gurobi_convex_solve(
        cvx_problem_false, predual=1
    )
    cvx_problem_true = atp_mosek.create_problem(
        data, days=days, add_trade_cost=False, add_market_cost=True
    )
    cvx_cost_true, cvx_time_true = cvxpy_gurobi_convex_solve(
        cvx_problem_true, predual=1
    )
    print("Market Cost Timings - Gurobi CVXPY")
    print(
        f"Time without market impact cost: {cvx_time_false}, Time With Market Impact Cost: {cvx_time_true}"
    )
    # assert cvx_cost_false >= cvx_cost_true
    print(
        f"Value without market impact: {-cvx_cost_false}, Value With Market impact (should be smaller) {-cvx_cost_true}"
    )


def gurobipy_market_impact_cost_test(data, time_limit=30):
    cvx_problem = atp_mosek.create_problem(
        data, days=100, add_trade_cost=True, add_market_cost=True
    )
    cvx_cost, cvx_time = cxvpy_mosek_convex_solve(cvx_problem)
    print(f"Cvxpy Mosek Cost: {cvx_cost}, cvxpy time {cvx_time}")

    # gur_prob.setParam("premiqcpform", 2)
    res = {}
    for predual in [0, 1, 2]:
        gur_problem, gur_weigths = atp_gurobi.make_gurobi(
            data=data, add_trade_cost=True, add_market_cost=True
        )

        gur_cost, gur_time = gurobipy_solve(
            gur_problem, time_limit=time_limit, predual=predual
        )
        res[predual] = (gur_cost, gur_time)
        print(f"Gurobi Predual: {predual}, Cost: {cvx_cost}, cvxpy time {cvx_time}")
    ic(res)


def gurobi_mip_test(
    data, trade_cost=True, market_cost=False, time_limit=600, predual=-1
):
    """Test gurobi mixed integer with and without tradecost and market cost
    Compare gap somehow to greedy or at least solution. I will run benchmark tool to get it.
    1. Day and see if 10 day is possible.

    #model.setParam('premiqcpform', 1)
    #model.setParam('presparsify', 1)
    see hvad der sker hvis vi starter fra k_largest
    """

    cvx_problem = atp_mosek.create_problem(
        data,
        days=100,
        add_trade_cost=trade_cost,
        add_market_cost=market_cost,
        mip_K=160,
    )
    cvx_cost, cvx_time = cvxpy_gurobi_convex_solve(
        cvx_problem, time_limit=time_limit, predual=predual
    )

    gur_problem, _ = atp_gurobi.make_gurobi(data, add_trade_cost=trade_cost, K=160)
    gpy_cost, gpy_time = gurobipy_solve(
        gur_problem, time_limit=time_limit, predual=predual
    )
    print(f"MIP Time Used Cvxpy Gurobi: {cvx_time}, Time Used gurobipy: {gpy_time}")
    print(
        f"MIP Cost Achieved Cvxpy Gurobi: {-cvx_cost}, MIP Cost Achieved gurobipy: {gpy_cost}, diff = {np.abs(-cvx_cost - gpy_cost)}"
    )
    # assert np.allclose(np.abs(-cvx_cost - gpy_cost), 0.0)


def gurobipy_mip_test(
    data, trade_cost=True, market_cost=True, time_limit=600, predual=-1
):
    """Test gurobi mixed integer with and without tradecost and market cost
    Compare gap somehow to greedy or at least solution. I will run benchmark tool to get it.
    1. Day and see if 10 day is possible.

    #model.setParam('premiqcpform', 1)
    #model.setParam('presparsify', 1)
    see hvad der sker hvis vi starter fra k_largest
    """

    gur_problem, _ = atp_gurobi.make_gurobi(data, add_trade_cost=trade_cost, K=160)
    gpy_cost, gpy_time = gurobipy_solve(gur_problem, time_limit=time_limit)
    print(f"MIP Time Used gurobipy: {gpy_time}")
    print(f"MIP Cost Achieved gurobipy: {gpy_cost}")
    # assert np.allclose(np.abs(-cvx_cost - gpy_cost), 0.0)


def atp_parts_test():
    """Test the atp model when we delete each of the parts and look at runtime"""
    pass


def header_print(s):
    print(s)


def market_impact_test():
    all_data = atp_mosek.get_default()
    header_print("1 Day Test")
    cvxpy_gurobi_market_impact_cost_test(all_data, days=1)
    cvxpy_gurobi_market_impact_cost_test(all_data, days=2)


def mip_that_worked(data=None, add_trade_cost=True, add_market_cost=False):
    # 1 day klargest get 0.0012766665920330275 (may have different constraint violation) relative quality of 0.001
    # Girobi Best objective 1.275945380754e-03, best bound 1.277907649797e-03, gap 0.1538%
    # Hence klargest is fine
    #

    # 10 day
    # Explored 2375 nodes (3797951 simplex iterations) in 3001.27 seconds (2608.02 work units)
    # Thread count was 10 (of 10 available processors)
    # predual -1
    ##
    ic(add_trade_cost, add_market_cost)
    if data is None:
        data = atp_mosek.get_default_day()
    res = {}
    for vals in [0, 1, 2]:
        model_mip, weights = atp_gurobi.make_gurobi(
            data, add_trade_cost=add_trade_cost, add_market_cost=add_market_cost, K=160
        )
        # model_mip.setParam("predual", 1)
        model_mip.setParam("premiqcpform", vals)
        atp_gurobi.solve(model_mip, time_limit=120)
        res[vals] = model_mip.ObjVal
    ic(res)
    print(res)


def initial_weights_set_test(data_prev, data, time_limit=300):
    value, weights = atp_mosek.run_k_largest(
        data_prev, trade_cost=True, market_cost=False, K=160
    )
    print("Previous value", value)
    prev_output_weights = weights[0, :]
    print(prev_output_weights)
    K = len(np.nonzero(prev_output_weights)[0])
    # assert False
    data["weights"] = prev_output_weights
    print("initial weights used", prev_output_weights)
    value_2, weights_2 = atp_mosek.run_k_largest(
        data, trade_cost=True, market_cost=False, K=160
    )
    print("Value with some initial values", value_2)
    D_new, N_new = data["expected_return"].shape
    initial_weights_tiled = np.tile(prev_output_weights, (D_new, 1))
    gur_prob, gurobi_weights = atp_gurobi.make_gurobi(
        data,
        add_trade_cost=False,
        K=160,
        initial_weights=None
        # initial_weights=initial_weights_tiled
        # starting_weights=weights,
        # initial_weights=weights_2,
    )
    gur_prob.setParam("premiqcpform", 2)
    gurobipy_solve(gur_prob, time_limit=time_limit)
    # print(np.sum(gur_prob.X))
    print(f"Comparison K largest {value_2} - Gurobi {gur_prob.ObjVal}")
    if np.isneginf(gur_prob.objVal):
        print("Gurobi Found notthing")
        return
    gurobi_output = gurobi_weights["weights"].getAttr("x")
    # nz = gurobi_output.nonzero()
    print("nonzero outputs gurobi", np.count_nonzero(gurobi_output, axis=1))
    print(
        f"Comparison K largest output size {weights_2.shape} - Gurobi {gurobi_weights['weights'].shape}"
    )
    return
    # print(gurobi_weights)


def test_with_w0(
    first_date,
    second_date,
    add_trade_cost=True,
    add_market_cost=True,
    K=160,
    use_initial_weights=True,
    time_limit=900,
    days=10,
):
    ic(
        "input",
        first_date,
        second_date,
        add_trade_cost,
        add_market_cost,
        K,
        use_initial_weights,
        time_limit,
        days,
    )
    data_dict = atp_mosek.get_all_data()
    # first_date = "2016-05-17"
    # second_date = "2016-05-18"
    D0 = data_dict[first_date]
    D1 = data_dict[second_date]
    if days < 10:
        D0 = atp_mosek.data_subset(D0, days, 2000)
        D1 = atp_mosek.data_subset(D1, days, 2000)
    res_prev = atp_mosek.run_k_largest(
        D0, trade_cost=add_trade_cost, market_cost=add_market_cost, K=K
    )
    print("Previous value", res_prev["sparse_value"])
    initial_weights = res_prev["sparse_weights"][0, :]
    # print(initial_weights)
    K = len(np.nonzero(initial_weights)[0])
    # assert False
    D1["weights"] = initial_weights
    # assert False
    kl_start = time.time()
    res_cur = atp_mosek.run_k_largest(
        D1, trade_cost=add_trade_cost, market_cost=add_market_cost, K=160
    )
    kl_end = time.time()
    klarg_time = kl_end - kl_start
    print("Value with some initial values", res_cur["sparse_value"])
    D_new, N_new = D1["expected_return"].shape
    prev_output_weights = res_cur["sparse_weights"][0, :]
    initial_weights_tiled = np.tile(prev_output_weights, (D_new, 1))
    starting_weights = None
    if use_initial_weights:
        starting_weights = initial_weights_tiled
    gur_prob, gurobi_weights = atp_gurobi.make_gurobi(
        D1,
        add_trade_cost=add_trade_cost,
        add_market_cost=add_market_cost,
        K=160,
        # starting_weights=None
        starting_weights=starting_weights
        # initial_weights=initial_weights_tiled
        # starting_weights=weights,
        # initial_weights=weights_2,
    )
    gur_prob.setParam("premiqcpform", 2)
    gurobipy_solve(gur_prob, time_limit=time_limit)
    # print(np.sum(gur_prob.X))
    print(
        f"Comparison K largest {res_cur['sparse_value']} - Gurobi {gur_prob.ObjVal} - full value {res_cur['full_value']}"
    )
    gurobi_output = None
    gurobi_sparsity = None
    if np.isneginf(gur_prob.objVal):
        print("Gurobi Found nothing")
    else:
        gurobi_output = gurobi_weights["weights"].getAttr("x")
        # nz = gurobi_output.nonzero()
        print("nonzero outputs gurobi", np.count_nonzero(gurobi_output, axis=1))
        gurobi_sparsity = np.count_nonzero(gurobi_output, axis=1)
        print(
            f"Comparison K largest output size {res_cur['sparse_weights'].shape} - Gurobi {gurobi_weights['weights'].shape}"
        )
    out = dict(
        gurobi_val=gur_prob.ObjVal,
        gurobi_sparsity=gurobi_sparsity,
        k_largest_val=res_cur["sparse_value"],
        k_largest_time=klarg_time,
        first_date=first_date,
        second_date=second_date,
        K=K,
        time_limit=time_limit,
        use_initial_weights=use_initial_weights,
        days=days,
        add_market_cost=add_market_cost,
        add_trade_cost=add_trade_cost,
        gurobi_output=gurobi_output,
    )
    np.save(
        f"test_w0_gurobi_per_5_{first_date}_{second_date}_{use_initial_weights}_{add_trade_cost}_{add_market_cost}.npy",
        out,
    )
    ic(out)
    return out


def test_mc():
    # day_data = atp_mosek.get_default_day()
    # all_data = atp_mosek.get_default()
    # small_data = atp_mosek.data_subset(all_data, 1, 1000)
    cons_data = atp_mosek.get_default_with_w0()
    day_data = atp_mosek.data_subset(cons_data, 1, 2000)

    # dum_data = atp_mosek.dum_data(all_data, 1, 123)
    # cvxpy_gurobipy_test(dum_data, add_trade_cost=False, add_market_cost=True)
    cvxpy_gurobipy_test(
        day_data, add_trade_cost=True, add_market_cost=True, time_limit=60
    )

    # gurobipy_market_impact_cost_test(day_data)


def time_gurobi_convex_full_problem(data, add_trade_cost=True, add_market_cost=True):
    gur_problem, _ = atp_gurobi.make_gurobi(
        data, add_trade_cost=add_trade_cost, add_market_cost=add_market_cost, K=-1
    )
    gpy_cost, gpy_time = gurobipy_solve(gur_problem, time_limit=300, predual=1)
    return gpy_time


def gurobi_convex_full_problem_test(add_trade_cost=True, add_market_cost=True):
    days = [1, 2, 5, 10]
    data_sets = atp_mosek.get_all_data()
    assets = np.arange(0, 1100, 100)
    res_dict = {key: {} for key in data_sets.keys()}
    # assets = [3, 5, 12]
    print(assets)
    if add_trade_cost and add_market_cost:
        for data_key, data in data_sets.items():
            for N in assets:
                data_subset = atp_mosek.data_subset(data, 1, N)
                time_used = time_gurobi_convex_full_problem(
                    data_subset, add_market_cost=add_market_cost
                )
                res_dict[data_key][N] = time_used
        np.save(
            f"./gurobi_convex_full_problem_times_with_market_cost",
            res_dict,
        )
    elif add_trade_cost and not add_market_cost:
        for data_key, data in data_sets.items():
            for D in days:
                data_subset = atp_mosek.data_subset(data, D, 2000)
                time_used = time_gurobi_convex_full_problem(
                    data_subset, add_market_cost=add_market_cost
                )
                res_dict[data_key][D] = time_used
        np.save(
            f"./gurobi_convex_full_problem_times_without_market_cost_days22",
            res_dict,
        )
    elif not add_trade_cost and not add_market_cost:
        for data_key, data in data_sets.items():
            for D in days:
                data_subset = atp_mosek.data_subset(data, D, 2000)
                time_used = time_gurobi_convex_full_problem(
                    data_subset, add_trade_cost=False, add_market_cost=False
                )
                res_dict[data_key][D] = time_used
        np.save(
            f"./gurobi_convex_full_problem_times_without_market_and_trade_cost_days",
            res_dict,
        )

def read_command_line_inputs():
    parser = argparse.ArgumentParser(description='Test parameters for test_with_w0 function')

    parser.add_argument('--first_date', type=str, 
                        help='First date in YYYY-MM-DD format', required=True)
    parser.add_argument('--second_date', type=str, 
                        help='Second date in YYYY-MM-DD format', required=True)
    parser.add_argument('--add_trade_cost', type=bool, default=True, 
                        help='Boolean to add trade cost')
    parser.add_argument('--add_market_cost', type=bool, default=True, 
                        help='Boolean to add market cost')
    parser.add_argument('--K', type=int, default=160, 
                        help='Integer value for K')
    parser.add_argument('--use_initial_weights', type=bool, default=True, 
                        help='Boolean to use initial weights')
    parser.add_argument('--time_limit', type=int, default=900, 
                        help='Time limit for the test')
    parser.add_argument('--days', type=int, default=10, 
                        help='Number of days for the test')

    args = parser.parse_args()

    return args

def main():
    """
    Example
     python gurobi_comparison.py --days 1 --first_date 2020-03-13 --second_date 2020-03-16 --time_limit 30
     """
    args = read_command_line_inputs()
    test_with_w0(
        args.first_date,
        args.second_date,
        args.add_trade_cost,
        args.add_market_cost,
        args.K,
        args.use_initial_weights,
        args.time_limit,
        args.days
    )

def old_main():
    # day_data = atp_mosek.get_default_day()
    # cvxpy_gurobi_market_impact_cost_test(day_data, days=1)
    # test_mc()
    # test_with_w0()
    # test_with_w0("2020-03-13", "2020-03-16", True, True, 160, 900)
    res = test_with_w0(
        "2016-05-17",
        "2016-05-18",
        add_trade_cost=False,  # True,
        add_market_cost=False,  # True,
        K=160,
        time_limit=1800,
        use_initial_weights=False,
        days=10,
    )
    ic(res)
    # gurobi_convex_full_problem_test(add_trade_cost=False, add_market_cost=False)
    # all_data = atp_mosek.get_default()
    # gurobipy_mip_test(all_data, trade_cost=True, predual=-1, time_limit=3000)
    # cvxpy_gurobipy_test(day_data, include_trade_cost=False, market_cost=True)
    if False:
        prev_data, data = atp_mosek.get_default_consecutive()
        days = 1
        prev_day_data = atp_mosek.data_subset(prev_data, days, 2000)
        day_data = atp_mosek.data_subset(data, days, 2000)
        # cvxpy_gurobipy_test(day_data, True, True)
        # 5 days problem is solvable in 5-6 minuttes - better than klargest
        initial_weights_set_test(prev_day_data, day_data, time_limit=60 * days)
    # all_data = atp_mosek.get_default()
    # cvxpy_gurobipy_test(all_data, include_trade_cost=True)
    # mip_that_worked()

if __name__ == "__main__":
    main()
