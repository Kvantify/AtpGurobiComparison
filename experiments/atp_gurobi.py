### Test Mixed Integer
from icecream import ic
import cvxpy as cp
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB
from types import SimpleNamespace as sn


def get_known_data():
    pass


def compute_delta_weights(x, initial_weights):
    D, N = x.shape
    first_day_diff = x[0, :].reshape((1, N)) - initial_weights
    if D == 1:
        return first_day_diff, None
        # return np.reshape(abs_first_day_trade, (1, N))
    rem_day_diff = x[1:, :] - x[:-1, :]  # shape (D-1) x N
    # delta_weights = np.vstack([abs_first_day_trade, abs_day_trade])
    return first_day_diff, rem_day_diff


def compute_benchmark_normalization(
    benchmark_index, official_risk_measure, target_risk
):
    standard_deviation = np.sqrt(
        benchmark_index.T @ official_risk_measure @ benchmark_index
    )
    clipped_std = np.maximum(standard_deviation, 1.0e-16)
    benchmark_normalization_factors_ = target_risk / clipped_std
    return benchmark_normalization_factors_


def add_trade_cost_bad(m, weights, initial_weights):
    assert False
    first_day_diff, rem_day_diff = compute_delta_weights(weights, initial_weights)
    D, N = weights.shape
    trade_params = m.addMVar(shape=(D, N))
    m.addConstr(trade_params[0, :] - first_day_diff >= 0)
    m.addConstr(trade_params[0, :] + first_day_diff >= 0)
    m.addConstr(trade_params[1:, :] - rem_day_diff >= 0)
    m.addConstr(trade_params[1:, :] + rem_day_diff >= 0)
    return trade_params


def vectorized_rotatedq_cone(model, x1, x2, x3):
    N = x1.shape[0]
    assert x1.shape == x2.shape
    assert x2.shape == x3.shape
    # model.update()
    for i in range(N):
        model.addQConstr(x1[i].item() * x2[i].item() >= x3[i].item() * x3[i].item())


def vectorized_soc_cone(model, cost_param, aux, abs_delta):
    N = aux.shape[0]
    for i in range(N):
        a = cost_param[i].item()
        b = aux[i].item()
        c = abs_delta[i].item()

        # a+b >= (a-b, 2c)
        model.addQConstr((a + b) * (a + b) >= (a - b) * (a - b) + 4 * c * c)
        # reshape(reshape(weights[0, 0:2], (1, 2), F) + -[[11. 11.]], (1, 2), F) <= var3218
        # SOC(
        #    reshape(var3228 + var3229, (2,), F),
        #    Vstack(
        #        reshape(var3228 + -var3229, (1, 2), F),
        #        reshape(Promote(2.0, (1, 2)) @ var3218, (1, 2), F),
        #    ),
        # )
        model.addQConstr((1 + c) * (1 + c) >= (1 - c) * (1 - c) + 4 * b * b)


def add_market_cost_soc(model, weights, w0):
    D, N = weights.shape
    print("Adding Market Costs", D, N)
    # s = m.addMVar(shape=(D, N))
    # first_day_diff, rest_diff = compute_delta_weights(weights, initial_weights)
    auxillary = []
    bound = GRB.INFINITY
    for day in range(D):
        weights_day = weights[day, :]
        weights_day_prev = weights[day - 1, :] if day > 0 else w0
        # ic(weights_day_prev)
        # ic(weights_day.shape, weights_day_prev.shape)
        assert weights_day.shape == weights_day_prev.shape
        delta = weights_day - weights_day_prev
        delta_day = model.addMVar(
            shape=(N,),
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=GRB.INFINITY,
            name=f"|delta|_mc_{day}",
        )
        model.addConstr(delta_day - delta >= 0.0, name=f"buy_mc_{day}")
        model.addConstr(delta_day + delta >= 0.0, name=f"sell_mc_{day}")

        t_day = model.addMVar(N, lb=0, vtype=GRB.CONTINUOUS, name=f"t_day_{day}")
        s_day = model.addMVar(N, lb=0, vtype=GRB.CONTINUOUS, name=f"s_day_{day}")
        # to be scaled by gamma

        # (s_i, t_i, delta_i) \in Q_r
        # model.mosek.constraint(
        #    mosek.Expr.hstack(s_day, t_day, delta_day),
        #    mosek.Domain.inRotatedQCone(),
        # )
        # vectorized_rotatedq_cone(model, s_day, t_day, delta_day)
        # (delta_i, 1/8, s_i) \in Q_r
        # model.mosek.constraint(
        #    mosek.Expr.hstack(
        #        delta_day,
        #        mosek.Expr.constTerm(problem.number_of_stocks, 1.0 / 8.0),
        #        s_day,
        #    ),
        #    mosek.Domain.inRotatedQCone(),
        # )
        # vectorized_rotatedq_cone(model, delta_day, np.ones(N) * 0.125, s_day)
        vectorized_soc_cone(model, t_day, s_day, delta_day)
        auxillary.append(t_day)
    return auxillary


def make_abs_poly_parameters(model, weights, w0):
    D, N = weights.shape
    ic("Adding Market Impact Costs - Linear Interpolation", D, N)
    # s = m.addMVar(shape=(D, N))
    # first_day_diff, rest_diff = compute_delta_weights(weights, initial_weights)
    auxillary = []
    for day in range(D):
        weights_day = weights[day, :]
        weights_day_prev = weights[day - 1, :] if day > 0 else w0
        # ic(weights_day_prev)
        # ic(weights_day.shape, weights_day_prev.shape)
        assert weights_day.shape == weights_day_prev.shape
        delta = weights_day - weights_day_prev
        delta_day = model.addMVar(
            shape=(N,),
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=GRB.INFINITY,
            name=f"|delta|_mc_{day}",
        )
        model.addConstr(delta_day - delta >= 0.0, name=f"buy_mc_{day}")
        model.addConstr(delta_day + delta >= 0.0, name=f"sell_mc_{day}")

        abs_poly = model.addMVar(
            shape=(N,), lb=0, ub=5, vtype=GRB.CONTINUOUS, name=f"FuncApprox_{day}"
        )
        for i in range(N):
            model.addGenConstrPow(
                delta_day[i],
                abs_poly[i],
                1.5,
                options="FuncPieces=100",
            )
        auxillary.append(abs_poly)
        # auxillary.append(aux)
        # c1 = m.addConstr(aux - delta >= 0.0, name=f"buy_{day}")
        # c2 = m.addConstr(aux + delta >= 0.0, name=f"sell_{day}")
        # ic(c1, c2)
    return auxillary


def make_abs_value_parameters(m, weights, initial_weights):
    D, N = weights.shape
    # ic("Adding Trading Costs", D, N)
    # s = m.addMVar(shape=(D, N))
    # first_day_diff, rest_diff = compute_delta_weights(weights, initial_weights)
    auxillary = []
    for day in range(D):
        weights_day = weights[day, :]
        weights_day_prev = weights[day - 1, :] if day > 0 else initial_weights
        # ic(weights_day_prev)
        # ic(weights_day.shape, weights_day_prev.shape)
        assert weights_day.shape == weights_day_prev.shape
        delta = weights_day - weights_day_prev
        aux = m.addMVar(shape=(N,), lb=0, ub=GRB.INFINITY, name=f"abs_delta_{day}")
        auxillary.append(aux)
        c1 = m.addConstr(aux - delta >= 0.0, name=f"buy_{day}")
        c2 = m.addConstr(aux + delta >= 0.0, name=f"sell_{day}")
        # ic(c1, c2)
    return auxillary


def add_cardinality_constraint(y, K):
    return cp.sum(y) <= K


def add_mixed_integer_bigM_constraints(weights, model):
    D, N = weights.shape
    integer_constraints = []
    integer_variables = []
    for d in range(D):
        y = cp.Variable(shape=weights.shape, boolean=True, name=f"BooleanWeights_{d}")
        integer_variables.append(y)
        cardinality_constraint = add_cardinality_constraint(y, K)
        bigM_constraint = weights[d, :] <= cp.multiply(y, M)
        # ic(cardinality_constraint, bigM_constraint)
        integer_constraints += [cardinality_constraint, bigM_constraint]
    return integer_constraints


def add_risk_measures(m, weights, data, alpha=0.1):
    # print("add risk measures")
    # ic(weights.shape)
    off = data["official_risk_measure"]
    act = data["active_risk_measure"]
    target = data["target_risk"]
    benchmark_normalization_factors = compute_benchmark_normalization(
        data["benchmark_index"], off, target
    )
    # print(data.benchmark_index)
    # print(benchmark_normalization_factors)
    D = len(target)
    # off_con = []
    # act_con = []
    for day in range(D):
        # print("Adding risk", day)
        day_weights = weights[day, :]
        off_risk = day_weights @ off @ day_weights
        off_target = target[day] ** 2
        # ic(target, off_target)
        o_con = m.addConstr(off_risk <= off_target, f"off_risk_{day}")
        bench = benchmark_normalization_factors[day] * data["benchmark_index"]
        vec = day_weights - bench
        act_risk = vec @ act @ vec
        ac_target = (alpha * target[day]) ** 2
        a_con = m.addConstr(act_risk <= ac_target, f"acc_risk_{day}")


def indicator_cardinality(m, weights, K, initial_bits, params):
    # logging.info("Adding Cardinality Constraints")
    D, N = weights.shape
    # assert D == 1
    for day in range(D):
        # logging.info(f"Set cardinality constraing day {day} - nonzeros to set {initial_bits.sum()}")
        # print("initial_bits", initial_bits)
        day_weights = weights[day, :]
        y = m.addVars(
            N, name=f"cardinality_{day}", vtype=GRB.BINARY
        )  # , obj=initial_bits
        if initial_bits is not None:
            y.Start = initial_bits[day, :]
            ic(day, initial_bits[day, :].sum())
        params[f"ind_{day}"] = y
        m.update()

        # y = m.addMVar(N, name=f"cardinality_{day}", vtype=GRB.BINARY)
        m.addConstrs((y[i] == 0) >> (day_weights[i] == 0) for i in range(N))
        m.addConstr(y.sum() <= K)


def make_market_cost(abs_pow_trade_deltas, gamma):
    # trade_cost = (trade_params @ atp_data.trading_costs_beta).sum()
    market_cost = 0
    for mp in abs_pow_trade_deltas:
        market_cost = market_cost + (mp * gamma).sum()
    return market_cost


def make_trade_cost(abs_trade_deltas, beta):
    # ic("Gurobipy - Adding trade cost")
    # trade_cost = (trade_params @ atp_data.trading_costs_beta).sum()
    trade_cost = 0
    for tp in abs_trade_deltas:
        trade_cost = trade_cost + (tp * beta).sum()
    return trade_cost
    # objective = income - trade_cost


# Create an empty model
def make_gurobi(
    data, add_trade_cost=True, add_market_cost=False, K=-1, starting_weights=None
):
    m = gp.Model("portfolio")
    # ic(data)
    # atp_data = sn(**data.copy())
    # ic(atp_data.target_risk)
    D, N = data["expected_return"].shape  # atp_data.expected_retur%n.shape
    # if initial_weights is None:
    #    initial_weights = np.zeros((D, N))
    # if initial_weights is None:
    #    initial_weights = np.zeros(D, N)  # = np.tile(initial_weights, (D_new, 1))
    ic("Make Gurobi", D, N)
    weights = m.addMVar(
        shape=(D, N),
        name="weights",
        lb=0.0,
        ub=GRB.INFINITY,
    )
    params = {}
    params["weights"] = weights
    if starting_weights is not None:
        ic(starting_weights.std())
        assert weights.shape == starting_weights.shape
        weights.Start = starting_weights
    else:
        ic(starting_weights)

    income = (weights * data["expected_return"]).sum()
    objective = income
    """
    if include_trade_cost:
        # ic("Gurobipy - Adding trade cost")
        trade_params = add_trade_cost(m, weights, atp_data.weights)
        # trade_cost = (trade_params @ atp_data.trading_costs_beta).sum()
        beta = atp_data.trading_costs_beta.ravel()
        trade_cost = 0
        for tp in trade_params:
            trade_cost = trade_cost + (tp * beta).sum()
        objective = income - trade_cost
    else:
        objective = income
    """

    if add_trade_cost:
        # create the trade cost from abs_vars which are at least abs_vars_i > = |w_i - w_{i-1}|
        abs_vars = make_abs_value_parameters(m, weights, data["weights"])
        beta = data["trading_costs_beta"].ravel()
        tc = make_market_cost(abs_vars, beta)
        objective = objective - tc
        # assert False
        # tc = 0.0
        # for tp in abs_vars:
        #    tc = tc + (tp * beta).sum()
        # objective = objective - tc
    if add_market_cost:
        gamma = data["trading_costs_gamma"].ravel()
        # ic(gamma)
        # abs_poly_params = make_abs_poly_parameters(m, weights, data["weights"])
        abs_poly_params = add_market_cost_soc(m, weights, data["weights"])
        mc = make_market_cost(abs_poly_params, gamma)
        objective = objective - mc

    alpha = 0.1
    add_risk_measures(m, weights, data, alpha)

    if K > 0:
        ic("Add Cardinality constraint -> MIP")
        # if initial_weights is None:
        #    ic((initial_weights > 0)).sum(axis=1)

        # assert False
        # initial_indices = data["weights"].ravel() > 0
        # ic(initial_indices.sum())
        bits = None
        if starting_weights is not None:
            bits = starting_weights > 0

        indicator_cardinality(m, weights, K, bits, params)
        # assert False

    m.setObjective(objective, GRB.MAXIMIZE)
    # m.update()
    # ic(weights.Start.std())
    return m, params


def solve(m, time_limit=300):
    m.setParam("TimeLimit", time_limit)
    m.optimize()
    return m


def test():
    from atp_mosek import load_npy_file, data_subset, dum_data

    filename3 = "scaled_data_file_as_dict_2020_03_16.npy"
    all_data = load_npy_file(filename=filename3)
    ic(all_data["target_risk"])
    dummy_data_2_3 = dum_data(all_data, 2, 3)
    # all_data['trading_costs_gamma'] = all_data['trading_costs_gamma'] * 0.0
    day_data = data_subset(all_data, 1, 2000)
    ic(day_data["target_risk"])
    model, weights = make_gurobi(day_data, add_trade_cost=False, K=-1)
    # model, weights = make_gurobi(dummy_data_2_3, include_trade_cost=True, K=-1)
    model, weights = make_gurobi(all_data, add_trade_cost=True, K=-1)

    start = time.time()
    model.setParam("predual", 1)
    solve(model, time_limit=60)
    end = time.time()
    print(f"Time Used: {end - start} - Objective {model.objVal}")


if __name__ == "__main__":
    test()
