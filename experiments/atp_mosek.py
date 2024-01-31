from icecream import ic
import h5py
import time

# import mosek
import cvxpy as cp
import numpy as np
import copy


def add_field_from_fileobj(data, fieldname):
    try:
        data_variable = np.array(data[fieldname])
    except KeyError as e:
        print(e)
        print(f"Key: {fieldname} is not a valid key in input data file.")
        return None
    except Exception as e:
        print(e)
        print(f"Unknown  exception in reading  field {fieldname}.")
        return None
    else:
        return data_variable


def load_input_data_from_file(data: h5py.File):
    weights = add_field_from_fileobj(data, "weights")
    stock_ids = add_field_from_fileobj(data, "stock_ids")
    if stock_ids is not None:
        stock_ids = stock_ids.astype(np.str_)
    target_risk = add_field_from_fileobj(data, "target_risk")
    expected_return = add_field_from_fileobj(data, "expected_return")
    benchmark_index = add_field_from_fileobj(data, "benchmark_index")
    active_risk_measure = add_field_from_fileobj(data, "active_risk_measure")
    official_risk_measure = add_field_from_fileobj(data, "official_risk_measure")
    trading_costs_beta = add_field_from_fileobj(data, "trading_costs/beta")
    trading_costs_gamma = add_field_from_fileobj(data, "trading_costs/gamma")
    output = dict(
        weights=weights,
        stock_ids=stock_ids,
        target_risk=target_risk,
        expected_return=expected_return,
        benchmark_index=benchmark_index,
        active_risk_measure=active_risk_measure,
        official_risk_measure=official_risk_measure,
        trading_costs_beta=trading_costs_beta,
        trading_costs_gamma=trading_costs_gamma,
    )
    return output


def load_npz_file(filename="2008_11_26.npz"):
    np_data = np.load(filename)
    weights = add_field_from_fileobj(np_data, "initial_weights")
    N = len(weights)
    stock_ids = add_field_from_fileobj(np_data, "stock_ids")
    if stock_ids is not None:
        stock_ids = stock_ids.astype(np.str_)
    target_risk = add_field_from_fileobj(np_data, "target_risk").reshape(
        1,
    )
    expected_return = add_field_from_fileobj(np_data, "expected_return").reshape(-1, N)
    benchmark_index = add_field_from_fileobj(np_data, "benchmark_index")
    active_risk_measure = add_field_from_fileobj(np_data, "active_risk_measure_matrix")
    official_risk_measure = add_field_from_fileobj(
        np_data, "official_risk_measure_matrix"
    )
    trading_costs_beta = add_field_from_fileobj(np_data, "trading_costs_beta")
    trading_costs_gamma = add_field_from_fileobj(np_data, "trading_costs_gamma")
    output = dict(
        weights=weights,
        stock_ids=stock_ids,
        target_risk=target_risk,
        expected_return=expected_return,
        benchmark_index=benchmark_index,
        active_risk_measure=active_risk_measure,
        official_risk_measure=official_risk_measure,
        trading_costs_beta=trading_costs_beta,
        trading_costs_gamma=trading_costs_gamma,
    )
    return output


def load_npy_file(filename="data_file_as_dict.npy"):
    np_data = np.load(filename, allow_pickle=True).item()
    weights = add_field_from_fileobj(np_data, "weights")
    N = len(weights)
    stock_ids = add_field_from_fileobj(np_data, "stock_ids")
    if stock_ids is not None:
        stock_ids = stock_ids.astype(np.str_)
    target_risk = add_field_from_fileobj(np_data, "target_risk")
    expected_return = add_field_from_fileobj(np_data, "expected_return")
    benchmark_index = add_field_from_fileobj(np_data, "benchmark_index")
    active_risk_measure = add_field_from_fileobj(np_data, "active_risk_measure")
    official_risk_measure = add_field_from_fileobj(np_data, "official_risk_measure")
    trading_costs_beta = add_field_from_fileobj(np_data, "trading_costs_beta")
    trading_costs_gamma = add_field_from_fileobj(np_data, "trading_costs_gamma")
    # official_risk_measure = official_risk_measure
    # ic(official_risk_measure.shape)
    output = dict(
        weights=weights,
        stock_ids=stock_ids,
        target_risk=target_risk,
        expected_return=expected_return,
        benchmark_index=benchmark_index,
        active_risk_measure=active_risk_measure,
        official_risk_measure=official_risk_measure,
        trading_costs_beta=trading_costs_beta,
        trading_costs_gamma=trading_costs_gamma,
    )
    return output
    ic(filename)
    ic(
        np.abs(expected_return).max(),
        active_risk_measure.max(),
        official_risk_measure.max(),
        active_risk_measure.min(),
        official_risk_measure.min(),
    )
    ic(
        np.abs(expected_return).mean(),
        np.abs(active_risk_measure).mean(),
        np.abs(official_risk_measure).mean(),
    )
    ic(
        np.abs(trading_costs_beta).mean(),
        np.abs(trading_costs_gamma).mean(),
        np.abs(benchmark_index).mean(),
        np.abs(weights).min(),
    )
    return output


def dum_data(data, D, N):
    output = dict(
        weights=data["weights"][:N] * 0 + 11,
        stock_ids=data["stock_ids"][:N],
        target_risk=data["target_risk"][:D] * 0 + 100,
        expected_return=data["expected_return"][:D, :N] * 0 + 3,
        benchmark_index=data["benchmark_index"][:N] * 0,
        active_risk_measure=data["active_risk_measure"][:N, :N],
        official_risk_measure=data["official_risk_measure"][:N, :N],
        trading_costs_beta=data["trading_costs_beta"][:N] * 0 + 6,
        trading_costs_gamma=data["trading_costs_gamma"][:N] * 0 + 8,
    )
    return output


def data_subset(data, D, N, scale=True):
    num_assets = len(data["stock_ids"])
    asset_scale = 1.0
    if N < num_assets and scale:
        asset_scale = N / num_assets
    else:
        asset_scale = 1.0
    ic(num_assets, D, N, asset_scale)
    output = dict(
        weights=data["weights"][:N],
        stock_ids=data["stock_ids"][:N],
        target_risk=data["target_risk"][:D] * asset_scale,
        expected_return=data["expected_return"][:D, :N],
        benchmark_index=data["benchmark_index"][:N],
        active_risk_measure=data["active_risk_measure"][:N, :N],
        official_risk_measure=data["official_risk_measure"][:N, :N],
        trading_costs_beta=data["trading_costs_beta"][:N],
        trading_costs_gamma=data["trading_costs_gamma"][:N],
    )
    return output


def example():
    filename = "2020-03-16-new.hdf"
    data_file = h5py.File(filename, "r")
    # ic(data_file)
    raw_data = load_input_data_from_file(data_file)
    all_data = copy.deepcopy(raw_data)
    scale_factor = 1.0 / raw_data["target_risk"][0]
    all_data["weights"] *= scale_factor
    all_data["trading_costs_gamma"] *= np.sqrt(scale_factor)
    all_data["target_risk"] *= scale_factor
    ic(all_data["target_risk"])

    ic(raw_data["expected_return"].shape)
    day_data = data_subset(all_data, 1, 2000)
    ic(day_data["expected_return"].shape)

    # ic(data)
    dummy_data_2_3 = dum_data(raw_data, 2, 3)

    ic(dummy_data_2_3)
    print("data loaded")


def get_data(filename="2020-03-16-new.hdf"):
    assert False
    data_file = h5py.File(filename, "r")
    # ic(data_file)
    raw_data = load_input_data_from_file(data_file)
    all_data = copy.deepcopy(raw_data)
    scale_factor = 1.0 / raw_data["target_risk"][0]
    all_data["weights"] *= scale_factor
    all_data["trading_costs_gamma"] *= np.sqrt(scale_factor)
    all_data["target_risk"] *= scale_factor
    return all_data


def get_default():
    filename3 = "scaled_data_file_as_dict_ATP_new_dateless.2020-03-16.npy"
    all_data = load_npy_file(filename=filename3)
    return all_data


def get_all_data():
    filenames = [
        "scaled_data_file_as_dict_ATP_new_dateless.2016-05-17.npy",
        "scaled_data_file_as_dict_ATP_new_dateless.2016-05-18.npy",
        "scaled_data_file_as_dict_ATP_new_dateless.2012-09-05.npy",
        "scaled_data_file_as_dict_ATP_new_dateless.2012-09-06.npy",
        "scaled_data_file_as_dict_ATP_new_dateless.2020-03-13.npy",
        "scaled_data_file_as_dict_ATP_new_dateless.2020-03-16.npy",
    ]

    def extract_date(date):
        return date.split(".")[-2]

    data_dict = {extract_date(x): load_npy_file(x) for x in filenames}
    return data_dict


def get_default_consecutive():
    f1 = "scaled_data_file_as_dict_ATP_new_dateless.2020-03-13.npy"
    f2 = "scaled_data_file_as_dict_ATP_new_dateless.2020-03-16.npy"
    data1 = load_npy_file(filename=f1)
    data2 = load_npy_file(filename=f2)
    return data1, data2


def data_with_w0(
    first_date, cons_date, add_trade_cost=True, add_market_cost=True, K=160
):
    data_dict = get_all_data()
    D0 = data_dict[first_date]
    D1 = data_dict[cons_date]
    res = run_k_largest(D0, trade_cost=add_trade_cost, market_cost=add_market_cost, K=K)
    print("Previous value", res["sparse_value"])
    initial_weights = res["sparse_weights"][0, :]
    print(initial_weights)
    K = len(np.nonzero(initial_weights)[0])
    # assert False
    D1["weights"] = initial_weights
    return D1


def get_default_day():
    all_data = get_default()
    day_data = data_subset(all_data, 1, 2000)
    return day_data


def get_new_data():
    pass


def pretty_print_prog(prog):
    print("Objective:", " ".join(str(prog.objective).split()))
    ic(prog.objective)
    print("Num Constraints", len(prog.constraints))
    for i, cns in enumerate(prog.constraints):
        print(f"Constraint {i}:", " ".join(str(cns).split()))
    print("Variables:\n", prog.param_dict)
    print("Parameters:\n", prog.param_dict)


def print_program(name, prog):
    stars = "*" * 20
    print(f"{stars} {name} {stars}")
    pretty_print_prog(prog)
    # print(prog)
    print(stars)


def print_mosek(data, shorten=False):
    for key, item in data.items():
        if key == shorten and "K_dir":
            cones = item
            cone_keys = ["dp3", "q", "s"]
            for cone_key in cone_keys:
                counts = np.unique(np.array(cones[cone_key]), return_counts=True)
                ic(cone_key, counts)
        else:
            ic(key, item)


def print_cone_program(prob, solver=cp.MOSEK):
    red_cone = cp.reductions.Dcp2Cone()
    p2, l2 = red_cone.apply(prob)
    print_program("Dcp2Cone", p2)
    # prob_data, chain, inverse_data = prob.get_problem_data(solver)
    # print_mosek(prob_data)
    # ic(chain)


def matrix_sqrt(x):
    # MOSEK is much faster with an eigenvalue based sqrt
    w, V = np.linalg.eigh(x)
    # ic(w.min())
    assert w.min() > -1e-10
    w = np.clip(w, a_min=0, a_max=None)
    # ic(w.min())
    res = np.diag(np.sqrt(w)) @ V.T
    return res


def add_quad_constraints_soc(
    weights, data, benchmark_normalization_factors, target_risk, constraints, alpha
):
    # Defining risk constraint and objective
    official_risk_sqrt = matrix_sqrt(data["official_risk_measure"])
    active_risk_sqrt = matrix_sqrt(data["active_risk_measure"])
    D = len(alpha)
    # off = weights @ O
    # s = time.time()
    for i in range(D):
        risk = cp.SOC(
            target_risk[i],
            official_risk_sqrt @ weights[i, :].T,
            constr_id=f"SOC_Official_Risk_{i}",
        )
        risk2 = cp.SOC(
            target_risk[i] * alpha[i],
            active_risk_sqrt
            @ (
                weights[i, :]
                - benchmark_normalization_factors[i] * data["benchmark_index"]
            ).T,
            constr_id=f"SOC_Active_Risk_{i}",
        )
        constraints += [risk, risk2]
    # print('add quad const soc time', time.time() - s)


def add_quad_constraints_quad(
    x, data, benchmark_normalization_factors, target_risk, constraints, alpha
):
    # Defining risk constraint and objective
    D = len(alpha)
    for i in range(D):
        risk = cp.quad_form(
            x[i, :], data["official_risk_measure"]
        )  # , constr_id=f'Quad_Off_risk_{i}')
        risk2 = cp.quad_form(
            x[i, :] - benchmark_normalization_factors[i] * data["benchmark_index"],
            data["active_risk_measure"],
        )  # , constr_id=f'Quad_Acc_Risk_{i}') # change to SOC?
        constraints += [risk <= target_risk[i] ** 2]  # variance constraint
        constraints += [
            risk2 <= (alpha[i] * target_risk[i]) ** 2
        ]  # variance constraint


def add_power_cones(delta, constraints):
    z = cp.Variable(delta.shape, name="MarketCost")
    constraints += [
        cp.constraints.PowCone3D(
            z, np.ones(z.shape), delta, 2.0 / 3.0, constr_id="market_cost_p3"
        )
    ]
    return z


def delta_weights(x, initial_weights):
    D, N = x.shape
    abs_first_day_trade = cp.reshape(x[0, :], (1, N)) - initial_weights
    if D == 1:
        return cp.reshape(abs_first_day_trade, (1, N))
    abs_day_trade = x[1:, :] - x[:-1, :]  # shape (D-1) x N
    delta_weights = cp.vstack([abs_first_day_trade, abs_day_trade])
    return delta_weights


def trade_cost(delta, beta):
    return cp.sum(cp.abs(delta) @ beta)


def market_cost(delta, gamma):
    # ic("atp mosek market cost", gamma)
    return cp.sum(cp.power(cp.abs(delta), 1.5) @ gamma)


def compute_benchmark_normalization(
    benchmark_index, official_risk_measure, target_risk
):
    standard_deviation = np.sqrt(
        benchmark_index.T @ official_risk_measure @ benchmark_index
    )
    clipped_std = np.maximum(standard_deviation, 1.0e-16)
    benchmark_normalization_factors_ = target_risk / clipped_std
    return benchmark_normalization_factors_


def add_mixed_integer_bigM_constraints(weights, M, K):
    D, N = weights.shape
    integer_constraints = []
    integer_variables = []
    for d in range(D):
        y = cp.Variable(shape=(N,), boolean=True, name=f"BooleanWeights_{d}")
        integer_variables.append(y)
        cardinality_constraint = cp.sum(y) <= K
        # cp.multiply(weights[d, :], y) == 0 #
        bigM_constraint = weights[d, :] <= cp.multiply(y, M)
    integer_constraints += [cardinality_constraint, bigM_constraint]
    return integer_constraints, integer_variables


def add_mixed_integer_quad_constraints(weights, K):
    D, N = weights.shape
    integer_constraints = []
    integer_variables = []
    for d in range(D):
        y = cp.Variable(shape=(N,), boolean=True, name=f"BooleanWeights_{d}")
        integer_variables.append(y)
        cardinality_constraint = cp.sum(y) <= K
        # cp.multiply(weights[d, :], y) == 0 #
        # bigM_constraint = weights[d, :] <= cp.multiply(y, M)
        integer_constraint = weights[d, :] * y <= y
    integer_constraints += [cardinality_constraint, integer_constraint]
    return integer_constraints, integer_variables


def add_mixed_integer_bigM_constraints(weights, M, K):
    D, N = weights.shape
    integer_constraints = []
    integer_variables = []
    for d in range(D):
        y = cp.Variable(shape=(N,), boolean=True, name=f"BooleanWeights_{d}")
        integer_variables.append(y)
        cardinality_constraint = cp.sum(y) <= K
        # cp.multiply(weights[d, :], y) == 0 #
        bigM_constraint = weights[d, :] <= cp.multiply(y, M)
    integer_constraints += [cardinality_constraint, bigM_constraint]
    return integer_constraints, integer_variables


def create_problem(
    data,
    days=-1,
    scale=1,
    add_trade_cost=True,
    add_market_cost=True,
    use_power_cone=False,
    mip_K=-1,
    bigM=1,
    deleted_indices=None,
):
    assert scale == 1
    # sigma = np.eye(N) # Y.cov().to_numpy()
    # if scale <= 0:
    #    scale = 1.0 / data["target_risk"][0]
    #    assert False
    exp_return = data["expected_return"][:days, :]
    D, N = exp_return.shape
    # ic("cvxpy shape", D, N, scale)
    # active_risk_measure = data['active_risk_measure']
    benchmark_index = data["benchmark_index"]
    official_risk_measure = data["official_risk_measure"]
    beta = data["trading_costs_beta"].reshape(N, 1)
    gamma = data["trading_costs_gamma"].reshape(N, 1) / np.sqrt(scale)
    initial_weights = data["weights"].reshape(1, N) * scale
    target_risk = data["target_risk"][:days] * scale
    # target_risk = data["target_risk"][:days] * scale

    alpha = np.ones(D) * 0.1

    x = cp.Variable(shape=(D, N), name="weights")
    # start_value = np.broadcast_to(initial_weights.reshape(1, N), (D, N))
    # x.value = start_valuenew_exp
    benchmark_normalization_factors_ = compute_benchmark_normalization(
        benchmark_index, official_risk_measure, target_risk
    )

    delta_w = delta_weights(x, initial_weights)
    # objective = cp.sum(cp.multiply(exp_return, x))
    if add_trade_cost:
        print("adding trade cost ")
        objective = cp.sum(cp.multiply(exp_return, x)) - trade_cost(delta_w, beta)
    else:
        objective = cp.sum(cp.multiply(exp_return, x))
    # make constraints
    constraints = [x >= 0]
    # weight_upper_bound = cp.Parameter(shape=x.shape, value=10 * np.ones(x.shape), name='param_weight_uppper_bounds')
    # constraints += [x <= weight_upper_bound]

    if add_market_cost:
        print("add market cost")
        if use_power_cone:
            pow_cone_variables = add_power_cones(delta_w, constraints)
            objective = objective - cp.sum(pow_cone_variables @ gamma)
        else:
            objective = objective - market_cost(delta_w, gamma)

    add_quad_constraints_soc(
        x, data, benchmark_normalization_factors_, target_risk, constraints, alpha
    )
    if mip_K > 0:
        print("Adding integer K constraint")
        integer_cons, integer_vars = add_mixed_integer_bigM_constraints(x, bigM, mip_K)
        constraints += integer_cons
        # objective = objective - cp.sum(integer_vars[0]) * 1e-5

    if deleted_indices is not None:
        ic("fixing weights", deleted_indices.sum())
        assert deleted_indices.shape == exp_return.shape

        constraints += [cp.sum(cp.multiply(deleted_indices, x)) <= 0]
        # assert False

    objective = cp.Minimize(-objective)
    # Solving the problem with several solvers
    prob = cp.Problem(objective, constraints)

    return prob


def solve_problem(problem, optimizer, mosek_params, save_file=None, warm_start=False):
    start = time.time()
    print("Solving using", optimizer)

    problem.solve(
        solver=optimizer,
        verbose=False,
        mosek_params=mosek_params,
        save_file=save_file,
        ignore_dpp=True,
        warm_start=warm_start,
    )
    end = time.time()
    print(f"Time Taken {optimizer} -: {end-start} seconds")


def find_k_largest(weights, k):
    sorted = np.argsort(-weights)
    return sorted[:, :k]


def run_k_largest(data, trade_cost, market_cost, K):
    problem = create_problem(
        data,
        days=1000,
        add_trade_cost=trade_cost,
        add_market_cost=market_cost,
        use_power_cone=False,
        mip_K=-1,
    )
    problem.solve(solver=cp.MOSEK, verbose=False)  # , mosek_params=mosek_options)
    full_value = problem.value
    optimized_weights = problem.var_dict["weights"].value
    D, N = optimized_weights.shape
    good_indices = find_k_largest(optimized_weights, K)
    # ic(good_indices)
    # assert False
    bad_weights = np.ones(optimized_weights.shape)
    rows = np.arange(D).reshape(-1, 1)
    bad_weights[rows, good_indices] = 0
    # bad_weights[good_indices] = 0
    # assert False
    ic(bad_weights.sum())
    problem_sparse = create_problem(
        data,
        days=1000,
        add_trade_cost=trade_cost,
        add_market_cost=market_cost,
        use_power_cone=False,
        mip_K=-1,
        deleted_indices=bad_weights,
    )
    problem_sparse.solve(
        solver=cp.MOSEK, verbose=False
    )  # , mosek_params=mosek_options)
    sparse_optimized_weights = problem_sparse.var_dict["weights"].value
    ic(-problem_sparse.value, ((sparse_optimized_weights) > 0).sum())
    # assert False
    # ic(sparse_optimized_weights)
    return {
        "sparse_value": -problem_sparse.value,
        "sparse_weights": sparse_optimized_weights,
        "full_value": -full_value,
    }


def basic_example(K=-1):
    all_data = get_data()
    D = 1
    N = 2000
    runtime = 60
    test_data = data_subset(all_data, D, N)
    # ic(test_data)
    problem = create_problem(
        test_data,
        days=2,
        scale=-1,
        add_trade_cost=True,
        add_market_cost=True,
        use_power_cone=True,
        mip_K=K,
    )
    mosek_options = dict(
        MSK_DPAR_MIO_MAX_TIME=runtime,
        MSK_DPAR_OPTIMIZER_MAX_TIME=runtime,
    )

    problem.solve(solver=cp.MOSEK, verbose=True, mosek_params=mosek_options)
    tmp = problem.var_dict["weights"]
    ic(tmp)
    ic(tmp.value.nonzero()[1].shape)


def npz_example(K=-1):
    D = 1
    N = 2000
    runtime = 60
    all_data = load_npz_file()
    # all_data_old = get_data()
    # ic(all_data)
    # ic(all_data_old)
    test_data = data_subset(all_data, D, N)
    # ic(test_data)
    problem = create_problem(
        test_data,
        days=2,
        scale=-1,
        add_trade_cost=True,
        add_market_cost=True,
        use_power_cone=True,
        mip_K=K,
    )
    mosek_options = dict(
        MSK_DPAR_MIO_MAX_TIME=runtime,
        MSK_DPAR_OPTIMIZER_MAX_TIME=runtime,
    )

    problem.solve(solver=cp.MOSEK, verbose=True, mosek_params=mosek_options)
    tmp = problem.var_dict["weights"]
    ic(tmp)
    ic(tmp.value.nonzero()[1].shape)


def npy_example(K=-1):
    D = 1
    N = 2000
    runtime = 180
    all_data = load_npy_file("2020_03_16.npy")
    # all_data_old = get_data()
    # ic(all_data)
    # ic(all_data_old)
    test_data = data_subset(all_data, D, N)
    # ic(test_data)
    problem = create_problem(
        test_data,
        days=2,
        scale=-1,
        add_trade_cost=True,
        add_market_cost=True,
        use_power_cone=True,
        mip_K=K,
    )
    mosek_options = dict(
        MSK_DPAR_MIO_MAX_TIME=runtime,
        MSK_DPAR_OPTIMIZER_MAX_TIME=runtime,
    )

    problem.solve(solver=cp.MOSEK, verbose=True, mosek_params=mosek_options)
    tmp = problem.var_dict["weights"]
    ic(tmp)
    ic(tmp.value.nonzero()[1].shape)


def test_k_largest():
    day_data = get_default_day()
    run_k_largest(day_data, True, False, 160)


if __name__ == "__main__":
    # npz_example()

    # print("No constraint")
    # basic_example(-1)
    # print("Pick 160 assets")
    # basic_example(160)
    # print("No constraint")
    # npy_example(-1)
    # print("Pick 160 assets")
    # npy_example(160)
    test_k_largest()
