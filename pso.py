import numpy as np


def calc_score(x, score_func):
    return score_func(x)


def check_constraint(constraint_ueq, x):
    for constraint_func in constraint_ueq:
        if constraint_func(x) > 0:
            return 0
    return 1


def update_velocity(initial_velocity,
                    l_best, n_best, g_best,
                    x, c0, c1, c2, c3):
    # print("=" * 50)
    # print(initial_velocity[0])
    # print((g_best - x)[0])
    randomness = np.random.rand(3, *initial_velocity.shape)
    init_effect = c0 * initial_velocity
    local_effect = c1 * randomness[0] * (l_best - x)
    nghbr_effect = c2 * randomness[1] * (n_best - x)
    global_effect = c3 * randomness[2] * (g_best - x)
    new_velocity = init_effect + local_effect + nghbr_effect + global_effect
    # print(new_velocity[0])
    return new_velocity


def update_x(x, velocity):
    return x + velocity


def update_local_best(x, x_score, old_l_best,
                      old_l_best_score, constraints=[]):
    tmp1 = np.stack((old_l_best_score, x_score))
    mask = np.argmin(tmp1, axis=0)
    for idx, xs in enumerate(x):
        if mask[idx] == 1:
            mask[idx] = check_constraint(constraints, xs)
    m_range = list(range(len(mask)))
    tmp2 = np.stack((old_l_best, x))
    l_best_score = tmp1[mask, m_range]
    l_best = tmp2[mask, m_range]
    return l_best, l_best_score


def get_global_best(l_best, l_best_score):
    current_best_score_i = np.argmin(l_best_score)
    g_best = l_best[current_best_score_i]
    g_best_score = l_best_score[current_best_score_i]
    return g_best, g_best_score


# the nan solution doesn't look good (wasn't)
# used inf
def get_nghbr_best(l_best, l_best_score, nghbrh_matrix):
    l_best_score = np.where(l_best_score == 0, np.inf, l_best_score)
    tmp = l_best_score[None, :, None] * nghbrh_matrix[:, :]
    # tmp = np.where(tmp == 0, np.nan, tmp)
    n_best_indices = np.nanargmin(tmp,
                                  axis=1)[0, :]
    n_best = l_best[n_best_indices]
    n_best_score = l_best_score[n_best_indices]
    return n_best, n_best_score


def epoch(x, v, l_best, l_best_score,
          adj_matrix, c0, c1, c2, c3,
          opt_func, ranges, constraints):
    x_score = calc_score(x, opt_func)
    new_l_best, new_l_best_score = update_local_best(x, x_score, l_best,
                                                     l_best_score, constraints)

    new_g_best, _ = get_global_best(new_l_best,
                                    new_l_best_score)
    new_g_best = np.tile(new_g_best, (x.shape[0], 1))

    new_n_best, _ = get_nghbr_best(new_l_best,
                                   new_l_best_score,
                                   adj_matrix)
    new_v = update_velocity(v, new_l_best,
                            new_n_best,
                            new_g_best,
                            x, c0, c1, c2, c3)
    new_x = update_x(x, new_v)
    new_x = np.clip(new_x, *ranges)
    new_v = np.where(np.any(new_x == ranges[0]) or np.any(new_x == ranges[1]),
                     0, new_v)
    return new_x, new_v, new_l_best, new_l_best_score


def run(opt_func, adj_matrix,
        n=10, dim=2,
        epochs=10, c0=1,
        c1=0.5, c2=0.5, c3=0.5,
        ub=[], lb=[], constraints=[]):
    lb, ub = np.array(lb), np.array(ub)
    x = np.random.uniform(low=lb, high=ub, size=(n, dim))
    assert np.all(ub > lb)
    v_range = (ub - lb)/2
    v = np.random.uniform(low=-v_range, high=v_range, size=(n, dim))

    l_best = np.zeros(x.shape)
    l_best_score = np.inf * np.ones(x.shape[0])

    ranges = (lb, ub)

    g_best_log = []

    for i in range(epochs):
        x, v, l_best, l_best_score = epoch(x, v, l_best,
                                           l_best_score, adj_matrix,
                                           c0, c1, c2, c3, opt_func,
                                           ranges, constraints)
        g_best_log.append(get_global_best(l_best, l_best_score)[1])
        # if (i % 500) == 0:
        # print(i)
    return (l_best, l_best_score,
            *get_global_best(l_best, l_best_score), g_best_log)
