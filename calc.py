import numpy as np
import sys
from time import perf_counter as tpc


import teneva


from show import show_function_big
from show import show_function_small
from show import show_random_small
from show import show_random_small_hist
from utils import Log
from utils import folder_ensure


# Benchmarks with known global minimum:
FUNCTIONS_SMALL = [
    'Ackley',
    'Alpine',
    'Dixon',
    'Exponential',
    'Grienwank',
    'Michalewicz',
    'Qing',
    'Rastrigin',
    'Schaffer',
    'Schwefel',
]


# Benchmarks with known global minimum and with known explicit TT-cores:
FUNCTIONS_BIG = [
    'Exponential',
    'Grienwank',
    'Qing',
    'Rastrigin',
    'Schwefel',
]


def calc_function_big(d=100, n=2**10, k=100, mode='tt'):
    t_full = tpc()

    if mode == 'tt':
        log = Log('result/logs_calc/function_big.txt')
    else:
        log = Log('result/logs_calc/function_big_qtt.txt')

    mode_name = 'function_big' if mode == 'tt' else 'function_big_qtt'

    log(f'---> CALC | {mode_name} | d: {d:-3d} | n: {n:-8d} | k: {k:-4d}\n')

    data = {}

    for func in teneva.func_demo_all(d, names=FUNCTIONS_BIG):
        # Set the grid:
        func.set_grid(n, kind='cheb')

        # Build the TT-cores:
        func.cores()
        Y = func.Y
        r = teneva.erank(Y)

        # Find the value of the TT-tensor near argmin of the function:
        y_min_real = func.y_min
        x_min_real = func.x_min
        i_min_real = teneva.poi_to_ind(x_min_real,
            func.a, func.b, func.n, func.kind)
        y_min_appr = func.get_ind(i_min_real)

        # Find min/max values for TT-tensor by optima_tt:
        t = tpc()
        if mode == 'tt':
            i_min, y_min = teneva.optima_tt(Y, k)[:2]
        else:
            i_min, y_min = teneva.optima_qtt(Y, k)[:2]
        t = tpc() - t

        # Calculate the error:
        e_val = np.abs(y_min - y_min_appr)
        e_ind = np.max(np.abs(i_min - i_min_real))

        name = func.name
        data[name] = {'t': t, 'r': r, 'e_val': e_val, 'e_ind': e_ind}

        text = ''
        text += name + ' ' * max(0, 12-len(name)) +  ' | '
        text += f'r: {r:-4.1f} | '
        text += f't: {t:-7.3f} | '
        text += f'e_min: {e_val:-7.1e}'
        log(text)

    if mode == 'tt':
        fpath = 'result/data/function_big.npz'
    else:
        fpath = 'result/data/function_big_qtt.npz'
    np.savez_compressed(fpath, data=data, d=d, n=n, k=k)

    t_full = tpc() - t_full
    log(f'\n===> DONE | {mode_name} | Time: {t_full:-10.3f}\n')

    show_function_big(mode)


def calc_function_small(d=6, n=16, k=100):
    t_full = tpc()

    log = Log('result/logs_calc/function_small.txt')
    log(f'---> CALC | function_small | d: {d:-3d} | n: {n:-8d} | k: {k:-4d}\n')

    data = {}

    for func in teneva.func_demo_all(d, names=FUNCTIONS_SMALL):
        # Set the grid:
        func.set_grid(n, kind='cheb')

        # Build the TT-approximation by the TT-CROSS method:
        Y = teneva.tensor_rand(func.n, r=1)
        Y = teneva.cross(func.get_f_ind, Y, e=1.E-16, m=1.E+7, dr_max=2)
        Y = teneva.truncate(Y, e=1.E-16, r=12)
        r = teneva.erank(Y)

        # Generate full tensor and find its min/max values:
        Y_full = teneva.full(Y)
        y_min_real = np.min(Y_full)
        y_max_real = np.max(Y_full)

        # Find min/max values for TT-tensor by optima_tt:
        t = tpc()
        i_min, y_min, i_max, y_max = teneva.optima_tt(Y, k)
        t = tpc() - t

        # Calculate the errors:
        e_min = np.abs(y_min - y_min_real)
        e_max = np.abs(y_max - y_max_real)

        name = func.name
        data[name] = {'t': t, 'r': r, 'e_min': e_min, 'e_max': e_max}

        text = ''
        text += name + ' ' * max(0, 12-len(name)) +  ' | '
        text += f'r: {r:-4.1f} | '
        text += f't: {t:-7.3f} | '
        text += f'e_min: {e_min:-7.1e} | '
        text += f'e_max: {e_max:-7.1e}'
        log(text)

    np.savez_compressed('result/data/function_small.npz', data=data,
        d=d, n=n, k=k)

    t_full = tpc() - t_full
    log(f'\n===> DONE | function_small | Time: {t_full:-10.3f}\n')

    show_function_small()


def calc_random_small(d_=[4,6], n_=[5,20], r_=[1,5], k=100, rep=100):
    t_full = tpc()

    log = Log('result/logs_calc/random_small.txt')
    log(f'---> CALC | random_small | k: {k:-4d}\n')

    data = {}

    for d in range(d_[0], d_[1]+1):
        data[d] = {}

        for r in range(r_[0], r_[1]+1):
            n = [np.random.choice(range(n_[0], n_[1]+1)) for _ in range(d)]
            t = 0.
            e_min = []
            e_max = []

            for _ in range(1, rep+1):
                # Create random TT-tensor of shape n and rank r:
                Y = teneva.tensor_rand(n, r)

                # Generate full tensor and find its min/max values:
                Y_full = teneva.full(Y)
                i_min_real = np.unravel_index(np.argmin(Y_full), n)
                i_max_real = np.unravel_index(np.argmax(Y_full), n)
                y_min_real = Y_full[i_min_real]
                y_max_real = Y_full[i_max_real]

                # Find min/max values for TT-tensor by optima_tt:
                t_cur = tpc()
                i_min, y_min, i_max, y_max = teneva.optima_tt(Y, k)
                t += tpc() - t_cur

                # Calculate the errors:
                e_min.append(abs(y_min - y_min_real))
                e_max.append(abs(y_max - y_max_real))

            t /= rep
            e_min = np.max(e_min)
            e_max = np.max(e_max)

            data[d][r] = {'t': t, 'e_min': e_min, 'e_max': e_max}

            text = ''
            text += f'd: {d:-4d} | '
            text += f'r: {r:-3d} | '
            text += f't: {t:-8.3f} * {rep:3d} | '
            text += f'e_min: {e_min:-7.1e} | '
            text += f'e_max: {e_max:-7.1e}'
            log(text)

    np.savez_compressed('result/data/random_small.npz', data=data,
        d_=d_, n_=n_, r_=r_, k=k, rep=rep)

    t_full = tpc() - t_full
    log(f'\n===> DONE | random_small | Time: {t_full:-10.3f}\n')

    show_random_small()


def calc_random_small_hist(d=6, n=16, r=3, k_=[1, 10, 25], rep=10000):
    t_full = tpc()

    log = Log('result/logs_calc/random_small_hist.txt')
    log(f'---> CALC | random_small_hist |\n')

    data = {}

    for k in k_:
        t = 0.
        e_min = []
        e_max = []

        for _ in range(1, rep+1):
            # Create random TT-tensor of shape n and rank r:
            Y = teneva.tensor_rand([n]*d, r)

            # Generate full tensor and find its min/max values:
            Y_full = teneva.full(Y)
            i_max_real = np.unravel_index(np.argmax(np.abs(Y_full)), [n]*d)
            y_max_real = Y_full[i_max_real]

            # Find max value for TT-tensor by optima_tt:
            t_cur = tpc()
            i_max, y_max = teneva.optima_tt_max(Y, k)
            t += tpc() - t_cur

            # Calculate the error:
            e_max.append(abs(y_max / y_max_real))

        t /= rep

        data[k] = {'t': t, 'e_max': e_max}

        text = ''
        text += f'k: {k:-4d} | '
        text += f't: {t:-8.3f} * {rep:3d} | '
        text += f'e_max: {np.mean(e_max):-7.1e}'
        log(text)

    np.savez_compressed('result/data/random_small_hist.npz', data=data,
        d=d, n=n, r=r, k_=k_, rep=rep)

    t_full = tpc() - t_full
    log(f'\n===> DONE | random_small_hist | Time: {t_full:-10.3f}\n')

    show_random_small_hist()


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/data')
    folder_ensure('result/plot')
    folder_ensure('result/logs_calc')
    folder_ensure('result/logs_show')

    mode = sys.argv[1] if len(sys.argv) > 1 else None

    if mode == 'function_big':
        calc_function_big()
    elif mode == 'function_small':
        calc_function_small()
    elif mode == 'random_small':
        calc_random_small()
    elif mode == 'random_small_hist':
        calc_random_small_hist()
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
