import numpy as np
import sys
from time import perf_counter as tpc


import teneva


from show import show_function_big
from show import show_function_small
from show import show_random_big
from show import show_random_small
from show import show_random_stat
from utils import Log
from utils import folder_ensure
from utils import rand_for_opt


def calc_function_big():
    # TODO
    raise NotImplmentedError()


def calc_function_small(d=6, n=16):
    t_full = tpc()

    log = Log('result/logs/function_small.txt')
    log('---> CALC | function_small | \n')

    data = {}

    for func in teneva.func_demo_all(d, dy=0.5):
        # Set the uniform grid:
        func.set_grid(n, kind='uni')

        # Build the TT-approximation by the TT-CROSS method:
        Y = teneva.rand(func.n, r=1)
        Y = teneva.cross(func.get_f_ind, Y, m=1.E+5, cache={})
        Y = teneva.truncate(Y, e=1.E-8)
        r = teneva.erank(Y)

        # Generate full tensor and find its min/max values:
        Y_full = teneva.full(Y)
        y_min_real = np.min(Y_full)
        y_max_real = np.max(Y_full)

        # Find min/max values for TT-tensor by optima_tt:
        t = tpc()
        i_min, i_max = teneva.optima_tt(Y)
        y_min = teneva.get(Y, i_min)
        y_max = teneva.get(Y, i_max)
        t = tpc() - t

        # Calculate the errors:
        e_min = np.abs((y_min - y_min_real) / y_min_real)
        e_max = np.abs((y_max - y_max_real) / y_max_real)

        name = func.name
        data[name] = {'t': t, 'r': r, 'e_min': e_min, 'e_max': e_max}

        text = ''
        text += name + ' ' * max(0, 15-len(name)) +  ' | '
        text += f'd={d:-2d} | '
        text += f'r={r:-5.1f} | '
        text += f't={t:-7.2f} | '
        text += f'e_min={e_min:-7.1e} | '
        text += f'e_max={e_max:-7.1e}'
        log(text)

    np.savez_compressed('result/data/function_small.npz', data=data,
        d=d, n=n)

    t_full = tpc() - t_full
    log(f'\n===> DONE | function_small | Time: {t_full:-10.3f}\n')

    show_function_small()


def calc_random_big(ds=[10,25,50], n=10, r=2, ss=[1., 0.1, 0.01], rep=1):
    t_full = tpc()

    log = Log('result/logs/random_big.txt')
    log('---> CALC | random_big | \n')

    data = {}

    for d in ds:
        data[d] = {}

        for s in ss:
            t = 0.
            e = []

            for k in range(1, rep+1):
                # Create random TT-tensor of shape n with known i_min, y_min:
                i_min_real = [2] * d
                Y = rand_for_opt([n]*d, r, i_min_real, y_min=1., y_min_scale=s)
                y_min_real = teneva.get(Y, i_min_real)
                erank_real = teneva.erank(Y)

                # Find min value for TT-tensor by optima_tt:
                t_cur = tpc()
                i_min, i_max = teneva.optima_tt(Y)
                y_min = teneva.get(Y, i_min)
                t += tpc() - t_cur

                # Calculate the error:
                e.append(np.abs((y_min - y_min_real) / y_min_real))

            t /= rep
            e_avg = np.mean(e)
            e_max = np.max(e)

            data[d][str(s)] = {'t': t, 'e_avg': e_avg, 'e_max': e_max, 's': s,
                'r': erank_real}

            text = ''
            text += f'd={d:-3d} | '
            text += f's={s:-5.3f} | '
            text += f'r={erank_real:-5.1f} | '
            text += f't={t:-7.2f} * {rep:2d} | '
            text += f'e avg={e_avg:-7.1e} | '
            text += f'e max={e_max:-7.1e}'
            log(text)

    np.savez_compressed('result/data/random_big.npz', data=data,
        ds=ds, n=n, r=r, ss=ss, rep=rep)

    t_full = tpc() - t_full
    log(f'\n===> DONE | random_big | Time: {t_full:-10.3f}\n')

    show_random_big()


def calc_random_small(d_=[4,6], n_=[5,20], r_=[1,5], rep=10):
    t_full = tpc()

    log = Log('result/logs/random_small.txt')
    log('---> CALC | random_small | \n')

    data = {}

    for d in range(d_[0], d_[1]+1):
        data[d] = {}

        for r in range(r_[0], r_[1]+1):
            n = [np.random.choice(range(n_[0], n_[1]+1)) for _ in range(d)]
            t = 0.
            e_min = []
            e_max = []

            for k in range(1, rep+1):
                # Create random TT-tensor of shape n and rank r:
                Y = teneva.rand(n, r)

                # Generate full tensor and find its min/max values:
                Y_full = teneva.full(Y)
                i_min_real = np.unravel_index(np.argmin(Y_full), n)
                i_max_real = np.unravel_index(np.argmax(Y_full), n)
                y_min_real = Y_full[i_min_real]
                y_max_real = Y_full[i_max_real]

                # Find min/max values for TT-tensor by optima_tt:
                t_cur = tpc()
                i_min, i_max = teneva.optima_tt(Y)
                y_min = teneva.get(Y, i_min)
                y_max = teneva.get(Y, i_max)
                t += tpc() - t_cur

                # Calculate the errors:
                e_min.append(np.abs((y_min - y_min_real) / y_min_real))
                e_max.append(np.abs((y_max - y_max_real) / y_max_real))

            t /= rep
            e_min = np.max(e_min)
            e_max = np.max(e_max)

            data[d][r] = {'t': t, 'e_min': e_min, 'e_max': e_max}

            text = ''
            text += f'd={d:-2d} | '
            text += f'r={r:-2d} | '
            text += f't={t:-7.2f} * {rep:2d} | '
            text += f'e_min={e_min:-7.1e} | '
            text += f'e_max={e_max:-7.1e}'
            log(text)

    np.savez_compressed('result/data/random_small.npz', data=data,
        d_=d_, n_=n_, r_=r_, rep=rep)

    t_full = tpc() - t_full
    log(f'\n===> DONE | random_small | Time: {t_full:-10.3f}\n')

    show_random_small()


def calc_random_stat():
    # TODO
    raise NotImplmentedError()


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/data')
    folder_ensure('result/logs')

    mode = sys.argv[1] if len(sys.argv) > 1 else None

    if mode == 'function_big':
        calc_function_big()
    elif mode == 'function_small':
        calc_function_small()
    elif mode == 'random_big':
        calc_random_big()
    elif mode == 'random_small':
        calc_random_small()
    elif mode == 'random_stat':
        calc_random_stat()
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
