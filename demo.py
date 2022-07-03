import numpy as np
import sys
from time import perf_counter as tpc


import teneva


from utils import Log
from utils import rand_for_opt


def demo_function_big():
    # TODO
    raise NotImplmentedError()


def demo_function_small(name='Brown', d=6, n=16, dy=0.5):
    t_full = tpc()

    log = Log()
    log('---> DEMO | function_small | \n')

    func = teneva.func_demo_all(d, dy=dy, names=[name])[0]

    # Set the uniform grid:
    func.set_grid(n, kind='uni')

    # Build the TT-approximation by the TT-CROSS method:
    Y = teneva.rand(func.n, r=1)
    Y = teneva.cross(func.get_f_ind, Y, m=1.E+5, dr_max=1, cache={})
    Y = teneva.truncate(Y, e=1.E-8)
    r_real = teneva.erank(Y)

    # Generate full tensor and find its min/max values:
    Y_full = teneva.full(Y)
    y_min_real = np.min(Y_full)
    y_max_real = np.max(Y_full)

    # Find min/max values for TT-tensor by optima_tt:
    log('\n... log from "optima_tt": \n')
    t = tpc()
    i_min, i_max = teneva.optima_tt(Y, log=True)
    y_min = teneva.get(Y, i_min)
    y_max = teneva.get(Y, i_max)
    t = tpc() - t
    log('\n... end log. \n')

    # Calculate the errors:
    e_min = np.abs((y_min - y_min_real) / y_min_real)
    e_max = np.abs((y_max - y_max_real) / y_max_real)

    name = func.name

    text = ''
    text += name + ' ' * max(0, 15-len(name)) +  ' | '
    text += f'd={d:-2d} | '
    text += f'r={r_real:-5.1f} | '
    text += f't={t:-7.2f} | '
    text += f'e_min={e_min:-7.1e} | '
    text += f'e_max={e_max:-7.1e}'
    log(text)

    t_full = tpc() - t_full
    log(f'\n===> DONE | function_small | Time: {t_full:-10.3f}\n')


def demo_random_big(d=20, n=10, r=2, s=1.):
    t_full = tpc()

    log = Log()
    log('---> DEMO | random_big | \n')

    # Create random TT-tensor of shape n with known i_min, y_min:
    i_min_real = [2] * d
    Y = rand_for_opt([n]*d, r, i_min_real, y_min=1., y_min_scale=s)
    y_min_real = teneva.get(Y, i_min_real)
    erank_real = teneva.erank(Y)

    # Find min value for TT-tensor by optima_tt:
    log('\n... log from "optima_tt": \n')
    t = tpc()
    i_min, i_max = teneva.optima_tt(Y, log=True)
    y_min = teneva.get(Y, i_min)
    t = tpc() - t
    log('\n... end log. \n')

    # Calculate the error:
    e = np.abs((y_min - y_min_real) / y_min_real)

    text = ''
    text += f'd={d:-2d} | '
    text += f's={s:-4.2f} | '
    text += f'r={erank_real:-5.1f} | '
    text += f't={t:-7.2f} | '
    text += f'e={e:-7.1e}'
    log(text)

    t_full = tpc() - t_full
    log(f'\n===> DONE | random_big | Time: {t_full:-10.3f}\n')


def demo_random_small(d=5, n_=[10, 10], r=5):
    t_full = tpc()

    log = Log()
    log('---> DEMO | random_small | \n')

    n = [np.random.choice(range(n_[0], n_[1]+1)) for _ in range(d)]

    # Create random TT-tensor of shape n and rank r:
    Y = teneva.rand(n, r)

    # Generate full tensor and find its min/max values:
    Y_full = teneva.full(Y)
    i_min_real = np.unravel_index(np.argmin(Y_full), n)
    i_max_real = np.unravel_index(np.argmax(Y_full), n)
    y_min_real = Y_full[i_min_real]
    y_max_real = Y_full[i_max_real]

    # Find min/max values for TT-tensor by optima_tt:
    log('\n... log from "optima_tt": \n')
    t = tpc()
    i_min, i_max = teneva.optima_tt(Y, log=True)
    y_min = teneva.get(Y, i_min)
    y_max = teneva.get(Y, i_max)
    t = tpc() - t
    log('\n... end log. \n')

    # Calculate the errors:
    e_min = np.abs((y_min - y_min_real) / y_min_real)
    e_max = np.abs((y_max - y_max_real) / y_max_real)

    text = ''
    text += f'd={d:-2d} | '
    text += f'r={r:-2d} | '
    text += f't={t:-7.2f} | '
    text += f'e_min={e_min:-7.1e} | '
    text += f'e_max={e_max:-7.1e}'
    log(text)

    t_full = tpc() - t_full
    log(f'\n===> DONE | random_small | Time: {t_full:-10.3f}\n')


def demo_random_stat():
    # TODO
    raise NotImplmentedError()


if __name__ == '__main__':
    np.random.seed(42)

    mode = sys.argv[1] if len(sys.argv) > 1 else None

    if mode == 'function_big':
        demo_function_big()
    elif mode == 'function_small':
        demo_function_small()
    elif mode == 'random_big':
        demo_random_big()
    elif mode == 'random_small':
        demo_random_small()
    elif mode == 'random_stat':
        demo_random_stat()
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
