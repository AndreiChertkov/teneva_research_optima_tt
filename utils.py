import numpy as np
import os


import teneva


class Log:
    def __init__(self, fpath=None):
        self.fpath = fpath
        self.is_new = True

    def __call__(self, text):
        print(text)
        if self.fpath:
            with open(self.fpath, 'w' if self.is_new else 'a') as f:
                f.write(text + '\n')
        self.is_new = False


def folder_ensure(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)


def rand_for_opt(n, r, i_min, y_min, y_min_scale=1., norm_scale=1.E+5):
    Y = teneva.rand(n, r)
    Y = teneva.mul(Y, Y)
    # Y = teneva.mul(Y, norm_scale/teneva.norm(Y))
    Y = teneva.add(Y, y_min * (1. + y_min_scale))
    y = teneva.get(Y, i_min)
    D = teneva.tensor_delta(n, i_min, y_min - y)
    Y = teneva.add(Y, D)
    # Y = teneva.truncate(Y, 1.E-16)
    return Y


def tex_auto_end():
    text = ''
    text += '\n'
    text += '% ' + 'AUTO GENERATED DATA | END' + '\n'
    text += '% ' + '=' * 40 + '\n'
    return text


def tex_auto_start(text_info=''):
    text = ''
    text += '% ' + '-' * 40 + '\n'
    text += '% ' + 'AUTO GENERATED DATA | START' + '\n'
    text += text_info
    text += '\n\n'
    return text


def tex_err_val(e, pref=' ', post=' '):
    if e > 0:
        e = f'{e:-8.2e}'
    else:
        e = '0' + ' '*7
    return pref + e + post


def tex_multirow(text, num):
    return '\\multirow{' + str(num) +'}{*}{' + str(text) + '}' + '\n'


def tex_row_end():
    return '\\\\' + '\n'


def tex_row_line():
    return '\\hline' + '\n\n'
