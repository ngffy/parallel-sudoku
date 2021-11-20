import argparse
from math import exp
from numba import cuda, boolean, int64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
from random import shuffle

threads = 512

def get_squares(board):
    """
    Returns a copy of the array where board[0] is the first 3x3 square,
    board[1] is the second, and so on
    """
    return board.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(9, 3, 3)


def get_board(squares):
    """
    Inverse of get_squares
    """
    return squares.reshape(3,3,3,3).swapaxes(1,2).reshape(9,9)


def empty_squares(board, mask):
    return np.sum(board.mask)


def random_fill(board):
    subboards = get_squares(board)

    for sb in subboards:
        unused = list(set(range(1, 10)) - set(sb.data.flatten()))
        shuffle(unused)
        with np.nditer(sb.mask, flags=['multi_index']) as it:
            for x in it:
                if x == False:
                    continue
                i, j = it.multi_index
                sb.data[i][j] = unused.pop()

    return get_board(subboards)


@cuda.jit(device=True)
def temperature(x):
    return x


@cuda.jit(device=True)
def neighbor(board, mask, rng_states, tx):
    i = int(xoroshiro128p_uniform_float32(rng_states, tx) * 9)

    # This is ugly but I couldn't get the reshape method to work
    sb = board[3* (i // 3):3*(i // 3) + 3,3*(i%3):3*(i%3)+3]
    sbm = mask[3* (i // 3):3*(i // 3) + 3,3*(i%3):3*(i%3)+3]

    # This is kinda ugly but it was easy to write
    while True:
        j1 = int(xoroshiro128p_uniform_float32(rng_states, tx) * 3)
        k1 = int(xoroshiro128p_uniform_float32(rng_states, tx) * 3)
        if sbm[j1][k1] == True:
            break

    while True:
        j2 = int(xoroshiro128p_uniform_float32(rng_states, tx) * 3)
        k2 = int(xoroshiro128p_uniform_float32(rng_states, tx) * 3)
        if sbm[j2][k2] == True and not (j1 == j2 and k1 == k2):
            break

    # sb[j1][k1], sb[j2][k2] = sb[j2][k2], sb[j1][k1]
    tmp = sb[j1][k1]
    sb[j1][k1] = sb[j2][k2]
    sb[j2][k2] = tmp


@cuda.jit(device=True)
def P(e_old, e_new, t):
    if e_new < e_old:
        return 1
    return exp(-(e_new - e_old) / t)


@cuda.jit(device=True)
def E(b):
    e = 0
    for row in b:
        present = cuda.local.array(9, boolean)
        for i in row:
            present[i] = True
        for i in present:
            if i == True:
                e -= 1
    for col in b.T:
        present = cuda.local.array(9, boolean)
        for i in col:
            present[i] = True
        for i in present:
            if i == True:
                e -= 1
    return e


@cuda.jit
def kernel(boards, mask, rng_states, outputs):
    tx = cuda.threadIdx.x
    board = boards[tx]

    imax = 1000
    for i in range(0, imax):
        t = temperature(1 - (i + 1) / imax)

        b = cuda.local.array((9, 9), int64)
        for i in range(0, 9):
            for j in range(0, 9):
                b[i][j] = board[i][j]
        neighbor(b, mask, rng_states, tx)

        if P(E(board), E(b), t) >= xoroshiro128p_uniform_float32(rng_states, tx):
            for i in range(0, 9):
                for j in range(0, 9):
                    board[i][j] = b[i][j]


    for i in range(0, 9):
        for j in range(0, 9):
            outputs[tx][i][j] = board[i][j]


parser = argparse.ArgumentParser(description="Solve a sudoku puzzle")
parser.add_argument('filename', help="""a txt file with an unfinished sudoku
puzzle inside. All numbers should be delimited with spaces and all empty squares
should be marked with -""")
args = parser.parse_args()

board = np.genfromtxt(args.filename, dtype=int, delimiter=' ',
        missing_values='-', usemask=True)

# NOTE: probably need to change seed
rng_states = create_xoroshiro128p_states(threads, seed=12)

board = random_fill(board)
board_copies = np.tile(board.data, (threads, 1, 1))
outputs = np.zeros(board_copies.shape)
kernel[1,threads](board_copies, board.mask, rng_states, outputs)
print(outputs[np.any(outputs != 0, axis=(1,2))])
