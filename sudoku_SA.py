import argparse
from math import exp, log
from numba import cuda, boolean, int16, int64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
from random import shuffle, getrandbits

threads = 512

@cuda.jit(device=True)
def temperature(x, t_max):
    alpha = 0.99985
    return t_max * alpha ** x


@cuda.jit(device=True)
def get_square_dims(i):
    return ((3 * (i // 3), 3 * (i // 3) + 3), (3 * (i % 3), 3 * (i % 3) + 3))


@cuda.jit(device=True)
def random_fill_cuda(board, mask, rng_states, tx):
    for i in range(0, 9):
        dims = get_square_dims(i)
        sb = board[dims[0][0]:dims[0][1], dims[1][0]:dims[1][1]]
        sbm = mask[dims[0][0]:dims[0][1], dims[1][0]:dims[1][1]]

        present = cuda.local.array(10, boolean)
        for j in range(0, 10):
            present[j] = False

        for j in range(0, 3):
            for k in range(0, 3):
                if sbm[j][k] == 0:
                    present[sb[j][k]] = True

        for j in range(0, 3):
            for k in range(0, 3):
                if sbm[j][k] == 0:
                    continue
                r = 1 + int(xoroshiro128p_uniform_float32(rng_states, tx) * 9)
                while present[r]:
                    r = 1 + int(xoroshiro128p_uniform_float32(rng_states, tx) * 9)
                sb[j][k] = r
                present[r] = True


@cuda.jit(device=True)
def neighbor(board, mask, rng_states, tx):
    # FIXME: Something here is making the program hang once a solution is found
    # in some instances
    i = int(xoroshiro128p_uniform_float32(rng_states, tx) * 9)

    # This is ugly but I couldn't get the reshape method to work
    dims = get_square_dims(i)
    sb = board[dims[0][0]:dims[0][1], dims[1][0]:dims[1][1]]
    sbm = mask[dims[0][0]:dims[0][1], dims[1][0]:dims[1][1]]

    # This is kinda ugly but it was easy to write
    while True:
        r = int(xoroshiro128p_uniform_float32(rng_states, tx) * 9)
        j1 = r // 3
        k1 = r % 3
        if sbm[j1][k1] == True:
            break

    while True:
        r = int(xoroshiro128p_uniform_float32(rng_states, tx) * 9)
        j2 = r // 3
        k2 = r % 3
        if sbm[j2][k2] == True and not (j1 == j2 and k1 == k2):
            break

    sb[j1][k1], sb[j2][k2] = sb[j2][k2], sb[j1][k1]


@cuda.jit(device=True)
def P(e_old, e_new, t):
    if e_new < e_old:
        return 1
    if t < 0.001:
        return 0
    p = (e_new - e_old) / t
    return exp(-p)


@cuda.jit(device=True)
def E(b):
    e = 0
    present = cuda.local.array(10, boolean)
    for row in b:
        for i in range(0, 10):
            present[i] = False
        for i in row:
            present[i] = True
        for i in present:
            if i == True:
                e -= 1
    for col in b.T:
        for i in range(0, 10):
            present[i] = False
        for i in col:
            present[i] = True
        for i in present:
            if i == True:
                e -= 1
    return e


@cuda.jit(device=True)
def init_t_max(board, mask, rng_states, tx):
    s = 30
    samples = cuda.local.array(s, int16)
    mean = 0

    b = cuda.local.array((9, 9), int16)
    for i in range(0, s):
        board_copy(board, b)
        neighbor(b, mask, rng_states, tx)

        samples[i] = E(b)
        mean += samples[i]

    mean /= s

    var = 0
    for i in range(0, s):
        var += (samples[i] - mean) ** 2
    var /= s

    return var ** (1 / 2)


@cuda.jit(device=True)
def board_copy(a, b):
    for i in range(0, 9):
        for j in range(0, 9):
            b[i][j] = a[i][j]


@cuda.jit
def kernel(init_board, mask, rng_states, output):
    tx = cuda.threadIdx.x

    board = cuda.local.array((9, 9), int16)
    board_copy(init_board, board)

    output_access = cuda.shared.array(1, int64)
    cuda.atomic.compare_and_swap(output_access, 0, 1)
    cuda.syncthreads()

    b = cuda.local.array((9, 9), int16)
    imax = 100000
    while cuda.atomic.compare_and_swap(output_access, 1, 1):
        random_fill_cuda(board, mask, rng_states, tx)
        t_max = init_t_max(board, mask, rng_states, tx)
        for i in range(0, imax):
            t = temperature(i, t_max)

            board_copy(board, b)
            neighbor(b, mask, rng_states, tx)

            old = E(board)
            new = E(b)
            p = P(old, new, t)
            if p >= xoroshiro128p_uniform_float32(rng_states, tx):
                board_copy(b, board)

            if new == -162 and cuda.atomic.compare_and_swap(output_access, 1, 0):
                board_copy(board, output)

            if 0 == cuda.atomic.compare_and_swap(output_access, 0, 0):
                break


parser = argparse.ArgumentParser(description="Solve a sudoku puzzle")
parser.add_argument('filename', help="""a txt file with an unfinished sudoku
puzzle inside. All numbers should be delimited with spaces and all empty squares
should be marked with -""")
args = parser.parse_args()

board = np.genfromtxt(args.filename, dtype=int, delimiter=' ',
        missing_values='-', usemask=True)

rng_states = create_xoroshiro128p_states(threads, seed=12, subsequence_start=getrandbits(16))

output = np.zeros(board.shape)
cuda_board = cuda.to_device(board.data)
cuda_mask = cuda.to_device(board.mask)

kernel[1,threads](cuda_board, cuda_mask, rng_states, output)

print(output)
