import argparse
import heapq
from itertools import permutations
from numba import cuda, int8
import numpy as np
from queue import SimpleQueue

BLOCKS = 200

def get_squares(board):
    """
    Returns a reshaped array where board[0] is the first 3x3 square, board[1]
    is the second, and so on
    """
    return board.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(9, 3, 3)


@cuda.jit(device=True)
def empty_squares(board):
    return np.sum(board.mask)


@cuda.jit
def solve(boards):
    board = boards[cuda.blockIdx.x]

    c = 0
    h = [(empty_squares(board), c, board)]

    # Boards with fewer empty squares are searched first
    while h != []:
        e, _, b = heapq.heappop(h)
        if e == 0:
            # Board is solved
            print(b)
            break

        it = np.nditer(b.mask, flags=['multi_index'])
        # Iterate through every square on board, skip if it is already filled
        for x in it:
            if x == False:
                continue

            i, j = it.multi_index
            s = 3 * (i // 3) + j // 3
            # Get row, column, and 3x3 square the current square is in
            row = b.data[i]
            col = b.data[:, j]
            square = b.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(9, 3, 3)[s]
            # square = get_squares(b.data)[s]
            guesses = np.setdiff1d(np.arange(1,10), row)
            guesses = np.setdiff1d(guesses, col)
            guesses = np.setdiff1d(guesses, square)
            for g in guesses:
                # Add a new board copy to the queue for every possible guess
                b[i][j] = g
                c += 1
                n = b.copy()
                heapq.heappush(h, (empty_squares(n), c, n))

            break


def bfs_expand(b):
    expansions = []
    p = set(range(1, 10))
    with np.nditer(b.mask, flags=['multi_index']) as it:
        for x in it:
            if x == False:
                continue

            i, j = it.multi_index
            s = 3 * (i // 3) + j // 3
            # Get row, column, and 3x3 square the current square is in
            row = b.data[i]
            col = b.data[:, j]
            square = get_squares(b.data)[s]
            guesses = p - set(row) - set(col) - set(square.flatten())
            for g in guesses:
                # Add a new board copy to the queue for every possible guess
                b[i][j] = g
                n = b.copy()
                expansions.append(n)

            break

    return expansions


parser = argparse.ArgumentParser(description="Solve a sudoku puzzle")
parser.add_argument('filename', help="""a txt file with an unfinished sudoku
puzzle inside. All numbers should be delimited with spaces and all empty squares
should be marked with -""")
args = parser.parse_args()

board = np.genfromtxt(args.filename, dtype=int, delimiter=' ',
        missing_values='-', usemask=True)

c = 0
bfs_queue = [(0, c, board)]
while bfs_queue != []:
    p, _, b = heapq.heappop(bfs_queue)
    # p is the number of squares that will be filled in from the start
    # Might want to make it dependent on how empty the board is to start
    if p == 9:
        break

    for i in bfs_expand(b):
        c += 1
        heapq.heappush(bfs_queue, (p+1, c, i))

new_boards = np.stack([b for (_,_,b) in bfs_queue])
solve[BLOCKS, 1](new_boards)
