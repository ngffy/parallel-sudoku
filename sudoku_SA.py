import argparse
from numba import cuda
import numpy as np
from random import shuffle

threads = 1024

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


@cuda.jit
def kernel(board, mask, output):
    pass


parser = argparse.ArgumentParser(description="Solve a sudoku puzzle")
parser.add_argument('filename', help="""a txt file with an unfinished sudoku
puzzle inside. All numbers should be delimited with spaces and all empty squares
should be marked with -""")
args = parser.parse_args()

board = np.genfromtxt(args.filename, dtype=int, delimiter=' ',
        missing_values='-', usemask=True)

board = random_fill(board)
board_copies = np.tile(board, (threads, 1, 1))
