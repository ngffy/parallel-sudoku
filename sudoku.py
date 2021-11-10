import argparse
import heapq
import numpy as np

def get_squares(board):
    """
    Returns a reshaped array where board[0] is the first 3x3 square, board[1]
    is the second, and so on
    """
    return board.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(9, 3, 3)


def empty_squares(board):
    return np.sum(board.mask)


parser = argparse.ArgumentParser(description="Solve a sudoku puzzle")
parser.add_argument('filename', help="""a txt file with an unfinished sudoku
puzzle inside. All numbers should be delimited with spaces and all empty squares
should be marked with -""")
args = parser.parse_args()

board = np.genfromtxt(args.filename, dtype=int, delimiter=' ',
        missing_values='-', usemask=True)

p = set(range(1, 10))
c = 0
h = [(empty_squares(board), c, board)]

# Boards with fewer empty squares are searched first
while h != []:
    e, _, b = heapq.heappop(h)
    if e == 0:
        # Board is solved
        print(b)
        break

    # Iterate through every square on board, skip if it is already filled
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
                c += 1
                n = b.copy()
                heapq.heappush(h, (empty_squares(n), c, n))

            break
