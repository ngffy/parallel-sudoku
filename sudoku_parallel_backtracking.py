import argparse
import numpy as np

import math
from numba import cuda

BOARD_SIZE = 9
SUBBOARD_SIZE = 3


@cuda.jit
def backtrack(boards, num_boards, empty, num_empty, solved, output):
    ''' Function for each thread to solve a different board using backtracking algorithm

        boards: array of multiple boards, with each board stored one after the other (array of arrays)
        num_boards: number of boards in boards array (int)
        empty: array that stores indices of empty spaces (array of ints)
        num_empty: array that stores number of empty spaces in each board in boards (array of ints)
        solved = whether solution is found (bool)
        output = array of solved board (array of ints)
    '''
    index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    while not solved and index < num_boards:
        empty_index = 0

        current_board = boards + index * (BOARD_SIZE*BOARD_SIZE)
        current_empty = empty + index * (BOARD_SIZE*BOARD_SIZE)
        current_num_empty = num_empty[index]

        while empty_index >= 0 and empty_index < current_num_empty:
            current_board[current_empty[empty_index]] += 1

            if not valid_board(current_board, current_empty[empty_index]):
            #   Board is invalid, so backtrack
              if (current_board[current_empty[empty_index]] >= BOARD_SIZE):
                  current_board[current_empty[empty_index]] = 0
                  empty_index -= 1
            else:
              empty_index += 1

        # Solved board has been found
        if empty_index == current_num_empty:
            solved = True
            output = current_board.copy()

        index += cuda.gridDim.x * cuda.blockDim.x

    return output


@cuda.jit
def backtrack_kernel(num_blocks, num_threads, boards, num_boards, empty, num_empty, solved, output):
    '''
        num_blocks: number of blocks
        num_threads: number of threads per block
        boards: array of multiple boards, with each board stored one after the other (array of ints)
        num_boards: number of boards in boards array (int)
        empty: array that stores indices of empty spaces (array of ints)
        num_empty: array that stores number of empty spaces in each board in boards (array of ints)
        solved = whether solution is found (bool)
        output = array of solved board (array of ints)
    '''
    backtrack[num_blocks, num_threads](boards, num_boards, empty, num_empty, solved, output)


@cuda.jit
def breadth_first(old_boards, new_boards, total_boards, board_index, empty, empty_count):
    ''' old_boards:
        new_boards:
        total_boards: number of boards in total (int)
        board_index:
        empty:
        empty_count:
    '''
    index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    while index < total_boards:
        found = False

        start = index * BOARD_SIZE * BOARD_SIZE
        end = start + (BOARD_SIZE*BOARD_SIZE)
        for i in range(start, end):
            if found:
                break
            elif old_boards[i] == 0:
                found = True

                row = (i - start) / BOARD_SIZE
                col = (i - start) % BOARD_SIZE

                works = True
                # Check which numbers can be input at this row, col location
                for num in range(1, BOARD_SIZE+1):
                    # Check if can be placed in row
                    for c in range(BOARD_SIZE):
                        if old_boards[row * BOARD_SIZE + c + start] == num:
                            works = False
                    # Check if can be placed in col
                    for r in range(BOARD_SIZE):
                        if old_boards[r * BOARD_SIZE + col + start] == num:
                            works = False
                    # Check if can be placed in box
                    for r in range(SUBBOARD_SIZE * (row / SUBBOARD_SIZE), SUBBOARD_SIZE):
                        for c in range(SUBBOARD_SIZE * (col / SUBBOARD_SIZE), SUBBOARD_SIZE):
                            if old_boards[r * BOARD_SIZE + c + start] == num:
                                works = False

                    if works:
                        next_board_index = cuda.atomic.add(board_index, 1)
                        empty_index = 0
                        for r in range(BOARD_SIZE):
                            for c in range(BOARD_SIZE):
                                new_boards[next_board_index * BOARD_SIZE * BOARD_SIZE + r * BOARD_SIZE + c] = old_boards[start + r * BOARD_SIZE + c]
                                if old_boards[start + r * BOARD_SIZE + c] == 0 and r != row and c != col:
                                    empty[empty_index + BOARD_SIZE * BOARD_SIZE * next_board_index] = r * BOARD_SIZE + c
                                    empty_index += 1

                        empty_count[next_board_index] = empty_index
                        new_boards[next_board_index * BOARD_SIZE * BOARD_SIZE + row * BOARD_SIZE + col] = num

        index += cuda.blockDim.x * cuda.gridDim.x


def breadth_first_kernel(num_blocks, num_threads, old_boards, new_boards, total_boards, board_index, empty, empty_count):
    breadth_first[num_blocks, num_threads](old_boards, new_boards, total_boards, board_index, empty, empty_count)


def valid_board(board):
    # Check that rows are valid
    for r in range(BOARD_SIZE):
        seen = np.zeros(BOARD_SIZE)
        for c in range(BOARD_SIZE):
            num = board[r * BOARD_SIZE + c]

            if num != 0:
                if seen[num - 1]:
                    return False
                else:
                    seen[num - 1] = 1

    # Check that columns are valid
    for c in range(BOARD_SIZE):
        seen = np.zeros(BOARD_SIZE)
        for r in range(BOARD_SIZE):
            num = board[r * BOARD_SIZE + c]

            if num != 0:
                if seen[num - 1]:
                    return False
                else:
                    seen[num - 1] = 1

    # Check that subboards are valid
    for r_id in range(SUBBOARD_SIZE):
        for c_id in range(SUBBOARD_SIZE):
            seen = np.zeros(BOARD_SIZE)

            for r in range(SUBBOARD_SIZE):
                for c in range(SUBBOARD_SIZE):
                    row = r_id * SUBBOARD_SIZE + r
                    col = c_id * SUBBOARD_SIZE + c
                    num = board[row * BOARD_SIZE + col]

                    if num != 0:
                        if seen[num - 1]:
                            return False
                        else:
                            seen[num - 1] = 1

    return True


def valid_board_at_index(board, index):
    row = index / BOARD_SIZE
    col = index % BOARD_SIZE

    # Check entire board if index < 0
    if index < 0:
        return valid_board(board)
    
    # Check that number itself is valid
    if board[index] < 1 or board[index] > BOARD_SIZE:
        return False

    # Check row is valid
    seen = np.zeros(BOARD_SIZE)
    for c in range(BOARD_SIZE):
        num = board[row * BOARD_SIZE + c]

        if num != 0:
            if seen[num - 1]:
                return False
            else:
                seen[num - 1] = 1
    
    # Check column is valid
    seen = np.zeros(BOARD_SIZE)
    for r in range(BOARD_SIZE):
        num = board[r * BOARD_SIZE + col]

        if num != 0:
            if seen[num - 1]:
                return False
            else:
                seen[num - 1] = 1

    # Check subboard is valid
    r_id = row / SUBBOARD_SIZE
    c_id = col / SUBBOARD_SIZE
    seen = np.zeros(BOARD_SIZE)
    for r in range(SUBBOARD_SIZE):
        for c in range(SUBBOARD_SIZE):
            r_loc = r_id * SUBBOARD_SIZE + r
            c_loc = c_id * SUBBOARD_SIZE + c
            num = board[r_loc * BOARD_SIZE + c_loc]

            if num != 0:
                if seen[num - 1]:
                    return False
                else:
                    seen[num - 1] = 1

    return True


def main():
    parser = argparse.ArgumentParser(description="Solve a sudoku puzzle")
    parser.add_argument('filename', help="""a txt file with an unfinished sudoku
    puzzle inside. All numbers should be delimited with spaces and all empty squares
    should be marked with -""")
    args = parser.parse_args()

    board = np.genfromtxt(args.filename, dtype=int, delimiter=' ',
            missing_values='-', filling_values='0', usemask=False)

    board = board.flatten()


    threads = 32
    blocks = board.size + (threads - 1) // threads

    # Calculate the max number of boards that will need to be searched through
    max_boards_bfs = int(math.pow(2, 26))    # TODO not sure why this is it, just reused from someone's code

    # Initialize data to be input into kernel function
    new_boards = np.empty(max_boards_bfs, dtype=object)
    old_boards = np.empty(max_boards_bfs, dtype=object)
    old_boards[0] = board
    empty = np.empty(BOARD_SIZE*BOARD_SIZE, dtype=np.float32)
    empty_count = 0
    board_index = 0
    total_boards = 1

    # Breadth first search on the initial board
    breadth_first_kernel(blocks, threads, old_boards, new_boards, total_boards, board_index, empty, empty_count)

    count = 0
    num_iterations = 18     # Subject to change

    # Iterate through and perform more BFS to generate more boards
    # NOTE I also don't know why we should switch the new and old boards
    # Taken from https://github.com/vduan/parallel-sudoku-solver/blob/master/src/CudaSudoku.cc
    for i in range(num_iterations):
        if i % 2 == 0:
            breadth_first_kernel(blocks, threads, new_boards, old_boards, total_boards, board_index, empty, empty_count)
        else:
            breadth_first_kernel(blocks, threads, old_boards, new_boards, total_boards, board_index, empty, empty_count)

    if num_iterations % 2 == 1:
        new_boards = old_boards

    solved = False
    output = np.empty(BOARD_SIZE*BOARD_SIZE, dtype=np.float32)

    backtrack_kernel(blocks, threads, new_boards, count, empty, empty_count, solved, output)

    print(output)
main()