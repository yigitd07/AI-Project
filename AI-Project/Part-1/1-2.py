#Given a Sudoku puzzle of 6×6 grid, implement a CSP algorithm with backtracking to solve an example of the puzzle.


def print_board(board):
    for i in range(6):
        if i == 2 or i == 4:
            print("-" * 13)
        for j in range(6):
            if j == 3:
                print("|", end=" ")
            print(board[i][j], end=" ")
        print()

def find_empty(board):
    for i in range(6):
        for j in range(6):
            if board[i][j] == 0:
                return (i, j)  
    return None

def is_valid(board, num, pos):
    row, col = pos

    
    for j in range(6):
        if board[row][j] == num and j != col:
            return False

    
    for i in range(6):
        if board[i][col] == num and i != row:
            return False

    
    box_row = (row // 2) * 2
    box_col = (col // 3) * 3
    for i in range(box_row, box_row + 2):
        for j in range(box_col, box_col + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True

def backtracking(board):
    find = find_empty(board)
    if not find:
        return True  
    else:
        row, col = find

    for num in range(1, 7):
        if is_valid(board, num, (row, col)):
            board[row][col] = num

            if backtracking(board):
                return True

            board[row][col] = 0  

    return False

if __name__ == "__main__":
    sudoku_board = [
        [0, 0, 0, 0, 6, 0],
        [0, 0, 0, 0, 0, 3],
        [0, 0, 0, 1, 0, 0],
        [4, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [0, 3, 0, 0, 0, 0]
    ]

    print("Initial Sudoku:")
    print_board(sudoku_board)

    if backtracking(sudoku_board):
        print("\nSolved Sudoku:")
        print_board(sudoku_board)
    else:
        print("No solution exists.")
