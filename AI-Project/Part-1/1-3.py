#Implement a Monte Carlo Tree Search solver for a 3x3 Tic-Tac-Toe game.

import math
import random
import copy

class TicTacToe:
    def __init__(self):
        self.board = [[' ']*3 for _ in range(3)]
        self.current_player = 'X'

    def get_legal_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == ' ']

    def make_move(self, row, col):
        if self.board[row][col] != ' ':
            return False
        self.board[row][col] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    def winner(self):
        lines = self.board + [list(x) for x in zip(*self.board)]  # rows and columns
        lines.append([self.board[i][i] for i in range(3)])  # main diagonal
        lines.append([self.board[i][2-i] for i in range(3)])  # other diagonal
        for line in lines:
            if line == ['X'] * 3:
                return 'X'
            if line == ['O'] * 3:
                return 'O'
        if all(cell != ' ' for row in self.board for cell in row):
            return 'Draw'
        return None

    def clone(self):
        new_game = TicTacToe()
        new_game.board = copy.deepcopy(self.board)
        new_game.current_player = self.current_player
        return new_game

    def print_board(self):
        for row in self.board:
            print(' | '.join(row))
            print('-'*9)

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, c_param=1.4):
        choices = [
            (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices.index(max(choices))]

def mcts(root_state, iter_limit=1000):
    root = Node(root_state)

    for _ in range(iter_limit):
        node = root
        state = root_state.clone()

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            state.make_move(*node.move)

        # Expansion
        legal_moves = state.get_legal_moves()
        tried_moves = [child.move for child in node.children]
        for move in legal_moves:
            if move not in tried_moves:
                state.make_move(*move)
                child_node = Node(state.clone(), node, move)
                node.children.append(child_node)
                node = child_node
                break

        # Simulation
        rollout_state = state.clone()
        while rollout_state.winner() is None:
            move = random.choice(rollout_state.get_legal_moves())
            rollout_state.make_move(*move)

        # Backpropagation
        result = rollout_state.winner()
        while node is not None:
            node.visits += 1
            if result == 'Draw':
                node.wins += 0.5
            elif result == root_state.current_player:
                node.wins += 1
            node = node.parent

    return sorted(root.children, key=lambda c: c.visits)[-1].move

# Example run: MCTS vs Human
if __name__ == "__main__":
    game = TicTacToe()
    while game.winner() is None:
        game.print_board()
        if game.current_player == 'X':
            print("AI (X) is thinking...")
            move = mcts(game, iter_limit=1000)
        else:
            row = int(input("Enter row (0-2): "))
            col = int(input("Enter col (0-2): "))
            move = (row, col)
        game.make_move(*move)

    game.print_board()
    print("Winner:", game.winner())
