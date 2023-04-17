import copy
import itertools
import random
import numpy as np

AGENTS_PIECES = {"player_0": 1, "player_1": 2}
GOOD_POSITION_VALUE = 10
WIN_POSITION_VALUE = 1000
        
class Connect4:
    ''' Some methods assume that the board has a 6x7 size'''

    def __init__(self, nb_rows=6, nb_cols=7, board: np.ndarray = None):
        self.rows = nb_rows
        self.cols = nb_cols
        self.board = board if board is not None else np.zeros((nb_rows, nb_cols))
        self.game_over = False
        self.turn = 0
        self.winner = None

    def play_move(self, col, piece):
        row = max(np.where(self.board[:, col] == 0)[0])
        self.board[row, col] = piece

    def get_available_col(self):
        return np.where(self.board[:, self.cols-1] == 0)[0]
    
    def move_is_valid(self, col):
        return self.board[0, col] == 0

    def check_wins(self, piece):
        win_position = str(piece)*4
        # Check horizontal locations for win
        for c, r in itertools.product(range(self.cols-3), range(self.rows)):
            if self.board[r][c] == piece and self.board[r][c+1] == piece and self.board[r][c+2] == piece and self.board[r][c+3] == piece:
                return True

        # Check vertical locations for win
        for c, r in itertools.product(range(self.cols), range(self.rows-3)):
            if self.board[r][c] == piece and self.board[r+1][c] == piece and self.board[r+2][c] == piece and self.board[r+3][c] == piece:
                return True

        # Check positively sloped diaganols
        for c, r in itertools.product(range(self.cols-3), range(self.rows-3)):
            if self.board[r][c] == piece and self.board[r+1][c+1] == piece and self.board[r+2][c+2] == piece and self.board[r+3][c+3] == piece:
                return True

        # Check negatively sloped diaganols
        for c, r in itertools.product(range(self.cols-3), range(3, self.rows)):
            if self.board[r][c] == piece and self.board[r-1][c+1] == piece and self.board[r-2][c+2] == piece and self.board[r-3][c+3] == piece:
                return True
        
    def play(self):
        '''Allows you to play as Player 2 against Minimax'''

        while not self.game_over:
            if self.turn == 0:
                col = best_move(self, self.depth)
                self.play_move(col, 1)

                if self.check_wins(1):
                    self.game_over = True
                    break
                self.turn = 1 - self.turn
            if self.turn == 1:
                available_cols = self.get_available_col()
                col = random.choice(available_cols)
                self.play_move(col, 2)

                if self.check_wins(1):
                    self.game_over = True
                    break
                self.turn = 1 - self.turn
                

class AlphaBetaAgent:
    def __init__(self, action_space, observation_space, depth: int = 3):
        self.action_space = action_space
        self.observation_space = observation_space
        self.depth = depth
    
    def get_action(self, observation):
        connect_4_copy = Connect4(board=get_board(observation))
        return best_move(connect_4_copy, self.depth)
    
    def update(self, obs, action, reward, terminated, next_obs):
        pass

def get_board(observation, agents_pieces: dict = AGENTS_PIECES):
    return sum(
        observation["observation"].T[idx]* agent_value 
        for idx, (_, agent_value) in enumerate(AGENTS_PIECES.items())
    ).T

def static_value(connect_four: Connect4): 
    '''Evalue statiquement un plateau de jeu'''
    S = 0
    signe = {1: 1, 2: -1}
    is_good = {
    1: {'1110', '1101', '1011', '0111'},
    2: {'2220', '2202', '2022', '0222'}
    }
    is_win = {1: '1111', 2: '2222'}

    for p in [1, 2]:

        # Checks rows
        for r in range(connect_four.rows):
            row = ''.join(str(int(i)) for i in connect_four.board[r, :])
            S += sum(row.count(good_position)*signe[p]*GOOD_POSITION_VALUE for good_position in is_good[p])
            S += row.count(is_win[p])*signe[p]*WIN_POSITION_VALUE

        # Checks columns
        for c in range(connect_four.cols):
            col = ''.join(str(int(i)) for i in connect_four.board[:, c])
            S += sum(col.count(good_position)*signe[p]*GOOD_POSITION_VALUE for good_position in is_good[p])
            S += col.count(is_win[p])*signe[p]*WIN_POSITION_VALUE

        # Checks positively sloped diagonals
        for c, r in itertools.product(range(connect_four.cols-3), range(connect_four.rows-3)):
            diag = ''.join(str(int(connect_four.board[r+i][c+i])) for i in range(4))
            S += sum(diag.count(good_position)*signe[p]*GOOD_POSITION_VALUE for good_position in is_good[p])
            S += diag.count(is_win[p])*signe[p]*WIN_POSITION_VALUE
    
        # Checks negatively sloped diagonals
        for c, r in itertools.product(range(connect_four.cols-3), range(3, connect_four.rows)):
            diag = ''.join(str(int(connect_four.board[r-i][c+i])) for i in range(4))
            S += sum(diag.count(good_position)*signe[p]*GOOD_POSITION_VALUE for good_position in is_good[p])
            S += diag.count(is_win[p])*signe[p]*WIN_POSITION_VALUE

    return S

def get_children(connect_four: Connect4):
    '''Returns a dictionnary {move:obtained_child}'''
    dict_children = {}
    for col in range(connect_four.cols):
        if connect_four.move_is_valid(col):
            child = copy.deepcopy(connect_four)
            children_piece = connect_four.turn + 1
            child.play_move(col, children_piece)
            child.turn = 1 - connect_four.turn
            if child.check_wins(children_piece) or not moves(child):
                child.game_over = True
            dict_children[col] = child
    if not dict_children and not connect_four.game_over:
        connect_four.game_over = True
    return dict_children


def moves(connect_four: Connect4):
    '''Returns an array of available moves from current state'''
    L = {col for col in range(connect_four.cols) if connect_four.move_is_valid(col)}
    if not L:
        connect_four.game_over = True
    return L


def minimax(connect_four: Connect4, depth: int, alpha: float, beta: float, maximizingPlayer: bool = None):
    '''Returns best possible score obtained from the root'''
    if maximizingPlayer is None:
        maximizingPlayer = not bool(connect_four.turn)

    if depth == 0 or connect_four.game_over:
        return static_value(connect_four)

    elif maximizingPlayer:
        maxEval = float('-inf')
        for child in get_children(connect_four).values():
            score = minimax(child, depth - 1, alpha, beta, False)
            maxEval = max(maxEval, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return maxEval

    else:
        minEval = float('inf')
        for child in get_children(connect_four).values():
            score = minimax(child, depth - 1, alpha, beta, True)
            minEval = min(minEval, score)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return minEval


def best_move(connect_four: Connect4, depth: int = 3):
    '''Returns best possible move from current state, based on alphabeta prunning algorithm'''
    # Player 1 has here turn 0 and plays as MAX
    if connect_four.game_over:
        return
    scores = [float('inf')]*connect_four.cols if connect_four.turn else [float('-inf')]*connect_four.cols

    children = get_children(connect_four)
    for playable_move in children.keys():
        scores[playable_move] = minimax(
            children[playable_move],
            depth=depth, 
            alpha=-float('inf'),
            beta=float('inf'), 
            maximizingPlayer=not children[playable_move].turn
        )

    hyp_move = scores.index(max(scores))
    # Unplayable moves get NaN score
    for col in range(connect_four.cols):
        if col not in children.keys():
            scores[col] = float('nan')

    if connect_four.turn != 0:
        return scores.index(min(scores))
            # Children score is then maximized
    if abs(np.nanmean(scores)) != float('inf'):
        if max(scores) == int(np.nanmean(scores)):
            return (
                3
                if not max(scores) and 3 in children
                else int(random.choice(list(children.keys())))
            )
        else:
            return hyp_move
