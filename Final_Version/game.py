import numpy as np
import cv2
from MCTS import *
import matplotlib.pyplot as plt


class Game(object):
    def __init__(self, net=None, eval_net=None):
        self.size = 13
        self.board = Board(self.size)
        self.grid_size = 35
        self.next_player = 1
        self.moves = [(-1, -1)]
        self.net = net
        self.eval_net = eval_net

    # def add_move(self, board):
    #     self.boards.append(board)
    #     self.next_player = 3 - self.next_player
    #
    # def move(self, x, y):
    #     board_clone = self.boards[-1].board.copy()
    #     if board_clone[x][y] != 0:
    #         return
    #     new_board = Board(board=board_clone)
    #     new_board.do_move(x, y, self.next_player)
    #     self.moves.append((x, y))
    #     self.add_move(new_board)
    # 自对弈 收集数据
    def start_self_play(self):
        datas, node = [], Node(None, None)
        mc = MCTS(self.net)
        move_count = 0
        states, mcts_probs, cur_players = [], [], []
        self.init_board()
        while True:
            if move_count < 8:
                action, next_node, pi = mc.search(self.board, node, temp=1)
            else:
                action, next_node, pi = mc.search(self.board, node)
            states.append(self.board.get_state())
            mcts_probs.append(pi)
            cur_players.append(self.board.cur_player)
            x, y = self.board.location_to_move(action)
            self.board.do_move(x, y)
            next_node.parent = None
            node = next_node
            move_count += 1
            end, winner = self.board.game_end()
            if end:
                winner_score = np.zeros(len(cur_players))
                if winner != -1:
                    winner_score[np.array(cur_players) == winner] = 1.0
                    winner_score[np.array(cur_players) != winner] = -1.0
                return winner, zip(states, mcts_probs, winner_score)

    def init_board(self):
        self.board = Board(self.size)

    # 评估算法 将现在训练的神经网络和历史最好的神经网络进行对弈 如果胜率高达80%就将当前网络替换成最佳网络继续训练
    def evaluate(self):
        result = [0, 0, 0]
        for i in range(10):
            self.init_board()
            if i % 2 == 0:
                players = {
                    'WHITE': (MCTS(self.net), 'cur_net'),
                    'BLACK': (MCTS(self.eval_net), 'best_net')
                }
            else:
                players = {
                    'WHITE': (MCTS(self.eval_net), 'best_net'),
                    'BLACK': (MCTS(self.net), 'cur_net')
                }
            node = Node(None, None)
            while True:
                cur_player = 'BLACK' if self.board.cur_player == 1 else 'WHITE'
                action, next_node, _ = players[cur_player][0].search(self.board, node)
                x, y = self.board.location_to_move(action)
                self.board.do_move(x, y)

                end, winner = self.board.game_end()
                if end:
                    plt.imshow(self.board.show_board())
                    plt.show()
                    c_player = 'BLACK' if winner == 1 else 'WHITE'
                    if winner == -1:
                        result[0] += 1
                    elif players[c_player][1] == 'cur_net':
                        result[1] += 1
                    else:
                        result[2] += 1
                    break
                next_node.parent = None
                node = next_node
        print("win: {}, lose: {}, tie:{}".format(
            result[1], result[2], result[0]))
        return result

    def bind_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xx, yy = int(round(float(x) / self.grid_size)) - 1, int(round(float(y) / self.grid_size)) - 1
            self.board.do_move(yy, xx)
        # if event == cv2.EVENT_RBUTTONDOWN:
        #     self.roll_back()

    def get_cur_img(self):
        return self.board.show_board(self.grid_size)


class Board(object):
    def __init__(self, size=13, board=None):
        self.size = size
        if board is None:
            self.board = np.zeros((size, size), dtype=np.uint8)
        else:
            self.board = board
        self.invalid_states = []
        self.valid_states = list(range(self.size ** 2))
        self.last_move = (-1, -1)
        self.cur_player = 1
        self.next_player = 1

    def show_board(self, grid_size=35, last_move=None):
        last_move = self.last_move
        canvas = np.ones(((self.size + 1) * grid_size, (self.size + 1) * grid_size, 3), dtype=np.uint8) * 100
        # 画横线
        for i in range(1, self.size + 1):
            cv2.line(canvas, (grid_size, grid_size * i), (grid_size * self.size, grid_size * i), color=(0, 0, 0),
                     thickness=2)
        # 画竖线
        for i in range(1, self.size + 1):
            cv2.line(canvas, (i * grid_size, grid_size), (grid_size * i, grid_size * self.size), color=(0, 0, 0),
                     thickness=2)
        for x in range(2):
            for y in range(2):
                cv2.circle(canvas, (4 * grid_size + x * 6 * grid_size, 4 * grid_size + y * 6 * grid_size),
                           int(grid_size / 8), color=(0, 0, 0), thickness=2)
        cv2.circle(canvas, (7 * grid_size, 7 * grid_size), int(grid_size / 8), color=(0, 0, 0), thickness=2)
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == 1:
                    cv2.circle(canvas, ((y + 1) * grid_size, (x + 1) * grid_size), int(grid_size / 2.2),
                               color=(0, 0, 0), thickness=-1)
                if self.board[x][y] == 2:
                    cv2.circle(canvas, ((y + 1) * grid_size, (x + 1) * grid_size), int(grid_size / 2.2),
                               color=(200, 200, 200), thickness=-1)
        if last_move is not None and 0 <= last_move[0] <= self.size and 0 <= last_move[1] <= self.size:
            x, y = last_move[0], last_move[1]
            if self.board[x][y] == 1:
                cv2.circle(canvas, ((y + 1) * grid_size, (x + 1) * grid_size), int(grid_size / 3),
                           color=(200, 200, 200), thickness=2)
            if self.board[x][y] == 2:
                cv2.circle(canvas, ((y + 1) * grid_size, (x + 1) * grid_size), int(grid_size / 3), color=(0, 0, 0),
                           thickness=2)

        return canvas

    def get_state(self):
        cur_states = np.zeros((4, self.size, self.size))
        cur_states[0][np.where(self.board == 1)] = 1
        cur_states[1][np.where(self.board == 2)] = 1
        cur_states[2][self.last_move[0]][self.last_move[1]] = 1

        cur_states[3] = np.full((self.size, self.size), 2 - self.cur_player)
        return cur_states

    def move_to_location(self, x, y):
        location = y * self.size + x
        return location

    def location_to_move(self, action):
        x = action % self.size
        y = action // self.size
        return x, y

    def clone(self):
        c_board = Board(self.size)
        c_board.invalid_states = self.invalid_states.copy()
        c_board.valid_states = self.valid_states.copy()
        c_board.board = self.board.copy()
        return c_board

    def do_move(self, x, y):
        if self.board[x, y] != 0:
            return
        self.board[x][y] = self.next_player
        self.cur_player = self.next_player
        self.last_move = (x, y)
        self.next_player = 3 - self.next_player
        self.invalid_states.append(self.move_to_location(x, y))
        self.valid_states.remove(self.move_to_location(x, y))

    def _get_piece(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.board[x][y]
        return 0

    def has_winner(self):
        move_color = self.cur_player
        x, y = self.last_move
        for i in range(x - 4, x + 5):
            if self._get_piece(i, y) == \
                    self._get_piece(i + 1, y) == \
                    self._get_piece(i + 2, y) == \
                    self._get_piece(i + 3, y) == \
                    self._get_piece(i + 4, y) == move_color:
                return True, move_color
        for j in range(y - 4, y + 5):
            if self._get_piece(x, j) == \
                    self._get_piece(x, j + 1) == \
                    self._get_piece(x, j + 2) == \
                    self._get_piece(x, j + 3) == \
                    self._get_piece(x, j + 4) == move_color:
                return True, move_color
        j = y - 4
        for i in range(x - 4, x + 5):
            if self._get_piece(i, j) == \
                    self._get_piece(i + 1, j + 1) == \
                    self._get_piece(i + 2, j + 2) == \
                    self._get_piece(i + 3, j + 3) == \
                    self._get_piece(i + 4, j + 4) == move_color:
                return True, move_color
            j += 1

        i = x + 4
        for j in range(y - 4, y + 5):
            if self._get_piece(i, j) == \
                    self._get_piece(i - 1, j + 1) == \
                    self._get_piece(i - 2, j + 2) == \
                    self._get_piece(i - 3, j + 3) == \
                    self._get_piece(i - 4, j + 4) == move_color:
                return True, move_color
            i -= 1
        return False, -1

    def game_end(self):
        win, winner = self.has_winner()
        if win:
            return True, winner
        elif not len(self.valid_states):
            return True, -1
        return False, -1


if __name__ == '__main__':
    net = Net()
    eval_net = Net()
    game = Game(net, eval_net)
    board_img = game.get_cur_img()
    cv2.imshow('board_img', board_img)
    cv2.setMouseCallback('board_img', game.bind_click)
    while True:
        board_img = game.get_cur_img()
        cv2.imshow('board_img', board_img)
        end, winner = game.board.game_end()
        if end:
            if winner == 1:
                cv2.putText(board_img, "Black PLayer Win!",
                            (game.grid_size * game.size // 6, game.grid_size * game.size // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color=(255, 255, 255), thickness=2)
                cv2.imshow('board_img', board_img)
                cv2.waitKey(0)
                break
            elif winner == 2:
                cv2.putText(board_img, "White Player Win!",
                            (game.grid_size * game.size // 6, game.grid_size * game.size // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color=(255, 255, 255), thickness=2)
                cv2.imshow('board_img', board_img)
                cv2.waitKey(0)
                break
            else:
                cv2.putText(board_img, "Draw!", (game.grid_size * game.size // 6, game.grid_size * game.size // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color=(255, 255, 255), thickness=2)
                cv2.imshow('board_img', board_img)
                cv2.waitKey(0)
                break
        cv2.waitKey(33)
