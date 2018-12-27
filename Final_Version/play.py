import cv2
import game
import MCTS
import policy_value_net


def play():
    g = game.Game()
    net = policy_value_net.Net()
    mc = MCTS.MCTS(net=net)
    node = MCTS.Node(None, None)
    while True:
        img = g.get_cur_img()
        cv2.imshow('board_img', img)
        cv2.setMouseCallback('board_img', g.bind_click)
        cv2.waitKey(33)
        while True:
            before_len = len(g.board.valid_states)
            board_img = g.get_cur_img()
            cv2.imshow('board_img', board_img)
            cv2.waitKey(33)
            now_len = len(g.board.valid_states)
            if now_len < before_len:
                board_img = g.get_cur_img()
                cv2.imshow('board_img', board_img)
                cv2.waitKey(33)
                action, next_node, _ = mc.search(g.board, node)
                x, y = g.board.location_to_move(action)
                print(action)
                print(x, y)
                g.board.do_move(action)
                node = MCTS.Node(None, None)


if __name__ == '__main__':
    play()
