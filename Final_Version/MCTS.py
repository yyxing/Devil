import numpy as np
from policy_value_net import Net

"""
    蒙特卡洛树搜索
    用于自我对弈和人机对弈
    自我对弈：自我对弈中再下每一步之前将当前局面喂进神经网络，得出当前局面的胜率和接下来每一步的概率，然后进行扩展和backup 以上步骤执行1600次以后
    输出下一步的位置
    然后根据子节点访问的次数进行softmax函数的概率转换
    最后根据加了狄利克雷噪声的概率分布随机取一个值作为下一步
"""


# 搜索树节点 每个节点包含其父子几点 这个节点的先验概率 访问次数 Q值 和 W值
class Node(object):
    def __init__(self, p, parent):
        self.p = p
        self.parent = parent
        self.N = 0
        self.Q = 0
        self.W = 0
        self.children = {}

    # 根据uct值选择值最大的子节点
    def select(self):
        return max(self.children.items(), key=lambda c: c[1].uct())

    # uct值
    def uct(self, c_puct=5.0):
        return self.Q + c_puct * self.p * (np.sqrt(self.parent.N)) / (self.N + 1)

    # 扩展节点
    def expand(self, act_probs):
        for act, prob in act_probs:
            if act not in self.children:
                self.children[act] = Node(prob, self)

    # 沿着扩展到叶子节点的路线回溯将边的统计数据更新
    def backup(self, leaf_value):
        self.W += leaf_value
        self.N += 1
        self.Q = self.W / self.N

    # 判断是否为叶子节点
    def is_leaf(self):
        return len(self.children) == 0


# 蒙特卡洛树
class MCTS(object):
    def __init__(self, net: Net):
        self.net = net
        self.board = None
        self.root = None

    def search(self, board, node, temp=1e-3):
        self.root = node
        self.board = board
        for _ in range(1600):
            node = self.root
            board = self.board.clone()
            while not node.is_leaf():
                action, node = node.select()
                x, y = board.location_to_move(action)
                board.do_move(x, y)
            act_probs, value = self.net.get_value_policy(board)
            act, probs = map(np.array, zip(*act_probs))
            if node.parent is None:
                probs = self.dirichlet_noise(probs)
            act_probs = zip(act, probs)
            end, winner = board.game_end()
            if not end:
                node.expand(act_probs)
            else:
                if winner == -1:
                    value = 0.0
                else:
                    value = 1.0 if winner == board.cur_player else -1.0
            while node is not None:
                value = -value
                node.backup(value)
                node = node.parent
        action_items = np.zeros(board.size ** 2)
        for action, child in self.root.children.items():
            action_items[action] = child.N
        action, pi = self.decision(action_items, temp)
        for act, child in self.root.children.items():
            if action == act:
                return action, child, pi

    @staticmethod
    def dirichlet_noise(probs, eps=0.25, alpha=0.03):
        return (1 - eps) * probs + eps * np.random.dirichlet(np.full(len(probs), alpha))

    @staticmethod
    def decision(pi, temperature):
        pi = (1.0 / temperature) * np.log(pi + 1e-10)
        pi = np.exp(pi - np.max(pi))
        pi /= np.sum(pi)
        action = np.random.choice(len(pi), p=pi)
        return action, pi
