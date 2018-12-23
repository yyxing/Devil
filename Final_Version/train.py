import random
import numpy as np
from collections import deque
from game import Game
from policy_value_net import Net
import time


class Train(object):
    def __init__(self):
        self.net = Net()
        self.eval_net = Net()
        self.game = Game(self.net, self.eval_net)
        self.lr = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.size = 13
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 5000
        self.best_win_ratio = 0.8
        self.losses = []
        self.entropy = []

    # 取样
    def sample(self, data):
        extend_data = []
        for state, mcts_porb, winner in data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.size, self.size)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    # 通过取样收集指定盘数的数据
    def collect_selfplay_data(self, n_games=10):
        for i in range(n_games):
            winner, play_data = self.game.start_self_play()
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.sample(play_data)
            self.data_buffer.extend(play_data)

    # 策略提升 运行网络的train_step 通过kl散度来判断当前更新的效率还可以根据kl散度来动态调节学习率
    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        score_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.net.policy_value_by_batch(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.net.train_step(state_batch, score_batch, mcts_probs_batch,
                                                self.lr * self.lr_multiplier)
            new_probs, new_v = self.net.policy_value_by_batch(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        explained_var_old = (1 -
                             np.var(np.array(score_batch) - old_v.flatten()) /
                             np.var(np.array(score_batch)))
        explained_var_new = (1 -
                             np.var(np.array(score_batch) - new_v.flatten()) /
                             np.var(np.array(score_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    # 训练主程序通过5000轮的训练得到一个不错的AI
    def run(self):
        try:
            self.net.saver.save(self.net.sess, self.net.model_path)
            self.eval_net.saver.restore(self.eval_net.sess, self.net.model_path)
            for i in range(self.game_batch_num):
                s_time = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                    i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    self.losses.append(loss)
                    self.entropy.append(entropy)
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    result = self.game.evaluate()
                    win_ratio = (1.0 * result[1] + 0.5 * result[0]) / 10
                    self.net.saver.save(self.net.sess, self.net.model_path)
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!")
                        # update the best_policy
                        self.net.saver.save(self.net.sess, self.net.model_path)
                        self.eval_net.saver.restore(self.eval_net.sess, self.net.model_path)
                e_time = time.time()
                print(f"each batch takes {(e_time - s_time)} seconds")
        except KeyboardInterrupt:
            self.net.saver.save(self.net.sess, self.net.model_path)
            import matplotlib.pyplot as plt
            plt.plot(self.losses, range(i))
            plt.xlabel("self-play game num")
            plt.ylabel("loss")
            plt.show()


if __name__ == '__main__':
    train = Train()
    train.run()
