import tensorflow as tf
import os


class Net(object):
    def __init__(self, size=11):
        tf.reset_default_graph()
        self.size = size
        self.model_path = './model/net_{}_{}_model'.format(size, size)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.l2_const = 1e-4
        self._create_policy_value_net()
        self._loss_train_op()
        self.saver = tf.train.Saver()
        self.restore_model()

    # 构建策略价值网络
    def _create_policy_value_net(self):
        with tf.variable_scope("input_tensor"):
            # 当前棋盘输入
            self.state_input = tf.placeholder(tf.float32, shape=[None, 4, self.size, self.size])
            # self.state_reshape = tf.transpose(self.state_input, [0, 2, 3, 1])
            # 某一局对局的最后的得分
            self.winner_score = tf.placeholder(tf.float32, shape=[None], name='winner')
            self.winner_z = tf.reshape(self.winner_score, shape=[-1, 1])
            # 通过蒙特卡洛树搜索得出的各个位置的概率 用于policy loss
            self.mcts_probs = tf.placeholder(tf.float32, shape=[None, self.size ** 2], name='mcts_probs')
        # 卷积层参数
        conv1 = tf.layers.conv2d(self.state_input, filters=32, kernel_size=3,
                                 strides=1, padding="SAME", data_format='channels_first',
                                 activation=tf.nn.relu, name="conv1")
        conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3,
                                 strides=1, padding="SAME", data_format='channels_first',
                                 activation=tf.nn.relu, name="conv2")
        conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size=3,
                                 strides=1, padding="SAME", data_format='channels_first',
                                 activation=tf.nn.relu, name="conv3")

        # 策略头 输出下一步各个位置的概率
        policy_net = tf.layers.conv2d(conv3, filters=4, kernel_size=1,
                                      strides=1, padding="SAME", data_format='channels_first',
                                      activation=tf.nn.relu, name="policy_net")
        policy_net_flat = tf.reshape(policy_net, shape=[-1, 4 * self.size * self.size])
        self.policy_net_out = tf.layers.dense(policy_net_flat, self.size * self.size, name="output")
        self.action_probs = tf.nn.softmax(self.policy_net_out, name="policy_net_proba")

        # 价值头 输出这一步的价值评估
        value_net = tf.layers.conv2d(conv3, filters=2, kernel_size=1, data_format='channels_first',
                                     name='value_conv', activation=tf.nn.relu)
        value_net = tf.layers.dense(tf.contrib.layers.flatten(value_net), 64, activation=tf.nn.relu)
        self.value = tf.layers.dense(value_net, units=1, activation=tf.nn.tanh)

    # 损失函数和优化操作
    # total_loss = policy_loss + value_loss + l2_loss
    # 这个损失函数作为整篇论文的最大亮点 使棋类游戏可以有一个通用的解法
    # l = (z - v) ^ 2 - π^T.dot(p) + c||θ||^2
    def _loss_train_op(self):
        # l2 loss
        l2_loss = 0
        for v in tf.trainable_variables():
            if 'bias' not in v.name.lower():
                l2_loss += tf.nn.l2_loss(v)
        # value loss
        value_loss = tf.reduce_mean(tf.square(self.winner_z - self.value))
        # policy loss
        # 输出概率和mcts概率的极大似然估计
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.mcts_probs, logits=self.policy_net_out)
        policy_loss = tf.reduce_mean(cross_entropy)
        self.entropy = policy_loss
        self.loss = value_loss + policy_loss + self.l2_const * l2_loss
        self.lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train_step(self, state_batch, score_batch, mcts_batch, lr):
        loss, entropy, _ = self.sess.run([self.loss, self.entropy, self.optimizer],
                                         feed_dict={
                                             self.state_input: state_batch,
                                             self.winner_score: score_batch,
                                             self.mcts_probs: mcts_batch,
                                             self.lr: lr
                                         })
        return loss, entropy

    def restore_model(self):
        if os.path.exists(self.model_path + '.meta'):
            self.saver.restore(self.sess, self.model_path)
            print('load {} successful'.format(self.model_path + '.meta'))
        else:
            self.sess.run(tf.global_variables_initializer())

    def policy_value_by_batch(self, state_batch):
        action_probs, value = self.sess.run([self.action_probs, self.value],
                                            feed_dict={self.state_input: state_batch})
        return action_probs, value

    # 根据当前局面
    def get_value_policy(self, board):
        valid_states = board.valid_states
        current_state = board.get_state()
        # print(current_state)
        value, act_probs = self.sess.run([self.value, self.action_probs], feed_dict={
            self.state_input: current_state.reshape(-1, 4, self.size, self.size)
        })
        # print(act_probs)
        # print(value)
        act_probs = zip(valid_states, act_probs.flatten()[valid_states])
        return act_probs, value[0][0]
