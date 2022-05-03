import os
import time
import gym
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, Lambda
# from tensorflow.keras.optimizers import Adam
# from keras.utils import plot_model
from datetime import datetime


# DDPG(deep deterministic policy gradient) + HER sparse rewards
##########################################################################################################
# Config
##########################################################################################################

ENV_NAME = 'FetchPickAndPlace-v1'
RANDOMSEED = 1                             # random seed

LEARNING_RATE_A = 0.001                    # learning rate for actor
LEARNING_RATE_C = 0.001                    # learning rate for critic
GAMMA = 0.9                                # reward discount
TAU = 0.01                                 # soft replacement
BATCH_SIZE = 256                           # update batchsize
TRAIN_CYCLES = 200                         # MAX training times
EACH_TRAIN_UPDATE_TIME = 50                # every time TRAIN update time
PRE_FILL_EPISODES = 300                    # number of episodes for training
MAX_EPISODE_STEPS = 50                     # number of steps for each episode
MEMORY_CAPACITY = 500                      # replay buffer size
TEST_EPISODES = 5                          # test the model per episodes
ACTION_VARIANCE = 3                        # control exploration
MAX_TEST_STEPS = 100                       # TRAIN steps

PLAY_MODEL = False

# replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
# as many HER replays as regular replays are used)
K_FUTURE = 4
P_FUTURE = 1 - (1. / (1 + K_FUTURE))

##########################################################################################################
# Base Class
##########################################################################################################
class Normalizer(object):
    def __init__(self, range):
        pass

    def update(self):
        pass

    def normalize(self):
        pass

##########################################################################################################
# Agent class
##########################################################################################################
class DDPG(object):
    """
    DDPG class
    """

    def __init__(self, action_number, state_number, action_bound, goal_number):
        # replay_buffer 用于储存跑的数据的数组：
        # 保存 (MEMORY_CAPACITY，[state, action, desired_goal, reward, next_state, next_achieved_goal, achieved_goal])
        self.replay_buffer        = []
        self.replay_buffer_point  = 0
        self.replay_buffer_remain = MEMORY_CAPACITY
        self.action_number        = action_number
        self.state_number         = state_number
        self.action_bound         = action_bound
        self.goal_number          = goal_number
        self.gamma                = GAMMA

        self.actor = self.create_actor()
        self.critic = self.create_critic()

        self.actor_target = self.create_actor()
        self.critic_target = self.create_critic()

        # 建立actor_target网络，并和actor参数一致，不能训练
        self.__copy_para(self.actor, self.actor_target)
        self.__set_model_2_eval_model(self.actor_target)

        self.__copy_para(self.critic, self.critic_target)
        self.__set_model_2_eval_model(self.critic_target)

        self.actor_opt = tf.keras.optimizers.Adam(LEARNING_RATE_A)
        self.critic_opt = tf.keras.optimizers.Adam(LEARNING_RATE_C)

        # 建立ema(exponential moving average)，滑动平均值, decay 旧平均值的比例, like polyak-averaged
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

    # 建立actor网络，输入s，输出a
    def create_actor(self):
        w_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        b_init = keras.initializers.Constant(0.1)

        actor_state_input_layer = Input(shape=(self.state_number), name="actor_state_input_layer")
        actor_goal_input_layer = Input(shape=(self.goal_number), name="actor_goal_input_layer")
        input = Concatenate()([actor_state_input_layer, actor_goal_input_layer])
        fc1 = Dense(units=256, activation="relu", kernel_initializer=w_init, bias_initializer=b_init,
                    name="actor_fc1_layer")(input)
        fc2 = Dense(units=256, activation="relu", kernel_initializer=w_init, bias_initializer=b_init,
                    name="actor_fc2_layer")(fc1)
        fc3 = Dense(units=256, activation="relu", kernel_initializer=w_init, bias_initializer=b_init,
                    name="actor_fc3_layer")(fc2)
        output = Dense(units=self.action_number, activation="tanh", kernel_initializer=w_init, bias_initializer=b_init,
                       name="actor_output_layer")(fc3)

        model = Model(inputs=[actor_state_input_layer, actor_goal_input_layer], outputs=output)
        # model.summary()
        # plot_model(model, show_shapes=True)
        return model

    # 建立Critic网络，输入s，a。输出 Q 值
    def create_critic(self):
        w_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        b_init = keras.initializers.Constant(0.1)

        critic_state_input_layer = Input(shape=(self.state_number), name="critic_state_input_layer")
        critic_action_input_layer = Input(shape=(self.action_number), name="critic_action_input_layer")
        critic_goal_input_layer = Input(shape=(self.goal_number), name="critic_goal_input_layer")
        input = Concatenate()([critic_state_input_layer, critic_action_input_layer, critic_goal_input_layer])
        fc1 = Dense(units=256, activation="relu", kernel_initializer=w_init, bias_initializer=b_init,
                    name="critic_fc1_layer")(input)
        fc2 = Dense(units=256, activation="relu", kernel_initializer=w_init, bias_initializer=b_init,
                    name="critic_fc2_layer")(fc1)
        fc3 = Dense(units=256, activation="relu", kernel_initializer=w_init, bias_initializer=b_init,
                    name="critic_fc3_layer")(fc2)
        output = Dense(units=1, kernel_initializer=w_init, bias_initializer=b_init, name="critic_output_layer")(fc3)

        model = Model(inputs=[critic_state_input_layer, critic_action_input_layer, critic_goal_input_layer], outputs=output)
        # model.summary()
        # plot_model(model, show_shapes=True)
        return model

    def __normalize_goal(self, data:list):
        pass

    def __set_model_2_eval_model(self, model):
        for layer in model.layers:
            layer.trainable = False

    def __set_model_2_train_model(self, model):
        for layer in model.layers:
            layer.trainable = True

    # 更新参数
    def __copy_para(self, from_model, to_model):
        for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
            j.assign(i)

    def __clip_obs(self, x):
        return np.clip(x, -200, 200)

    def store_2_replay_buffer(self, episode:list):
        step = deepcopy(episode)
        if self.replay_buffer_remain > 0:
            self.replay_buffer_remain -= 1
            self.replay_buffer.append(step)
        else:
            self.replay_buffer[self.replay_buffer_point] = episode
        self.replay_buffer_point = (self.replay_buffer_point + 1) % MEMORY_CAPACITY

    def choose_action_train(self, state, goal):
        state = state.reshape((1, -1))
        goal  = goal.reshape((1, -1))

        a = self.actor([state, goal])
        action = np.array(a[0])
        # Add exploration noise
        action = np.random.normal(action, ACTION_VARIANCE)

        return action

    def choose_action(self, state, goal):
        state = state.reshape((1, -1))
        goal  = goal.reshape((1, -1))
        a = self.actor([state, goal])
        action = np.array(a[0])

        return action

    def save_results(self, epoch):
        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        model_checkpoint_folder = self.get_result_folders()

        filepaths = {}
        for key, model_path in model_checkpoint_folder.items():
            filepaths[key] = os.path.join(model_path, "model-epoch_{:04d}_{}.hdf5".format(epoch, date_time))

        self.actor.save_weights(filepaths['actor'])
        self.critic.save_weights(filepaths['critic'])

        for key, model_folder in model_checkpoint_folder.items():
            self.delete_expire_file(model_folder)

        # torch.save({"actor_state_dict": self.actor.state_dict(),  #             "state_normalizer_mean": self.state_normalizer.mean,  #             "state_normalizer_std": self.state_normalizer.std,  #             "goal_normalizer_mean": self.goal_normalizer.mean,  #             "goal_normalizer_std": self.goal_normalizer.std}, "FetchPickAndPlace.pth")

    def get_result_folders(self):
        checkpoint_folder = './checkpoints'
        models = ['actor', 'critic']
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        model_checkpoint_folder = {}
        for model in models:
            model_folder = os.path.join(checkpoint_folder, model)
            model_checkpoint_folder[model] = model_folder
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
        return model_checkpoint_folder

    def delete_expire_file(self, model_folder, num_file_limit=10):
        files = os.listdir(model_folder)
        file_dts_idx = self.folder_file_sort(files)
        if len(files) > num_file_limit:
            for i in range(len(files)):
                if file_dts_idx[i] < (len(files) - num_file_limit):
                    os.remove(os.path.join(model_folder, files[i]))

    def folder_file_sort(self, files):
        file_dts = []
        for file_name in files:
            file_dts.append(datetime.strptime(file_name[-31:-5], '%Y-%m-%d %H:%M:%S.%f').timestamp())
        file_dts = np.array(file_dts)
        file_dts_idx = np.argsort(file_dts)
        return file_dts_idx

    def load_results(self):
        model_checkpoint_folder = self.get_result_folders()
        for key, model_folder in model_checkpoint_folder.items():
            files = os.listdir(model_folder)
            file_dts_idx = self.folder_file_sort(files)
            max_id = np.argmax(file_dts_idx)
            if key == 'actor':
                self.actor.load_weights(os.path.join(model_folder, files[max_id]))
            elif key == 'critic':
                self.critic.load_weights(os.path.join(model_folder, files[max_id]))

        # checkpoint = torch.load("FetchPickAndPlace.pth")
        # actor_state_dict = checkpoint["actor_state_dict"]
        # self.actor.load_state_dict(actor_state_dict)
        # state_normalizer_mean = checkpoint["state_normalizer_mean"]
        # self.state_normalizer.mean = state_normalizer_mean
        # state_normalizer_std = checkpoint["state_normalizer_std"]
        # self.state_normalizer.std = state_normalizer_std
        # goal_normalizer_mean = checkpoint["goal_normalizer_mean"]
        # self.goal_normalizer.mean = goal_normalizer_mean
        # goal_normalizer_std = checkpoint["goal_normalizer_std"]
        # self.goal_normalizer.std = goal_normalizer_std
        pass

    def train(self, env):
        # 1. reward reshape 和 goal 选取
        # HER sampling
        episode_indices = np.random.randint(0, len(self.replay_buffer), BATCH_SIZE)
        time_indices = np.random.randint(0, MAX_EPISODE_STEPS, BATCH_SIZE)

        states = []
        actions = []
        desired_goals = []
        rewards = []
        next_states = []
        next_achieved_goal = []
        achieved_goal = []

        # 做切片
        for episode, timestep in zip(episode_indices, time_indices):
            # [state, action, desired_goal, reward, next_state, next_achieved_goal, achieved_goal]
            states.append(deepcopy(self.replay_buffer[episode][timestep][0]))
            actions.append(deepcopy(self.replay_buffer[episode][timestep][1]))
            desired_goals.append(deepcopy(self.replay_buffer[episode][timestep][2]))
            rewards.append(deepcopy(self.replay_buffer[episode][timestep][3]))
            next_states.append(deepcopy(self.replay_buffer[episode][timestep][4]))
            next_achieved_goal.append(deepcopy(self.replay_buffer[episode][timestep][5]))
            achieved_goal.append(deepcopy(self.replay_buffer[episode][timestep][6]))

        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_achieved_goal = np.vstack(next_achieved_goal)
        next_states = np.vstack(next_states)
        achieved_goal = np.vstack(achieved_goal)

        her_indices = np.where(np.random.uniform(size=BATCH_SIZE) < P_FUTURE)
        future_offset = np.random.uniform(size=BATCH_SIZE) * (MAX_EPISODE_STEPS - time_indices)
        future_offset = future_offset.astype(int)
        future_time = (time_indices + 1 + future_offset)[her_indices]

        future_ag = []
        for episode, f_offset in zip(episode_indices[her_indices], future_time):
            if f_offset == MAX_EPISODE_STEPS:
                f_offset = MAX_EPISODE_STEPS - 1
            future_ag.append(deepcopy(self.replay_buffer[episode][f_offset][6]))
        future_ag = np.vstack(future_ag)

        desired_goals[her_indices] = future_ag
        rewards = np.expand_dims(env.compute_reward(next_achieved_goal, desired_goals, None), 1)

        states = self.__clip_obs(states)
        next_states = self.__clip_obs(next_states)
        desired_goals = self.__clip_obs(desired_goals)

        # 2. 更新
        # Critic：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        # loss = (y - q)^2
        with tf.GradientTape() as tape:
            action_next = self.actor_target([next_states, desired_goals])
            # action_next = np.clip(np.random.normal(action_next, ACTION_VARIANCE), self.action_bound[0],
            #                       self.action_bound[1])
            q_value_next = self.critic_target([next_states, action_next, desired_goals])
            y = rewards + self.gamma * q_value_next

            q_value = self.critic([states, actions, desired_goals])
            td_error = tf.losses.mean_squared_error(y, q_value)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor：
        # Actor的目标就是获取最多Q值的。
        # Exception
        with tf.GradientTape() as tape:
            action = self.actor([states, desired_goals])
            # action = np.clip(np.random.normal(action, ACTION_VARIANCE), self.action_bound[0], self.action_bound[1])
            q_value_next = self.critic([states, action, desired_goals])
            a_loss = -tf.reduce_mean(q_value_next)
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))


    def sync_network(self):
        """
        滑动平均更新
        """
        # 其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        paras = self.actor.trainable_weights + self.critic.trainable_weights  # 获取要更新的参数包括actor和critic的
        self.ema.apply(paras)  # 主要是建立影子参数, update shadow_variables
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))  # 用滑动平均赋值


##########################################################################################################
# test code
##########################################################################################################
def generate_episode(agent, env):
    env_dict = env.reset()  # 设置随机种子，为了能够重现
    state = env_dict["observation"]
    achieved_goal = env_dict["achieved_goal"]
    desired_goal = env_dict["desired_goal"]

    while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
        env_dict = env.reset()
        state = env_dict["observation"]
        achieved_goal = env_dict["achieved_goal"]
        desired_goal = env_dict["desired_goal"]

    episode = []
    for _ in range(MAX_EPISODE_STEPS):
        # env.render()
        action = agent.choose_action_train(state, desired_goal)
        env_dict, reward, done, info = env.step(action)

        next_state = env_dict["observation"]
        next_achieved_goal = env_dict["achieved_goal"]
        next_desired_goal = env_dict["desired_goal"]

        # [state, action, desired_goal, reward, next_state, next_achieved_goal, achieved_goal]
        step = [state.tolist(), action.tolist(), desired_goal.tolist(), reward, next_state.tolist(),
                next_achieved_goal.tolist(), achieved_goal.tolist()]
        episode.append(deepcopy(step))

        state = next_state.copy()
        achieved_goal = next_achieved_goal.copy()
        desired_goal = next_desired_goal.copy()
    return episode


if __name__ == '__main__':

    # 初始化环境
    env = gym.make(ENV_NAME)

    # 定义状态空间，动作空间，动作幅度范围
    state_number  = env.observation_space.spaces["observation"].shape[0]
    action_number = env.action_space.shape[0]
    action_bound  = [env.action_space.low[0], env.action_space.high[0]]
    goal_number   = env.observation_space.spaces["desired_goal"].shape[0]

    agent = DDPG(action_number, state_number, action_bound, goal_number)

    if PLAY_MODEL == True:
        now = datetime.now()
        print("[DEBUG]test agent {} {}".format(now.strftime("%m/%d/%Y, %H:%M:%S"), 0))
        test_count = 0
        agent.load_results()
        while TEST_EPISODES > test_count:
            test_count += 1
            env_dict = env.reset()
            for i in range(MAX_TEST_STEPS):
                env.render()
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
                env_dict, _, done, _ = env.step(agent.choose_action(state, desired_goal))
                if done:
                    break
        exit()

    # pre-fill replay-buffer
    for _ in range(PRE_FILL_EPISODES):
        episode = generate_episode(agent, env)
        agent.store_2_replay_buffer(episode)

    # 训练部分
    for train_cycle in range(TRAIN_CYCLES):
        now = datetime.now() # 统计时间
        print("[DEBUG]TRAIN_CYCLES {} {}".format(now.strftime("%m/%d/%Y, %H:%M:%S"), train_cycle))

        episode = generate_episode(agent, env)
        agent.store_2_replay_buffer(episode)
        for _ in range(EACH_TRAIN_UPDATE_TIME):
            agent.train(env)

        agent.sync_network()
        # store agent train results
        agent.save_results(train_cycle)

    # test agent
    now = datetime.now()
    print("[DEBUG]test agent {} {}".format(now.strftime("%m/%d/%Y, %H:%M:%S"), train_cycle))
    test_count = 0
    while TEST_EPISODES > test_count:
        test_count += 1
        env_dict = env.reset()
        for i in range(MAX_TEST_STEPS):
            env.render()
            state = env_dict["observation"]
            achieved_goal = env_dict["achieved_goal"]
            desired_goal = env_dict["desired_goal"]
            env_dict, _, done, _ = env.step(agent.choose_action(state, desired_goal))
            if done:
                break