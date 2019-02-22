from vizdoom import *
import numpy as np
import time,random
import tensorflow as tf
from collections import deque
import skimage
from skimage import transform, color, exposure
import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
import objgraph
warnings.filterwarnings('ignore')

import os


class agent :

    def __init__ (self,sess,input_dim,output_dim,batch_size,tau,buffer_size):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.tau = tau
        self.buffer = deque(maxlen=buffer_size)
        self.inp_layer = tf.placeholder(tf.float32, [None, *self.input_dim])
        self.sess = sess
        self.outputs = self.createNetwork()
        self.gamma = 0.99 #Discount factor

        self.network_params = tf.trainable_variables()
        # self.t_outputs = self.createNetwork()
        # self.t_network_params = tf.trainable_variables()[len(self.network_params):]

        '''Target Network implementation'''
        #
        # print(len(self.t_network_params))
        # print(len(self.network_params))
        #
        # self.update_target_op = [self.t_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.t_network_params[i], 1 - self.tau)) for i in range(len(self.network_params))]

        self.t_out = tf.placeholder(tf.float32,[None],'Target_out')
        self.action_buffer = []
        self.action_vec = tf.placeholder(tf.float32,[None,3],name='Action_vector')

        self.q_vector = tf.reduce_sum(tf.multiply(self.outputs,self.action_vec))

        self.loss = tf.reduce_mean(tf.square(self.q_vector-self.t_out))
        self.frames = []

        self.optimizer = tf.train.AdamOptimizer(0.002).minimize(self.loss)
        self.decay_rate = 0.95
        self.stop_prob = 0.01
        self.start_prob = 1.0



    def train(self):
        """
        1. Fetch everything saved in replay buffer.
        2. Compute target values for each observation in buffer by computing r + gamma*q_target(s')
        3. Compute update i.e q(s) - r + gamma*q_target(s')
        4. Reduce this loss using the optimiser.
        :return:
        """

        index = self.sample()
        # samples = [self.buffer[i] for i in index]
        # inp = [samples[i][0] for i in range(len(samples))]
        # # inp =np.array(inp)
        # # inp = list(inp[:,0,:,:,:])
        #
        # tar = [samples[i][3] for i in range(len(samples))]
        # tar = np.array(tar)
        # tar = list(tar[:,0,:,:,:])

        action_ = []
        inp = []
        tar = []
        done = []
        rew = []
        act_vec = []
        # print('1')
        for i in index:
            inp.append(self.buffer[i][0])
            tar.append(self.buffer[i][3])
            rew.append(self.buffer[i][2])
            act = np.zeros(3)
            act[self.buffer[i][1]] = 1
            act_vec.append(act)
            # action_.append(self.buffer[i][1])
            done.append(self.buffer[i][4])
        # target_predictions = self.sess.run(self.t_outputs, feed_dict = {self.inp_layer:tar})
        # print('2')
        model_predictions = self.sess.run(self.outputs, feed_dict = {self.inp_layer:tar})


        target = np.zeros(np.shape(model_predictions))

        tar_q = []

        for i in range(self.batch_size):
            if(done[i]):
                target[i][np.argmax(act_vec[i])] = rew[i]
                tar_q.append(rew[i])
            else:
                act = np.argmax(model_predictions[i])
                target[i][np.argmax(act_vec[i])] = rew[i] + self.gamma*(model_predictions[i][act])
                tar_q.append(rew[i]+self.gamma*(np.max(model_predictions[i][act])))


        # print(np.shape(tar_q))
        _,l = self.sess.run([self.optimizer,self.loss],feed_dict={self.inp_layer:tar,self.action_vec:act_vec,self.t_out:tar_q})
        # self.sess.run(self.update_target_op)
        return l

    def choose_act(self,state,decay_step,isRand,newEp):
        explore_prob = self.stop_prob + (self.start_prob-self.stop_prob)*np.exp(-decay_step*self.decay_rate)
        img = skimage.transform.resize(state, (84,84))
        if(newEp):
            self.frames = [img,img,img,img]
            self.action_buffer = np.stack(self.frames, axis=2)
            # self.action_buffer = np.expand_dims(self.action_buffer, axis=0)
            act = self.sess.run(self.outputs, feed_dict={self.inp_layer: self.action_buffer.reshape((1,84,84,4))})
        elif(explore_prob>np.random.rand() or isRand):
            act = np.random.randint(0,3)
            return act,self.action_buffer.tolist()
        else:
            # img = np.reshape(img, (1, 84, 84, 1))
            self.frames[:-1] = self.frames[1:]
            self.frames[-1] = img
            self.action_buffer = np.stack(self.frames,axis = 2)
            act = self.sess.run(self.outputs, feed_dict={self.inp_layer: self.action_buffer.reshape((1,84,84,4))})
        # print("ACT BUF SIZE: ",np.shape(self.action_buffer.tolist()))
        return np.argmax(act[0]),self.action_buffer.tolist()
    def add(self,experience):
        self.buffer.append(experience)
        return len(self.buffer)
    def sample(self):
        size = len(self.buffer)
        index = np.random.choice(np.arange(size),size = self.batch_size,replace=False)
        return index
    def createNetwork(self):
        cnn_layer_1 = tf.nn.elu(tf.layers.batch_normalization(self.cnn_layer(self.inp_layer,shape=[8,8,4,16],stride=4),training = True, epsilon = 1e-5))
        cnn_layer_2 = tf.nn.elu(tf.layers.batch_normalization(self.cnn_layer(cnn_layer_1,shape=[4,4,16,32],stride=2),training = True, epsilon = 1e-5))
        # cnn_layer_3 = tf.nn.elu(tf.layers.batch_normalization(self.cnn_layer(cnn_layer_2,shape=[4,4,64,128],stride=2),training = True, epsilon = 1e-5))
        flat_layer = tf.layers.flatten(cnn_layer_2)
        h1 = tf.nn.elu(self.make_layer(flat_layer,256))
        output = self.make_layer(h1, self.output_dim)

        return output



    def make_layer(self, input_layer, out_size):
        inp_size = int(input_layer.get_shape()[1])
        W = self.init_wts([inp_size,out_size])
        b = self.init_bias([out_size])
        return tf.matmul(input_layer,W)+b


    def init_bias(self,shape):
        return tf.Variable(tf.constant(0.1), shape, name='BIAS')
    def init_wts(self,shape):
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.Variable(initializer(shape))
    def conv2d(self,inp,ker,stride):
        return tf.nn.conv2d(input=inp,filter=ker, strides=[1,stride,stride,1], padding='SAME')
    def max_pool(inp):
        return tf.nn.max_pool(inp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def cnn_layer(self,inp,shape,stride):
        wts = self.init_wts(shape)
        bs = self.init_bias([shape[3]])
        return self.conv2d(inp, wts, stride)+bs



if __name__ == "__main__":

    game = DoomGame()
    game.load_config('defend_the_center.cfg')
    # game.load_config('basic2.cfg')
    game.init()
    actions = np.identity(game.get_available_buttons_size(), dtype=int).tolist()
    state = game.get_state().screen_buffer
    i = 0
    print('Action Size : ',game.get_available_buttons_size())
    episodes = 1000
    state_size = [84, 84, 4]
    brain = agent(None, input_dim=state_size, output_dim=3, batch_size=64, tau=0.001, buffer_size=10000)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:


        output_size = 3
        brain.sess = sess
        sess.run(tf.global_variables_initializer())
        decay_step = 0
        buff_size = 0
        act_1 = np.zeros(3)
        for i in range(episodes):
            # print(objgraph.show_most_common_types())
            step = 0
            print('Episode Number : ',i)
            game.new_episode()
            count = 0
            total_ep_rew = 0
            l = []
            while not game.is_episode_finished() and step<200:
                decay_step+=1
                step += 1
                state = game.get_state()
                # print('STATE : ',state.)
                img = state.screen_buffer
                misc = state.game_variables
                # print(img)



                if (buff_size > 100):
                    l.append(brain.train())
                    a_1, s_1 = brain.choose_act(img, decay_step, False, False)
                elif (count==0):
                    a_1, s_1 = brain.choose_act(img, decay_step, False, True)
                else :
                    a_1, s_1 = brain.choose_act(img, decay_step, True, False)
                # a_1,s_1 = brain.choose_act(img,decay_step)
                act_1 = np.zeros(3)
                act_1[a_1] = 1
                # a = random.choice(actions)
                # print(action)
                # print(act)




                if (count > 0):
                    # print(game.is_episode_finished())
                    # print(r_0)
                    buff_size = brain.add((s_0, a_0, r_0, s_1, game.is_episode_finished()))
                    # print('BUFFER_SIZE : ',buff_size)


                r_1 = game.make_action(act_1.tolist())

                # print('GAME STATE:',game.get_state())

                total_ep_rew += r_1

                # if game.is_episode_finished() :
                #      print('Total Reward : ', total_ep_rew)

                s_0 = s_1
                a_0 = a_1
                act_0 = act_1
                r_0 = r_1

                # print(state.game_variables)
                # print("\treward:", reward)
                count = 1
                # time.sleep(0.02)

            img = np.ones((84, 84), dtype = np.int)
            if (buff_size > 100):
                l.append(brain.train())
                print('Training loss : ',sum(l)/len(l))
                a_1, s_1 = brain.choose_act(img, decay_step, False, True)
            else:
                a_1, s_1 = brain.choose_act(img, decay_step, True, True)
            if (count > 0):
                buff_size = brain.add((s_0, a_0, r_0, s_1, game.is_episode_finished()))
                print(len(brain.buffer))
            print("Result:", game.get_total_reward())
            if(i%5 == 0):
                save_path = saver.save(sess, "./models_rut/brain.ckpt")
            # time.sleep(2)