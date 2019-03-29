import os
import threading
from collections import deque
from copy import deepcopy
from time import sleep

import numpy as np

from Player import Player
from PolicyValueCNN import PolicyValueCNN
from ReplayBuffer import ReplayBuffer
from World import World

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

def get_average_value(list):
    total = 0
    for element in list:
        total += element
    return total/len(list)

def generator_worker(train, index):
    print("Generator worker started")
    while True:
        samples, pec1, pec2 = train.players[index].play_two_player_game()
        print("One pair samples ganerated, pec: {} {}".format(pec1, pec2))
        train.replay_buffer.add_objects(samples)

#def evaluator_worker(train):
#    print("Evaluator worker started")
#    while True:
#        # if need evaluator work, just evaluate for 10 times
#        if train.finish_latest_test == False:
#            latest_scores = []
#            best_scores = []

#            for i in range(5):
#                print("Evaluator in round " + str(i+1))
#                best_scores.append( train.best_player_evl.start_first_play() )
#                print("Best player score: " + str(get_average_value(best_scores)))
#                best_scores.append( train.best_player_evl.start_second_play() )
#                print("Best player score: " + str(get_average_value(best_scores)))
#                _ = train.best_player_evl.get_play_data()

#                latest_scores.append( train.latest_player.start_first_play() )
#                print("Latest player score: " + str(get_average_value(latest_scores)))
#                latest_scores.append( train.latest_player.start_second_play() )
#                print("Latest player score: " + str(get_average_value(latest_scores)))
#                _ = train.latest_player.get_play_data()   

#            if get_average_value(latest_scores) > get_average_value(best_scores):
#                train.network.copy_latest_to_best()
#                train.network.save_model(os.path.join(os.getcwd() ,"latest"))
#                print("Got new best player")

#            latest_scores.clear()
#            best_scores.clear()
#            train.finish_latest_test = True
#        # if no need to work, just wait
#        else:
#            sleep(1)

#def optimizer_worker(train):
#    print("Optimizer worker started")
#    while train.replay_buffer.ready() == False:
#        sleep(10)
#    while True:
#        # if evaluator is working, update train network until evaluator work end
#        if train.finish_latest_test == False:
#            for _ in range(30):
#                sampled = train.replay_buffer.get_samples(train.batch_size)
#                state_batch, action_batch, value_batch = zip(*sampled)
#                state_batch_reshaped = np.reshape(state_batch, (-1,train.obs_shape))
#                action_batch_reshaped = np.reshape(action_batch, (-1,train.act_shape))
#                value_batch_reshaped = np.reshape(value_batch, (-1,1))
#                loss, entropy = train.network.train_step(state_batch_reshaped, action_batch_reshaped, value_batch_reshaped, train.learning_rate)
#                sleep(0.1)
#            print(loss, entropy)
#        # evaluator end, set new latest player to evaluator
#        else:
#            train.network.copy_train_to_latest()
#            train.network.save_model(os.path.join(os.getcwd() ,"latest"))
#            print("Latest model updated")
#            train.finish_latest_test = False

def optimizer_worker(train):
    print("Optimizer worker started")
    while train.replay_buffer.ready() == False:
        sleep(10)
    while True:
        for _ in range(5):
            sampled = train.replay_buffer.get_samples(train.batch_size)
            state_batch, action_batch, value_batch = zip(*sampled)
            # SB need to read file from state.txt file to prevent json file is too big
            states = []
            for state_path in state_batch:
                world = World(state_path)
                state = world.current_state()
                states.append(state)
            # need change visit count to probability distribution
            actions = []
            for visit_count in action_batch:
                visit_count_array = np.array(visit_count)
                total_visit = np.sum(visit_count_array)
                probability_distribution = visit_count_array / total_visit
                actions.append(probability_distribution)
            state_batch_reshaped = np.reshape(states, (-1,train.obs_shape[1], train.obs_shape[0], train.obs_shape[2]))
            action_batch_reshaped = np.reshape(actions, (-1,train.act_shape))
            value_batch_reshaped = np.reshape(value_batch, (-1,1))
            loss, entropy = train.network.train_step(state_batch_reshaped, action_batch_reshaped, value_batch_reshaped, train.learning_rate)
        print(loss, entropy)
        train.network.save_model(os.path.join(os.getcwd() ,"latest"))

    

class Train(object):
    def __init__(self, model_file = None):
        if model_file is not None:
            self.model_file = model_file

        self.obs_shape = (210,120,12)
        self.act_shape = 147
        
        self.build_network(model_file)

        self.batch_size = 32
        self.replay_buffer = ReplayBuffer(element_count=3, buffer_size=1000, ready_size=100, file_name="replay.json")

        self.learning_rate = 0.00001
        
        self.players = []
        self.player_count = 6

        self.generator_lock = threading.Lock()

        for index in range(self.player_count):
            worker_path = os.path.join( os.getcwd(), "Players", "MCTSPlayer"+str(index+1) )
            player = Player(worker_path=worker_path, policy_value_function=self.network.policy_value, network_worker=True, train=self)
            self.players.append(player)
        #self.player = MctsPlayer(self.env_manger, self.obs_shape, self.act_shape, self.network.policy_value)

    def build_network(self, model_file):
        self.network = PolicyValueCNN(self.obs_shape, self.act_shape, model_file)

    def thread_train(self):

        for index in range( len(self.players) ):
            t = threading.Thread(name='generator_worker', target=generator_worker, args= (self, index, ))
            t.start()

        t = threading.Thread(name='optimizer_worker', target=optimizer_worker, args= (self, ))
        t.start()

if __name__ == "__main__":
    train = Train()
    train.thread_train()
