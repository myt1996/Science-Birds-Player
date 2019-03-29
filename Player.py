import os
import random
import shutil
import subprocess
import sys
import threading
from time import sleep

import numpy as np

from PolicyValueCNN import PolicyValueCNN
from World import World

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
##os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

width = 210
height = 120
channel = 12
action_count = 147

def read_action(action_str):
    actions_visit = action_str.split("_")
    visits = []
    for i in range(len(actions_visit)):
        try:
            value = int(actions_visit[i])
            visits.append(value)
        except:
            print("one result file broken")
            break
    return visits

def game(worker_path):
    os.system(os.path.join(worker_path, "MCTSPlayer.exe") + " " + worker_path)

def network_thread(player, temp_folder):

    print("Network thread started for " + temp_folder)

    state_path = os.path.join(temp_folder, "State.txt")
    policy_value_path = os.path.join(temp_folder, "PolicyValue.txt")
    policy_value_temp = os.path.join(temp_folder, "PolicyValueTemp.txt")

    while True:
        sleep(0.05)
        if os.access(state_path, os.F_OK) and os.access(state_path, os.R_OK):
            try:
                world = World(state_path, width, height)
                state = world.current_state()
                os.remove(state_path)

                policy, value = player.policy_value(state)
                #print(policy, value)

                policy_str = ""
                for i in range(action_count):
                    policy_str += str(policy[i])
                    if i!=action_count-1:
                        policy_str += "_"
                value_str = str(value)

                with open(policy_value_temp,'w') as f:
                    f.write(policy_str + "\n")
                    f.write(value_str)

                if os.access(policy_value_path, os.F_OK):
                    os.remove(policy_value_path)

                shutil.move(policy_value_temp, policy_value_path)

            except Exception as e:
                print(str(e))

class Player(object):
    def __init__(self, worker_path=None, policy_value_function=None, network_worker=False, train=None):

        if worker_path is not None:
            self.worker_path = worker_path
        else:
            self.worker_path = os.path.join( os.getcwd(), "MCTSPlayer" )
        print("init player on path {}".format(self.worker_path))

        self.temp_floder = os.path.join( worker_path, "Data", "Temp" )
        self.result_path = os.path.join( worker_path, "Data", "Result" )

        self.policy_value_function = policy_value_function

        #for test
        #self.network = PolicyValueCNN([210,120,12], 147)
        #self.policy_value_function = self.network.policy_value

        self.game_played = 0
        self.result_count = 0

        if network_worker:
            self.network_worker()

        self.train = train

    def start_play(self):

        if os.path.isdir(self.result_path) == True:
            count = 0
            for _,_,files in os.walk(self.result_path):
                for _ in files:
                    count += 1
            self.result_count = count
        
        sbexe = subprocess.Popen([os.path.join(self.worker_path, "MCTSPlayer.x86_64"), self.worker_path])
        try:
            sbexe.wait(timeout=300000)
        except subprocess.TimeoutExpired:
            sbexe.kill()

        #p = Process(name='game', target=game, args = (self.worker_path,))
        #p.start()
        #p.join(timeout=300)
        #if p.is_alive():
        #    p.terminate()

    def collect_result(self):
        if os.path.isdir(self.result_path) == True:
            count = 0
            for _,_,files in os.walk(self.result_path):
                for _ in files:
                    count += 1
        count_temp = self.result_count+1
        new_results = []
        while(count_temp<=count):
            new_results.append( os.path.join(self.result_path, (str(count_temp)+".txt")) )
            count_temp+=1
        state_paths = []
        actions = []
        pecs = []
        for now_result_file in new_results:
            with open(now_result_file, "r") as f:
                state_paths.append(f.readline().strip('\n'))
                actions.append(f.readline().strip('\n'))
                pecs.append(f.readline().strip('\n'))
        return state_paths, actions, pecs

    def generator_a_game(self):
        self.train.generator_lock.acquire()

        #pig_count_min = random.randint(1,3)
        #pig_count_max = pig_count_min + 4
        pig_count_min = 1
        pig_count_max = 10

        parameter_file_name = "parameters.txt"
        f = open(parameter_file_name, "w")
        f.write("1\n")
        f.write("\n")
        f.write("" + str(pig_count_min) + "," + str(pig_count_max) + "\n")
        f.write("30\n")
        f.close()

        over_flag = False
        while over_flag == False:
           try:
               import Generator
               sys.modules.pop('Generator')
               over_flag = True
           except NameError:
               print("one level fail")
               if os.path.isfile("level-04.xml"):
                    os.remove("level-04.xml")


        if os.path.isfile(os.path.join(self.worker_path, "MCTSPlayer_Data", "StreamingAssets", "Levels","level-04.xml")):
            os.remove(os.path.join(self.worker_path, "MCTSPlayer_Data", "StreamingAssets", "Levels","level-04.xml"))
        shutil.move("level-04.xml", os.path.join(self.worker_path, "MCTSPlayer_Data", "StreamingAssets", "Levels", "level-04.xml"))   

        self.train.generator_lock.release()
        

    def play_a_game(self):
        self.generator_a_game()
        self.start_play()
        state_path,action,pec = self.collect_result()
        self.game_played+=1
        return state_path,action,pec

    def play_two_player_game(self):

        finish_one_game = False
        while not finish_one_game:
            self.generator_a_game()
            self.start_play()
            state_path1,action1,pec1 = self.collect_result()
            if len(pec1) > 0 :
                finish_one_game = True
        self.game_played+=1
        
        finish_one_game = False
        while not finish_one_game:
            self.generator_a_game()
            self.start_play()
            state_path2,action2,pec2 = self.collect_result()
            if len(pec2) > 0 :
                finish_one_game = True
        self.game_played+=1

        state_paths = []
        action_strs = []
        pecs = []

        final_pec1 = float(pec1[len(pec1)-1])
        final_pec2 = float(pec2[len(pec2)-1])

        if final_pec1 > final_pec2:
            for index in range( len(pec1) ):
                pec1[index] = 1
            for index in range( len(pec2) ):
                pec2[index] = -1
        elif final_pec1 < final_pec2:
            for index in range( len(pec1) ):
                pec1[index] = -1
            for index in range( len(pec2) ):
                pec2[index] = 1
        else:
            for index in range( len(pec1) ):
                pec1[index] = 0
            for index in range( len(pec2) ):
                pec2[index] = 0

        state_paths.extend(state_path1)
        state_paths.extend(state_path2)
        action_strs.extend(action1)
        action_strs.extend(action2)
        pecs.extend(pec1)
        pecs.extend(pec2)

        actions = []
        for action_str in action_strs:
            actions.append( read_action(action_str) )

        return zip(state_paths, actions, pecs), final_pec1, final_pec2

    def policy_value(self, state):
        state_reshaped = np.reshape(state, (-1,height,width,channel))
        policy, value = self.policy_value_function(state_reshaped)
        policy_reshaped = np.reshape(policy, (action_count))
        value_reshaped = np.array(value).mean()
        return policy_reshaped, value_reshaped

    def network_worker(self):
        t = threading.Thread(name='network_thread', target=network_thread, args= (self, self.temp_floder, ))
        t.start()

if __name__ == "__main__":
    #player = Player( worker_path = "D:\ScienceBirdsPlayer\MCTSPlayer", network_worker= True)
    #state_path,action,pec = player.play_a_game()
    while(1):
        i=1
    #state_paths, actions, pecs = player.play_a_game()
