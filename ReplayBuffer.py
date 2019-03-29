import json
import os.path
import random
import shutil
import time
from collections import deque
from threading import Lock
# why python document said deque is thread safe but i still need a lock!!

import numpy as np


class ReplayBuffer(object):
    def __init__(self, element_count=3, buffer_size=100, ready_size=50, file_name=None, old_backup_folder="Old"):
        self.objects = deque()
        self.lock = Lock()
        self.old_objects = []

        self.element_count = element_count
        self.buffer_size = buffer_size
        self.reday_size = ready_size

        self.random = random.Random()

        self.file_name = file_name

        if file_name is not None:
            self.restore_from_file(file_name)

        self.old_backup_floder = old_backup_folder
        if not os.path.exists(os.path.join( os.getcwd(), self.old_backup_floder)):
            os.mkdir( os.path.join( os.getcwd(), self.old_backup_floder) )
        #time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        if os.path.isdir(os.path.join( os.getcwd(), self.old_backup_floder)) == True:
            count = 0
            for _,_,files in os.walk(os.path.join( os.getcwd(), self.old_backup_floder)):
                for _ in files:
                    count += 1
        self.old_filename = os.path.join( os.getcwd(), self.old_backup_floder, "sb_twopigs_bk{}.json".format(count+1))

    def ready(self):
        if len(self.objects) > self.reday_size:
            return True
        else:
            return False

    def add_objects(self, objects, check=True):
        self.lock.acquire()
        if not check:
            self.objects.extend(objects)
        else:
            add_count = 0
            for object in objects:
                if len(object) == self.element_count:
                    self.objects.append(object)
                    add_count+=1
                else:
                    print("{}th object is invalid".format(add_count))
            print("Add {} objects to replay buffer".format(add_count))
        self.auto_remove()
        self.save_to_file()
        self.lock.release()

    def get_samples(self, sample_count, weights=None):
        if weights == None:
            sampled = self.random.sample(self.objects, sample_count)
        else:
            objects_to_sample = []
            if len(weights) == self.buffer_size:
                for i in range(self.buffer_size):
                    for _ in range(weights[i]):
                        objects_to_sample.append(self.objects[i]) # make a new list incuding weight[i] object[i] for sampling
            sampled = self.random.sample(objects_to_sample, sample_count)
        return sampled

    def auto_remove(self):
        remove_count = 0
        while len(self.objects) > self.buffer_size:
            object = self.objects.popleft()
            self.old_objects.append(object)
            remove_count += 1
        if remove_count != 0:
            print("Remove {} objects from replay buffer to old buffer".format(remove_count))
        return remove_count

    def save_to_file(self, file=None, old_backup=True):
        # Save only support element of int/float or string or list
        if file is None:
            file = self.file_name
        #if os.path.exists(file):
        #    print("do nothing")
        #else:
        if True:
            with open(file,'w') as f:
                json_strs = []
                for object in self.objects:
                    json_str = json.dumps(object) + "\n"
                    json_strs.append(json_str)
                f.writelines(json_strs)
                print("Save {} new samples to new file".format(len(json_strs)))

            if len(self.old_objects) > 0:
                old_filename = self.old_filename
                with open(old_filename,'w') as f:
                    old_json_strs = []
                    for object in self.old_objects:
                        old_json_str = json.dumps(object) + "\n"
                        old_json_strs.append(old_json_str)
                    f.writelines(old_json_strs)
                    print("Save {} old samples to backup file {}".format(len(old_json_strs), old_filename))
        return

    def restore_from_file(self, file=None):
        # Restore only support element of int/float or string
        if file is None:
            file = self.file_name
        if os.path.exists(file):
            objects = []
            with open(file,'r') as f:
                for line in f:
                    line = line.strip()
                    object = json.loads(line)
                    objects.append(object)
            self.objects.extend(objects)
            print("Restore {} samples from file".format(len(objects)))
        else:
            print("Restore file not exists")
        return

if __name__ == "__main__":
    replay_buffer = ReplayBuffer(element_count=3, buffer_size=1000, ready_size=100, file_name= os.path.join(os.getcwd(), "replay.json")) # "Old", "sb_twopigs.json_bk{}".format(25)))

    """ for object in replay_buffer.objects:
        state_path = object[0]
        try:
            array = np.loadtxt(state_path)
            if np.shape(array) != (120,210):
                print(state_path)
        except:
            print(state_path) """
    
    save_index = 1
    replay_states_floder = os.path.join( os.getcwd(), "ReplayStates" )
    if os.path.isdir(replay_states_floder) == True:
        count = 0
        for _,_,files in os.walk(replay_states_floder):
            for _ in files:
                count += 1
    save_index += count
    
    for index in range( len(replay_buffer.objects) ):
        state_path = replay_buffer.objects[index][0]
        if state_path.endswith("txt"):
            filename = str(save_index)+".array"
            new_state_path = os.path.join( replay_states_floder, filename )
            shutil.copy(state_path, new_state_path)
            replay_buffer.objects[index][0] = new_state_path
            #print(new_state_path)
            save_index+=1

    replay_buffer.save_to_file(os.path.join(os.getcwd(), "replay_backup.json" ))# "ReplayBackup", "replay_backup_{}.json".format(27)))

    #samples = []
    #for _ in range(200):
    #    samples.append([np.array([random.randint(0,5),random.randint(5,10)]),[1,2,3],list()])
    #replay_buffer.add_objects(objects = samples)
    #sampled = replay_buffer.get_samples(32)
    #print(sampled)
