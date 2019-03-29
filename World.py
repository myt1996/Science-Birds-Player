import numpy as np


class World(object):

    def __init__(self, world_file=None, width=210, height=120):
        self.width = width
        self.height = height

        self.states = np.zeros((self.height, self.width))
        self.path = world_file

        if world_file is not None:
            self.read_state(world_file)

    def read_state(self, state_path=None):
        if state_path is None:
            state_path = self.path
        array = np.loadtxt(state_path, dtype=int)
        if np.shape(array) == (120, 210):
            self.states = array
        # f = open(state_path, "r")
        # for h in range(self.height):
        #     line = f.readline().rstrip()
        #     line_states = line.split(" ")
        #     for w in range(self.width):
        #         self.states[h,w] = int(line_states[w])
        # f.close()


    def current_state(self):

        world_state = np.zeros((self.height, self.width, 12))

        if self.states.shape==(self.height, self.width):

            for i in range(self.height):
                for j in range(self.width):
                    state = self.states[i,j]
                    #find slingshot
                    if state ==1:
                        world_state[i,j,0] = 1#10
                    #find a bird
                    if state ==11:
                        world_state[i,j,1] = 1#10
                    if state ==12:
                        world_state[i,j,2] = 1#10
                    if state ==13:
                        world_state[i,j,3] = 1#10
                    if state ==14:
                        world_state[i,j,4] = 1#10
                    if state ==15:
                        world_state[i,j,5] = 1#10

                    #find a pig
                    if state == 21:
                        world_state[i,j,6] = 1#6

                    #find a wood block
                    if state == 31:
                        world_state[i,j,7] = 1#2

                    #find a stone block
                    if state == 32:
                        world_state[i,j,8] = 1#3

                    #find a ice block
                    if state == 33:
                        world_state[i,j,9] = 1#1

                    #find a platform
                    if state == 41:
                        world_state[i,j,10] = 1#4

                    #find a tnt
                    if state == 51:
                        world_state[i,j,11] = 1#5

                    if state != 1 and state != 11 and state != 12 and state != 13 and state != 14 and state != 15  and state != 21 and state != 31 and state != 32 and state != 33 and state != 41 and state != 51 and state != 0:
                        print("state file error")

        return world_state
