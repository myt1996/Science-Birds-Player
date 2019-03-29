# Science-Birds-Player

An AI player used to play game Science-Birds. See https://github.com/lucasnfe/Science-Birds for source code and more information about this game.
It is based on AlphaGo Zero(https://www.nature.com/articles/nature24270?sf123103138=1) and AlphaZero(http://science.sciencemag.org/content/362/6419/1140+) paper from DeepMind.

It can only run on Linux x86-64 now.
Python3.6 and Tensorflow 1.x is need. See https://www.tensorflow.org/install for install of tensorflow. (Notice that tensorflow 0.x is not tested and tensorflow 2.x does not work.)

To use it, please extract Players/MCTSPlayer.zip to Players folder.
Copy MCTSPlayer folder in the same Players folder and rename to MCTSPlayer+player_index, for example, MCTSPlayer1 to MCTSPlayer6.
Please make sure that the count of MCTSPlayer is also the count of threads you want to use in train.

Modify Train.player_count in Train.py if need. Defualt it is set as 6.
It take a lot of GPU memory and defualt is on CPU mode (even you are using tensorflow-gpu), to use GPU, delete os.environ["CUDA_VISIBLE_DEVICES"] = '-1' in the start part of Train.py.

Run python Train.py. (Or python3 Train.py if need)
It takes a lot of time to train it.
