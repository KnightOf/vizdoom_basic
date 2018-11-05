from env import *
from config import *
from memory import *
from net import *

from random import sample
import itertools as it
import torch
import torch.nn.functional as F
import torch.nn as nn
from time import time, sleep
import skimage.color, skimage.transform
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange
import matplotlib.pyplot as plt


print("GPU is ->", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#print(torch.cuda.get_device_name(0))

"""
device = ("cpu")
"""

class Trainer:
    def __init__(self):
        self.game = init_doom(visable=False)
        # find game available action
        n = self.game.get_available_buttons_size()
        actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.action_available = actions
        #self.model = Net(len(actions)).to(device)
        #self.net = Net(len(actions))
        #self.model = self.net.cuda()
        self.model = Net(len(actions))
                             
        #loss
        self.criterion = nn.MSELoss()
        #bp
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         learning_rate)
        self.eps = epsilon 
        self.memory = ReplayMemory(replay_memory_size)

    def perform_learning_step(self):
        collect_scores = []
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_scores = []
            train_episodes_finished = 0
            print("Training...")
            self.game.new_episode()
            # trange show the long process text
            for learning_step in trange(learning_step_per_epoch, leave=False, ascii = True):
                #while not self.game.is_episode_finished():
                s1 = self.preprocess(self.game.get_state().screen_buffer)
                s1 = s1.reshape([1, 1, resolution[0], resolution[1]])

                action_index = self.choose_action(s1)
                reward = self.game.make_action(
                    self.action_available[action_index], frame_repeat)

                isterminal = self.game.is_episode_finished()
                s2 = self.preprocess(
                    self.game.get_state()
                    .screen_buffer) if not isterminal else None

                self.memory.add_transition(s1, action_index, s2, isterminal,
                                           reward)

                self.learn_from_memory()

                if self.game.is_episode_finished():
                    score = self.game.get_total_reward()
                    train_scores.append(score)
                    collect_scores.append(score)
                    train_episodes_finished += 1
                    # next start
                    self.game.new_episode()
            print("%d training episodes played." % train_episodes_finished)
            train_scores = np.array(train_scores)

            '''
            print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
        ''''''
        loop over
        '''
        self.show_score(collect_scores)
        self.save_model()

        self.game.close()

    '''
    test step
    '''

    def watch_model(self):
        self.load_model()
        self.game = init_doom(visable=True)
        totalgb = 0
        good = 0
        for _ in range(watch_step_per_epoch):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                state = self.preprocess(self.game.get_state().screen_buffer)
                state = state.reshape([1, 1, resolution[0], resolution[1]])
                action_index = self.choose_action(state)            
                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                self.game.set_action(self.action_available[action_index])
                #totalgb+=1
                #print(self.action_available[action_index],self.answer_check())
                '''
                if  self.action_available[action_index] == self.answer_check():    
                    good+=1
                    #print(good)
                elif self.answer_check()==[0,1,0] and self.action_available[action_index] == [0,1,1]:
                    good+=1
                elif self.answer_check()==[1,0,0] and self.action_available[action_index] == [1,0,1]:
                    good+=1
                #else:
                    #print("BADDDDDDD")
                '''
                for _ in range(frame_repeat):
                    self.game.advance_action()
                sleep(0.2) 
                
            sleep(0.5) 
            score = self.game.get_total_reward()
            print("Total score: ", score)
        #print(good,"\t",totalgb,"\t",good/totalgb)
        #return good/totalgb
        self.game.close()

    def answer_check(self):
        #training_data_answer = np.zeros([100,100])
        #move_step = 50
        #pic_per_episode = 50

        obj = self.game.get_state().labels
        #print(obj)
        if len(obj)>1:
            distance = obj[0].object_position_y - obj[1].object_position_y
        else: 
            return [0,0,1]
                
        #monster - player
        #print(abs(distance))
        if abs(distance) > 10:
            if distance > 0:
                #print("left",left)
                #training_data_answer[move_step,pic_per_episode] = 0b0100
                return [1,0,0]
            elif distance < 0:
                #print("right",right)
                #training_data_answer[move_step,pic_per_episode] = 0b0010
                return [0,1,0]
        else:
            #print("shoot",shoot)
            #training_data_answer[move_step,pic_per_episode] = 0b0001
            return [0,0,1]
        """
        img = self.game.get_state().screen_buffer
                
        current_name = './training_data/'+str(move_step)+'-'+str(pic_per_episode)
        plt.imshow(img)
        plt.axis('OFF')
        plt.savefig(current_name,bbox_inches='tight')
        
        return reaction
        """
    
    '''
    save model
    '''

    def save_model(self):
        torch.save(self.model, model_savefile)

    '''
    load model
    '''

    def load_model(self):
        print("Loading model from: ", model_savefile)
        
        self.model = torch.load(model_savefile)
        
    '''
    show score
    '''

    def show_score(self, scores):
        import time
        localtime = time.localtime()
        timeString = time.strftime("%m%d%H", localtime)
        timeString = './' + 'score_' + str(timeString) + '.jpg'

        plt.plot(scores)
        plt.xlabel('episodes')
        plt.ylabel('total reward')
        plt.savefig(timeString)
        plt.show()

    '''
    Subsampling image and convert to numpy types
    '''

    def preprocess(self, img):
        img = skimage.transform.resize(img, resolution)
        img = img.astype(np.float32)
        return img

    '''
    bp using
    '''

    def back_propagation(self, state, target_q):
        s1 = torch.from_numpy(state)
        target_q = torch.from_numpy(target_q)
        s1, target_q = Variable(s1), Variable(target_q)
        output = self.model(s1)
        loss = self.criterion(output, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    '''
    convert to tensor data
    '''

    def tensor_type(self, state):
        state = torch.from_numpy(state)
        state = Variable(state)
        #print(state.shape)
        return self.model(state)

    '''
    choose action
    '''

    #(64, 1, 30, 45)
    def choose_action(self, state):
        #state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        if self.eps > np.random.uniform():
            action_index = np.random.randint(0, len(self.action_available) - 1)
            return action_index
        else:

            q = self.tensor_type(state)
            m, index = torch.max(q, 1)
            action = index.data.numpy()[0]
            #print('eps == ' ,self.eps,'action ==',action)
            return action

    '''
    learn
    '''

    def learn_from_memory(self):
        # decrease explore rate
        self.exploration_rate()
        if self.memory.size > batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(batch_size)
            #convert numpy type
            target_q = self.tensor_type(s1).data.numpy()
            #q_eval = self.model(s1).gather(1, a)
            q = self.tensor_type(s2).data.numpy()
            #q_next = self.model(s2).detach() # won't do BP
            q2 = np.max(q, axis=1)
            #q_target = r + gamma*(q_next(1)[0].view(batch_size, 1))
            target_q[np.arange(target_q.shape[0]),
                     a] = r + gamma * (1 - isterminal) * q2
            self.back_propagation(s1, target_q)

    '''
    decrease explorate
    '''

    def exploration_rate(self):

        if self.eps > min_eps:
            self.eps *= dec_eps
    
#watch_models = 10
#arr = np.zeros([watch_models])
trainer = Trainer()
trainer.perform_learning_step()
trainer.watch_model()

'''
for i in range(watch_models):
    arr[i]=trainer.watch_model()
'''


#print("epochs,*learning_step_per_epoch:",epochs,'*',learning_step_per_epoch)
#print("average:",arr.sum()/watch_models)
#print(arr)

