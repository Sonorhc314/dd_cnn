from option import *
from enviroment_setup import *
import numpy as np
import math
from matplotlib import pyplot as plt

class QLearningOptionsAgent():
    def __init__(self, gold_positions, env, e = 0.1):
        self.domain = env.env_master.domain
        self.q_table = np.ones((8, 22, 4))*(0)
        for action in range(4):
            self.q_table[gold_positions[0]][gold_positions[1]][action] = 0
        self.gamma = 1
        self.learning_rate = 0.1
        self.num_actions = 4 #primitives
        self.num_prim_actions = 4
        self.e = e
        self.actions = ["n", "s", "w", "e"]
        self.total_reward=0
        self.t = 0
        self.trajectories=[]
        #===================
        
        # Correspondance between column index in Q matrix and definition of option
        self.idOption2Option = {}
        self.idOption2OptionTerminalState = {}
        
        # Redefining the possible options from state s : actions + options
        #self.available_actions = {}
        
    def update_env(self, env):
        self.env = env
        
    def find_agent_location(self, array):
        loc = np.where(array == 2)
        try:
            return [loc[0][0], loc[1][0]]
        except:
            print("can't find the agent ...")
            return [None, None]
        
    def add_option(self, option):
        #adds option : updates available actions from state,
        #Q matrix and nb_visits matrix, correspondance between column of Qmatrix and new option definition

        id_option = self.num_actions
        
        self.num_actions+=1

        self.idOption2Option[id_option] = option
        self.idOption2OptionTerminalState[id_option] = option.terminal_states
        
        q_table_buf = - math.inf * np.ones((8, 22, self.num_actions))
#         print(f"q_table_buf shape is {q_table_buf.shape}")
#         print(f"q_table shape is {self.q_table.shape}")
        
        q_table_buf[:, :, :-1] = self.q_table
        
        num_states = self.q_table.shape[0]
        Q_option = np.zeros((num_states, 1))
        for row in range(len(self.q_table)):
            for col in range(len(self.q_table[0])):
                if ((row, col) in option.input_set):
                    q_table_buf[row][col][id_option] = 0
        self.q_table = q_table_buf

        #nb_visits_option = np.zeros((num_states, 1))
        #self.nb_visits = np.hstack([self.nb_visits, nb_visits_option])
        
    def valid_pos(self, candidate_position):
        if self.domain[candidate_position[0]][candidate_position[1]] not in [1]:
            return 1
        return 0 
                
        return new_position
    
    def update_table(self, action_index, state, density):
        done=False
        reward_episode=0
        t = 0
        trajectory = [] #moves_made, local trajectory
        if action_index<self.num_prim_actions:
            density.add_to_one_p_bag(state)
            #primitive action
            nexts, reward, done, info = env.step(action_index)
            #print(reward)
            new_pos = self.find_agent_location(env.env_master.domain)
            next_max_q = max(self.q_table[new_pos[0]][new_pos[1]])
            new_val = (1-self.learning_rate)*self.q_table[state[0]][state[1]][action_index]+self.learning_rate*(reward + self.gamma*next_max_q)
            self.q_table[state[0]][state[1]][action_index] = new_val
            trajectory.append([state, action_index, reward])
            
            reward_episode += self.gamma ** t * reward
            t += 1
            state=new_pos
            
        else:
            #*** ACTION IS AN OPTION***
            #print("option start")

            state_0 = state

            id_option = action_index
            option = self.idOption2Option[action_index]

            reward_option = 0
            t_option = 0
            
#             option_graph = np.ones((8,22))
            
#             for row in range(len(self.q_table)):
#                 for col in range(len(self.q_table[0])):
#                     if ((row, col) in option.input_set):
#                         option_graph[row][col] = 0
            
#             option_graph[option.terminal_states[0][0]][option.terminal_states[0][1]] = 0.5
            
#             plt.figure(figsize=(len(self.q_table), len(self.q_table[0])))
#             plt.imshow(option_graph, cmap='gray', vmin=0, vmax=1)
#             plt.xticks([])
#             plt.yticks([])
#             plt.show()
            
#             plt.imshow(self.domain)
#             plt.show()
#             plt.clf()
            
            print("option start")
            while (not(state in option.terminal_states)) \
                    and ((state[0], state[1]) in option.input_set):
                
                if (done or (self.t+t)>1000):  # time horizon:
                    # time of episod > Tmax
                    # or we did not reach the big goal (absorbing state)
                    break
                density.add_to_one_p_bag(state)
                #while subgoal not reached
                #and still in input set of option

                # Following option
                action = option.take_action(state)
#                 print(f"state is {state}")
#                 print(f"action is {action}")

                # Simulating next state and reward
                print(f"state is {state}, action is {action}")
                nexts, reward, done, info = env.step(action)
                
                new_pos = self.find_agent_location(env.env_master.domain)

                trajectory.append([state, action, reward])

                # Updating reward
                reward_episode += self.gamma ** t * reward

                # Updating reward option
                reward_option += self.gamma ** t_option * reward

                # Updating current state
                #option.Q[state[0]][state[1]][action] =  
                state = new_pos
                t += 1
                t_option += 1
                #print("option")
            print("option end")
            print("")
            learned_value = reward_option + self.gamma**t_option * self.q_table[state[0], state[1], :].max()
            option.init_policy(self.trajectories)
            self.q_table[state_0[0], state_0[1], id_option] = (1 - self.learning_rate) * self.q_table[state_0[0], state_0[1], id_option] + self.learning_rate * learned_value
        self.total_reward+=reward_episode
        self.t += t
        #print(reward_episode)
        return done, trajectory, state
    
    def episode_t_renew(self):
        self.t=0
    
    def choose_action(self, state, density):
        self.t=0
        
        done = False
        trajectory=[]
        
        while (not done and self.t<1000):
        
            row_needed = self.q_table[state[0]][state[1]]

            if np.random.uniform(0, 1) > self.e:
                action_index = np.random.choice(np.flatnonzero(row_needed==row_needed.max()))#np.argmax(row_needed)
            else:
                actions_indices_available=[]
                for index in range(len(row_needed)): #[2,3,2,3,-math,3]
                    if row_needed[index]!=-math.inf:
                        actions_indices_available.append(index)

                action_index = np.random.choice(actions_indices_available)
            
            done, moves_made, state = self.update_table(action_index, state, density)
            #print(trajectory)
            for states in moves_made:
                trajectory.append(states)
            self.e-=0.00001
        self.trajectories.append(trajectory)
        return density, self.trajectories