import diverse_density

class Option():
    def __init__(self, env, goal, trajectories, size_option = None, gamma = 1):
        self.epsilon = 0.1
        #self.initializing_set = set()
        self.input_set=set()
        for trajectory in trajectories:
            found_subgoal = False
            for (t,step) in enumerate(trajectory):
                if found_subgoal: break
                state = step[0]
                if state[0] == goal[0] and state[1] == goal[1]:
                    #print("foiund")
                    if size_option==None:
                        size_option = t
                    found_subgoal = True
                    for s in range(max(0,t-size_option),t):
                        #print(trajectory[s][0])
                        self.input_set.add((trajectory[s][0][0], trajectory[s][0][1]))
                        
        self.terminal_states = [goal]
        
        # initialize QLearning matrix of option
        self.Q = np.zeros((8, 22, 4))
        #self.Q = np.random.rand(8,22,4)
        
        self.nb_visits = np.zeros((8, 22, 4))

        self.policy = self.Q.argmax(axis=1)

        self.gamma = gamma
        self.size_option = size_option
        
        
    def init_policy(self, trajectories):
        #computes the policy of an option

        #building dataset of samples from all trajectories (experience replay)
        #https: // datascience.stackexchange.com / questions / 20535 / what - is -experience - replay - and -what - are - its - benefits

        dataset = []
        for trajectory in trajectories:
            for (t,step) in enumerate(trajectory[:-1]):
                state = step[0]
                if state in self.terminal_states:
                    self.size_option = t
                    for s in range(max(0,t-self.size_option),t):
                        step2 = trajectory[s]
                        #print(f"trajectory is {trajectory[s+1][0]}")
                        dataset.append([step2[0],step2[1],trajectory[s+1][0]])
        dataset = np.array(dataset)
#         print("dataset 0 is")
#         print(dataset[0])
        # shuffling samples

        num_samples = dataset.shape[0]
        perm = np.random.permutation(num_samples)
        dataset = dataset[perm,:]

        # updating Q matrix of option

        for t in range(num_samples):
            #print(dataset[t,0])
            #print(dataset[0])

            state = dataset[t, 0]
            action =  int(dataset[t, 1])
            next_state = dataset[t, 2]

            reward_option = self.reward_option(next_state)

            learned_value = reward_option + self.gamma * self.Q[next_state[0], next_state[1], :].max()

            learning_rate = 1 / (self.nb_visits[state[0], state[1], action] + 1)
            self.Q[state[0], state[1], action] = (1 - learning_rate) * self.Q[state[0], state[1], action] + learning_rate * learned_value
            self.nb_visits[state[0], state[1], action] +=1

        # updating policy of option

        #self.policy = self.Q.argmax(axis=1)


    def take_action(self, state):
        
        self.epsilon-=0.00001
        
        #returns action to take following the policy from state

        if (state[0], state[1]) in self.input_set:
            if np.random.uniform(0, 1) > self.epsilon:
                random_max = np.random.choice(np.where(self.Q[state[0]][state[1]] == self.Q[state[0]][state[1]].max())[0])
                return random_max#np.argmax(self.Q[state[0]][state[1]])
            else:
                return np.random.choice(len(self.Q[state[0]][state[1]]))
        else:
            print("State not in the input set of option. Can't take action.")
            exit()


    def reward_option(self, next_state):
        #returs the reward of the option for reaching next_state
#         print(next_state)
#         print(self.input_set)
        if next_state in self.terminal_states:
            return 10
        elif not((next_state[0], next_state[1]) in self.input_set):
            return -1
        else:
            return -1
    