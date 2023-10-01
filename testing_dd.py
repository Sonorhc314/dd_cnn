from q_learning_options import *
from option import *
from enviroment_setup import *
import numpy as np
import math
from matplotlib import pyplot as plt
import random



def find_agent_location(array):
    loc = np.where(array == 2)
    try:
        return [loc[0][0], loc[1][0]]
    except:
        print("can't find the agent ...")
        return [None, None]
    
def find_goal_location(array):
    loc = np.where(array == 3)
    try:
        return [loc[0][0], loc[1][0]]
    except:
        return None
    
file_path1 = 'gridworlds.txt'
file_path2 = 'densities.txt'

iterations=60
agents=1
option_iteration=10
current_option_iteration=0
# lambd = 0.9
# threshold_peak = 0.9999 * lambd/(1-lambd) 
threshold_peak = 0

rewards = np.zeros(iterations)
rewards_nooptions = np.zeros(iterations)

for agent_iteration in range(agents):
        
    summed_density=np.zeros((8, 22))
    env.reset(start_same=True)
    domain = env.env_master.domain
    
    plt.imshow(domain)
    plt.show()
    plt.clf()
        
    #file.close()
    
    gold_positions=find_goal_location(domain)
    agent = QLearningOptionsAgent(gold_positions, env)
    agent_nooptions = QLearningOptionsAgent(gold_positions, env)
    density = DD(domain)
    density_nooptions = DD(domain)
    start_xy = find_agent_location(domain)

    all_trajectories = []
    
    #true_maps.append(domain)
    selected_subgoals=[]
    

    for iteration in range(iterations):
        #print("iteration")
        env.reset(start_same=True)
        domain = env.env_master.domain
        
        agent.update_env(env)
        state = find_agent_location(domain)
        #print("iteration1")
        density, trajectories = agent.choose_action(state, density)
        
        #----------------------------
        env.reset(start_same=True)
        agent_nooptions.update_env(env)
        domain = env.env_master.domain
        state = find_agent_location(domain)
        density_nooptions, trajectories_nooptions = agent_nooptions.choose_action(state, density_nooptions)
        #----------------------------
        
        #print("iteration2")
        total_reward = agent.total_reward
        total_reward_nooptions = agent_nooptions.total_reward
        
        # print(total_reward_nooptions)
        # print(total_reward)
        # print("")
        
        density.add_to_p_bags()
        density.update_desity()

        taboo_concepts = state
        
        all_trajectories = trajectories
        
        #print(density.density[5][5])
        
        #print(taboo_concepts)

        if current_option_iteration==option_iteration:
            option_iteration=10
            density.plot_gridworld()
            for row in range(8):
                for col in range(22):
                    if density.density[row][col] <= threshold_peak:
                        #if not([row, col]  taboo_concepts): #admissible states : not too close to each start or end of all succ traj
                        if [row, col] != taboo_concepts:
                            selected_subgoals.append([row, col])
                            

            #select chosen subgoal candidate, and create option (max one per run)
            #print(selected_subgoals)
            if len(selected_subgoals)>0:
                chosen_subgoal = random.choice(selected_subgoals)
                #print(chosen_subgoal)
                # trajectories that will be mined to create the option
                if len(all_trajectories) < 100:
                    memory = all_trajectories
                else:
                    memory = all_trajectories[-100:] #last trajectories
                    
                #print(memory)

                option = Option(env, chosen_subgoal, memory)
                option.init_policy(memory) #compute option policy with experience replay
                agent.add_option(option)
#                 print("\nOption created. Terminal states : ")
#                 for s in option.terminal_states: env.describe_state(s)
            current_option_iteration=0

        current_option_iteration+=1
        agent.episode_t_renew()
        rewards[iteration] += total_reward
        rewards_nooptions[iteration] += total_reward_nooptions
        #print(total_reward)
        agent.total_reward=0
        agent_nooptions.total_reward=0
        #print(total_reward_nooptions)
        total_reward=0
        total_reward_nooptions=0
        
    summed_density+=density.density

rewards_options = rewards