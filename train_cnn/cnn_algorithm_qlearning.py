from q_leaning_cnn_algorithm import *

current_option_iteration=0
option_iteration=10
threshold_peak=0.6

rewards_algorithm = np.zeros(iterations)

for agent_iteration in range(agents):
    
    env.reset(start_same=True)
    domain = env.env_master.domain
    
    print(domain.shape)
    #domain = domain.astype('float32')

    #print(my_test)
    
    #print(domain)
    domains=[]
    domains.append(domain.tolist())
    domains = np.array(domains)
    #print(domains)
    

    map_encoded = autoencoder_conv.encoder(domains).numpy()
    density_map_decoded = autoencoder_conv.decoder(map_encoded)[0].numpy()
    
    
    plt.imshow(domain)
    plt.show()
    plt.clf()
        
    #file.close()
    
    gold_positions=find_goal_location(domain)
    agent = QLearningAlgorithmAgent(gold_positions, env)
    start_xy = find_agent_location(domain)
    
    #print(density_map_decoded)
    plt.imshow(density_map_decoded)
    plt.show()
    plt.clf()
    
    selected_subgoals=[]
    for row in range(9):
        for col in range(23):
            if density_map_decoded[row][col] <= threshold_peak:
                if [row, col] != taboo_concepts:
                    selected_subgoals.append([row, col])

    for iteration in range(iterations):
        #print("iteration")
        env.reset(start_same=True)
        domain = env.env_master.domain
        
        agent.update_env(env)
        state = find_agent_location(domain)
        #print("iteration1")
        trajectories = agent.choose_action(state)
        
        #print("iteration2")
        total_reward = agent.total_reward
        

        taboo_concepts = state
        
        all_trajectories = trajectories
        
        #print(density.density[5][5])
        
        #print(taboo_concepts)

        if current_option_iteration==option_iteration:
            agent.vipe_options()
            
            option_iteration=30
                            
            #select chosen subgoal candidate, and create option (max one per run)
            #print(selected_subgoals)
            if len(selected_subgoals)>0:
                if len(all_trajectories) < 100:
                    memory = all_trajectories
                else:
                    memory = all_trajectories[-100:] #last trajectories
                    
                #print(memory)
                
            #agent.num_actions=4
                
            for subgoal in selected_subgoals:
                option = Option(env, subgoal, memory)
                option.init_policy(memory) #compute option policy with experience replay
                agent.add_option(option)
                
            current_option_iteration=0

        current_option_iteration+=1
        agent.episode_t_renew()
        rewards_algorithm[iteration] += total_reward
        print(total_reward)
        agent.total_reward=0
        #print(total_reward_nooptions)
        total_reward=0