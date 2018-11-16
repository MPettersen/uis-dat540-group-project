import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.neural_network import MLPClassifier as MLP

import gym
import numpy as np
import pandas as pd
import time

env = gym.make('CartPole-v0')

POPULATION_SIZE = 50
GENERATIONS = 15
MUTATION_RATE = 0.001
ITER_NUM = np.inf 

def mlp():
    return MLP(
    batch_size=1, max_iter=1, solver='sgd', activation='relu', learning_rate='invscaling', 
    hidden_layer_sizes=1, random_state=1
    )

neural_network = np.full(POPULATION_SIZE,None)
for i, _ in enumerate(neural_network):
    neural_network[i] = mlp()
population = np.full(POPULATION_SIZE, None)

for i, j in enumerate(neural_network):
    population[i] = j.partial_fit(np.array([env.observation_space.sample()]),
    np.array([env.action_space.sample()] ),
    classes = np.arange(env.action_space.n))
    #print(population[i].coefs_)

score_set = np.full(POPULATION_SIZE, 0)
score = score_set
for i in range(GENERATIONS):
    for k, j in enumerate(population):
        env.reset()
        reward_count = 0
        action = env.action_space.sample()
        action_prev = action
        action_count = 0
        #print(j.coefs_)
        for _ in range(1000000):
            #env.render()
            #time.sleep(0.05)
            observation, reward, done, _ = env.step(action)
            reward_count += reward
            if done:
                break
            action_predict = j.predict(observation.reshape(1,-1))
            action = action_predict[0]
            if action == action_prev:
                action_count += 1
                if action_count == 5:
                    action = env.action_space.sample()
            action_prev = action
            population[i] = j.partial_fit(observation.reshape(1,-1), action_predict)
        score_set[k] = reward_count
    score = score_set
    score_set = (score_set / score_set.sum()).cumsum()
    print(score.mean())
    
    new_population_num = 0
    new_generation = population
    while(new_population_num < POPULATION_SIZE):
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        parent1 = None
        parent2 = None
        for k, j in enumerate(score_set):
            if j > rand1:
                parent1 = neural_network[k]
                #print(neural_network[k].coefs_)
                break
        for k, j in enumerate(score_set):
            if j > rand2:
                parent2 = neural_network[k]
                break
        child_num = 0
        child = None
        while(child_num < 2):
            child = parent1
            #print(parent1.coefs_[0])
            weight_start = np.random.randint(len(parent1.coefs_))
            weight_end = np.random.randint(len(parent1.coefs_))
            if weight_start > weight_end:
                temp = weight_start
                weight_start = weight_end
                weight_end = temp
            bias_start = np.random.randint(len(parent1.intercepts_))
            bias_end = np.random.randint(len(parent1.intercepts_))
            if bias_start > bias_end:
                temp = bias_start
                bias_start = bias_end
                bias_end = temp
            child.coefs_[weight_start:weight_end] = parent2.coefs_[weight_start:weight_end]
            child.intercepts_[bias_start:bias_end] = parent2.intercepts_[bias_start:bias_end]
            
            new_generation[new_population_num] = child

            child_num += 1
            new_population_num += 1
    population = new_generation



print(score.mean())
    