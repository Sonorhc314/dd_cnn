from enviroment_setup import *
import numpy as np
import math
from matplotlib import pyplot as plt

class DD():
    def __init__(self, domain):
        self.density=math.inf*np.ones((8, 22))
        self.positive_bags=[]
        self.negative_bags=[]
        self.one_bag_p=[]
        self.one_bag_n=[]
        self.domain=domain
        self.n = 5 #initiation set

    def PInstance(self, Point, Instance):
        distance = np.linalg.norm(np.array(Instance) - np.array(Point), 2)
        Prob = np.exp(-distance)
        return Prob

    # Function to calculate the probability of Point being the true concept given a positive bag
    def ppositive(self, Point, PBag):
        #print(PBag)
        size_pbag = len(PBag)
        temp = np.zeros(size_pbag)
        for i in range(size_pbag):
            temp[i] = self.PInstance(Point, PBag[i])
        Prob = 1 - np.prod(1 - temp)
        return Prob

    # Function to calculate the probability of Point being the true concept given a negative bag
    def pnegative(self, Point, NBag):
        size_nbag = len(NBag)
        temp = np.zeros(size_nbag)
        for i in range(size_nbag):
            temp[i] = self.PInstance(Point, NBag[i])
        Prob = np.prod(1 - temp)
        return Prob

    # Function to calculate the diverse density given a concept Point and bags PBag, NBag
    def densitydiv(self, Point, Pbags, Nbags):
        temp = []
        for bag in Pbags:
            temp.append(self.ppositive(Point, bag))
        
        if Nbags:
            for bag in Nbags:
                temp.append(self.pnegative(Point, bag))
        
        Prob = -np.sum(np.log(np.maximum(temp, 1e-8)))  # Obtain the log of the diverse density with overflow check
        return Prob
        
    def add_to_one_p_bag(self, state):
        self.one_bag_p.append(state)
        
    def add_to_one_n_bag(self, state):
        self.one_bag_n.append(state)
        
    def add_to_p_bags(self):
        #self.one_bag_p.pop() #remove last element(goal)
        self.positive_bags.append(self.one_bag_p)
        self.one_bag_p=[]
        
    def add_to_n_bags(self):
        self.one_bag_n.pop() #remove last element(goal)
        self.negative_bags.append(self.one_bag_n)
        self.one_bag_n=[]
    
    def update_desity(self):
        for row in range(len(self.density)):
            for col in range(len(self.density[0])):
                if self.domain[row][col]!=1:
                    self.density[row][col]=self.densitydiv((row,col),self.positive_bags,self.negative_bags)
    
    def generate_shade(self, i, j):
        # Generate a random grayscale value between 0 and 1 
        gray_shade = self.density[i][j]
        return (gray_shade, gray_shade, gray_shade)
    
    def plot_gridworld(self):
        # Create a grid with random grayscale shades
       # print("asasdwdadwadw")
        grid = np.zeros((len(self.density), len(self.density[0]), 3))
        max_ij=(0,0)
        max_val=math.inf
        for i in range(len(self.density)):
            for j in range(len(self.density[0])):
                grid[i, j] = self.generate_shade(i,j)
                if(grid[i,j,0]<max_val and (i!=1 or j!=1)):
                    max_ij=(i,j)
                    max_val=grid[i,j,0]
                
        
                #print(grid[i][j])
                
        #print(grid)
        # Plot the gridworld
        #print(grid)
        #print(f"max is {grid[max_ij[0], max_ij[1]]}")
        #grid[max_ij[0], max_ij[1]]=(10000,10000,10000)
        plt.figure(figsize=(len(self.density), len(self.density[0])))
        plt.imshow(grid, cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        #print(grid)
    