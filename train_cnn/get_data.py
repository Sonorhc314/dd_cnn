import copy
from .. import testing_dd
from testing_dd import *
#getting data from files

file_path1 = 'gridworlds.txt'
file_path2 = 'densities.txt'

with open(file_path1, 'r') as textFile:
    index=0
    map_array=[]
    one_array = []
    for line in textFile:
        lines = line.split()
        lines = [eval(i) for i in lines]
        if len(lines)!=0:
            one_array.append(lines)
        else:
            map_array.append(one_array)
            one_array = []
        index+=1
        
with open(file_path2, 'r') as textFile:
    index=0
    dd_array=[]
    one_array = []
    for line in textFile:
        lines = line.split()
        lines = [(eval(i)) for i in lines]
        #print(lines)
        if len(lines)!=0:
            lines.append(1.0) #forgot about the right wall
            one_array.append(lines)
        else:
            lines_floor = [1.0 for i in range(23)] #forgot about the floor
            one_array.append(lines_floor)
            dd_array.append(one_array)
            one_array = []
        index+=1
        
processed_density=copy.deepcopy(dd_array)
for array_index in range(len(processed_density)):
    for row in range(len(processed_density[array_index])):
        for col in range(len(processed_density[array_index][row])):
            #a=1
#             if processed_density[array_index][row][col]==1:
#                 processed_density[array_index][row][col]=1
            if processed_density[array_index][row][col]>0.45:
                processed_density[array_index][row][col]=1
    processed_density[array_index]


# clipped_data = np.clip(dd_array[9], 0, 1)
# #print(clipped_data)        
print(processed_density[0])
density.density=processed_density[632]
density.plot_gridworld()

density.density=dd_array[632]
density.plot_gridworld()

# print(clipped_data.tolist())
# print(np.count_nonzero(clipped_data==0))

# clipped_data = np.clip(map_array[0], 0, 1)
# print(len(map_array[0]))
plt.imshow(np.array(map_array[632]))
plt.show()
plt.clf()