import random
import time
import matplotlib.pyplot as plt

#set line width
plt.rcParams['lines.linewidth'] = 2
#set font size for titles
plt.rcParams['axes.titlesize'] = 16
#set font size for labels on axes
plt.rcParams['axes.labelsize'] = 16
#set size of numbers on x-axis
plt.rcParams['xtick.labelsize'] = 10
#set size of numbers on y-axis
plt.rcParams['ytick.labelsize'] = 10
#set size of ticks on x-axis
plt.rcParams['xtick.major.size'] = 5
#set size of ticks on y-axis
plt.rcParams['ytick.major.size'] = 5
#set size of markers
plt.rcParams['lines.markersize'] = 10
#set number of examples shown in legends
plt.rcParams['legend.numpoints'] = 1
#set the font size globally
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams['axes.labelsize'] = 26 
plt.rcParams['axes.titlesize'] = 26 
plt.rcParams["figure.figsize"] = (15,10)

## -------------------------- ##
## Experiment is rolling a die once.
## Repeat the experiment 10000 times. This is the simulation!
## Count how many times a 1 shows up among 10000 sims.
## Find the prob of rolling a 1.
def prob_dice(side):
    dice = ['.',':',':.','::','::.',':::']
    Nsims = 10000
    one_count = 0
    for i in range(Nsims):
        roll = random.choice(dice)
        if roll == side:
            one_count += 1
    print(one_count/Nsims)

# prob_dice('.')
# prob_dice('::')


## -------------------------- ##
## Experiment is rolling a die 7 times and keeping track of ::
## Repeat the experiment 10000 times. This is the simulation!
## Count how many times a 1 shows up at least 3 times.
## Find the prob of a die coming up as 1 at least 3 times out of 7 rolls.

def prob_dice_atleast(Nrolls, n_at_least):
    """ Nrolls is how many dice rolls to do
        n_at_least is how many times '::' should come up in the Nrolls """
    dice = ['.',':',':.','::','::.',':::']
    Nsims = 1000000
    how_many_matched = []
    for i in range(Nsims):
        matched = 0
        for i in range(Nrolls):
            roll = random.choice(dice)
            if roll == '::':
                matched += 1
        how_many_matched.append(matched)
    
    at_least_this = 0
    for i in how_many_matched:
        if i >= n_at_least:
            at_least_this += 1
    print(at_least_this/len(how_many_matched))


# prob_dice_atleast(7, 3)        
# prob_dice_atleast(1, 1) # should match prob_dice('::')       
    

## -------------------------- ##
## Water runs through a faucet somewhere between 
## 1 gallons per minute and 3 gallongs per minute. 
## 1. Experiment is getting a random value between 1 and 3. Then 
##    calculating the time it takes to fill the pool with that value.
## 2. Repeat the experiment 10000 times. This is the simulation!
## 3. Print the average time it takes to fill the pool. 

def fill_pool(size):
    flow_rate = []
    fill_time = []
    Npoints = 10000
    for i in range(Npoints):
        r = 1+2*random.random() # random number between 1 and 3
        flow_rate.append(r)
        fill_time.append(size/r)

    print('avg flow_rate:', sum(flow_rate)/len(flow_rate))
    print('avg fill_time', sum(fill_time)/len(fill_time))
    plt.figure()    
    plt.scatter(range(Npoints),flow_rate,s=1) # try sorted(flow_rate)
    plt.figure()    
    plt.scatter(range(Npoints),fill_time,s=1) # try sorted(fill_time)

# fill_pool(2)
# fill_pool(600)







