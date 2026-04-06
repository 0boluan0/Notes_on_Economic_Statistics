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

########################
## Plotting many lines 
########################
nVals = []
linear = []
quadratic = []
cubic = []
exponential = []

for i in range(0, 30):
    nVals.append(i)
    linear.append(i)
    quadratic.append(i**2)
    cubic.append(i**3)
    exponential.append(1.5**i)

# #### Plotting one line
# plt.plot(nVals, linear)


# ##### order of data points matters
testSamples = [0,5,3,6,15,2,1,4,25,20,7,21,22,23,9,8,24,10,12,11]
testValues =  [0,25,9,36,225,4,1,16,625,400,49,441,484,529,81,64,576,100,144,121]
## plot connects the points
# plt.plot(testSamples, testValues)
## scatter plot does not connect the points
# plt.scatter(testSamples, testValues)

# ##### Plotting many lines
# plt.plot(nVals, linear)
# plt.plot(nVals, quadratic)
# plt.plot(nVals, cubic)
# plt.plot(nVals, exponential)


# ###### Plotting two lines on one plot
# plt.figure('expo')
# plt.plot(nVals, exponential)
# plt.figure('lin')
# plt.plot(nVals, linear)
# plt.figure('quad')
# plt.plot(nVals, quadratic)
# plt.figure('cube')
# plt.plot(nVals, cubic)
# newExpo = []
# for i in range(30):
#     newExpo.append(1.6**i)
# plt.figure('expo')
# plt.plot(nVals, newExpo)


################
## Temperature with axes options
################
###### Plotting temperatures and changing xaxis
months = range(1, 13, 1)
temps = [28,32,39,48,59,68,75,73,66,54,45,34]
# plt.plot(months, temps)

# # ## Add axes, labels, and a title
# plt.title('Ave. Temperature in Boston')
# plt.xlabel('Month')
# plt.ylabel('Degrees F')

# # #### Start axis at 1 to 12
# plt.xlim(1, 12)
# # ### Change x axes labels
# plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12))
# # #### Change x axes labels to custom labels
# plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12),
#             ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))

# # #### add/remove grid lines
# plt.grid()


###################################
## Temperatures for many cities 
###################################

###### Plotting multiple lines with labels
# months = range(1, 13, 1)
# boston = [28,32,39,48,59,68,75,73,66,54,45,34]
# plt.plot(months, boston, label = 'Boston')
# phoenix = [54,57,61,68,77,86,91,90,84,73,61,54]
# plt.plot(months, phoenix, label = 'Phoenix')
# # Add labels and title
# plt.title('Ave. Temperatures')
# plt.xlabel('Month')
# plt.ylabel('Degrees F')
# # Change x axis labels to custom labels
# plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12),
#           ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))

# plt.legend(loc = 'best', fontsize=20) # position it automatically

###### Plotting multiple lines and changing their line style
# months = range(1, 13, 1)           
# boston = [28,32,39,48,59,68,75,73,66,54,45,34]
# plt.plot(months, boston, 'b-', label = 'Boston')
# phoenix = [54,57,61,68,77,86,91,90,84,73,61,54]
# plt.plot(months, phoenix, 'r--', label = 'Phoenix')
# msp = [16,19,34,48,59,70,75,73,64,60,37,21]
# plt.plot(months, msp, 'g-.', label = 'Minneapolis')
# plt.legend(loc = 'best', fontsize=20)
# plt.title('Ave. Temperatures')
# plt.xlabel('Month')
# plt.ylabel('Degrees F')
# plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12),
#           ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))

# # ###### Plotting with keywords (same plot as below)
# months = range(1, 13, 1)           
# boston = [28,32,39,48,59,68,75,73,66,54,45,34]
# plt.plot(months, boston, label = 'Boston',\
#           color = 'b', linestyle = '-')
# phoenix = [54,57,61,68,77,86,91,90,84,73,61,54]
# plt.plot(months, phoenix, label = 'Phoenix',\
#           color = 'r', linestyle = '--')
# msp = [16,19,34,48,59,70,75,73,64,60,37,21]
# plt.plot(months, msp, label = 'Minneapolis',\
#           color = 'g', linestyle = '-.')
# plt.legend(loc = 'best', fontsize=20)
# plt.title('Ave. Temperatures')
# plt.xlabel('Month')
# plt.ylabel(('Degrees F'))
# plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12),
#           ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))

###### Plotting with styled markers (same plot as above)
# months = range(1, 13, 1)           
# boston = [28,32,39,48,59,68,75,73,66,54,45,34]
# plt.plot(months, boston, '.b-', label = 'Boston')
# phoenix = [54,57,61,68,77,86,91,90,84,73,61,54]
# plt.plot(months, phoenix, 'or--', label = 'Phoenix')
# msp = [16,19,34,48,59,70,75,73,64,60,37,21]
# plt.plot(months, msp, '*g-.', label = 'Minneapolis')
# plt.legend(loc = 'best', fontsize=20)
# plt.title('Ave. Temperatures')
# plt.xlabel('Month')
# plt.ylabel(('Degrees F'))
# plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12),
#           ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))

###### Plotting with keywords, change width
# months = range(1, 13, 1)           
# boston = [28,32,39,48,59,68,75,73,66,54,45,34]
# plt.plot(months, boston, label = 'Boston',\
#           color = 'b', linestyle = '-', linewidth = 2)
# phoenix = [54,57,61,68,77,86,91,90,84,73,61,54]
# plt.plot(months, phoenix, label = 'Phoenix',\
#           color = 'r', linestyle = '--', linewidth = 10)
# msp = [16,19,34,48,59,70,75,73,64,60,37,21]
# plt.plot(months, msp, label = 'Minneapolis',\
#           color = 'g', linestyle = '-.', linewidth = 20)
# plt.legend(loc = 'best', fontsize=20)
# plt.title('Ave. Temperatures')
# plt.xlabel('Month')
# plt.ylabel(('Degrees F'))
# plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12),
#           ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))

###### Using subplots
# months = range(1, 13, 1)           
# boston = [28,32,39,48,59,68,75,73,66,54,45,34]
# plt.subplot(2,1,1)
# # plt.ylim(0, 100)
# plt.plot(months, boston, 'b-')
# plt.ylabel('Degrees F')
# plt.title('Boston vs. Phoenix')
# plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12),
#           ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
# phoenix = [54,57,61,68,77,86,91,90,84,73,61,54]
# plt.subplot(2,1,2)
# # plt.ylim(0, 100)
# plt.plot(months, phoenix, 'r--')
# plt.ylabel('Degrees F')
# plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12),
#           ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))

###### Using subplots
# months = range(1, 13, 1)           
# boston = [28,32,39,48,59,68,75,73,66,54,45,34]
# plt.subplot(2,2,1)
# plt.ylim(0, 100)
# plt.plot(months, boston, 'b-')
# plt.ylabel('Degrees F')
# plt.title('Boston')
# plt.xticks((1,3,5,7,9,11),('Jan','Mar','May','Jul','Sep','Nov'))

# phoenix = [54,57,61,68,77,86,91,90,84,73,61,54]
# plt.subplot(2,2,2)
# plt.ylim(0, 100)
# plt.plot(months, phoenix, 'r--')
# plt.title('Phoenix')
# plt.xticks((1,3,5,7,9,11),('Jan','Mar','May','Jul','Sep','Nov'))

# msp = [16,19,34,48,59,70,75,73,64,60,37,21]
# plt.subplot(2,2,3)
# plt.ylim(0, 100)
# plt.plot(months, msp, 'g-.')
# plt.ylabel('Degrees F')
# plt.title('Minneapolis')
# plt.xticks((1,3,5,7,9,11),('Jan','Mar','May','Jul','Sep','Nov'))

####################################
## US Population Example 
####################################

###### Read file data and plot it
def getUSPop(fileName):
    inFile = open(fileName, 'r')
    dates, pops = [], []
    for l in inFile:
        line = ''
        for c in l:
            if c in '0123456789 ':
                line += c
        line = line.split(' ')
        dates.append(int(line[0]))
        pops.append(int(line[1]))
    return dates, pops

# dates, pops = getUSPop('lec25_USPopulation.txt')
# plt.plot(dates, pops)
# plt.title('Population in What Is Now U.S.\n' +\
#           '(Native Am. Excluded Before 1860)')
# plt.xlabel('Year')
# plt.ylabel('Population')

####### Change the scale to semilog
# plt.semilogy()   


####################################
## Country Population Example 
####################################

###### Read file data from many countries and plot it
def getCountryPops(fileName):
    inFile = open(fileName, 'r')
    pops = []
    for l in inFile:
        line = l.split('\t')
        l = line[2]
        pop = ''
        for c in l:
            if c in '0123456789':
                pop += c
        pops.append(int(pop))
    return pops

pops = getCountryPops('lec25_countryPops.txt')


# ## Plot populations 
# plt.plot(pops)
# plt.title('Population Size of Countries July 2017')
# plt.ylabel('Population')
# plt.xlabel('Country Rank Based on Size')
# plt.semilogy()

# ## Investigate the first digits
pops = getCountryPops('lec25_countryPops.txt')
firstDigits = []
for p in pops:
    firstDigits.append(int(str(p)[0]))
# print(firstDigits)    

### Plot the fist digits, as found in order in the file
# plt.plot(firstDigits)

### Plot the histogram to show Benford's law
# plt.hist(firstDigits, bins = 9)


####################################
## Comparing Cities Example 
####################################
def getCities():
    inFile = open('lec25_temperatures.csv')
    cities = []
    for l in inFile:
        c = l.split(',')[0]
        if c not in cities:
            cities.append(c)
    return cities

def CtoF(c):
    return (c * 9/5) + 32

def getTempsForCity(city):
    inFile = open('lec25_temperatures.csv')
    temps = []
    dates = []
    for l in inFile:
        data = l.strip().split(',')
        c = data[0]
        tem = data[1]
        date = data[2]
        if c == city:
            temps.append(CtoF(float(tem)))
            dates.append(date)
    return temps, dates

def getAverageTemps():
    cities = getCities()[1:]
    xPts = range(len(cities))
    aveTemp = []
    cityLabels = []
    for c in cities:
        temps, dates = getTempsForCity(c)
        aveTemp.append(sum(temps)/len(temps))
        cityLabels.append(c[0:2])
        print(c[0:2], sum(temps)/len(temps))
        
    plt.figure('Temps')
    plt.scatter(xPts, aveTemp)
    plt.title('Ave. Temperatures')
    plt.xlabel('City')
    plt.ylabel(('Degrees F'))
    plt.xticks(xPts, cityLabels)

## print average temperatures for all cities (and plot them)
# getAverageTemps()

def getAvgTempForYear(tem, dat, y):
    yearlyTemps = []
    for i in range(len(tem)):
        if y == dat[i][:4]:
            yearlyTemps.append(tem[i])
    return sum(yearlyTemps)/len(yearlyTemps), y

## List of temps and a corresponding list of dates for a specific city
temps,dates = getTempsForCity('SEATTLE')
## zip makes tuples of (0th elem from temps and 0th from dates)
##                     (1st from temps and 1st from dates)
##                     (ith from temps and ith from dates), etc.
# print(list(zip(temps, dates)))

## average temperatures for one year
# print(getAvgTempForYear(temps, dates, '1961'))

            
##### plot average temperatures for a few different cities
def getTempsByYearForCity(city):
    temps, dates = getTempsForCity(city)
    averages = []
    years = []
    for y in range(1961,2016):
        tem = getAvgTempForYear(temps, dates, str(y))[0]
        averages.append(tem)
        years.append(str(y))
    return averages, years

if False:
    plt.close()
    for c in ('BOSTON','PHOENIX', 'MIAMI', 'SAN DIEGO'):
        
        av, yr = getTempsByYearForCity(c)
        xPts = range(len(yr))
        plt.figure('Temps by City')
        plt.plot(xPts, av, label = c)
        plt.title('Ave. Temperatures')
        plt.xlabel('Years since 1961')
        plt.ylabel(('Degrees F'))
        plt.legend(loc = 'best')
        
        
##### plot yearly average temperature for a city, including range

def getTempsForYearRange(tem, dat, y):
    yearly = []
    for i in range(len(tem)):
        if y == dat[i][:4]:
            yearly.append(tem[i])
    return sum(yearly)/len(yearly), max(yearly), min(yearly), y
            
def getTempsByYearForCityRange(city):
    temps, dates = getTempsForCity(city)
    averages = []
    maxes = []
    mins = []
    years = []
    for y in range(1961,2000):
        tem, mx, mn, y = getTempsForYearRange(temps, dates, str(y))
        averages.append(tem)
        maxes.append(mx)
        mins.append(mn)
        years.append(str(y))
    return averages, maxes, mins, years

if False:
    plt.close()
    for c in ('BOSTON','SAN DIEGO', 'MIAMI'):  # try for BOSTON, SAN DIEGO, MIAMI
        av, mx, mn, yr = getTempsByYearForCityRange(c)
        xPts = range(len(yr))
        plt.figure('Temps by City: '+c)
        plt.ylim(0, 100)
        plt.plot(xPts, av, label = 'mean')
        plt.plot(xPts, mx, label = 'max')
        plt.plot(xPts, mn, label = 'min')        
        plt.title('Temperature Range: ' + c)
        plt.xlabel('Years since 1961')
        plt.ylabel(('Degrees F'))
        plt.legend(loc = 'best')
        
        
## look at number of days with a particular temperature by city

def getDayDistributionForCity(city, year):
    # assume a range of temperatures from 0 to 100
    temps, dates = getTempsForCity(city)
    newTemps = []
    for i in range(len(dates)):
        if year == dates[i][:4]:
            newTemps.append(temps[i])
    ## want to map temperature to number of occurences
    d = [0]*100
    for t in newTemps:
        tRound = round(t)
        d[tRound] += 1
    return d

if False:
    plt.close()
    for c in ('BOSTON','SAN DIEGO', 'MIAMI'):  # try for BOSTON, SAN DIEGO, MIAMI
        ans = getDayDistributionForCity(c, '1961')
        temps = []
        for i in range(100):
            temps.append(i)
        plt.figure('Distribution of Temps by City: '+c)
        plt.bar(temps, ans)
       
        plt.title('Temperature Distribution: ' + c)
        plt.xlabel('Temperature')
        plt.ylabel(('Number of days'))


if False:
    plt.close()
    for c in ('SAN DIEGO',):  # try for BOSTON, SAN DIEGO
        plt.figure('Distribution of Temps by City')
        for y in ('1961','2015'): # also check ('1961','2015')
            ans = getDayDistributionForCity(c, y)
            temps = []
            for i in range(100):
                temps.append(i)
            if y == '1961':
                plt.bar(temps, ans, color = 'blue', label = y, alpha=0.5)
            else:
                plt.bar(temps, ans, color = 'red', label = y, alpha=0.5)
       
        plt.title('Temperature Distribution: ' + c)
        plt.xlabel('Temperature')
        plt.ylabel(('Number of days'))
        plt.legend(loc = 'best')

if False:
    plt.close()
    for c in ('BOSTON',):  # try for BOSTON, SAN DIEGO
        plt.figure('Distribution of Temps by City')
        for y in ('1961', '2015'):
            ans = getDayDistributionForCity(c, y)
            temps = []
            for i in range(100):
                temps.append(i)
            if y == '1961':
                plt.subplot(2,1,1)
                plt.ylim(0,20)
                plt.xlabel('Temperature')
                plt.ylabel(('Number of days'))
                plt.bar(temps, ans, color = 'blue', label = y)
            else:
                plt.subplot(2,1,2)
                plt.ylim(0,20)
                plt.xlabel('Temperature')
                plt.ylabel(('Number of days'))
                plt.bar(temps, ans, color = 'red', label = y)
       
        #plt.title('Temperature Distribution: ' + c)
        plt.legend(loc = 'best')

    


    

