#######################################
## EXAMPLE: getting grades using lists, do NOT do it this way...
#######################################
def get_grade_list(student, name_list, grade_list):
    """ student is a str
        name_list and grade_list are lists """
    i = name_list.index(student)
    grade = grade_list[i]
    return (student, grade)

names = ['Ana', 'John', 'Denise', 'Katy']
grade = ['B', 'A', 'A', 'B']
# can add more lists for course number, etc.
# print(get_grade_list('John', names, grade))

#######################################
## EXAMPLE: getting grades using list of lists, do NOT do it this way...
#######################################
def get_grades(who, what, data):
    """ who and what are strings
        data is a list containing lists of student info """
    for stud in data:
        if stud[0] == who:
            for info in stud[1:]:
                if info[0] == what:
                    return who, info
  
eric = ['eric', ['ps', [8, 4, 5]], ['mq', [6, 7]]]
ana = ['ana', ['ps', [10, 10, 10]], ['mq', [9, 10]]]
john = ['john', ['ps', [7, 6, 5]], ['mq', [8, 5]]]

grades = [eric, ana, john]

# print(get_grades('eric', 'mq', grades))
# print(get_grades('ana', 'ps', grades))

#######################################
## EXAMPLE: getting grades using dictionaries
#######################################
def get_grade_dict(student, grade_dict):
    return grade_dict[student]

# d = {'Ana':(2.00,'B'), 'John':(6.0001,'A'), 'Denise':(20.002,'A'), 'Katy':(9.01,'B')}
# print(get_grade_dict('John', d))


#####################
## DICTIONARY OPERATIONS 
######################
# # create dicts
# my_dict = {}
# d = {4:16}
# grades = {'Ana':'B', 'Matt':'A', 'John':'B', 'Katy':'A'}

# # get dict entries
# grades = {'Ana':'B', 'Matt':'A', 'John':'B', 'Katy':'A'}
# print(grades['John'] )
# #print(grades['Grace'])  # this gives an error since 'Grace' is not in dict

# # add, edit, remove entries
# grades = {'Ana':'B', 'Matt':'A', 'John':'B', 'Katy':'A'}
# grades['Grace'] = 'A' 
# grades['Grace'] = 'C' 	
# del(grades['Ana'])

# # test if keys are in dict
# grades = {'Ana':'B', 'Matt':'A', 'John':'B', 'Katy':'A'}
# print('John' in grades)
# print('Daniel' in grades)
# print('B' in grades)

# # iterate over keys and values
# grades = {'Ana':'B', 'Matt':'A', 'John':'B', 'Katy':'A'}
# print(grades.keys())
# print(list(grades.keys()))

# print(grades.values())
# print(list(grades.values()))

# # iterate over keys and values at the same time
# grades = {'Ana':'B', 'Matt':'A', 'John':'B', 'Katy':'A'}
# print(grades.items() )
# print(list(grades.items()))

# for k,v in grades.items():
#     print(f"key {k} has value {v}")


################## YOU TRY IT #########################
def find_grades(grades, students):
    """ grades is a dict mapping student names (str) to grades (str)
        students is a list of student names 
    Returns a list containing the grades for students (in the same order) """
    # your code here
  

d = {'Ana':'B', 'Matt':'C', 'John':'B', 'Katy':'A'}
# print(find_grades(d, ['Matt', 'Katy'])) # returns ['C', 'A']

########################################################


################## YOU TRY IT #########################
def find_in_L(Ld, k):
    """ L is a list of dicts
        k is an int
    Returns True if k is a key in any dicts of L and False otherwise """
    # your code here

  
d1 = {1:2, 3:4, 5:6}
d2 = {2:4, 4:6}
d3 = {1:1, 3:9, 4:16, 5:25}

# print(find_in_L([d1, d2, d3], 2))  # returns True
# print(find_in_L([d1, d2, d3], 25))  # returns False

########################################################


########################### YOU TRY IT ######################
def count_matches(d):
    """ d is a dict
    Returns how many entries in d have the key equal to its value """
    # your code here


d = {1:2, 3:4, 5:6}
# print(count_matches(d))   # prints 0

d = {1:2, 'a':'a', 5:5}
# print(count_matches(d))   # prints 2

##############################################################

############# YOU TRY IT ###################
my_d ={'Ana':{'mq':[10], 'ps':[10,10]}, 
       'Fredo':{'ps':[7,8], 'mq':[8]},
       'Eric':{'mq':[3], 'ps':[0]}      }

def get_average(data, what):
    """ data is a dict like my_d above
        what is 'ps' or 'mq'
        Returns the average of all elements in data that match 'what' """
    all_data = []
    for stud in data.keys():
        pass
        # Which one of the below is correct? 
        # A) all_data = all_data + data[stud][what]
        # B) all_data.append(data[stud][what]) 
        # C) all_data = all_data + data[stud[what]]
        # D) all_data.append(data[stud[what]]) 

    return sum(all_data)/len(all_data)

# print(get_average(my_d, 'mq') )   # prints 7.0

###########################################################

#######################################
## COMPLEX EXAMPLE: frequency dictionary of song lyrics
#######################################
song = "RAH RAH AH AH AH ROM MAH RO MAH MAH"

def generate_word_dict(song):
    """ song is a string
    Returns a dictionary whose:
    * keys are song words
    * values are the frequency of each key in song
    """
    # remove special characters and convert to lowercase
    song_words = song.lower()
    words_list = song_words.split()
    word_dict = {}
    for w in words_list:
        if w in word_dict:
            # seen word again, so add one to frequency
            word_dict[w] += 1
        else:
            # first time seeing word, insert a dict entry with freq 1
            word_dict[w] = 1
    # return is a dict mapping str:int like {'word1':1, 'word2':3}
    return word_dict

word_dict = generate_word_dict(song)
# print(word_dict)
     
def find_frequent_word(word_dict):
    """ word_dict is a frequency dict mapping string:int
    Using word_dict, returns a tuple of 
    (list of keys in word_dict with the max freq, max freq)
    """
    # a list in case there is more than one word occuring that often
    words = []
    highest = max(word_dict.values())  # this is an int
    # loop to find words occurring with `highest` freq
    for k,v in word_dict.items():
        # k is a word and v is its frequency
        if v == highest:
            # word in dict has a value that matches `highest` so append it
            words.append(k)
    # return looks like (['word1', 'word2'], 4)
    return (words, highest)
    
most_freq = find_frequent_word(word_dict)
# print(most_freq)


def occurs_often(word_dict, x):
    """ word_dict is a frquency dict
        x is an int
    Side effect warning, this function mutates word_dict here modifies word_dict.

    Returns the list of tuples in order of highest freq to lowest freq > x. 
    Each tuple is (list of keys in word_dict with some freq, some freq)
    """
    freq_list = []
    word_freq_tuple = find_frequent_word(word_dict)
    # repeat for the frequencies greater than 'x'
    while word_freq_tuple[1] > x:
        # extract most frequent word(s) using function we wrote
        word_freq_tuple = find_frequent_word(word_dict)
        # keep track of most common words, append them in order
        freq_list.append(word_freq_tuple)
        # remove every entry that matches words in `word_freq_tuple`
        # so that you are left with next most frequent words
        for word in word_freq_tuple[0]:
            del(word_dict[word])
    return freq_list

# print(occurs_often(word_dict, 2))

# pick a song by uncommenting your favorite
#song = "I threw a wish in the well Dont ask me Ill never tell I looked to you as it fell And now youre in my way  Id trade my soul for a wish Pennies and dimes for a kiss I wasnt looking for this But now youre in my way  Your stare was holdin Ripped jeans skin was showin Hot night wind was blowin Where do you think youre going baby  Hey I just met you And this is crazy But heres my number So call me maybe  Its hard to look right At you baby But heres my number So call me maybe  Hey I just met you And this is crazy But heres my number So call me maybe  And all the other boys Try to chase me But heres my number So call me maybe  You took your time with the call I took no time with the fall You gave me nothing at all But still youre in my way  I beg and borrow and steal Have foresight and its real I didnt know I would feel it But its in my way  Your stare was holdin Ripped jeans skin was showin Hot night wind was blowin Where you think youre going baby  Hey I just met you And this is crazy But heres my number So call me maybe  Its hard to look right At you baby But heres my number So call me maybe  Hey I just met you And this is crazy But heres my number So call me maybe  And all the other boys Try to chase me But heres my number So call me maybe  Before you came into my life I missed you so bad I missed you so bad I missed you so so bad  Before you came into my life I missed you so bad And you should know that I missed you so so bad bad bad  Its hard to look right At you baby But heres my number So call me maybe  Hey I just met you And this is crazy But heres my number So call me maybe  And all the other boys Try to chase me But heres my number So call me maybe  Before you came into my life I missed you so bad I missed you so bad I missed you so so bad  Before you came into my life I missed you so bad And you should know that  So call me maybe"
#song = "Because you know Im all about that bass  Bout that bass no treble Im all about that bass Bout that bass no treble Im all about that bass Bout that bass no treble Im all about that bass Bout that bass bass bass bass  Yeah its pretty clear I aint no size two But I can shake it shake it like Im supposed to do Cause I got that boom boom that all the boys chase And all the right junk in all the right places I see the magazine workin that Photoshop We know that stuff aint real come on now make it stop If you got beauty beauty just raise em up Cause every inch of you is perfect from the bottom to the top Yeah my mama she told me dont worry about your size Shoo wop wop shaooh wop wop She says Boys like a little more booty to hold at night That booty uh that booty booty You know I wont be no stick figure silicone Barbie doll So if that what youre into then go head and move along Because you know Im all about that bass Bout that bass no treble Im all about that bass Bout that bass no treble Im all about that bass Bout that bass no treble Im all about that bass Bout that bass Hey Im bringing booty back Go head and tell them skinny girls that No Im just playing I know you think youre fat But Im here to tell you Every inch of you is perfect from the bottom to the top Yeah my mama she told me dont worry about your size Shoo wop wop shaooh wop wop She says Boys like a little more booty to hold at night That booty booty uh that booty booty You know I wont be no stick figure silicone Barbie doll So if thats what youre into then go head and move along Because you know Im all about that bass Bout that bass no treble Im all about that bass Bout that bass no treble Im all about that bass Bout that bass no treble Im all about that bass Bout that bass You know Im all about that bass Bout that bass no treble I said Im all about that bass Bout that bass no treble Im all about that bass Bout that bass no treble Im all about that bass Bout that bass Because you know Im all about that bass Bout that bass no treble Im all about that bass Bout that bass no treble Im all about that bass Bout that bass no treble Im all about that bass Bout that bass Hey Im all about that bass Bout that bass Hey Im all about that bass Bout that bass Hey Yeah yeah ohh You know you like this bass Hey "
#song = "It might seem crazy what Im about to say Sunshine shes here you can take a break Im a hot air balloon that could go to space With the air like I dont care baby by the way  Uh  Because Im happy Clap along if you feel like a room without a roof Because Im happy Clap along if you feel like happiness is the truth Because Im happy Clap along if you know what happiness is to you Because Im happy Clap along if you feel like thats what you wanna do  Here come bad news talking this and that yeah Well give me all you got and dont hold it back yeah Well I should probably warn you Ill be just fine yeah No offense to you dont waste your time Heres why  Because Im happy Clap along if you feel like a room without a roof Because Im happy Clap along if you feel like happiness is the truth Because Im happy Clap along if you know what happiness is to you Because Im happy Clap along if you feel like thats what you wanna do  Hey Go Uh  Happy Bring me down Cant nothing Bring me down My levels too high Bring me down Cant nothing Bring me down I said let me tell you now Bring me down Cant nothing Bring me down My levels too high Bring me down Cant nothing Bring me down I said  Because Im happy Clap along if you feel like a room without a roof Because Im happy Clap along if you feel like happiness is the truth Because Im happy Clap along if you know what happiness is to you Because Im happy Clap along if you feel like thats what you wanna do  Because Im happy Clap along if you feel like a room without a roof Because Im happy Clap along if you feel like happiness is the truth Because Im happy Clap along if you know what happiness is to you Because Im happy Clap along if you feel like thats what you wanna do  Hey Go Uh  Happy repeats Bring me down cant nothing Bring me down my levels too high Bring me down cant nothing Bring me down I said let me tell you now  Because Im happy Clap along if you feel like a room without a roof Because Im happy Clap along if you feel like happiness is the truth Because Im happy Clap along if you know what happiness is to you Because Im happy Clap along if you feel like thats what you wanna do   Because Im happy Clap along if you feel like a room without a roof Because Im happy Clap along if you feel like happiness is the truth Because Im happy Clap along if you know what happiness is to you Because Im happy Clap along if you feel like thats what you wanna do  Hey Cmon"
#song = "Oh  oh oh Oh  Yeah  Im gonna take my horse to the old town road Im gonna ride til I cant no more Im gonna take my horse to the old town road Im gonna ride til I cant no more Kio  Kio   I got the horses in the back Horse tack is attached Hat is matte black Got the boots thats black to match Ridin on a horse  ha You can whip your Porsche I been in the valley You aint been up off that porch  now  Cant nobody tell me nothin You cant tell me nothin Cant nobody tell me nothin You cant tell me nothin  Ridin on a tractor Lean all in my bladder Cheated on my baby You can go and ask her My life is a movie Bull ridin and boobies Cowboy hat from Gucci Wrangler on my booty  Cant nobody tell me nothin You cant tell me nothin Cant nobody tell me nothin You cant tell me nothin  Yeah  Im gonna take my horse to the old town road Im gonna ride til I cant no more Im gonna take my horse to the old town road Im gonna ride til I cant no more   Hat down  cross town  livin like a rock star Spent a lot of money on my brand new guitar Babys got a habit diamond rings and Fendi sports bras Ridin down Rodeo in my Maserati sports car Got no stress  Ive been through all that Im like a Marlboro Man so I kick on back Wish I could roll on back to that old town road I wanna ride til I cant no more   Yeah  Im gonna take my horse to the old town road Im gonna ride til I cant no more Im gonna take my horse to the old town road Im gonna ride til I cant no more"
song = "Hey  I was doing just fine before I met you I drink too much and thats an issue but Im okay Hey  you tell your friends it was nice to meet them But I hope I never see them again I know it breaks your heart Moved to the city in a broke down car And four years  no calls Now youre looking pretty in a hotel bar And I cant stop No  I cant stop So baby pull me closer in the backseat of your Rover That I know you cant afford Bite that tattoo on your shoulder Pull the sheets right off the corner Of the mattress that you stole From your roommate back in Boulder We aint ever getting older We aint ever getting older We aint ever getting older You look as good as the day I met you I forget just why I left you  I was insane Stay and play that Blink 182 song That we beat to death in Tucson  okay I know it breaks your heart Moved to the city in a broke down car And four years  no call Now Im looking pretty in a hotel bar And I cant stop No  I cant stop So baby pull me closer in the backseat of your Rover That I know you cant afford Bite that tattoo on your shoulder Pull the sheets right off the corner Of the mattress that you stole From your roommate back in Boulder We aint ever getting older We aint ever getting older We aint ever getting older So baby pull me closer in the backseat of your Rover That I know you cant afford Bite that tattoo on your shoulder Pull the sheets right off the corner Of the mattress that you stole From your roommate back in Boulder We aint ever getting older We aint ever getting older we aint ever getting older We aint ever getting older we aint ever getting older We aint ever getting older we aint ever getting older We aint ever getting older We aint ever getting older No we aint ever getting older"

song_dict = generate_word_dict(song)
# print("***** WORDS IN SONG *****")
# song_words = song.lower()
# print(song_words.split())

# print("***** WORD FREQUENCIES *****")
# print(song_dict)

# print("***** MOST COMMON WORD *****")
# print(find_frequent_word(song_dict))

# print("***** TOP MOST COMMON WORDS *****")
# print(occurs_often(song_dict, 6))


#####################################################
############### ANSWERS TO YOU TRY IT ###################
########################################################
def find_grades(grades, students):
    """ grades is a dict mapping student names (str) to grades (str)
        students is a list of student names 
    Returns a list containing the grades for students (in the same order) """
    # your code here
    L = []
    for s in students:
        g = grades[s]
        L.append(g)
    return L

# d = {'Ana':'B', 'Matt':'C', 'John':'B', 'Katy':'A'}
# print(find_grades(d, ['Matt', 'Katy'])) # returns ['C', 'A']


def find_in_L(Ld, k):
    """ L is a list of dicts
        k is an int
    Returns True if k is a key in any dicts of L and False otherwise """
    # your code here
    for d in Ld:
        if k in d:
            return True
    return False

d1 = {1:2, 3:4, 5:6}
d2 = {2:4, 4:6}
d3 = {1:1, 3:9, 4:16, 5:25}

# print(find_in_L([d1, d2, d3], 2))  # returns True
# print(find_in_L([d1, d2, d3], 25))  # returns False


def count_matches(d):
    """ d is a dict
    Returns how many entries in d have the key equal to its value """
    # your code here
    count = 0
    for k,v in d.items():
        if k == v:
            count += 1
    return count

# d = {1:2, 3:4, 5:6}
# print(count_matches(d))   # prints 0

# d = {1:2, 3:3, 5:5}
# print(count_matches(d))   # prints 2


######################################################
############## AT HOME ###################
######################################################

def is_inverse(d1, d2):
    """ d1 and d2 are dicts 
    Assume values of d1 and d2 are unique and immutable
    Returns True if d1's keys are values in d2 and d1's 
    values are keys in d2 """
    pass

# d1 = {1:2, 3:4}
# d2 = {2:1, 4:3}
# print(is_inverse(d1, d2))  # prints True

# d1 = {1:2, 3:4}
# d2 = {2:1, 4:3, 5:6}
# print(is_inverse(d1, d2))  # prints False
 
# d1 = {1:2, 3:4}
# d2 = {1:2, 2:1}
# print(is_inverse(d1, d2))  # prints False


def add_to_d(d, L):
    """ d is a dict
        L is a list of tuples
    Mutates d with new entries whose key is the first element of a 
    tuple in L and the associated value is the second element of a 
    tuple in L. If the key is already in d, do nothing to its value. 
    If the key cannot be added, raise a ValueError. Returns None. """
    pass
    
# d = {}
# L = [(1,2), (3,4)]
# add_to_d(d, L)
# print(d)   # d is mutated to be {1: 2, 3: 4}

# d = {1:1}
# L = [(1,2), (3,4)]
# add_to_d(d, L)
# print(d)   # d is mutated to be {1: 1, 3: 4}

# d = {1:1}
# L = [(3,4), ([1,2,3], 5)]
# add_to_d(d, L)   
# # raises a ValueError because its trying to add a list (mutable obj) as key


################################################################
#################### ANSWERS TO AT HOME #############
################################################################
# def is_inverse(d1, d2):
#     """ d1 and d2 are dicts 
#     Assume values of d1 and d2 are unique and immutable
#     Returns True if d1's keys are values in d2 and d1's 
#     values are keys in d2, and vice versa """
#     for k1,v1 in d1.items():
#         if (k1 not in d2.values()) or (v1 not in d2.keys()):
#             return False
#     for k2,v2 in d2.items():
#         if (k2 not in d1.values()) or (v2 not in d1.keys()):
#             return False
#     return True

    
# def add_to_d(d, L):
#     """ d is a dict
#         L is a list of tuples
#     Mutated d with new entries whose key is the first element of a 
#     tuple in L and the associated value is the second element of a 
#     tuple in L. If the key is already in d, do nothing to its value. 
#     If the key cannot be added, raise a ValueError. Does not return anything. """
#     for t in L:
#         try:
#             if t[0] not in d:
#                 d[t[0]] = t[1]
#         except:
#             raise ValueError


