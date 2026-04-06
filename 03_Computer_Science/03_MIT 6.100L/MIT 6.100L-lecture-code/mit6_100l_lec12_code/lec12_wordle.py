######## WORDLE ###########
######## ASSUME YOU ARE GIVEN CODE FROM HERE.... ############
import random

words = """abroad
accept
access
across
acting
action
active
actual
advice
advise
affect
afford
afraid
agency
agenda
almost
always
amount
animal
annual
answer
anyone
anyway
appeal
appear
around
arrive
artist
aspect
assess
assist
assume
attack
attend
august
author
avenue
backed
barely
battle
beauty
became
become
before
behalf
behind
belief
belong
berlin
better
beyond
bishop
border
bottle
bottom
bought
branch
breath
bridge
bright
broken
budget
burden
bureau
button
camera
cancer
cannot
carbon
career
castle
casual
caught
center
centre
chance
change
charge
choice
choose
chosen
church
circle
client
closed
closer
coffee
column
combat
coming
common
comply
copper
corner
costly
county
couple
course
covers
create
credit
crisis
custom
damage
danger
dealer
debate
decade
decide
defeat
defend
define
degree
demand
depend
deputy
desert
design
desire
detail
detect
device
differ
dinner
direct
doctor
dollar
domain
double
driven
driver
during
easily
eating
editor
effect
effort
eighth
either
eleven
emerge
empire
employ
enable
ending
energy
engage
engine
enough
ensure
entire
entity
equity
escape
estate
ethnic
exceed
except
excess
expand
expect
expert
export
extend
extent
fabric
facing
factor
failed
fairly
fallen
family
famous
father
fellow
female
figure
filing
finger
finish
fiscal
flight
flying
follow
forced
forest
forget
formal
format
former
foster
fought
fourth
French
friend
future
garden
gather
gender
german
global
golden
ground
growth
guilty
handed
handle
happen
hardly
headed
health
height
hidden
holder
honest
impact
import
income
indeed
injury
inside
intend
intent
invest
island
itself
jersey
joseph
junior
killed
labour
latest
latter
launch
lawyer
leader
league
leaves
legacy
length
lesson
letter
lights
likely
linked
liquid
listen
little
living
losing
lucent
luxury
mainly
making
manage
manner
manual
margin
marine
marked
market
martin
master
matter
mature
medium
member
memory
mental
merely
merger
method
middle
miller
mining
minute
mirror
mobile
modern
modest
module
moment
morris
mostly
mother
motion
moving
murder
museum
mutual
myself
narrow
nation
native
nature
nearby
nearly
nights
nobody
normal
notice
notion
number
object
obtain
office
offset
online
option
orange
origin
output
oxford
packed
palace
parent
partly
patent
people
period
permit
person
phrase
picked
planet
player
please
plenty
pocket
police
policy
prefer
pretty
prince
prison
profit
proper
proven
public
pursue
raised
random
rarely
rather
rating
reader
really
reason
recall
recent
record
reduce
reform
regard
regime
region
relate
relief
remain
remote
remove
repair
repeat
replay
report
rescue
resort
result
retail
retain
return
reveal
review
reward
riding
rising
robust
ruling
safety
salary
sample
saving
saying
scheme
school
screen
search
season
second
secret
sector
secure
seeing
select
seller
senior
series
server
settle
severe
should
singer
signal
signed
silent
silver
simple
simply
single
sister
slight
smooth
social
solely
sought
sounds
source
soviet
speech
spirit
spoken
spread
spring
square
stable
status
steady
stolen
strain
stream
street
stress
strict
strike
string
strong
struck
studio
submit
sudden
suffer
summer
summit
supply
surely
survey
switch
symbol
system
taking
talent
target
taught
tenant
tender
tennis
thanks
theory
thirty
though
threat
thrown
ticket
timely
timing
tissue
toward
travel
treaty
trying
twelve
twenty
unable
unique
united
unless
unlike
update
useful
valley
varied
vendor
versus
victim
vision
visual
volume
walker
wealth
weekly
weight
wholly
window
winner
winter
within
wonder
worker
wright
writer
yellow"""

random.seed(0)
def new_word(words):
    """ words is a multi-line string.
    Returns a 6 letter word as a str. """
    words_list = get_word_list(words)
    return random.choice(words_list)

######## ....ASSUME YOU ARE GIVEN CODE UP TO HERE ############

################################################
######## THE CODE BELOW IS BUGGY #############
################################################
def get_word_list(words_str):
    """ words_str is a multi-line string.
    Returns a list whose elements are lowercase words """
    return words.split('\n')

def is_a_real_word(s, word_list):
    """ s is a string
        word_list is a list of words
    Returns True is s is in word_list and False otherwise """
    return (s in word_list)

def is_correct_len(guess, length):
    """ guess is a str
        length is an int
    Returns True if guess has length number of characters. """
    return (len(guess) == length)

def make_wordle(guess, secret):
    """ guess and secret are 6 letter words
    Returns a result where:
    * a guess' letter in the correct place is capitalized
    * a guess' letter in the secret but not in the correct place is lowercase
    * a guess' letter not in the secret is not shown 
    For example: if guess is "struck" and the secret is "strike" 
    then the return is "ST   k"
    """
    result = ""
    guessed = []
    for i in range(len(secret)):
        if guess[i] == secret[i]:
            result += guess[i].upper()
            guessed.append(guess[i])
        elif guess[i] in secret:
            result += guess[i]
        else:
            result += ' '
    return result
 
def play_game():
    """ Plays the game.
    0) Generates a word_list, a new secret word, and sets up 6 guesses.
    1) Asks the user for a 6 letter word as a guess
    2) Creates a wordle, i.e. the word that the user guessed, but with the
       following replacements: 
       a) guessed letters in the correct position as secret are capitalized
       b) guessed letters in the secret but not in the correct position are lowercase
       c) all other guessed letters are represented as a space
       For example: if guess is "struck" and the secret is "strike" then the
       user is presented with "ST   k".
    3) The user has 6 guesses to guess the secret word. """
    word_list = get_word_list(words)
    secret = new_word
    wordle_len = 6
    n_guesses = 6
    win = False
    
    print(f"You have {n_guesses} to guess a 6-letter word.")
    guess = input("Guess: ")
    
    while guess != secret:
        # if is_a_real_word(guess, word_list) and is_correct_len(guess, wordle_len):
        #     result = make_wordle(guess, secret)
        #     print(result)
        #     n_guesses -= 1
        # else:
        #     print("Not a word or not the correct length.")
        result = make_wordle(guess, secret)
        print(result)
        print(f"You have {n_guesses} guesses left.")
        if n_guesses > 0:
            guess = input("Guess: ")
        elif guess == secret:
            win = True
            break
        else:
            break
    if win:
        print('YOU WIN')
    else:
        print('YOU LOSE')

play_game()





################################################
######## FIXES TO THE BUGGY CODE #############
################################################

def get_word_list(words_str):
    """ words_str is a multi-line string.
    Returns a list whose elements are lowercase words """
    return words.split('\n')

def is_a_real_word(s, word_list):
    """ s is a string
        word_list is a list of words
    Returns True is s is in word_list and False otherwise """
    return (s in word_list)

def is_correct_len(guess, length):
    """ guess is a str
        length is an int
    Returns True if guess has length number of characters. """
    return (len(guess) == length)

def make_wordle(guess, secret):
    """ guess and secret are 6 letter words
    Returns a result where:
    * a guess' letter in the correct place is capitalized
    * a guess' letter in the secret but not in the correct place is lowercase
    * a guess' letter not in the secret is not shown 
    For example: if guess is "struck" and the secret is "strike" 
    then the return is "ST   k"
    """
    result = ""
    guessed = []
    for i in range(len(secret)):
        if guess[i] == secret[i]:
            result += guess[i].upper()
            guessed.append(guess[i])
        elif guess[i] in secret and guess[i] not in guessed:
            result += guess[i]
        else:
            result += ' '
    return result

def play_game():
    """ Plays the game.
    0) Generates a word_list, a new secret word, and sets up 6 guesses.
    1) Asks the user for a 6 letter word as a guess
    2) Creates a wordle, i.e. the word that the user guessed, but with the
       following replacements: 
       a) guessed letters in the correct position as secret are capitalized
       b) guessed letters in the secret but not in the correct position are lowercase
       c) all other guessed letters are represented as a space
       For example: if guess is "struck" and the secret is "strike" then the
       user is presented with "ST   k".
    3) The user has 6 guesses to guess the secret word. """
    word_list = get_word_list(words)
    secret = new_word(words)
    wordle_len = 6
    n_guesses = 6
    win = False
    
    print(f"You have {n_guesses} to guess a 6-letter word.")
    guess = input("Guess: ")
    
    while guess != secret:
        # if is_a_real_word(guess, word_list) and is_correct_len(guess, wordle_len):
        #     result = make_wordle(guess, secret)
        #     print(result)
        #     n_guesses -= 1
        # else:
        #     print("Not a word or not the correct length.")
        result = make_wordle(guess, secret)
        print(result)
        n_guesses -= 1
        print(f"You have {n_guesses} guesses left.")
        if n_guesses > 0:
            guess = input("Guess: ")
        elif guess == secret:
            win = True
            break
        else:
            break
    if win or guess==secret:
        print('YOU WIN')
    else:
        print('YOU LOSE')
    
# play_game()




