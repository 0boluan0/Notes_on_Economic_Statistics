# for running unit tests on 6.100A/L student code

import sys
import unittest
import os
import string
from unittest.mock import patch
from unittest import TestCase
import re
import random
#pulled from http://stackoverflow.com/questions/20567497/overwrite-built-in-function
outputstr=""
class MyStream(object):
    def __init__(self, target):
        self.target = target

    def write(self, s):
        global outputstr
        outputstr+=s
        #self.target.write(s)
        return s
    def flush(self):
        pass

WILDCARD = "!"
HIDDEN = "*"

store = sys.stdout
sys.stdout = MyStream(sys.stdout)

def Dprint(*args):
    """
    Prints output to sys.__stdout__ then reverts the output of print to
    wherever it was before debugging. Useful for adding print statements in the
    testers to see if your game is printing the lines you think it should be.
    """
    global store
    sys.stdout = store
    print(*args)
    sys.stdout = MyStream(sys.stdout)


input_string = (letter for letter in ["h", "e", "i"])
def make_input(self):
    return next(input_string)

def output_to_file(test_case_name, word_to_guess, guessed_letters, student_output, correct_output):
    with open('run_game_test_results.txt', 'a+') as f:
        f.write("=============================================================\n")
        f.write("RESULTS FOR TEST CASE: %s\n"%test_case_name)
        f.write("WORD USED IN TEST: %s\n"%word_to_guess)
        f.write("GUESSED LETTERS IN ORDER OF GUESS: %s\n"%guessed_letters)
        f.write('************************\n')
        f.write("YOUR OUTPUT:\n")
        f.write('************************\n')
        f.write(student_output+'\n')
        f.write('************************\n')
        f.write("POSSIBLE CORRECT OUTPUT:\n")
        f.write('************************\n')
        f.write(correct_output+'\n')
        f.write("=============================================================\n\n\n")

def compare_results(expected, actual):
    '''
    Used for comparing equality of student answers with staff answers
    '''
    def almost_equal(x,y):
        if x == y or x.replace(' ', '') == y.replace(' ',''):
            return True
        return False

    exp = expected.strip()
    act = actual.strip()
    return almost_equal(exp, act)


# A class that inherits from unittest.TestCase, where each function
# is a test you want to run on the student's code. For a full description
# plus a list of all the possible assert methods you can use, see the
# documentation: https://docs.python.org/3/library/unittest.html#unittest.TestCase
class TestPS2(unittest.TestCase):

    # TODO Add HangmanOracle tests

    def test_has_player_won(self):
        self.assertTrue(student.has_player_won('face', ['f','c','a','e']))
        self.assertFalse(student.has_player_won('moves', ['o','c','a','v','e']))

    def test_has_player_won_repeated_letters(self):
        self.assertTrue(student.has_player_won('bass', ['a','s','b','e']),
            "Failed with repeated letters")
        self.assertFalse(student.has_player_won('rare', ['f','t','r','e']),
            "Failed with repeated letters")

    def test_has_player_won_empty_string(self):
        self.assertTrue(student.has_player_won('', ['f','c','y','e']),
            "Failed with the empty string")

    def test_has_player_won_empty_list(self):
        self.assertFalse(student.has_player_won('code', []),
            "Failed with the empty list")

    def test_get_word_progress(self):
        self.assertTrue(compare_results(student.get_word_progress('face', ['f','c','a','e']), 'face'))
        self.assertTrue(compare_results(student.get_word_progress('moves', ['o','c','a','v','e']), HIDDEN+'ove'+HIDDEN))

    def test_get_word_progress_repeated_letters(self):
        self.assertTrue(compare_results(student.get_word_progress('bass', ['a','s','b','e']), 'bass'),
            "Failed with repeated letters")
        self.assertTrue(compare_results(student.get_word_progress('rare', ['f','t','r','e']), 'r'+HIDDEN+'re'),
            "Failed with repeated letters")

    def test_get_word_progress_empty_string(self):
        self.assertTrue(compare_results(student.get_word_progress('', ['f','c','y','e']), ''),
            "Failed with the empty string")

    def test_get_word_progress_empty_list(self):
        self.assertTrue(compare_results(student.get_word_progress('code', []), HIDDEN*4),
            "Failed with the empty list")

    def test_get_available_letters(self):
        self.assertEqual(student.get_available_letters(['a','b','c','d']), 'efghijklmnopqrstuvwxyz')
        self.assertEqual(student.get_available_letters(['z','p','x','b', 'b']), 'acdefghijklmnoqrstuvwy')
        self.assertEqual(student.get_available_letters(['a','u','i','o','w']), 'bcdefghjklmnpqrstvxyz')

    def test_get_available_letters_empty_string(self):
        self.assertEqual(student.get_available_letters(list(string.ascii_lowercase)), '',
            "Failed to return the empty string")

    def test_get_available_letters_empty_list(self):
        self.assertEqual(student.get_available_letters([]), 'abcdefghijklmnopqrstuvwxyz',
            "Failed with the empty list")


    def test_play_game_short(self):
        correct='''Welcome to Hangman!
I am thinking of a word that is 2 letters long.
------
You have 10 guesses left.
Available letters: abcdefghijklmnopqrstuvwxyz
Please guess a letter: h
Good guess: h*
------
You have 10 guesses left.
Available letters: abcdefgijklmnopqrstuvwxyz
Please guess a letter: e
Oops! That letter is not in my word: h*
------
You have 8 guesses left.
Available letters: abcdfgijklmnopqrstuvwxyz
Please guess a letter: i
Good guess: hi
------
Congratulations, you won!
Your total score for this game is: 22'''
        with unittest.mock.patch('builtins.input',  make_input):
            threw_exception =  False
            try:
                student.hangman("hi", False)
            except:
                threw_exception = True
            global outputstr
            student_output = outputstr[:]
            lines = re.split('\-{3,}',outputstr)
            outputstr =""
            try:
                self.assertFalse(threw_exception)
                if len(lines) > 4:
                    self.assertTrue("h"+HIDDEN in lines[1])
                    self.assertTrue("10 guesses" in lines[2])
                    self.assertTrue("h"+HIDDEN in lines[2])
                    self.assertTrue("8 guesses" in lines[3])
                    self.assertTrue("hi" in lines[3])
                    self.assertTrue("score" in lines[4])
                    self.assertTrue("22" in lines[4])
                else:
                    self.assertTrue(False)
            except Exception as e:
                output_to_file('test_play_game_short', 'hi', ["h", "e", "i"], student_output, correct)
                raise(e)
    def test_play_game_short_fail(self):
        correct='''Welcome to hangman!
I am thinking of a word that is 2 letters long.
--------------
You have 10 guesses left.
Available Letters: abcdefghijklmnopqrstuvwxyz
Please guess a letter: n
Oops! That letter is not in my word: **
--------------
You have 9 guesses left.
Available Letters: abcdefghijklmopqrstuvwxyz
Please guess a letter: e
Oops! That letter is not in my word: **
--------------
You have 7 guesses left.
Available Letters: abcdfghijklmopqrstuvwxyz
Please guess a letter: i
Good guess: *i
--------------
You have 7 guesses left.
Available Letters: abcdfghjklmopqrstuvwxyz
Please guess a letter: m
Oops! That letter is not in my word: *i
--------------
You have 6 guesses left.
Available Letters: abcdfghjklopqrstuvwxyz
Please guess a letter: u
Oops! That letter is not in my word: *i
--------------
You have 4 guesses left.
Available Letters: abcdfghjklopqrstvwxyz
Please guess a letter: k
Oops! That letter is not in my word: *i
--------------
You have 3 guesses left.
Available Letters: abcdfghjlopqrstvwxyz
Please guess a letter: l
Oops! That letter is not in my word: *i
--------------
You have 2 guesses left.
Available Letters: abcdfghjopqrstvwxyz
Please guess a letter: p
Oops! That letter is not in my word: *i
--------------
You have 1 guess left.
Available Letters: abcdfghjoqrstvwxyz
Please guess a letter: s
Oops! That letter is not in my word: *i
-------
Sorry, you ran out of guesses. The word was hi'''
        global input_string
        computer_guesses = ["n", "e", "i", "m","u","k", "l", "p", "s"]
        input_string = (letter for letter in computer_guesses)
        with unittest.mock.patch('builtins.input',  make_input):
            threw_exception =  False
            try:
                student.hangman("hi", False)
            except:
                threw_exception = True
            global outputstr
            lines = re.split('\-{3,}',outputstr)
            student_output = outputstr[:]
            outputstr =""
            try:
                self.assertFalse(threw_exception)
                if len(lines) > 8:
                    self.assertTrue("10 guesses" in lines[1])
                    self.assertTrue(HIDDEN*2 in lines[1])
                    self.assertTrue("9 guesses" in lines[2])
                    self.assertTrue(HIDDEN*2 in lines[2])
                    self.assertTrue("7 guesses" in lines[3])
                    self.assertTrue(HIDDEN+"i" in lines[3])
                    self.assertTrue("7 guesses" in lines[4])
                    self.assertTrue(HIDDEN+"i" in lines[4])
                    self.assertTrue("6 guesses" in lines[5])
                    self.assertTrue(HIDDEN+"i" in lines[5])
                    self.assertTrue("4 guesses" in lines[6])
                    self.assertTrue(HIDDEN+"i" in lines[6])
                    self.assertTrue("3 guesses" in lines[7])
                    self.assertTrue(HIDDEN+"i" in lines[7])
                    self.assertTrue("2 guesses" in lines[8])
                    self.assertTrue(HIDDEN+"i" in lines[8])
                    self.assertTrue("1 guess" in lines[9])
                    self.assertTrue(HIDDEN+"i" in lines[9])
                    self.assertTrue("abcdfghjoqrstvwxyz" in lines[9])
                    self.assertTrue("hi" in lines[10])

                else:
                    self.assertTrue(False)
            except Exception as e:
                output_to_file('test_play_game_short_fail', 'hi', computer_guesses, student_output, correct)
                raise(e)

    def test_play_game_wildcard(self):
        correct = '''Welcome to hangman!
I am thinking of a word that is 8 letters long
------
You have 10 guesses left
Available letters: abcdefghijklmnopqrstuvwxyz
Please guess a letter: k
Oops! That letter is not in my word: ********
------
You have 9 guesses left
Available letters: abcdefghijlmnopqrstuvwxyz
Please guess a letter: w
Good guess: w*******
------
You have 9 guesses left
Available letters: abcdefghijlmnopqrstuvxyz
Please guess a letter: i
Good guess: wi******
------
You have 9 guesses left
Available letters: abcdefghjlmnopqrstuvxyz
Please guess a letter: l
Good guess: wil*****
------
You have 9 guesses left
Available letters: abcdefghjmnopqrstuvxyz
Please guess a letter: d
Good guess: wild***d
------
You have 9 guesses left
Available letters: abcefghjmnopqrstuvxyz
Please guess a letter: c
Good guess: wildc**d
------
You have 9 guesses left
Available letters: abefghjmnopqrstuvxyz
Please guess a letter: !
Letter revealed: a
wildca+d
------
You have 6 guesses left
Available letters: befghjmnopqrstuvxyz
Please guess a letter: !
Letter revealed: r
Good guess: wildcard
------
Congratulations, you won!
Your total score for this game is: 55
'''
        global input_string
        computer_guesses = ["k", "w", "i", "l", "d", "c", WILDCARD, WILDCARD]
        input_string = (letter for letter in computer_guesses)
        with unittest.mock.patch('builtins.input',  make_input):
            threw_exception =  False
            try:
                student.hangman("wildcard", True)
            except:
                threw_exception = True
            global outputstr

            lines = re.split('\-{3,}',outputstr)
            student_output = outputstr[:]
            outputstr =""
            try:
                self.assertFalse(threw_exception)
                if len(lines) > 9:
                    self.assertTrue("10 guesses" in lines[1])
                    self.assertTrue(HIDDEN*8 in lines[1])
                    self.assertTrue("9 guesses" in lines[2])
                    self.assertTrue("w"+HIDDEN*7 in lines[2])
                    self.assertTrue("9 guesses" in lines[3])
                    self.assertTrue("wi"+HIDDEN*6 in lines[3])
                    self.assertTrue("9 guesses" in lines[4])
                    self.assertTrue("wil"+HIDDEN*5 in lines[4])
                    self.assertTrue("9 guesses" in lines[5])
                    self.assertTrue("wild"+HIDDEN*3+"d" in lines[5])
                    self.assertTrue("9 guesses" in lines[6])
                    self.assertTrue("wildc"+HIDDEN*2+"d" in lines[6])
                    self.assertTrue("9 guesses" in lines[7])
                    self.assertTrue("revealed" in lines[7]) # lost 2 guesses
                    self.assertTrue("6 guesses" in lines[8])
                    self.assertTrue("revealed" in lines[8]) # lost 2 guesses
                    self.assertTrue("score" in lines[9])
                    self.assertTrue("55" in lines[9])
                else:
                    self.assertTrue(False)
            except Exception as e:
                output_to_file('test_play_game_wildcard', 'wildcard', computer_guesses, student_output, correct)
                raise(e)

# Dictionary mapping function names from the above TestCase class to
# messages you'd like the student to see if the test fails.
failure_messages = {
    'test_has_player_won' : 'Your function has_player_won() does not return the correct result.',
    'test_has_player_won_repeated_letters' : 'Your function has_player_won() does not return the correct result for repeated letters.',
    'test_has_player_won_empty_string': 'Your function has_player_won() does not return the correct result for the empty string',
    'test_has_player_won_empty_list': 'Your function has_player_won() does not return the correct result for the empty list',
    'test_get_word_progress': 'Your function get_word_progress() does not return the correct result.',
    'test_get_word_progress_repeated_letters': 'Your function get_word_progress() does not return the correct result for repeated letters.',
    'test_get_word_progress_empty_string': 'Your function get_word_progress() does not return the correct result for the empty string.',
    'test_get_word_progress_empty_list': 'Your function get_word_progress() does not return the correct result for the empty list.',
    'test_get_available_letters': 'Your function get_available_letters() does not return the correct result.',
    'test_get_available_letters_empty_string': 'Your function get_available_letters() does not return the correct result for the empty string.',
    'test_get_available_letters_empty_list': 'Your function get_available_letters() does not return the correct result for the empty list.',
    'test_play_game_short': 'You do not play the game right for a two letter word and correct guesses',
    'test_play_game_short_fail': 'You do not play the game right for a two letter word and incorrect guesses',
    'test_play_game_wildcard': 'You do not play the game correctly with help. '
}

# Dictionary mapping function names from the above TestCase class to
# messages you'd like the student to see if their code throws an error.
error_messages = {
    'test_has_player_won' : 'Your function has_player_won() produces an error.',
    'test_has_player_won_repeated_letters' : 'Your function has_player_won() produces an error for repeated letters.',
    'test_has_player_won_empty_string': 'Your function has_player_won() produces an error for the empty string',
    'test_has_player_won_empty_list': 'Your function has_player_won() produces an error for the empty list',
    'test_get_word_progress': 'Your function get_word_progress() produces an error.',
    'test_get_word_progress_repeated_letters': 'Your function get_word_progress() produces an error for repeated letters.',
    'test_get_word_progress_empty_string': 'Your function get_word_progress() produces an error for the empty string.',
    'test_get_word_progress_empty_list': 'Your function get_word_progress() produces an error for the empty list.',
    'test_get_available_letters': 'Your function get_available_letters() produces an error.',
    'test_get_available_letters_empty_string': 'Your function get_available_letters() produces an error for the empty string.',
    'test_get_available_letters_empty_list': 'Your function get_available_letters() produces an error for the empty list.',
    'test_play_game_short': 'You do not play the game right for a two letter word and correct guesses',
    'test_play_game_short_fail': 'You do not play the game right for a two letter word and incorrect guesses',
    'test_play_game_wildcard': 'You do not play the game correctly with help.'
}

# Dictionary mapping function names from the above TestCase class to
# the point value each test is worth. Make sure these add up to 5!
# TODO update values depending on the number of Hangman test cases we use
point_values = {
    'test_has_player_won' : .40,
    'test_has_player_won_repeated_letters' : .40,
    'test_has_player_won_empty_string': .40,
    'test_has_player_won_empty_list': .40,
    'test_get_word_progress': .40,
    'test_get_word_progress_repeated_letters': .40,
    'test_get_word_progress_empty_string': .40,
    'test_get_word_progress_empty_list': .40,
    'test_get_available_letters': .40,
    'test_get_available_letters_empty_string': .40,
    'test_get_available_letters_empty_list': .40,
    'test_play_game_short': .35,
    'test_play_game_short_fail': .35,
    'test_play_game_wildcard': .30
}

# Subclass to track a point score and appropriate
# grade comment for a suit of unit tests
class Results_600(unittest.TextTestResult):

    # We override the init method so that the Result object
    # can store the score and appropriate test output.
    def __init__(self, *args, **kwargs):
        super(Results_600, self).__init__(*args, **kwargs)
        self.output = []
        self.points = 5

    def addFailure(self, test, err):
        test_name = test._testMethodName
        self.handleDeduction(test_name, failure_messages)
        super(Results_600, self).addFailure(test, err)

    def addError(self, test, err):
        test_name = test._testMethodName
        self.handleDeduction(test_name, error_messages)
        super(Results_600, self).addError(test, err)

    def handleDeduction(self, test_name, messages):
        point_value = point_values[test_name]
        message = messages[test_name]
        self.output.append('[-%s]: %s' % (point_value, message))
        self.points -= round(point_value,2)

    def getOutput(self):
        if len(self.output) == 0:
            return "All correct!"
        return '\n'.join(self.output)

    def getPoints(self):
        return self.points

if __name__ == '__main__':
    exec("import hangman as student")
    # os.remove("run_game_test_results.txt")

    sys.stdout = store
    print("Running unit tests")
    sys.stdout = MyStream(sys.stdout)
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPS2))
    result = unittest.TextTestRunner(verbosity=2, resultclass=Results_600).run(suite)

    output = result.getOutput()
    points = round(result.getPoints(),3)
    if points <=0:
        points=0.0
    sys.stdout = store

    print("\n\nProblem Set 2 Unit Test Results:")
    print(output)
    print("Points for these tests: %s/5\n (Please note that this is not your final pset score, additional test cases will be run on submissions)" % points)

