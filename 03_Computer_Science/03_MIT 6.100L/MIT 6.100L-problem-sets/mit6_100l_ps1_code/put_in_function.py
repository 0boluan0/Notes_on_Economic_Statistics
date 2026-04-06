#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 19:55:06 2017

@author: lauragustafson, knjohnso, carlosh
"""
def put_in_functions_a():
    try:
        FUNCTION_NAME = 'part_a'
        #change FUNCTION_NAME to be the name of the function that you want their code
        #to be wrapped in
    
        FUNCTION_HEADER = 'def %s(yearly_salary, portion_saved, cost_of_dream_home):' % (FUNCTION_NAME)
    
        STUDENT_FILE_NAME = 'ps1a.py'
        #change STUDENT_FILE_NAME to be the name of the file that their code will be in
    
    
        RETURN_VARIABLE = 'months'
        #change RETURN_VARIABLE to be the name of the variable you want the fucntion
        #to return
    
        RETURN_STATEMENT = '\treturn %s' % (RETURN_VARIABLE)
    
        NEW_FILE_NAME = 'ps1a_in_function.py'
        #change NEW_FILE_NAME to be the name of the output file
    
        START_LINE = "## Initialize other variables you need (if any) for your program below ##" 
        #The start line to grab for the student's function, everything else below this should be fine to copy and not include any input statements.
    
        new_lines = []
        lines = [line.rstrip('\n') for line in open('ps1a.py')]
        # look for the start line and find its index in lines 
        START_INDEX = [line.startswith(START_LINE) for line in lines].index(True)
        lines = lines[START_INDEX+1:]
     
        new_lines.append(FUNCTION_HEADER)
        for line in lines:
            new_lines.append('\t'+line)
        new_lines.append(RETURN_STATEMENT)
    
        with open(NEW_FILE_NAME, 'w') as new_file:
            new_file.write('\n'.join(new_lines))
    except:
        print('[PART A] Did you keep all the original comments the file came with? Did you initialize variables at the right places?')

def put_in_functions_b():
    try:
        FUNCTION_NAME = 'part_b'
        #change FUNCTION_NAME to be the name of the function that you want their code
        #to be wrapped in
    
        FUNCTION_HEADER = 'def %s(yearly_salary, portion_saved, cost_of_dream_home, semi_annual_raise):' % (FUNCTION_NAME)
    
        STUDENT_FILE_NAME = 'ps1b.py'
        #change STUDENT_FILE_NAME to be the name of the file that their code will be in
    
    
        RETURN_VARIABLE = 'months'
        #change RETURN_VARIABLE to be the name of the variable you want the fucntion
        #to return
    
        RETURN_STATEMENT = '\treturn %s' % (RETURN_VARIABLE)
    
        NEW_FILE_NAME = 'ps1b_in_function.py'
        #change NEW_FILE_NAME to be the name of the output file
    
        START_LINE = "## Initialize other variables you need (if any) for your program below ##" 
        #The start line to grab for the student's function, everything else below this should be fine to copy andd not include any input statements.
    
        new_lines = []
        lines = [line.rstrip('\n') for line in open('ps1b.py')]
        # look for the start line and find its index in lines
        START_INDEX = [line.startswith(START_LINE) for line in lines].index(True)
        lines = lines[START_INDEX+1:]
        
        new_lines.append(FUNCTION_HEADER)
        for line in lines:
            new_lines.append('\t'+line)
        new_lines.append(RETURN_STATEMENT)
    
        with open(NEW_FILE_NAME, 'w') as new_file:
            new_file.write('\n'.join(new_lines))
    except:
        print('[PART B] Did you keep all the original comments the file came with? Did you initialize variables at the right places?')

def put_in_functions_c():
    try:
        FUNCTION_NAME = 'part_c'
        #change FUNCTION_NAME to be the name of the function that you want their code
        #to be wrapped in
    
        FUNCTION_HEADER = 'def %s(initial_deposit):' % (FUNCTION_NAME)
    
        STUDENT_FILE_NAME = 'ps1c.py'
        #change STUDENT_FILE_NAME to be the name of the file that their code will be in
    
    
        RETURN_VARIABLE = 'r, steps'
        #change RETURN_VARIABLE to be the name of the variable you want the fucntion
        #to return
    
        RETURN_STATEMENT = '\treturn %s' % (RETURN_VARIABLE)
    
        NEW_FILE_NAME = 'ps1c_in_function.py'
        #change NEW_FILE_NAME to be the name of the output file
    
        START_LINE = "## Initialize other variables you need (if any) for your program below ##" 
        #The start line to grab for the student's function, everything else below this should be fine to copy andd not include any input statements.
    
        new_lines = []
        lines = [line.rstrip('\n') for line in open('ps1c.py')]
        # look for the start line and find its index in lines
        START_INDEX = [line.startswith(START_LINE) for line in lines].index(True)
        lines = lines[START_INDEX+1:]
    
        new_lines.append(FUNCTION_HEADER)
        for line in lines:
            new_lines.append('\t'+line)
        new_lines.append(RETURN_STATEMENT)
    
        with open(NEW_FILE_NAME, 'w') as new_file:
            new_file.write('\n'.join(new_lines))
    except:
        print('[PART C] Did you keep all the original comments the file came with? Did you initialize variables at the right places?')


put_in_functions_a()
put_in_functions_b()
put_in_functions_c()
