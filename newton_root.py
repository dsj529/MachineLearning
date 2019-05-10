'''
Created on May 9, 2019

@author: dsj529

a generalized implementation of Newton's method to find any degree root of a number.
'''
def newton_root(x, root, epsilon=7):
    def f(guess):
        return guess**root - x
    def df(guess):
        return root * guess**(root-1)
    
    guess = x/root
    i = 0
    epsilon = 10**-epsilon
    while abs(f(guess)) > epsilon:
        try:
            guess = guess - f(guess)/df(guess)
            i += 1
        except ZeroDivisionError:
            print('Derivative at {} == 0, iteration cannot proceed'
                  .format(guess))
    print('{x}**{root} is {guess:f}, {i} iterations needed'
          .format(root=root, x=x, guess=guess, i=i))
