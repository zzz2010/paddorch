# TEST CASES PROVIDED BELOW FUNCTIONS
# Einsum syntax matches numpy.Einsum syntax

import re
import numpy as np
import paddorch

def combo(n):
    #create a list that is the size of the tuple
    r = [0] * len(n)
    #good for general loops with conditions appearing elsewhere in the loop
    while True:
        #need to return a tuple so it cannot be modified
        yield(tuple(r))
        #number of digits
        p = len(r) - 1
        #add 1 digit
        r[p] += 1
        #n[p] is the maximum, each value of the tuple
        while r[p] == n[p]:
            #return this digit to 0
            r[p] = 0
            #move on to the next tuple element to the left
            p -= 1
            #if hit the last element in the tuple, end
            if p < 0:
                return
            #increment the next digit by 1
            r[p] += 1

def einsum(procedure, *pots):

    #Use regular expression to check for repeats
    regexp = re.compile(r"(.)\1")
    match = re.search(regexp, procedure)

    if '->' in procedure:
        if match:
            #CONDITION 1: '->' and repeated letter (return diagonal)
            l = len(pots[0])
            diag = []
            for i in range(0,l):
                diag.append(pots[0][i][i])
            #print(diag)
            return(diag)

        else:
            #CONDITION 8: Tensor dot multiplication and dimension specific broadcasting
            halves = procedure.split("->")
            tables = halves[0]
            tables = tables.split(",")
            broadcast = str(halves[1])
            broadcastList = []
            for letter in broadcast:
                broadcastList.append(letter)
            nameAndDims = {}
            broadcastDims = []
            dims = []
            for x in range(0,len(pots)):
                nameAndDims[tables[x]] = pots[x].shape
                dims.append(pots[x].shape)
            combinations = []
            uniqueTables = []
            originalTables = []
            flatTables = []
            flatDims = []
            uniqueDict = {}
            for i in range(0,len(dims)):
                for h in tables[i]:
                    flatTables.append(h)
                    if h not in originalTables:
                        originalTables.append(h)
                for dim in dims[i]:
                    flatDims.append(dim)
            uniqueTables = sorted(originalTables)
            for pos in range(0,len(flatTables)):
                uniqueDict[flatTables[pos]] = flatDims[pos]
            for z in uniqueTables:
                if z in uniqueDict.keys():
                    combinations.append(uniqueDict.get(z))
            keepGoing = True
            while keepGoing == True:
                for letter in broadcast:
                    if letter in broadcastList:
                        broadcastDims.append(uniqueDict.get(letter))
                        keepGoing = False
                    else:
                        print('projection char not found')

            #COMPUTE
            combos = combo(combinations) #Generate all combos needed
            broadcastCombos = combo(broadcastDims) #Generate all combos (dimensions) needed in the broadcast
            out = paddorch.zeros(broadcastDims)
            for bcomb in broadcastCombos:
                combos = combo(combinations) # generator gets exhausted on first loop through, need to reset it
                plug = 0
                for comb in combos:
                    skipCombo = False
                    if any(comb[uniqueTables.index(letter)] != bcomb[broadcastList.index(letter)] for letter in broadcastList):
                        skipCombo = True
                    if skipCombo == False:
                        forMultiplying = []
                        for v in range(0,len(tables)):
                            indices = []
                            for char in tables[v]:
                                indices.append([comb[uniqueTables.index(char)]])
                            forMultiplying.append( nest_index(pots[v],  indices   )  )
                        value = 1
                        for num in forMultiplying:
                            value *= num
                        plug += value
                    else:
                        pass
                out[bcomb] = plug
            #print(str(out))
            return(out)

    #CONDITION 2: REPEATED LETTER BUT NO  '->' (sum diagonal)
    else:
        l = len(pots[0])
        diag = []
        for i in range(0,l):
            diag.append(pots[0][i][i])
        #print(sum(diag))
        return(sum(diag))

def nest_index(x,indices):
    if isinstance(indices,int) :
        return x[indices]
    for ii in  indices :
        x=nest_index(x,ii)
    return x
#DATA
a = paddorch.FloatTensor(np.array([ 0.9, 0.1]))
c =paddorch.FloatTensor( np.array([ 0.1,  0.2,  0.3,  0.4]))
cab = paddorch.FloatTensor(np.array([[[ 0.2 ,  0.4 ,  0.4 ],[ 0.33,  0.33,  0.34]],
                                     [[ 0.1 ,  0.5 ,  0.4 ],[ 0.3 ,  0.1 ,  0.6 ]],[[ 0.01,  0.01,  0.98],
                                                                                    [ 0.2 ,  0.7 ,  0.1 ]],[[ 0.2 ,  0.1 ,  0.7 ],[ 0.9 ,  0.05,  0.05]]]))
b = paddorch.FloatTensor(np.array([0,0,1]))
# FUNCTION CALLS
print(einsum('a,c,cab,b->abc',a,c,cab,b))
print(einsum('a,c,cab,b->a',a,c,cab,b))
print(einsum('a,c,cab,b->b',a,c,cab,b))


a = paddorch.arange(60.).reshape(3,4,5)
b = paddorch.arange(24.).reshape(4,3,2)

print(einsum('ijk,jil->kl',a, b))

c = paddorch.arange(25).reshape(5,5)

print(einsum('ii', c) )
print(einsum('ii->i', c) )