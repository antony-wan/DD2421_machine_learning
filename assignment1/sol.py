from monkdata import monk1, monk2, monk3
from dtree import *
#from drawtree_qt5 import drawTree
import monkdata as m
import dtree as d
import random
from statistics import mean, variance, stdev
#import matplotlib.pyplot as plt

"""
def partition(data, fraction):
  ldata = list(data)
  random.shuffle(ldata)
  breakPoint = int(len(ldata) * fraction)
  return ldata[:breakPoint], ldata[breakPoint:]


def bestTreePerf(dataset,frac):
    training, validation = partition(dataset, frac)
    bestTree = d.buildTree(training, m.attributes)
    bestPerf = d.check(bestTree, validation)

    while "pruned is better":
        prunedTrees = d.allPruned(bestTree)
        if len(prunedTrees) == 0:
            return bestTree,bestPerf

        nextTree, nextPerf = bestFromPrunedTree(prunedTrees,validation)

        if nextPerf > bestPerf:
            bestPerf = nextPerf
            bestTree = nextTree
        else:
            return bestTree,bestPerf


def bestFromPrunedTree(prunedTrees, validation):
    maxPerf = 0
    for t in prunedTrees:
        nextPerf = d.check(t,validation)
        if nextPerf > maxPerf:
            maxPerf = nextPerf
            bestTree = t
    return bestTree, maxPerf
"""




def main():

  #ASSIGNMENT 1
  print("MONK-1 entropy : " + str(entropy(monk1)))
  print("MONK-2 entropy : " + str(entropy(monk2)))
  print("MONK-3 entropy : " + str(entropy(monk3)))


  #ASSIGNEMENT 3
  print("MONK-1 info gain")
  print("a1 : " + str(averageGain(monk1,m.attributes[0])))
  print("a2 : " + str(averageGain(monk1,m.attributes[1])))
  print("a3 : " + str(averageGain(monk1,m.attributes[2])))
  print("a4 : " + str(averageGain(monk1,m.attributes[3])))
  print("a5 : " + str(averageGain(monk1,m.attributes[4])))
  print("a6 : " + str(averageGain(monk1,m.attributes[5])))
  print('\n')

  print("MONK-2 info gain")
  print("a1 : " + str(averageGain(monk2,m.attributes[0])))
  print("a2 : " + str(averageGain(monk2,m.attributes[1])))
  print("a3 : " + str(averageGain(monk2,m.attributes[2])))
  print("a4 : " + str(averageGain(monk2,m.attributes[3])))
  print("a5 : " + str(averageGain(monk2,m.attributes[4])))
  print("a6 : " + str(averageGain(monk2,m.attributes[5])))
  print('\n')

  print("MONK-3 info gain")
  print("a1 : " + str(averageGain(monk3,m.attributes[0])))
  print("a2 : " + str(averageGain(monk3,m.attributes[1])))
  print("a3 : " + str(averageGain(monk3,m.attributes[2])))
  print("a4 : " + str(averageGain(monk3,m.attributes[3])))
  print("a5 : " + str(averageGain(monk3,m.attributes[4])))
  print("a6 : " + str(averageGain(monk3,m.attributes[5])))
  print('\n')


  #ASSIGNEMENT 5
  sub1 = select(monk1,m.attributes[4],1)
  sub2 = select(monk1,m.attributes[4],2)
  sub3 = select(monk1,m.attributes[4],3)
  sub4 = select(monk1,m.attributes[4],4)

  print("sub1 info gain")
  print("a1 : " + str(averageGain(sub1,m.attributes[0])))
  print("a2 : " + str(averageGain(sub1,m.attributes[1])))
  print("a3 : " + str(averageGain(sub1,m.attributes[2])))
  print("a4 : " + str(averageGain(sub1,m.attributes[3])))
  print("a5 : " + str(averageGain(sub1,m.attributes[4])))
  print("a6 : " + str(averageGain(sub1,m.attributes[5])))
  print

  print("sub2 info gain")
  print("a1 : " + str(averageGain(sub2,m.attributes[0])))
  print("a2 : " + str(averageGain(sub2,m.attributes[1])))
  print("a3 : " + str(averageGain(sub2,m.attributes[2])))
  print("a4 : " + str(averageGain(sub2,m.attributes[3])))
  print("a5 : " + str(averageGain(sub2,m.attributes[4])))
  print("a6 : " + str(averageGain(sub2,m.attributes[5])))
  print

  print("sub3 info gain")
  print("a1 : " + str(averageGain(sub3,m.attributes[0])))
  print("a2 : " + str(averageGain(sub3,m.attributes[1])))
  print("a3 : " + str(averageGain(sub3,m.attributes[2])))
  print("a4 : " + str(averageGain(sub3,m.attributes[3])))
  print("a5 : " + str(averageGain(sub3,m.attributes[4])))
  print("a6 : " + str(averageGain(sub3,m.attributes[5])))
  print

  print("sub4 info gain")
  print("a1 : " + str(averageGain(sub4,m.attributes[0])))
  print("a2 : " + str(averageGain(sub4,m.attributes[1])))
  print("a3 : " + str(averageGain(sub4,m.attributes[2])))
  print("a4 : " + str(averageGain(sub4,m.attributes[3])))
  print("a5 : " + str(averageGain(sub4,m.attributes[4])))
  print("a6 : " + str(averageGain(sub4,m.attributes[5])))
  print

  t=d.buildTree(m.monk1, m.attributes,2);
  #drawTree(t)

  t1=d.buildTree(m.monk1, m.attributes);

  print("Perf on test data (MONK-1): " + str(d.check(t1, m.monk1test)))
  print("Perf on training data (MONK-1): " + str(d.check(t1, m.monk1)))
  print("Error: " + str(1-d.check(t1, m.monk1test)))
  print

  t2=d.buildTree(m.monk2, m.attributes);
  print("Perf on test data (MONK-2): " + str(d.check(t2, m.monk2test)))
  print("Perf on training data (MONK-2): " + str(d.check(t2, m.monk2)))
  print("Error: " + str(1-d.check(t2, m.monk2test)))
  print

  t3=d.buildTree(m.monk3, m.attributes);
  print("Perf on test data (MONK-3): " + str(d.check(t3, m.monk3test)))
  print("Perf on training data (MONK-3): " + str(d.check(t3, m.monk3)))
  print("Error: " + str(1-d.check(t3, m.monk3test)))
  print

"""
  #ASSIGNMENT 7

fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
meanErrorMonk1 = []
meanErrorMonk3 = []
stdevErrorMonk1 = []
stdevErrorMonk3 = []

N = 100
print("MONK1 errors on pruned")
for i in range(0, len(fractions)):
    errors_test = []
    for j in range(0, N):
        bestTree, bestPerf = bestTreePerf(m.monk1, fractions[i])
        errors_test.append(1-check(bestTree, m.monk1test))
    meanErrorMonk1.append(mean(errors_test))
    stdevErrorMonk1.append(stdev(errors_test))
    print(str(fractions[i])+ " : err = " + str(meanErrorMonk1[i]) + " var = " + str(stdevErrorMonk1[i]))

print("MONK3 errors on pruned")

for i in range(0,len(fractions)):
    errors_validation = []
    errors_test = []
    for j in range(0, N):
        bestTree, bestPerf = bestTreePerf(m.monk3, fractions[i])
        errors_test.append(1-check(bestTree, m.monk3test))
    meanErrorMonk3.append(mean(errors_test))
    stdevErrorMonk3.append(stdev(errors_test))
    print(str(fractions[i])+ " : err = " + str(meanErrorMonk3[i]) + " var = " + str(stdevErrorMonk3[i]))

plt.figure(1)
plt.errorbar(fractions, meanErrorMonk1, yerr=stdevErrorMonk1, fmt='o')
plt.xlabel("Fraction between training data and validation data")
plt.ylabel("Classification error")
plt.title("Classification error")
plt.show()
plt.savefig("monk_01.png")

plt.figure(2)
plt.errorbar(fractions, meanErrorMonk3, yerr=stdevErrorMonk3, fmt='o')
plt.xlabel("Fraction between training data and validation data")
plt.ylabel("Classification error")
plt.title("Classification error")
plt.show()
plt.savefig("monk_02.png")


"""

main()
