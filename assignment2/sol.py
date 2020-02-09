import numpy , random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#ret = minimize (objective, start, bounds=B, constraints=XC)
#alpha = ret['x']

#objective: A function that takes a vector alpha as arg and return a scalar values.
#start: a vector with the initial guess of the alpha vector  can be : start = numpy.zeros(N)
#B: B is a list of pairs of the same length as the alpha vector, stating the lower and upper bounds
#   for the corresponding element in alpha    can be: bounds=[(0, C) for b in range(N)]
#                                          can be bounds=[(0, None) for b in range(N)].



POWER = 2

SIGMA = 3

C = 0.1

numpy.random.seed(100)
classA = numpy.concatenate ( (numpy.random.randn(10, 2) * 0.2 + [-1, 0],
                              numpy.random.randn(10, 2) * 0.2 + [1, 0])) 
classB = numpy.concatenate ( (numpy.random.randn(10, 2) * 0.2 + [0, 0],
                              numpy.random.randn(10, 2) * 0.2 + [2.5, 0])) 

inputs = numpy.concatenate (( classA , classB )) 
targets = numpy. concatenate ((numpy.ones(classA.shape[0]) , 
                               -numpy.ones(classB.shape[0])))
N = inputs.shape [0] # Number of rows (samples)
permute=list(range(N)) 
random.shuffle(permute) 
inputs = inputs [ permute , : ]
targets = targets [ permute ]

from operator import itemgetter

def preCompute(kernel):
    n = len(targets)
    P = numpy.zeros((n,n))
    for i in range(n):
        for j in range(n):
            P[i,j] = targets[i]*targets[j]*kernel(inputs[i],inputs[j])
    return P

def linearKernel(x,y):
    return numpy.dot(x,y)


def zerofun(alpha):
    return numpy.dot(alpha,targets)


def polynomialKernel(x,y):
    return (linearKernel(x,y) + 1)**POWER

def radialBasisKernel(x,y):
    exponent = ((numpy.linalg.norm(numpy.subtract(x,y)))**2)/(-2*SIGMA*SIGMA)
    return math.exp(exponent)

P1 = preCompute(polynomialKernel)

def objective(alpha):
    return 0.5*numpy.dot(alpha, numpy.dot(alpha, P1)) - numpy.sum(alpha)

start = numpy.zeros(N)
B=[(0, C) for b in range(N)]
XC={'type':'eq', 'fun':zerofun}

ret = minimize(objective, start, bounds=B, constraints=XC)
alpha = ret['x']

#extract the alpha non-zero
def alpha_no_zero(alpha):
    true_list = [x> 10**-5 for x in alpha ] 
    index_list = [i for i, x in enumerate(true_list) if x]
    alpha_extract = numpy.array(itemgetter(*index_list)(alpha))
    return alpha_extract

def targets_no_zero(targets, alpha):
    true_list = [x> 10**-5 for x in alpha ] 
    index_list = [i for i, x in enumerate(true_list) if x]
    targets_extract = numpy.array(itemgetter(*index_list)(targets))
    return targets_extract 

def inputs_no_zero(inputs, alpha):
    true_list = [x> 10**-5 for x in alpha ] 
    index_list = [i for i, x in enumerate(true_list) if x]
    inputs_extract = numpy.array(itemgetter(*index_list)(inputs))
    return inputs_extract 

alpha_extract = alpha_no_zero(alpha)
targets_extract = targets_no_zero(targets,alpha)
inputs_extract = inputs_no_zero(inputs,alpha)

#return the threshold b (equation 7)
def threshold(alpha_extract, targets_extract, inputs_extract, kernel):
    n = len(alpha_extract)
    b = numpy.zeros(n)
    index_margin = numpy.where(alpha_extract<C-10**-5)[0] #list of index of the margin value
    index1 = index_margin[0] #index of the first margin value
    for i in range(n):
        b[i] = numpy.dot(alpha_extract[i],numpy.dot(targets_extract[i],(kernel(inputs_extract[index1], inputs_extract[i]))))
    b = numpy.sum(b) - targets_extract[index1]
    return b

b = threshold(alpha_extract, targets_extract, inputs_extract, polynomialKernel)

#return a list with the margin points
def margin_list(alpha_extract):
    index_margin = numpy.where(alpha_extract<C-10**-7)[0] #list of index of the margin value
    n_index = index_margin.shape[0] #number of margin value
    list_margin = numpy.zeros((n_index,2))
    for i in range(n_index):
        list_margin[i] = inputs_extract[index_margin[i]]
    return list_margin

list_margin = margin_list(alpha_extract) 

def indicator(x,y):
    n = len(alpha_extract)
    ind = numpy.zeros(n)
    s = numpy.array([x,y])
    for i in range(n):      
        ind[i] = numpy.dot(alpha_extract[i], numpy.dot(targets_extract[i], polynomialKernel(s,inputs_extract[i]))) 
    ind = numpy.sum(ind) - b
    return ind

    

# Plot the point from classA in blue
plt.plot([p[0] for p in classA], [p[1] for p in classA],'b.')
# Plot the point from classA in red
plt.plot([p[0] for p in classB], [p[1] for p in classB],'r.') 
# Plot the margin points in magenta
plt.plot([p[0] for p in list_margin], [p[1] for p in list_margin],'m.') 


# Pot the Decision Boundary
xgrid=numpy.linspace(-5, 5) 
ygrid=numpy.linspace(-4, 4)
grid=numpy.array([[indicator(x, y) for x in xgrid ] for y in ygrid])

plt.contour( xgrid , ygrid , grid , 
            (-1.0, 0.0, 1.0), 
            colors=('red', 'black', 'blue'), 
            linewidths=(1, 3, 1))
plt.axis('equal') # Force same scale on both axes 
plt.savefig('svmplot.pdf') # Save a copy in a file 
plt.show() # Show the plot on the screen



'''
def main():
    print("working")
    x = numpy.array(1,2,3)
    print(x.length)

if __name__ == "__main__":
    main()
'''