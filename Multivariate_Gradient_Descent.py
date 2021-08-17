# USING "ENERGY_EFF.CSV" 
import numpy as np
import matplotlib.pyplot as mpl
import csv

file=csv.reader(open('energy_eff.csv'))
data=np.array(list(file)).astype(float)
def rescale(x):
    if type(x)!='numpy.ndarray': x=np.array(x)
    mean=np.mean(x)
    std=np.std(x)
    return((x-mean)/std)
for i in range(7):
    data[:,i]=rescale(data[:,i])

b=[0.01,0.01,0.01,0.01,0.01]
x=np.array([[data[i,0],data[i,1],data[i,2],data[i,3],data[i,4]] for i in range(len(data[:,1]))])
y6=data[:,6]
#y6p=b0*x0 + b1*x1 + b2*x2 + b3*x3 + b4*x4 + c

def prediction(x,b):
    y6p=[np.dot(xi,b) for xi in x ]
   # print('y6p created')
    return(y6p)
def error(y6,y6p):
    return( (np.sum((y6p-y6)**2))/len(y6)  )
def gradient_bi(xi,y6,y6p):
    gradient=(np.sum(xi*(y6p-y6)))/len(y6)
    return(gradient)

err,it,rate,max_it=100,0,0.001,5000

while (err>0.01)&(it<max_it):
    y6p  =prediction(x,b)
    for i in range(5):
        gradient=gradient_bi(x[:,i],y6,y6p)
        b[i]=b[i]-rate*gradient
    err=error(y6,y6p)
    it+=1
    print(it)

mpl.plot(y6,'b-',y6p,'r-')   
