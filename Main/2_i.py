#!/usr/bin/env python
# coding: utf-8

# ### Author :: Anup Patel (Mtech CSA - 15474)
# ### Library Import

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean


# ### Input File Read 

data=pd.read_csv("./../data/01_data_mars_opposition.csv")

# ### Sun Longitude 

mars_heliocentric_longitude=data.iloc[:,3:7]

s=data["ZodiacIndex"].values
degree=data["Degree"].values
minute=data["Minute"].values
seconds=data["Second"].values

mars_heliocentric_longitude_in_degree= s*30 +degree + (minute/60) + (seconds/3600)
mars_heliocentric_longitude_in_radian= mars_heliocentric_longitude_in_degree*math.pi/180.0

geocentric_latitude=data.iloc[:,7:9]

#Not Required for First Part
geocentric_latitude_in_radian=(geocentric_latitude["LatDegree"].values * math.pi/180 )+ (geocentric_latitude["LatMinute"].values *math.pi/(60*180.0))

# ### Average Sun Longitude 
mars_mean_longitude=data.iloc[:,9:13]

s_mean=mars_mean_longitude["ZodiacIndexAverageSun"].values
degree_mean=mars_mean_longitude["DegreeMean"].values
minute_mean=mars_mean_longitude["MinuteMean"].values
seconds_mean=mars_mean_longitude["SecondMean"].values

mars_mean_longitude_in_degree=s_mean*30 +degree_mean + (minute_mean/60) + (seconds_mean/3600.0)

mars_mean_longitude_in_radian=mars_mean_longitude_in_degree*math.pi/180


# ### Optimization 

alpha=mars_mean_longitude_in_radian
beta=mars_heliocentric_longitude_in_radian

def loss_function(params,args):
    x_list=[]
    y_list=[]
    r_list=[]
    a=params[0]
    b=params[1]
    alpha=args[0]
    beta=args[1]
    for i in range(len(data)):
        x=((-1-a)*np.sin(b) + (a*np.tan(alpha[i]) + np.tan(beta[i]))*np.cos(b))/(np.tan(alpha[i]) - np.tan(beta[i]))
        y=a*np.sin(b) + (np.tan(alpha[i])* (x - a*np.cos(b)))
        #print(x)
        r=np.sqrt(x**2 + y**2)
        x_list.append(x)
        y_list.append(y)
        r_list.append(r)
    #print(x_list)   
    ap=np.mean(r_list)
    gp=gmean(r_list)
    
    #print((math.log(ap,10) - math.log(gp,10)))
    return (math.log(ap,10) - math.log(gp,10))

def loss_function_variance(params,args):
    x_list=[]
    y_list=[]
    r_list=[]
    a=params[0]
    b=params[1]
    alpha=args[0]
    beta=args[1]
    for i in range(len(data)):
        x=((-1-a)*np.sin(b) + (a*np.tan(alpha[i]) + np.tan(beta[i]))*np.cos(b))/(np.tan(alpha[i]) - np.tan(beta[i]))
        y=a*np.sin(b) + (np.tan(alpha[i])* (x - a*np.cos(b)))
        #print(x)
        r=np.sqrt(x**2 + y**2)
        x_list.append(x)
        y_list.append(y)
        r_list.append(r)
    #print(x_list)   
    var=np.var(r_list)
    
    #print((math.log(ap,10) - math.log(gp,10)))
    return var

def optimizer(function,method_name,alpha,beta):
    
    
    a=[1.2]
    b=[0.2]
    initial_parameters = np.array(a+b) #Random Values
    #bound to avoid case of global Minima where Loss = 0
    bounds = [(0, np.inf) for _ in a] + [(-np.inf, np.inf)]
    
    parameters = minimize(function, initial_parameters,
                      args=[alpha,
                            beta
                            ],
                      method=method_name,bounds=bounds)
    #optimized_params, loss = parameters['x'], parameters['fun']
    #print(optimized_params1)
    #print(squared_error_loss1)
    return parameters['x'], parameters['fun']


# ### log(AM) -log(GM) Loss Function 

print("Optimizing Parameters .... ")
function_name=loss_function
optimized_params, loss= optimizer(function_name,'L-BFGS-B',alpha,beta)
print("Optimized Parameters Computed")


#optimized_params


print("Optimized Parameters = " + str(optimized_params))
print("Loss = " + str(loss))


# ### Testing of Result  (AM-GM)

#Plot to Check if Circle type structure is formed or not
a=optimized_params[0]
b=optimized_params[1]
x_list=[]
y_list=[]
r_list=[]
for i in range(len(data)):
        x=((-1-a)*np.sin(b) + (a*np.tan(alpha[i]) + np.tan(beta[i]))*np.cos(b))/(np.tan(alpha[i]) - np.tan(beta[i]))
        y=a*np.sin(b) + (np.tan(alpha[i])* (x - a*np.cos(b)))
        #print(x)
        r=np.sqrt(x**2 + y**2)
        x_list.append(x)
        y_list.append(y)
        r_list.append(r)


print("Radius = " + str(np.mean(r_list)))

import matplotlib.pyplot as plt
circle1 = plt.Circle((0, 0), np.mean(r_list),fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle1)
plt.scatter(x_list,y_list,color='red')
plt.savefig('plot.png')



