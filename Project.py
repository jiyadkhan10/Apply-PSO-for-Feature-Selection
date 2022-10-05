#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from copy import deepcopy 
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# In[5]:


df = pd.read_excel('Training_Data.xlsx')
df


# In[6]:


df.isnull().sum()


# In[7]:


df.dtypes


# # Preprocessing

# In[11]:





# In[12]:


def Preprocessing():
    df['Race']=df['Race'].fillna('White')
    df['DiabetesMellitus'].fillna('unknown',inplace=True)
    df['ChronicKidneyDisease'].fillna('unknown',inplace=True)
    df['Anemia'].fillna('unknown',inplace=True)
    df['ChronicObstructivePulmonaryDisease'].fillna('unknown',inplace=True)
    df['Depression '].fillna('unknown',inplace=True)
    df["EncounterId"] = df["EncounterId"].astype('str')
    df['EncounterId'] = df['EncounterId'].map(lambda rmv: rmv.lstrip('PH'))
    df['EncounterId'] = df['EncounterId'].map(lambda rmv: rmv.lstrip('W'))
    df['EncounterId'] = df['EncounterId'].map(lambda rmv: rmv.lstrip('V'))
    df['EncounterId'] = df['EncounterId'].map(lambda rmv: rmv.lstrip('D')) 
    df['PatientId']= labelencoder.fit_transform(df['PatientId'])
    df['EncounterId']= labelencoder.fit_transform(df['EncounterId'])
    df['DischargeDisposision']= labelencoder.fit_transform(df['DischargeDisposision'])
    df['Gender']= labelencoder.fit_transform(df['Gender'])
    df['Race']= labelencoder.fit_transform(df['Race'])
    df['DiabetesMellitus']= labelencoder.fit_transform(df['DiabetesMellitus'])
    df['ChronicKidneyDisease']= labelencoder.fit_transform(df['ChronicKidneyDisease'])
    df['Anemia']= labelencoder.fit_transform(df['Anemia'])
    df['Depression ']= labelencoder.fit_transform(df['Depression '])
    df['ChronicObstructivePulmonaryDisease']= labelencoder.fit_transform(df['ChronicObstructivePulmonaryDisease'])
    df['Age']= labelencoder.fit_transform(df['Age'])
    df['ChronicDiseaseCount']= labelencoder.fit_transform(df['ChronicDiseaseCount'])
    df['LengthOfStay']= labelencoder.fit_transform(df['LengthOfStay'])
    df['EmergencyVisit']= labelencoder.fit_transform(df['EmergencyVisit'])
    df['InpatientVisit']= labelencoder.fit_transform(df['InpatientVisit'])
    df['OutpatientVisit']= labelencoder.fit_transform(df['OutpatientVisit'])
    df['TotalVisits']= labelencoder.fit_transform(df['TotalVisits'])
    df['BMIMin']= labelencoder.fit_transform(df['BMIMin'])
    df['BMIMax']= labelencoder.fit_transform(df['BMIMax'])
    df['BMIMedian']= labelencoder.fit_transform(df['BMIMedian'])
    df['BMIMean']= labelencoder.fit_transform(df['BMIMean'])
    df['BPDiastolicMin']= labelencoder.fit_transform(df['BPDiastolicMin'])
    df['BPDiastolicMax']= labelencoder.fit_transform(df['BPDiastolicMax'])
    df['BPDiastolicMin']= labelencoder.fit_transform(df['BPDiastolicMin'])
    df['BPSystolicMedian']= labelencoder.fit_transform(df['BPSystolicMedian'])
    df['BPSystolicMean']= labelencoder.fit_transform(df['BPSystolicMean'])
    df['HeartRateMin']= labelencoder.fit_transform(df['HeartRateMin'])
    df['HeartRateMax']= labelencoder.fit_transform(df['HeartRateMax'])
    df['HeartRateMedian']= labelencoder.fit_transform(df['HeartRateMedian'])
    df['HeartRateMean']= labelencoder.fit_transform(df['HeartRateMean'])
    df['PulseRateMin']= labelencoder.fit_transform(df['PulseRateMin'])
    df['PulseRateMax']= labelencoder.fit_transform(df['PulseRateMax'])
    df['PulseRateMean']= labelencoder.fit_transform(df['PulseRateMean'])
    df['PulseRateMedian']= labelencoder.fit_transform(df['PulseRateMedian'])
    df['RespiratoryRateMin']= labelencoder.fit_transform(df['RespiratoryRateMin'])
    df['RespiratoryRateMax']= labelencoder.fit_transform(df['RespiratoryRateMax'])
    df['RespiratoryRateMedian']= labelencoder.fit_transform(df['RespiratoryRateMedian'])
    df['RespiratoryRateMean']= labelencoder.fit_transform(df['RespiratoryRateMean'])
    df['ACEInhibitors']= labelencoder.fit_transform(df['ACEInhibitors'])
    df['ARBs']= labelencoder.fit_transform(df['Race'])
    df['BetaBlockers']= labelencoder.fit_transform(df['BetaBlockers'])
    df['Diuretics']= labelencoder.fit_transform(df['Diuretics'])
    df['TotalMedicine']= labelencoder.fit_transform(df['TotalMedicine'])
    df['CardiacTroponin']= labelencoder.fit_transform(df['CardiacTroponin'])
    df['Hemoglobin']= labelencoder.fit_transform(df['Hemoglobin'])
    df['SerumSodium']= labelencoder.fit_transform(df['SerumSodium'])
    df['SerumCreatinine']= labelencoder.fit_transform(df['SerumCreatinine'])
    df['BNP']= labelencoder.fit_transform(df['BNP'])
    df['ReadmissionWithin_90Days ']= labelencoder.fit_transform(df['ReadmissionWithin_90Days'])




# In[13]:


Preprocessing()


# In[14]:


df.head()


# # Creating Population

# In[15]:


df.columns


# In[16]:


df['ReadmissionWithin_90Days'].unique


# In[17]:


cols=df.columns.to_list()
cols.remove("ReadmissionWithin_90Days")
cols


# In[18]:


def Create_population():
    col=df.columns.to_list()
    temp=[]
    index=0
    for i in range(0,40):
        listt=list()
        listt=random.sample(cols,30)
        if listt not in population:
            for j in range(0,len(col)):
                if col[j] in listt:
                    temp.append(1)
                else:
                    temp.append(0)
                
                    
            population.append(temp)
            temp=[]
        print(listt)
        print()
        print(col)
        print()
        print(population)


# In[19]:


population=[]

Create_population()

for i in range(0,40):
    print(len(population[i]))


# In[20]:


def pop_init(size,n_feat):
    populationn = [[random.randint(0, 1) for i in range(n_feat)] for i in range(size)]
    return populationn


# In[21]:


pop=pop_init(10,56)
print(pop)


# # Fitness Function

# In[43]:


#creating temporary dataframe
tempdf=pd.read_excel('Training_Data.xlsx')
tempdf.head()


# In[25]:


def SPLIT(df,label):
    X_train, X_test, Y_train, Y_test = train_test_split(df, label, test_size=0.25, random_state=38)
    return X_train, X_test, Y_train, Y_test


# In[26]:


df.head()


# In[27]:


label_col = df["ReadmissionWithin_90Days"]
label_col = np.where(label_col == 'Yes',1,0)


# In[51]:


index=3


# In[28]:


print(label_col)


# In[29]:


df.drop(["ReadmissionWithin_90Days"],axis = 1,inplace = True)


# In[30]:


rfmodel = RandomForestClassifier(n_estimators=192, random_state=2)


# In[31]:


X_train,X_test, Y_train, Y_test = SPLIT(df,label_col)


# In[32]:


fit=[]


# In[33]:


def fitness(data):
   
    temp_pop= (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))
    
    for i in range(0,len(temp_pop)):
        if temp_pop[i] > 0.5:
            temp_pop[i]=1
        else:
            temp_pop[i]=0
            
    cols = [column for (column, binary_value) in zip(df.columns, temp_pop) if binary_value]
    
    rfmodel.fit(X_train[cols],Y_train) 
    pred = rfmodel.predict(X_test[cols])
    score = accuracy_score(Y_test,pred)    
    print(score)                              
    return score


# In[34]:


print(fitness(pop[0]))


# In[36]:


print("Fitness before PSO Algo: ")
for i in pop:
    fitness(i)
    


# # PSO Algorithm

# In[38]:


def vel_iter():
    return [[random.random() for i in range(56)] for i in range(10)]


# In[39]:


def evalute_vel(weight,vel_iter,c1,r1,per_best,particle,c2,r2,global_best):
    return weight*np.array(vel_iter) + c1* r1* (np.array(per_best[0]) - np.array(particle)) + c2* r2* (np.array(global_best) - np.array(particle))


# In[40]:


def PSO(pop,i):
    weight=0.5  #constant inertia weight
    c1=1   #cognative constant
    c2=2   #social constant
    r1=0.001
    r2=0.007
    vel=vel_iter()
    global_best=pop[0]
    per_best=pop[0]
    velocity=[]
    score=[]
    
    velocity.append(evalute_vel(weight,vel[0],c1,r1,per_best,pop[0],c2,r2,global_best))
    velocity.append(evalute_vel(weight,vel[1],c1,r1,per_best,pop[1],c2,r2,global_best))
    velocity.append(evalute_vel(weight,vel[2],c1,r1,per_best,pop[2],c2,r2,global_best))
    velocity.append(evalute_vel(weight,vel[3],c1,r1,per_best,pop[3],c2,r2,global_best))
    velocity.append(evalute_vel(weight,vel[4],c1,r1,per_best,pop[4],c2,r2,global_best))
    velocity.append(evalute_vel(weight,vel[5],c1,r1,per_best,pop[5],c2,r2,global_best))
    velocity.append(evalute_vel(weight,vel[6],c1,r1,per_best,pop[6],c2,r2,global_best))
    velocity.append(evalute_vel(weight,vel[7],c1,r1,per_best,pop[7],c2,r2,global_best))
    velocity.append(evalute_vel(weight,vel[8],c1,r1,per_best,pop[8],c2,r2,global_best))
    velocity.append(evalute_vel(weight,vel[9],c1,r1,per_best,pop[9],c2,r2,global_best))
    
    for i in range(0,i):
        print('Calculated Velocity')
        
        for j in range(0,10):
            velocity[j]=evalute_vel(weight,vel[j],c1,r1,per_best,pop[j],c2,r2,global_best)
            print(velocity[j])
        print('Updated Position')   
        for k in range(0,10):
            pop[k]=np.array(velocity[k])+np.array(pop[k])
            print(pop[k])
            per_best=velocity(population[i])
        if per_best>fitness[i]:
            fitness[i]=per_bestfit
            per_best[i]=population[i]
          
    globalbest=personalbest[fitness.index(max(fitness))]
        
        print('Fitness score after PSO algo: ')
        
        for x in range(0,10):
            acc_score=fitness(pop[x])
            score.append(acc_score)
            print(acc_score)
            #print("Max Score: ")
            #print(max(acc_score))
            
      
    
    
 


# In[41]:


PSO(pop,10)


# In[ ]:





# In[49]:


print(listt)


# In[52]:


print(population[index])


# In[ ]:





# In[ ]:





# In[ ]:




