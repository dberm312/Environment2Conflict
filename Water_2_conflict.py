# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:37:15 2019

@author: Berma
"""
#%%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm
import pysal
import matplotlib.pyplot as plt
import datetime
#import geopandas
#%%
Water = pd.read_csv('https://storage.googleapis.com/global-surface-water-stats/gaul1-all-2018.csv')#import all water data
Water = Water[Water.ADM0_NAME.isin(['Kenya',
                                    'Ethiopia',
                                    'Somalia'])]#limit the countries 
Water.ADM1_NAME=Water.ADM1_NAME.apply(lambda x: x.replace(" ",''))
Water = Water[Water.year.isin(list(range(1997,2018)))]#limit the dates
Water_agg= Water.groupby(['ADM1_NAME','year']).mean()#Aggrigate the data
Water_agg.insert(8,'Perm_diff',(
        (Water_agg['permanent']-Water_agg['5yr_Permanent'])/Water_agg['5yr_Permanent']))#.fillna(0))
Water_agg.insert(9,'Seas_diff',(
        (Water_agg['seasonal']-Water_agg['5yr_Seasonal'])/Water_agg['5yr_Seasonal']))#.fillna(0))
Water_perm_diff = Water_agg.Perm_diff.unstack()#[5:]
Water_seas_diff = Water_agg.Seas_diff.unstack()#[5:]
Water_perm_diff.index = list(map((lambda x: x.replace(" ", '')),Water_perm_diff.index.tolist()))

#%%
loc = pd.DataFrame(columns = ['Region', 'Country', 'Latitude', 'Longitude'],
                   data = [['AddisAbaba', ' Ethiopia', 8.9806, 38.7578],
                           ['Afar', ' Ethiopia', 11.7559, 40.9587],
                           ['Amhara', ' Ethiopia', 11.3494, 37.9785],
                           ['BeneshangulGumu', ' Ethiopia', 10.7803, 35.5658],
                           ['DireDawa', ' Ethiopia', 9.6009, 41.8501],
                           ['Gambela', ' Ethiopia', 8.2472, 34.5916],
                           ['Hareri', ' Ethiopia', 9.3149, 42.1968],
                           ['SNNPR', ' Ethiopia', 6.5157, 36.9541],
                           ['Tigray', ' Ethiopia', 14.0323, 38.3166],
                           ['Awdal', ' Somalia', 10.6334, 43.3295],
                           ['Bakool', ' Somalia', 4.3657, 44.096],
                           ['Bari', ' Somalia', 10.1204, 49.6911],
                           ['Bay', ' Somalia', 2.4825, 43.4837],
                           ['Banadir', ' Somalia', 2.1187, 45.3369],
                           ['Galgaduud', ' Somalia', 5.185, 46.8253],
                           ['Gedo', ' Somalia', 3.5039, 42.2362],
                           ['Hiraan', ' Somalia', 4.321, 45.2994],
                           ['JubaHoose', ' Somalia', 0.224, 41.6012],
                           ['ShabelleHoose', ' Somalia', 1.8766, 44.2479],
                           ['JubaDhexe', ' Somalia', 2.078, 41.6012],
                           ['ShabelleDhexe', ' Somalia', 2.925, 45.904],
                           ['Mudug', ' Somalia', 6.5657, 47.7638],
                           ['Nugaal', ' Somalia', 8.2173, 49.2031],
                           ['Sanaag', ' Somalia', 10.3938, 47.7638],
                           ['Sool', ' Somalia', 8.7222, 47.7638],
                           ['Togdheer', ' Somalia', 9.4461, 45.2994],
                           ['WoqooyiGalbeed', ' Somalia', 9.5424, 44.096],
                           ['Oromia', ' Ethiopia', 7.9891, 39.3812],
                           ['Somali', ' Ethiopia', 6.6612, 43.7908],
                           ['Central', ' Kenya', -0.4026, 36.8951],
                           ['Coast', ' Kenya', -4.0427, 39.6649],
                           ['Eastern', ' Kenya', 2.4225, 38.0173],
                           ['Nairobi', ' Kenya', -1.2936, 36.8024],
                           ['NorthEastern', ' Kenya', 0.2848, 40.4206],
                           ['Nyanza', ' Kenya', -0.0646, 34.7558],
                           ['RiftValley', ' Kenya', 0.618, 35.1937],
                           ['Western', ' Kenya', 0.6371, 34.5326]])
#loc = loc.set_index('Region')
neigh =  KNeighborsClassifier(weights = 'distance')
neigh.fit(loc.drop(columns=['Country','Region']).values,loc.Region)
#%%
files = ["/Users/Berma/Documents/Kenya_pop.csv",
         "/Users/Berma/Documents/Somalia_pop.csv",
         "/Users/Berma/Documents/Ethiopia_pop.csv"]
pop =pd.DataFrame()
for i in files:
    pop= pop.append(pd.read_csv(i))
pop = pop.reset_index()
pop.insert(100,'ADM1_NAME',neigh.predict(pop[['INSIDE_Y','INSIDE_X']]))
pop = pop.groupby('ADM1_NAME').sum()
pop = pop[['UN_2000_E', 'UN_2005_E', 'UN_2010_E', 'UN_2015_E', 'UN_2020_E']]
pop.columns = list(range(2000,2021,5))
#%% Import Conflict data from the last 20 years from https://www.acleddata.com
for i in [['Battles'],['Explosions/Remote violence','Violence against civilians'],['Protests','Riots']]:
    Conflict = pd.read_csv("/Users/Berma/Downloads/1998-01-01-2017-12-31-Ethiopia-Kenya-Somalia.csv")
    Conflict.insert(6,'Season',Conflict['event_date'].apply(lambda x: np.floor((datetime.datetime.strptime(x, '%d %B %Y').month-1)/3)))
    Conflict_filter = Conflict[Conflict.source.isin(
            ['All Africa',
             'Shabelle Media Network',
             'AFP',
             'AP',
             'Reuters',
             'BBC News'])]#We only look at these news sources to ensure consistancy over time and stationary
    Conflict_filter=Conflict_filter[Conflict_filter.event_type.isin(i)]
    hold = neigh.predict(
            Conflict_filter[
                    ['latitude',
                     'longitude']].values)#Assigning all the conflict to respective areas
    Conflict_filter.insert(32,'ADM1_NAME',hold)
    del(hold)
    
    Conflict_agg = Conflict_filter.groupby(['year','ADM1_NAME']).count().data_id.unstack().fillna(0)
    Conflict_rolling_mean =Conflict_agg.rolling(3).mean().iloc[4:]
    Conflict_agg=Conflict_agg[4:]
    Conflict_diff = ((Conflict_agg-Conflict_rolling_mean)/Conflict_rolling_mean)#.fillna(0)
    
    y = pd.DataFrame(Conflict_diff.fillna(0).stack(),columns=['Con_diff'])
    X = pd.DataFrame(np.ones(len(y)),columns=['Constant'])
    X.index=Water_perm_diff.fillna(0).T[5:].stack().index
    X.insert(1,'Permanent water diff',Water_perm_diff.fillna(0).T[5:].stack().values)
    X.insert(2,'Seasonal water diff',Water_seas_diff.fillna(0).T[5:].stack().values)
    print(sm.OLS(y,X).fit().summary())

#%%seasonal adjustment

#%%
hold = pd.DataFrame(Conflict_diff.stack(),columns=['Conflict_diff'])
hold = hold.join(pd.DataFrame(Water_perm_diff.T.stack(),columns=['Water_perm_diff']),how='inner')
hold = hold.join(pd.DataFrame(Water_seas_diff.T.stack(),columns=['Water_seas_diff']),how='inner')
hold = hold.dropna()
y = hold.Conflict_diff
X =hold[['Water_perm_diff','Water_seas_diff']]
print(sm.OLS(y,sm.add_constant(X)).fit().summary())

print(pysal.model.spreg.ols.OLS(np.array([y.values]).T,
                                X.values,name_y='Conflict',
                                name_x=['Permanent water diff','Seasonal Water diff']).summary)

'''
hold = pd.DataFrame(Conflict_diff.unstack(),columns=['Conflict_diff'])
hold= hold.join(pd.DataFrame(Water_perm_diff.unstack(),columns=['Water_perm_diff']),how='inner')
hold = hold.join(pd.DataFrame(Water_seas_diff.unstack(),columns=['Water_seas_diff']),how='inner')
hold = hold.fillna(0)
y = hold.Conflict_diff
X =hold[['Water_perm_diff','Water_seas_diff']]
print(sm.OLS(y,sm.add_constant(X)).fit().summary())
print(pysal.model.spreg.ols.OLS(np.array([y.values]).T,
                                X.values,name_y='Change in Conflict',
                                name_x=['Permanent water diff','Seasonal Water diff']).summary)
'''
#%%
w = pysal.lib.io.open("/Users/Berma/Documents/Test.gal").read()
#%%
for j in [Conflict_diff,Water_perm_diff,Water_seas_diff]:
    hold = j.copy().T
    hold3 = []
    for i in range(2002,2018):
        hold1 = pysal.explore.esda.Moran(hold[i].fillna(0),w)
        hold3.append(hold1.I)
    plt.hist(hold3,cumulative=True,density=True,bins=100)
    plt.show()
#%%
for data in [Conflict_diff,Water_perm_diff,Water_seas_diff]:
    a = []
    b = []
    for i in w.neighbors:
        for k in range(2002,2018):
            if(np.isnan(Water_perm_diff[i][k])==False):
                a.append(Water_perm_diff[i][k])
                partb = []
                for j in w.neighbors[i]:
                    if(np.isnan(Conflict_diff[j][k])==False):
                        partb.append(Water_perm_diff[j][k])
                b.append(np.mean(partb))
    plt.scatter(a,b,s=1)
    plt.show()
#%%
hold= sm.add_constant(pd.DataFrame(
        (Conflict_filter.year+(Conflict_filter.Season/4)),
        columns = ['cat'])).groupby('cat').sum()
for i in [0,4,12,5*4]:
    plt.plot(hold.rolling(i).apply(np.mean).iloc[(i):])
#%%
#Conflict
df = pd.DataFrame(Conflict_agg.T.stack(),columns=['Conflict0'])
for k in [-1,-2]:
    hold = []
    for i,j in df.index:
        j=j+k
        try:
            hold.append(Conflict_agg[i][j])
        except:
            hold.append(np.nan)
    df.insert(len(df.columns),'Conflict'+str(k),hold)
#Neighbor Conflict
for k in [0,-1,-2]:
    hold = []
    for i,j in df.index:
        j=j+k
        try:
            hold.append(Conflict_agg[w.neighbors[i]].loc[j].mean())
        except:
            hold.append(np.nan)
    df.insert(len(df.columns),'NConflict'+str(l),hold)
#Perm Water
for k in [0,-1,-2]:
    hold = []
    for i,j in df.index:
        j=j+k
        try:
            hold.append(Water_agg['permanent'].unstack()[j][i])
        except:
            hold.append(np.nan)
    df.insert(len(df.columns),'PermW'+str(k),hold)
#Season Water
for k in [0,-1,-2]:
    hold = []
    for i,j in df.index:
        j=j+k
        try:
            hold.append(Water_agg['seasonal'].unstack()[j][i])
        except:
            hold.append(np.nan)
    df.insert(len(df.columns),'SeasW'+str(k),hold)
#%%
df1 = df.dropna(axis=0)
df1.pop('NConflict0')
train_dataset = df1.sample(frac=0.85,random_state=0)
test_dataset = df1.drop(train_dataset.index)
train_stats = train_dataset.describe()
train_stats.pop("Conflict0")
train_stats = train_stats.transpose()
print(train_stats)
train_labels = train_dataset.pop('Conflict0')
test_labels = test_dataset.pop('Conflict0')
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
#%%
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.leaky_relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.leaky_relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()
#%%
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
#%%
print(r2_score(test_labels,model.predict(test_dataset)))
#%%
reg = Lasso(alpha=0.1)
reg.fit(train_dataset,train_labels)
r2_score(test_labels,reg.predict(test_dataset))








