# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:42:57 2019

@author: Berma
"""
#%%
pop_change = pd.DataFrame(pop[2005]/pop[2000],columns=['00_05'])
pop_change.insert(1,'05_10',pop[2010]/pop[2005])
pop_change.insert(2,'10_15',pop[2015]/pop[2010])
pop_change.insert(3,'15_20',pop[2020]/pop[2015])
pop_change_change = pd.DataFrame(pop_change['05_10']/pop_change['00_05'],columns=['2005'])
pop_change_change.insert(1, '2010',pop_change['10_15']/pop_change['05_10'])
pop_change_change.insert(2, '2015',pop_change['15_20']/pop_change['10_15'])
#%%
df = pd.DataFrame(pop_change_change.stack(),columns = ['D_D_Pop'])#.index.rename(['ADM1_NAME','year'])
df.index=df.index.rename(['ADM1_NAME','year'])
#hold = pd.DataFrame(Water_perm_diff.T.rolling(5).mean().stack(),columns=['Water_perm_diff'])#.index.rename(['ADM1_NAME','year'])
#hold.index=hold.index.rename(['ADM1_NAME','year'])
hold = []
for i,j in df.index:
    hold.append(Water_perm_diff.T.rolling(5).mean()[i][int(j)])
df.insert(1,'Water_perm_diff',hold)

hold = []
for i,j in df.index:
    hold.append(Water_seas_diff.T.rolling(5).mean()[i][int(j)])
df.insert(1,'Water_Seas_diff',hold)
df = df.dropna()
y = df.pop('D_D_Pop')
#%%
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from scipy.spatial import Voronoi, voronoi_plot_2d
#%%
loc = pd.read_csv("/Users/Berma/Documents/Sub_sahara_loc.csv")
loc.ADM1_NAME= list(map(lambda x:x.replace(" ","").replace('-',''),loc.ADM1_NAME))
loc.ADM0_NAME= list(map(lambda x:x.replace(" ","").replace('-',''),loc.ADM0_NAME))
loc.City= list(map(lambda x:x.replace(" ",""),loc.City))
loc = loc.replace('#FIELD!',np.nan).dropna()
loc.Latitude = list(map(float,loc.Latitude))
loc.Longitude = list(map(float,loc.Longitude))
loc = loc[loc.Longitude<60]
loc = loc[loc.Longitude>-30]
loc = loc[loc.Latitude<30]
loc = loc[loc.ADM1_NAME!='Mukono']
#loc = loc[loc.ADM0_NAME!='Uganda']
#loc = loc[loc.ADM0_NAME!='Burundi']
loc = loc.set_index('ADM1_NAME')

cluster = MeanShift(1.5).fit(loc[['Latitude','Longitude']])
clusters = cluster.labels_
cluster.predict(loc[['Latitude','Longitude']])
loc.insert(4,'clusters',cluster.predict(loc[['Latitude','Longitude']]))
hold = loc.groupby('clusters').mean()
plt.scatter(hold.Longitude,hold.Latitude,s=1)
plt.show()
print(len(np.unique(clusters)))
#%%
hold1 = []
truthtable = []
for i in loc.index:
    if(loc.clusters[[i,'Zou']].isin(hold1)):
        hold1.append(loc.clusters[i])
        truthtable.append(True)
    else:
        truthtable.append(False)
#%%
vor = Voronoi(hold.values)
voronoi_plot_2d(vor,show_vertices=False)
plt.axis('square')
plt.ylim((-25,60))
plt.xlim((-33,30))
plt.show()
#%%
import numpy as np
from statsmodels.datasets import grunfeld
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)
# MultiIndex, entity - time
data = data.set_index(['firm','year'])
from linearmodels import PanelOLS
mod = PanelOLS(data.invest, data[['value','capital']], entity_effect=True)
res = mod.fit(cov_type='clustered', cluster_entity=True)
