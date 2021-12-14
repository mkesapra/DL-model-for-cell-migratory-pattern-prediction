#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 22:51:42 2021

@author: kesaprm
"""


from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import numpy as np
from matplotlib.lines import Line2D
import segPlots
import math
import scipy.sparse as sparse
import seaborn as sns
from statistics import mean
from pandas.plotting import autocorrelation_plot
from math import isclose


# read_file = pd.read_csv(r'M2_morphNmotility.txt')
# read_file.to_csv(r'M2_morphNmotility.csv',index=None)

df0 = pd.read_csv("M0_val_morphNmotility.txt") 
df0['imageName'] = 'M0'
df1 = pd.read_csv("M1_val_morphNmotility.txt") 
df1['imageName'] = 'M1'
df2 = pd.read_csv("M2_val_morphNmotility.txt") 
df2['imageName'] = 'M2'

#Append time-series speed values for M0
df0_speed = pd.read_csv("M0_val_Speed_allCells.txt")
df0_speed_arr = df0_speed.to_numpy()
df0['Speed_allCells'] =  df0_speed_arr.tolist()

#Append time-series persistence values for M0
df0_per = pd.read_csv("M0_val_Persistence_cellWise.txt")
df0_per_arr = df0_per.to_numpy()
df0['Per_allCells'] =  df0_per_arr.tolist()

# #Append time-series distance values for M0 
df0_dist = pd.read_csv("M0_val_Dist_travelled.txt")
df0_dist_arr = df0_dist.to_numpy()
df0['Dist_travelled'] =  df0_dist_arr.tolist()

# #Append time-series distance values for M0 
df0_cellAreaAll = pd.read_csv("M0_val_CellAreaAll.txt")
df0_cellAreaAll_arr = df0_cellAreaAll.to_numpy()
df0['CellAreaAll'] =  df0_cellAreaAll_arr.tolist()

# #Append time-series x vals for M0 
df0_cellCx = pd.read_csv("M0_val_CellCx.txt")
df0_cellCx_arr = df0_cellCx.to_numpy()
df0['CellCx'] =  df0_cellCx_arr.tolist()

# #Append time-series x vals for M0 
df0_cellCy = pd.read_csv("M0_val_CellCy.txt")
df0_cellCy_arr = df0_cellCy.to_numpy()
df0['CellCy'] =  df0_cellCy_arr.tolist()


#Append time-series speed values for M1
df1_speed = pd.read_csv("M1_val_Speed_allCells.txt")
df1_speed_arr = df1_speed.to_numpy()
df1['Speed_allCells'] =  df1_speed_arr.tolist()

#Append time-series persistence values for M1
df1_per = pd.read_csv("M1_val_Persistence_cellWise.txt")
df1_per_arr = df1_per.to_numpy()
df1['Per_allCells'] =  df1_per_arr.tolist()

# #Append time-series distance values for M1 - 5 frames dist
df1_dist = pd.read_csv("M1_val_Dist_travelled.txt")
df1_dist_arr = df1_dist.to_numpy()
df1['Dist_travelled'] =  df1_dist_arr.tolist()

# #Append time-series distance values for M0 
df1_cellAreaAll = pd.read_csv("M1_val_CellAreaAll.txt")
df1_cellAreaAll_arr = df1_cellAreaAll.to_numpy()
df1['CellAreaAll'] =  df1_cellAreaAll_arr.tolist()

# #Append time-series x vals for M1 
df1_cellCx = pd.read_csv("M1_val_CellCx.txt")
df1_cellCx_arr = df1_cellCx.to_numpy()
df1['CellCx'] =  df1_cellCx_arr.tolist()

# #Append time-series x vals for M1 
df1_cellCy = pd.read_csv("M1_val_CellCy.txt")
df1_cellCy_arr = df1_cellCy.to_numpy()
df1['CellCy'] =  df1_cellCy_arr.tolist()




# #Append time-series persistence values for M0 - 5 frames dist
# df1_per_5fr = pd.read_csv("M1_Persistence_5Fr_cellWise.txt")
# df1_per_5fr_arr = df1_per_5fr.to_numpy()
# df1['5fr_Per_allCells'] =  df1_per_5fr_arr.tolist()


#Append time-series speed values for M2
df2_speed = pd.read_csv("M2_val_Speed_allCells.txt")
df2_speed_arr = df2_speed.to_numpy()
df2['Speed_allCells'] =  df2_speed_arr.tolist()

#Append time-series persistence values for M2
df2_per = pd.read_csv("M2_val_Persistence_cellWise.txt")
df2_per_arr = df2_per.to_numpy()
df2['Per_allCells'] =  df2_per_arr.tolist()

# #Append time-series distance values for M0 - 5 frames dist
df2_dist = pd.read_csv("M2_val_Dist_travelled.txt")
df2_dist_arr = df2_dist.to_numpy()
df2['Dist_travelled'] =  df2_dist_arr.tolist()

# #Append time-series distance values for M0 
df2_cellAreaAll = pd.read_csv("M2_val_CellAreaAll.txt")
df2_cellAreaAll_arr = df2_cellAreaAll.to_numpy()
df2['CellAreaAll'] =  df2_cellAreaAll_arr.tolist()

# #Append time-series x vals for M2 
df2_cellCx = pd.read_csv("M2_val_CellCx.txt")
df2_cellCx_arr = df2_cellCx.to_numpy()
df2['CellCx'] =  df2_cellCx_arr.tolist()

# #Append time-series y vals for M2 
df2_cellCy = pd.read_csv("M2_val_CellCy.txt")
df2_cellCy_arr = df2_cellCy.to_numpy()
df2['CellCy'] =  df2_cellCy_arr.tolist()


# #Append time-series speed values for M0 - 5 frames dist
# df2_spd_5fr = pd.read_csv("M2_Speed_5Fr_cellWise.txt")
# df2_spd_5fr_arr = df2_spd_5fr.to_numpy()
# df2['Spd_5Fr_allCells'] =  df2_spd_5fr_arr.tolist()





df = pd.concat([df0,df1,df2], ignore_index=True)
num_frames = len(df1_per.columns)

# Create a new column Fil to have time series 0s and 1s. O if two values are not close to 2000px Area. 1 if two values are within 2000px area
df['Fil'] = df['CellAreaAll']*1
for k in range(0, df.CellAreaAll.size):
    #mean_val = np.mean(df.CellAreaAll[k])
    for j in range(1, len(df.CellAreaAll[k])):
        cur_val = df.CellAreaAll[k][j-1]
        next_val = df.CellAreaAll[k][j]
        if math.isclose(cur_val, next_val, abs_tol=5500):
            df['Fil'][k][j] = 1
        else:
            df['Fil'][k][j] = 0
        df['Fil'][k][0] = 1
        
# In the Fil column if an array has < 37 values, return False else if the array has all 1s then return true else return false
# This gives us the cells for which the area has not changed > 2000px across 37 frames
df['consider'] = ''
for k in range(0, df.Fil.size):
    if len(df.Fil[k]) != (num_frames + 1):
        df['consider'][k] = 'False';
    else:
        if np.count_nonzero(df.Fil[k]) == (num_frames +1):
            df['consider'][k] = 'True'
        else:
            df['consider'][k] = 'False'
# To check the number of cells with True(i.e the area not changed across): df.groupby('consider').count()



#columns = pd.DataFrame(df.loc[df['consider'] == 'True'],columns=['cell_num','ecc','sol','cmp','avgSpeed','directedness','persistence','cell_movm','VCL','VSL','VAC','ALH','cellSize','imageName','Speed_allCells','Per_allCells','Dist_travelled','CellCx','CellCy'])
columns = pd.DataFrame(df.loc[df['consider'] == 'True'],columns=['cell_num','ecc','sol','cmp','cellSize','imageName','Speed_allCells','Per_allCells','Dist_travelled','CellCx','CellCy','per','vicinity'])
columns.fillna(0, inplace = True)

# columns['normalized_vcl'] = (columns.VCL-columns.VCL.min())/(columns.VCL.max()-columns.VCL.min())
# columns['normalized_vsl'] = (columns.VSL-columns.VSL.min())/(columns.VSL.max()-columns.VSL.min())

# columns['LIN'] = (columns.VSL/columns.VCL)*100
# columns['STR'] = (columns.VSL/columns.VAC)*100
# columns['WOB'] = (columns.VAC/columns.VCL)*100

# #columns = columns[~columns.isin([np.nan, np.inf, -np.inf]).any(1)]
# columns['cos_cellMovm'] = np.cos(columns.cell_movm)



# columns['normalized_LIN'] = (columns.LIN-columns.LIN.min())/(columns.LIN.max()-columns.LIN.min())
# columns['normalized_cellSize'] = (columns.cellSize-columns.cellSize.min())/(columns.cellSize.max()-columns.cellSize.min())

# ##k means using motility params
# kmeans = KMeans(n_clusters=4)
# y = kmeans.fit_predict(columns[['normalized_LIN','normalized_cellSize','normalized_vcl']])
# columns['Cluster'] = y

# ## get centroids
# centroids = kmeans.cluster_centers_
# cen_x = [i[0] for i in centroids] 
# cen_y = [i[1] for i in centroids]
# cen_z = [i[2] for i in centroids]


# ## add to columns df
# columns['cen_x'] = columns.Cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2], 3:cen_x[3]})
# columns['cen_y'] = columns.Cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2], 3:cen_y[3]})
# columns['cen_z'] = columns.Cluster.map({0:cen_z[0], 1:cen_z[1], 2:cen_z[2], 3:cen_z[3]})


# colors = ['r', 'g', 'b','m']
# columns['color'] = columns.Cluster.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3]})

# ## Plot 3D scatter plot
# fig = plt.figure(figsize=(26,6))
# ax = fig.add_subplot(131, projection='3d')
# ax.scatter(columns.normalized_LIN, columns.normalized_cellSize, columns.normalized_vcl, c = columns.color, s=15)
# ax.scatter(cen_x, cen_y, cen_z, marker='^', c=colors, s=70)
# # plot lines
# for idx, val in columns.iterrows():
#     x = [val.normalized_LIN, val.cen_x,]
#     y = [val.normalized_cellSize, val.cen_y]
#     z = [val.normalized_vcl, val.cen_z]
#     plt.plot(x, y, z, c=val.color, alpha=0.2)

# ax.set_xlabel('Linearity')
# ax.set_ylabel('Cell Size')
# ax.set_zlabel('Total distance travelled')

# # legend
# legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i+1), 
#                    markerfacecolor=mcolor, markersize=6) for i, mcolor in enumerate(colors)]

# plt.legend(handles=legend_elements,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.title('Motility params clustering')
# plt.show()


##k means using morphological params - 4clusters
kmeans = KMeans(n_clusters=4)
y = kmeans.fit_predict(columns[['ecc','sol','cmp']])
columns['Cluster'] = y

## get centroids
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]
cen_z = [i[2] for i in centroids]


## add to columns df
columns['cen_x'] = columns.Cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2], 3:cen_x[3]})
columns['cen_y'] = columns.Cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2], 3:cen_y[3]})
columns['cen_z'] = columns.Cluster.map({0:cen_z[0], 1:cen_z[1], 2:cen_z[2], 3:cen_z[3]})


colors = ['r', 'g', 'b','m']
columns['color'] = columns.Cluster.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3]})

## Plot 3D scatter plot
fig = plt.figure(figsize=(26,6))
ax = fig.add_subplot(131, projection='3d')
ax.scatter(columns.ecc, columns.sol, columns.cmp, c = columns.color, s=15)
ax.scatter(cen_x, cen_y, cen_z, marker='^', c=colors, s=70)
# plot lines
for idx, val in columns.iterrows():
    x = [val.ecc, val.cen_x,]
    y = [val.sol, val.cen_y]
    z = [val.cmp, val.cen_z]
    plt.plot(x, y, z, c=val.color, alpha=0.2)

ax.set_xlabel('Eccentricity')
ax.set_ylabel('Solidity')
ax.set_zlabel('Compactness')

# legend
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i), 
                   markerfacecolor=mcolor, markersize=6) for i, mcolor in enumerate(colors)]

plt.legend(handles=legend_elements,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Morphological params clustering')
plt.show()

##k means using morphological parmas using 3 clusters
kmeans = KMeans(n_clusters=3)
y = kmeans.fit_predict(columns[['ecc','sol','cmp']])
columns['Cluster'] = y

## get centroids
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]
cen_z = [i[2] for i in centroids]


## add to columns df
columns['cen_x'] = columns.Cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
columns['cen_y'] = columns.Cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
columns['cen_z'] = columns.Cluster.map({0:cen_z[0], 1:cen_z[1], 2:cen_z[2]})


colors = ['r', 'g', 'b']
columns['color'] = columns.Cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})

## Plot 3D scatter plot
fig = plt.figure(figsize=(26,6))
ax = fig.add_subplot(131, projection='3d')
ax.scatter(columns.ecc, columns.sol, columns.cmp, c = columns.color, s=15)
ax.scatter(cen_x, cen_y, cen_z, marker='^', c=colors, s=70)
# plot lines
for idx, val in columns.iterrows():
    x = [val.ecc, val.cen_x,]
    y = [val.sol, val.cen_y]
    z = [val.cmp, val.cen_z]
    plt.plot(x, y, z, c=val.color, alpha=0.2)

ax.set_xlabel('Eccentricity')
ax.set_ylabel('Solidity')
ax.set_zlabel('Compactness')

# legend
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i), 
                   markerfacecolor=mcolor, markersize=6) for i, mcolor in enumerate(colors)]

plt.legend(handles=legend_elements,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Morphological params clustering')
plt.show()






## to get the cell count in each cluster: columns.groupby(columns.Cluster).size()
## column wise : to get the M0, M1, M2 counts in each cluster: columns.groupby(columns.imageName[columns.Cluster == 0]).size()
## row wise : to get the M0, M1, M2 counts in each cluster: columns.groupby(columns.Cluster[columns.imageName == 'M0']).size()


ylabel = 'Frequency'; c1 = 'Cluster 1';c2 = 'Cluster 2';c3 = 'Cluster 3';c4 = 'Cluster 4';labels=[c1,c2,c3,c4];
ecc1 = columns.ecc[columns.Cluster == 0]; ecc2 = columns.ecc[columns.Cluster == 1]; ecc3 =columns.ecc[columns.Cluster == 2]; ecc4 = columns.ecc[columns.Cluster == 3];
sol1 = columns.sol[columns.Cluster == 0]; sol2 = columns.sol[columns.Cluster == 1]; sol3 =columns.sol[columns.Cluster == 2]; sol4 = columns.sol[columns.Cluster == 3];
com1 = columns.cmp[columns.Cluster == 0]; com2 = columns.cmp[columns.Cluster == 1]; com3 =columns.cmp[columns.Cluster == 2]; com4 = columns.cmp[columns.Cluster == 3];
spd1 = columns.avgSpeed[columns.Cluster == 0]; spd2 = columns.avgSpeed[columns.Cluster == 1]; spd3 = columns.avgSpeed[columns.Cluster == 2]; spd4 = columns.avgSpeed[columns.Cluster == 3];
per1 = columns.persistence[columns.Cluster == 0]; per2 = columns.persistence[columns.Cluster == 1]; per3 = columns.persistence[columns.Cluster == 2]; per4 = columns.persistence[columns.Cluster == 3];
dir1 = columns.directedness[columns.Cluster == 0]; dir2 = columns.directedness[columns.Cluster == 1]; dir3 = columns.directedness[columns.Cluster == 2]; dir4 = columns.directedness[columns.Cluster == 3]; 
#cm1 = columns.cos_cellMovm[columns.Cluster == 0]; cm2 = columns.cos_cellMovm[columns.Cluster == 1]; cm3 = columns.cos_cellMovm[columns.Cluster == 2]; cm4 = columns.cos_cellMovm[columns.Cluster == 3]; 


lin1 = columns.normalized_LIN[columns.Cluster == 0]; lin2 = columns.normalized_LIN[columns.Cluster == 1]; lin3 = columns.normalized_LIN[columns.Cluster == 2]; lin4 = columns.normalized_LIN[columns.Cluster == 3];
str1 = columns.STR[columns.Cluster == 0]; str2 = columns.STR[columns.Cluster == 1]; str3 = columns.STR[columns.Cluster == 2]; str4 = columns.STR[columns.Cluster == 3];
wob1 = columns.WOB[columns.Cluster == 0]; wob2 = columns.WOB[columns.Cluster == 1]; wob3 = columns.WOB[columns.Cluster == 2]; wob4 = columns.WOB[columns.Cluster == 3];
vcl1 = columns.normalized_vcl[columns.Cluster == 0]; vcl2 = columns.normalized_vcl[columns.Cluster == 1]; vcl3 = columns.normalized_vcl[columns.Cluster == 2]; vcl4 = columns.normalized_vcl[columns.Cluster == 3];
cs1 = columns.normalized_cellSize[columns.Cluster == 0]; cs2 = columns.normalized_cellSize[columns.Cluster == 1]; cs3 = columns.normalized_cellSize[columns.Cluster == 2]; cs4 = columns.normalized_cellSize[columns.Cluster == 3];
alh1 = columns.ALH[columns.Cluster == 0]; alh2 = columns.ALH[columns.Cluster == 1]; alh3 = columns.ALH[columns.Cluster == 2]; alh4 = columns.ALH[columns.Cluster == 3];

data1 = [ecc1,ecc2,ecc3,ecc4]
data2 = [sol1,sol2,sol3,sol4]
data3 = [com1,com2,com3,com4]
data4 = [spd1,spd2,spd3,spd4]
data5 = [per1,per2,per3,per4]
data6 = [dir1,dir2,dir3,dir4]
#data7 = [cm1,cm2,cm3,cm4]

data8 = [lin1, lin2, lin3, lin4]
data9 = [str1, str2, str3, str4]
data10 = [wob1, wob2, wob3, wob4]
data11 = [vcl1,vcl2,vcl3,vcl4]
data12 = [alh1,alh2,alh3,alh4]
data13 = [cs1,cs2,cs3,cs4]



## box plots 


from numpy.random import default_rng
rng = default_rng(42)


fig = plt.figure()

ax = fig.add_axes([0, 0, 1, 1]) 

# to plot scatter points on the box plots
allData = np.array(data3)
xCenter = np.array(range(1,len(allData)+1))
spread = 0.5;
for i in range(0,np.size(allData)):
    ax.plot(rng.random(np.size(allData[i]))*spread -(spread/2) + xCenter[i], allData[i],'y.','linewidth', 0.01, marker='o', alpha=.5, ms=4)

bp = ax.boxplot(data3,notch=True,widths=0.15,labels=[c1,c2,c3,c4], patch_artist = True) 

for patch, color in zip(bp['boxes'], colors): 
    patch.set_facecolor(color) 
plt.ylabel('Compactness')
plt.title('Compactness')
plt.grid()
plt.show() 

### Swarm plots for Morph and Motility values

bplot=sns.boxplot(y='cmp', x='Cluster',
                 data=columns,
                 width=.42 , fliersize=0)
# bplot.set_xticks(range(0,3)) # <--- set the ticks first
# bplot.set_xticklabels(['Cluster 1','Cluster 2','Cluster 3','Cluster 4'])
bplot.set_xticklabels(bplot.get_xticklabels(),horizontalalignment='right') #,rotation=45

# iterate over boxes
for i,box in enumerate(bplot.artists):
    box.set_edgecolor('gray')
    box.set_facecolor('lightgray')

    # iterate over whiskers and median lines
    for j in range(6*i,6*(i+1)):
         bplot.lines[j].set_color('black')
# for i in range(0,4):
#     mybox = bplot.artists[i]
#     mybox.set_facecolor(colors[i])
#sns.pointplot(y = "cmp", x = "Cluster",data = columns)

sns.swarmplot(y = "cmp", x = "Cluster", zorder=2, size=4.2, 
              data = columns, palette=colors, dodge=True, alpha=0.8)


sns.set_style("ticks", {"xtick.major.size": 12, "ytick.major.size": 12})
sns.despine() #to remove top and right borders
plt.xlabel('Clusters',fontweight="bold",fontSize="16",fontname="Times New Roman")
plt.ylabel('cmp',fontweight="bold",fontSize="16",fontname="Times New Roman")
plt.title('cmp',fontweight="bold",fontSize="18",fontname="Times New Roman")
# plt.ylabel('Vicinity ($px^2$)',fontweight="bold",fontSize="16",fontname="Times New Roman")
# plt.title('Cell trajectories vicinity',fontweight="bold",fontSize="18",fontname="Times New Roman")
#plt.ylim(0,3800)
plt.show()


#sns.pointplot(x="cell_num", y="sol", hue="Cluster", data=columns, palette=colors)
# sns.jointplot(y = "cmp", x = "Cluster", 
#               data = columns)



###Speed and Persistence cluster wise time series data
y_mean_vals = []
x = range(1,num_frames+1)
cluster_spd = columns.Speed_allCells[columns.Cluster == 2]
cluster_size = cluster_spd.size


for k in range(0, cluster_size):
    y = cluster_spd.iloc[k]
    
    # for j in range(0,num_frames-1):
    #     y_sum = y_sum + cluster_spd.iloc[k][j] + cluster_spd.iloc[k+1][j]
    
    # y_sum = y_sum/cluster_size
    
    # y_mean_vals.append(y_sum)
    #poly_degree = 3
    # coeffs = np.polyfit(x, y, poly_degree)
    # poly_eqn = np.poly1d(coeffs)
    # y_hat = poly_eqn(x)
    
    
    plt.plot(x,y)
    #plt.plot(x,y_mean_vals)
    #plt.plot(x,y_hat, color='b', linewidth=2, marker='o')
    #y_mean_vals.append(mean(y))

# plt.plot(range(1,len(y_mean_vals)+1), y_mean_vals, 
#          color='b', linewidth=2, marker='o',label='Mean')

for i in range(0,num_frames):
    y_sum = 0
    for j in range(0, cluster_size):
        y_sum = y_sum + cluster_spd.iloc[j][i]
    y_mean_vals.append(y_sum/cluster_size)
    

plt.plot(x,y_mean_vals,color='b', linewidth=2, marker='>',label='Mean')


sns.despine() 

plt.xlabel('Time in frames',fontweight="bold",fontname="Times New Roman")
plt.ylabel('Cell Speed (px/frames)',fontweight="bold",fontname="Times New Roman")
plt.title('Cluster 3 - Speed',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.legend()
plt.xlim(1,36)
plt.ylim(0,20)
plt.show()

y2_spd = y_mean_vals


#Persistence
#columns.rename(columns={"5fr_Per_allCells": "Per_5fr_allCells"}, inplace = True)
y_mean_vals = []
num_frames = len(df0_per.columns)
x = range(1,num_frames+1)
cluster_per = columns.Per_allCells[columns.Cluster == 2]
cluster_size = cluster_per.size

for k in range(0, cluster_size):
    y = cluster_per.iloc[k]
    plt.plot(x,y)

for i in range(0,num_frames):
    y_sum = 0
    for j in range(0, cluster_size):
        y_sum = y_sum + cluster_per.iloc[j][i]
    y_mean_vals.append(y_sum/cluster_size)
    

plt.plot(x,y_mean_vals,color='b', linewidth=2, marker='>',label='Mean')


sns.despine()     
plt.xlabel('Time in frames',fontweight="bold",fontname="Times New Roman")
plt.ylabel('Cell Persistence',fontweight="bold",fontname="Times New Roman")
plt.title('Cluster 0 - Persistence',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.legend()
plt.xlim(5,36)
#plt.ylim(0,1)
plt.show()

y2_pers = y_mean_vals


#Distance Travelled
#columns.rename(columns={"5fr_Per_allCells": "Per_5fr_allCells"}, inplace = True)
y_mean_vals = []
num_frames = len(df0_per.columns)
x = range(1,num_frames+1)
cluster_per = columns.Dist_travelled[columns.Cluster == 3]
cluster_size = cluster_per.size

for k in range(0, cluster_size):
    cluster_per.iloc[k].pop(36)
    y = cluster_per.iloc[k]
    plt.plot(x,y)

for i in range(0,num_frames):
    y_sum = 0
    for j in range(0, cluster_size):
        y_sum = y_sum + cluster_per.iloc[j][i]
    y_mean_vals.append(y_sum/cluster_size)
    

plt.plot(x,y_mean_vals,color='b', linewidth=2, marker='>',label='Mean')


sns.despine()     
plt.xlabel('Time in frames',fontweight="bold",fontname="Times New Roman")
plt.ylabel('Distance travelled (px)',fontweight="bold",fontname="Times New Roman")
plt.title('Cluster 1 - Distance travelled',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.legend()
plt.xlim(1,36)
#plt.ylim(0,1)
plt.show()

y3_dist = y_mean_vals



#### Image wise motiltiy params

y_mean_vals = []
num_frames = len(df0_per.columns)
x = range(1,num_frames+1)
img_spd = df2.Per_allCells  #Per_allCellsSpeed_allCells
img_size = img_spd.size

for k in range(0, img_size):
    y = img_spd.iloc[k]
    plt.plot(x,y)

for i in range(0,num_frames):
    y_sum = 0
    for j in range(0, img_size):
        y_sum = y_sum + img_spd.iloc[j][i]
    y_mean_vals.append(y_sum/img_size)
    

plt.plot(x,y_mean_vals,color='b', linewidth=2, marker='>',label='Mean')


sns.despine()     
plt.xlabel('Time in frames',fontweight="bold",fontname="Times New Roman")
plt.ylabel('Persistence',fontweight="bold",fontname="Times New Roman")
plt.title('M0 image- Persistence',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.legend()
# plt.xlim(2,36)
# plt.ylim(0,0.8)
plt.show()

#y3_pers=y_mean_vals

#m0_spd = y_mean_vals

#Trajectories
#columns.rename(columns={"5fr_Per_allCells": "Per_5fr_allCells"}, inplace = True)
import statsmodels.api as sm


cluster_x = columns.CellCx[columns.Cluster == 3]
cluster_y = columns.CellCy[columns.Cluster == 3]
cluster_size = cluster_x.size
time = range(1,num_frames+1)
dw = [];

for k in range(0, cluster_size):
    x = cluster_x.iloc[k]
    y = cluster_y.iloc[k]
    dw.append(x)
    # res = sm.tsa.ARMA(y, (1,1)).fit(disp=-1)
    # sm.stats.acorr_ljungbox(res.resid, lags=[20], return_df=True)
    #plt.plot(x,y,linestyle='dotted',linewidth=1,marker='>',markersize=3)
    ##this is main ACF plot in slide 32
    autocorrelation_plot(x)
    #plt.acorr(y, maxlags = 20) 
    
    #Lag plot
    #pd.plotting.lag_plot(pd.Series(x),lag=1,c='r')

 

sns.despine() 

#Lag plot title
#plt.title('Cluster 0 - Lag Plot',fontweight="bold",fontSize="14",fontname="Times New Roman")

plt.xlabel('Lag',fontweight="bold",fontname="Times New Roman")
plt.ylabel('ACF',fontweight="bold",fontname="Times New Roman")
plt.title('Cluster 0 - Autocorrelation',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
#plt.legend()
#plt.xlim(1,36)
#plt.ylim(0,1)
plt.show()

##Lag plots

pd.plotting.lag_plot(pd.Series(cluster_y.iloc[1]),lag=1)
plt.show()

##Vicinity of cells
vicinity =[]
for k in range(0, cluster_size):
    #for i in range(0,num_frames):
    x = cluster_x.iloc[k]
    y = cluster_y.iloc[k]
    rangeX = np.max(x) - np.min(x)
    rangeY = np.max(y) - np.min(y)
    vicinity.append(rangeX * rangeY)
    plt.hist(vicinity)
# Trajectories title
plt.title('Cluster 3 - Vicinity',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.xlabel('x',fontweight="bold",fontname="Times New Roman")
plt.ylabel('y',fontweight="bold",fontname="Times New Roman")
plt.axis('tight')




###  Trajectories

# from matplotlib import animation
# from matplotlib.animation import FuncAnimation
# %matplotlib qt

# y1 = cluster_y.iloc[12]
# t = cluster_x.iloc[12]

# x,y=[], []

# fig = plt.figure(figsize=(12,8))
# axes = fig.add_subplot(1,1,1)
# axes.set_ylim(1145, 1165)  #--cluster 1  y1=cluster_y.iloc[12]
# axes.set_xlim(1065, 1085)
# axes.set_ylim(1490, 1560)  #--cluster 2  y1=cluster_y.iloc[5]
# axes.set_xlim(1580, 1660)
# # axes.set_ylim(120, 150)  --cluster 3  y1=cluster_y.iloc[8]
# # axes.set_xlim(2100, 2200) --cluster 3  t=cluster_x.iloc[8]

# ##cluster 2 - 5,11,14

# def animate(i):
#     x.append(t[i])
#     y.append((y1[i]))
    
#     axes.clear()
#     axes.plot(x,y, scaley=True, scalex=True,linestyle='dotted', linewidth=2, marker='>', markersize=2, color="g")
    
# ani = FuncAnimation(fig=fig, func=animate, interval=100)
# plt.show()

# f = r"/Users/kesaprm/Desktop/saveM1.gif" 
# writergif = animation.PillowWriter(fps=30) 
# ani.save(f, writer=writergif)

# # for k in range(0, cluster_size):
# #     #for i in range(0,num_frames):
# #     x = cluster_x.iloc[k]
# #     y = cluster_y.iloc[k]
# #     plt.plot(x,y,linestyle='dotted', linewidth=2, marker='>', markersize=2, c = 'm')

# def animate(i=cluster_size):
#     x = cluster_x.iloc[i]
#     y = cluster_y.iloc[i]
#     p = plt.plot(x,y, c = 'm') #note it only returns the dataset, up to the point i
    


# # Trajectories title
# # plt.title('Cluster 3 - Trajectories',fontweight="bold",fontSize="14",fontname="Times New Roman")
# # plt.xlabel('x',fontweight="bold",fontname="Times New Roman")
# # plt.ylabel('y',fontweight="bold",fontname="Times New Roman")
# # plt.axis('tight')
# #plt.xlim(960,1100)
# #plt.ylim(250,500)
# anim = FuncAnimation(fig, animate, interval = 10)

# anim.show() #12/8,25,5
import scipy


# y1 = cluster_y.iloc[5]
# t = cluster_x.iloc[5]

y1= pd.read_csv("B_CellCx.txt") 
t = pd.read_csv("B_CellCy.txt") 

y1 = y1.Var1
t = t.Var1

x=range(1,len(t))
slope =[]
#plt.plot(x,t)


#slope of the trajectories
for n in range(1,len(t)):
    rise = y1[n] - y1[n-1]
    run = t[n] - t[n-1]
    slope.append(rise/run)
plt.plot(t[1:],slope,'.-',color='b')

plt.xlabel('time in frames',fontweight="bold",fontname="Times New Roman")
plt.ylabel('Slope',fontweight="bold",fontname="Times New Roman")
plt.axis('tight')
plt.ylim(-100, 100)
plt.title('Spinning pattern')

# Plot x,y t plots
data_norm_to_0_1 = [number/scipy.linalg.norm(y1) for number in y1]
data_norm_to_0_2 = [number/scipy.linalg.norm(t) for number in t]


fig, ax = plt.subplots()
line, = ax.plot(data_norm_to_0_2, data_norm_to_0_1,'o-',color='g')

#ax.set_ylim(120, 150)  #--cluster 3  y1=cluster_y.iloc[1]
#ax.set_xlim(2100, 2200) #--cluster 3  t=cluster_x.iloc[1]


plt.xlabel('x',fontweight="bold",fontname="Times New Roman")
plt.ylabel('y',fontweight="bold",fontname="Times New Roman")
plt.axis('tight')
for n in range(len(t)):
    plt.title('Frame '+ str(n),fontweight="bold",fontSize="14",fontname="Times New Roman")
    line.set_data(data_norm_to_0_2[:n], data_norm_to_0_1[:n])
    #ax.axis([0, 1, 0, 1])
    fig.canvas.draw()
    fig.savefig('C2%03d.png' %n)
# # Writer = ani.writers['ffmpeg']
# # writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=100)
# writer = ani.FFMpegWriter(fps=30, codec='libx264')


# plt.rcParams['animation.ffmpeg_path'] = '/usr/local/Cellar/ffmpeg/4.4_2/bin/ffmpeg'
# anFun.save('myfirstAnimation.mp4', writer=writer, dpi=1)

import seaborn as sns

################################PATTERN VISUALIZATION - Deep learning
#0 - circular cells --12
#1 - protrusions -- 5
#2 - elongated -- 24

cluster_x = columns.CellCx[columns.Cluster == 0]
cluster_y = columns.CellCy[columns.Cluster == 0]
imageFrom = columns.imageName[columns.Cluster == 0]
# speed = columns.Speed_allCells[columns.Cluster == 2]
# pers = columns.Per_allCells[columns.Cluster == 0]
# vici = columns.vicinity[columns.Cluster == 0]


cluster_size = cluster_x.size


traj =[]

# #traj in the shaope of [[[x_11,y_11],[x_12,y_12]...,[x_136,y_136]],[[x_21,y_21]...[x_236,y_236]],...[[x_361,y_361],..[x_3636,y_3636]]]
# for k in range(0, cluster_size):
#     #for i in range(0,num_frames):
#     x = cluster_x.iloc[k]
#     y = cluster_y.iloc[k]
#     traj.append(np.column_stack((x, y)))

#traj in the shape of [[[0, 1, 2, 3, 4],[5, 6, 7, 8, 9]],[[10, 11, 12, 13, 14],[15, 16, 17, 18, 19]],[[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]

for k in range(0, cluster_size):
    x = cluster_x.iloc[k]
    y = cluster_y.iloc[k]
    traj.append([x,y])
    


d = {'traj': traj , 'pattern': 'Spinning','parentImg':imageFrom, 'patternNo': 9.}
cluster0 =  pd.DataFrame(data=d)

d = {'traj': traj , 'pattern': 'Wandering','parentImg':imageFrom, 'patternNo': 1.}
cluster1 =  pd.DataFrame(data=d)

d = {'traj': traj , 'pattern': 'Bidirectional','parentImg':imageFrom, 'patternNo': 2.}
cluster2 =  pd.DataFrame(data=d)

d = {'traj': traj , 'pattern': 'Mix','parentImg':imageFrom , 'patternNo': 3.}
cluster3 =  pd.DataFrame(data=d)

cluster0.traj = cluster0.traj.to_numpy().tolist()
cluster1.traj = cluster1.traj.to_numpy().tolist()
cluster2.traj = cluster2.traj.to_numpy().tolist()

#Append time-series persistence values for M0
df0_per = pd.read_csv("M0_val_Persistence_cellWise.txt")
df0_per_arr = df0_per.to_numpy()
df0['Per_allCells'] =  df0_per_arr.tolist()

#clusters = pd.concat([cluster0,cluster1,cluster2,cluster3], ignore_index=True)
clusters = pd.concat([cluster0,cluster1,cluster2], ignore_index=True)
#cluster_x.to_csv('clusters_validation_test.txt')

# to store list as a list. to_csv converted list to a string
clusters.to_pickle('clusters_validation.txt')

# dfData = pd.read_pickle("clusters_validation_test.txt") 


from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from keras.utils import normalize, to_categorical
from sklearn.utils import shuffle
 

# #k-fold cross validation using sklearn
# kf = KFold(n_splits=5, shuffle=True, random_state=2652124)
# k_trn =[]; k_tst =[];
# for trn, tst in kf.split(clusters.traj,clusters.patternNo):
#     print("%s %s" % (trn, tst))
#     k_trn.append(trn)
#     k_tst.append(tst)


#train,test, train_labels,test_labels =  X[train], X[test], y[train], y[test]
#train, test = train_test_split(data, test_size=0.2) # split 80:20 train:test 
train,test, train_labels,test_labels = train_test_split(clusters.traj,clusters.patternNo, shuffle=True, random_state=2652124)

#shuffle the training data to improve the val accuracy scoere -- Used instead of #np.random.shuffle(rank_3_tensor_train)
train, train_labels = shuffle(train, train_labels)

# tensor creation
rank_3_tensor_train = tf.constant([train,])
rank_3_tensor_train = tf.reshape(rank_3_tensor_train,[rank_3_tensor_train.shape[1], rank_3_tensor_train.shape[2], rank_3_tensor_train.shape[3]]) # 77 cells, 2 column arrays- x & y, 37 time points 

rank_3_tensor_test = tf.constant([test,])
rank_3_tensor_test = tf.reshape(rank_3_tensor_test,[rank_3_tensor_test.shape[1], rank_3_tensor_test.shape[2], rank_3_tensor_test.shape[3]]) # 77 cells, 2 column arrays- x & y, 37 time points 

#Not required to create tensors for labels, we use one-hot encode
# train_labels = tf.constant([train_labels,])
# train_labels = tf.reshape(train_labels,[train_labels.shape[1]]) # 77 cells, 2 column arrays- x & y, 37 time points 

# test_labels = tf.constant([test_labels,])
# test_labels = tf.reshape(test_labels,[test_labels.shape[1]]) # 77 cells, 2 column arrays- x & y, 37 time points 

# normalizing and using one hot encode
rank_3_tensor_train = normalize(rank_3_tensor_train)
rank_3_tensor_test = normalize(rank_3_tensor_test)

#to one hot labels
def to_one_hot(labels,dimension=46,features=2):
    results = np.zeros((len(labels),features, dimension))
    for j in range(1,features):        
        for i, label in enumerate(labels):           
            label = int(label)
            results[i, j, label] = 1.
        return results

#labels train and test
hot_train_labels = to_one_hot(train_labels)
hot_test_labels = to_one_hot(test_labels)


# #labels train and test using to_categorical
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(2, 37)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())
    return model
    
## Validating the approach by taking out 20 datapoints for validation -- Manual split of the training & validation data
# x_val = rank_3_tensor_train[:20]
# partial_x_train = rank_3_tensor_train[20:]

# y_val = train_labels[:20]
# partial_y_train = train_labels[20:]

# #k-fold cross validation from the text book
k = 5
num_val_samples = len(rank_3_tensor_train)//k
num_epochs = 500
all_scores = []
all_loss_histories = [];all_acc_histories = []; val_loss_histories =[];val_acc_histories =[]
#np.random.shuffle(rank_3_tensor_train)

validation_scores = []

for fold in range(k):
    #print('processing fold #', i)
    val_data = rank_3_tensor_train[num_val_samples * fold: num_val_samples * (fold+1)] # selects the validation-data partition
    val_labels = hot_train_labels[num_val_samples * fold: num_val_samples * (fold+1)] 
    
    partial_train_data = np.concatenate( [rank_3_tensor_train[:fold * num_val_samples],rank_3_tensor_train[(fold + 1) * num_val_samples:]], axis=0)
    partial_train_labels = np.concatenate( [hot_train_labels[:fold * num_val_samples], hot_train_labels[(fold + 1) * num_val_samples:]], axis=0)
    
    model =  build_model() # creates a brand-new instance of the model(untrained)
    history = model.fit(partial_train_data,
                    partial_train_labels,
                    validation_data=(val_data, val_labels),
                    epochs = num_epochs,
                    batch_size = 1,
                    verbose = 0)
    #val_mse, val_mae = model.evaluate(val_data, val_labels, verbose=0)
    #all_scores.append(val_mae)
    loss_hist = history.history['loss']
    all_loss_histories.append(loss_hist)
    
    val_loss = history.history['val_loss']
    val_loss_histories.append(val_loss)
    
    acc_hist = history.history['accuracy']
    all_acc_histories.append(acc_hist)
    
    val_hist = history.history['val_accuracy']
    val_acc_histories.append(val_hist)
    
    
avg_loss_hist = [ np.mean([x[i] for x in all_loss_histories]) for i in range(num_epochs)]
avg_val_loss_hist = [ np.mean([x[i] for x in val_loss_histories]) for i in range(num_epochs)]

avg_acc_hist = [ np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]
avg_val_acc_hist = [ np.mean([x[i] for x in val_acc_histories]) for i in range(num_epochs)]
    

# ##Traing the model wihtout shuffle and without k-fold cross validation

# history = model.fit(partial_train_data,
#                     partial_train_labels,
#                     epochs = num_epochs,
#                     batch_size = 1,
#                     verbose = 0)


##Plotting the training and validation loss
# loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs = range(1, len(avg_loss_hist) + 1)

plt.plot(epochs, avg_loss_hist, 'c-', label='Training loss')
plt.plot(epochs, avg_val_loss_hist, 'purple', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.xlim(0,20)
plt.legend()
plt.show()


def smooth_curve(points, factor=0.9): 
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else: 
            smoothed_points.append(point)
    return smoothed_points

smooth_loss_hist = smooth_curve(avg_loss_hist[10:])
smooth_val_loss_hist = smooth_curve(avg_val_loss_hist[10:])
plt.plot(range(1, len(smooth_loss_hist) + 1), smooth_loss_hist, 'c-', label='Training loss')
plt.plot(range(1, len(smooth_val_loss_hist) + 1), smooth_val_loss_hist, 'purple', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



##Plotting the training and validation accuracy
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

plt.plot(epochs, avg_acc_hist, 'co-', label='Training acc')
plt.plot(epochs, avg_val_acc_hist, 'purple', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


smooth_acc_hist = smooth_curve(avg_acc_hist[10:])
smooth_acc_loss_hist = smooth_curve(avg_val_acc_hist[10:])
plt.plot(range(1, len(smooth_acc_hist) + 1), smooth_acc_hist, 'c-', label='Training acc')
plt.plot(range(1, len(smooth_acc_loss_hist) + 1), smooth_acc_loss_hist, 'purple', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

###################################################################

y = cluster_y.iloc[5]
x = cluster_x.iloc[5]
xmin = np.min(x)
xmax = np.max(x)
ymin = np.min(y)
ymax = np.max(y)
fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
ax = axs[0]
hb = ax.hexbin(x, y, gridsize=50, cmap='inferno')
ax.axis([xmin, xmax, ymin, ymax])
ax.set_title("Hexagon binning")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('counts')

ax = axs[1]
hb = ax.hexbin(x, y, gridsize=50, bins='log', cmap='inferno')
ax.axis([xmin, xmax, ymin, ymax])
ax.set_title("With a log color scale")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(N)')

plt.show()


##Polynomail Regression

mymodel = np.poly1d(np.polyfit(x, y, 5))

myline = np.linspace(0, 0.2, 37)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()


##Joint Plots
kdeplot = sns.jointplot(x, y,  hue="species",kind='kde');
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
# get the current positions of the joint ax and the ax for the marginal x
pos_joint_ax = kdeplot.ax_joint.get_position()
pos_marg_x_ax = kdeplot.ax_marg_x.get_position()
# reposition the joint ax so it has the same width as the marginal x ax
kdeplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
# reposition the colorbar using new x positions and y positions of the joint ax
kdeplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
plt.show()


#single plots without blue
sns.kdeplot(x, y, hue="time", multiple="fill",levels=1, thresh=.2)

sns.jointplot(x,y, kind="kde",levels=1,palette='autumn')


traj = [x,y]

sns.heatmap(traj)

### ML codes from here----

from sklearn import preprocessing


##normalized trajectories
x_norm = preprocessing.normalize([x]) # (x-np.min(x))/(np.max(x)-np.min(x))
y_norm = preprocessing.normalize([y]) #(y-np.min(y))/(np.max(y)-np.min(y))


sns.jointplot(x_norm,y_norm, kind="kde",levels=1,palette='autumn')


########### Point patterns
import libpysal as ps
from pointpats import PointPattern, as_window
from pointpats import PoissonPointProcess as csr
import pointpats.quadrat_statistics as qs

pts = []

for i in range(0,np.size(x)):
    pts.append([x[i],y[i]])

juv_points = np.array(pts)

pp_juv = PointPattern(juv_points)

pp_juv.summary()
pp_juv.plot(window= True, title= "Point pattern")


q_r = qs.QStatistic(pp_juv,shape= "rectangle",nx = 3, ny = 3)
q_r.plot()

q_r.chi2 #chi-squared test statistic for the observed point pattern

q_r.df #degree of freedom
q_r.chi2_pvalue # analytical pvalue

#Rectangle quadrats & empirical sampling distribution 

csr_process = csr(pp_juv.window, pp_juv.n, 999, asPP=True)

q_r_e = qs.QStatistic(pp_juv,shape= "rectangle",nx = 3, ny = 3, realizations = csr_process)

q_r_e.chi2_r_pvalue

#Hexagon quadrats & analytical sampling distribution  1.2,28,10
q_h = qs.QStatistic(pp_juv,shape= "hexagon",lh = 5)
q_h.plot()

q_h.chi2 #chi-squared test statistic for the observed point pattern

q_h.df #degree of freedom

q_h.chi2_pvalue  # analytical pvalue
############ ML codes -----

##vector plots with position vector: P = xi+yj
u = [j - i for i, j in zip(x[: -1], x[1 :])]
v = [j - i for i, j in zip(y[: -1], y[1 :])]

u.insert(0,0)
v.insert(0,0)

#plt.quiver(x,y,u,v,[cmap=True])
plt.title('Cluster 1 - Position vector plot',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.xlabel('x',fontweight="bold",fontname="Times New Roman")
plt.ylabel('y',fontweight="bold",fontname="Times New Roman")
plt.show()


###Trajectories in 2D
for k in range(0, cluster_size):
    #for i in range(0,num_frames):
    x = cluster_x.iloc[k]
    y = cluster_y.iloc[k]
    plt.plot(x,y,linestyle='dotted', linewidth=2, marker='>', markersize=2, c = 'm')

plt.title('Cluster 3 - Trajectories',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.xlabel('x',fontweight="bold",fontname="Times New Roman")
plt.ylabel('y',fontweight="bold",fontname="Times New Roman")
plt.axis('tight')
plt.xlim(960,1100)
plt.ylim(250,500)
    
### Trajectories in 3D

def f(x,y):
   return np.sqrt(x**2+y**2)
        

#fig = plt.figure()
#ax = plt.axes(projection='3d')

# fig = plt.figure(figsize=(26,6))
# ax = fig.add_subplot(131, projection='3d')
for k in range(0, cluster_size):
    #for i in range(0,num_frames):
    x = np.array(cluster_x.iloc[k])
    y = np.array(cluster_y.iloc[k])
    X, Y = np.meshgrid(x, y)
    Z = (f(X,Y)) #np.sqrt(x**2+y**2)
    #ax.plot3D(x,y,time,linestyle='dotted', linewidth=2, marker='>', markersize=2, c = 'm')
    plt.contour(X,Y,Z , cmap='RdGy' )


# Trajectories title
plt.title('Cluster 2 - Trajectories',fontweight="bold",fontSize="14",fontname="Times New Roman")
ax.set_xlabel('x',fontweight="bold",fontname="Times New Roman")
ax.set_ylabel('y',fontweight="bold",fontname="Times New Roman")
ax.set_zlabel('t',fontweight="bold",fontname="Times New Roman")
plt.axis('tight')
plt.xlim(400,560)
plt.ylim(1300,1800)
plt.zlim(1,36)
plt.show()



## x vs y

for k in range(0, cluster_size):
    #for i in range(0,num_frames):
    x = cluster_x.iloc[k]
    y = cluster_y.iloc[k]
    #plt.plot(x,y,linestyle='dotted', linewidth=2, marker='>', markersize=2, c = 'g')
    plt.plot(x[:180],y[:180],linestyle='dotted', linewidth=2, marker='>', markersize=2, c = 'g')

# Trajectories title
plt.title('Cluster 0 - Trajectories',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.ylabel('x',fontweight="bold",fontname="Times New Roman")
plt.xlabel('y',fontweight="bold",fontname="Times New Roman")
plt.axis('tight')
plt.xlim(1200,1600)
plt.ylim(1400,1750)
plt.show()

#x vs t

time = range(0,num_frames+1)

for k in range(1, cluster_size):
    #for i in range(0,num_frames):
    x = cluster_x.iloc[10]
    y = cluster_y.iloc[10]
    plt.plot(time,x,linestyle='dotted', linewidth=2, marker='>', markersize=2,c = 'k')
    plt.plot(time,y,linestyle='dotted', linewidth=2, marker='>', markersize=2,c= 'y')

# Trajectories title
plt.title('Cluster 3 - Trajectories',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.ylabel('x',fontweight="bold",fontname="Times New Roman")
plt.xlabel('t',fontweight="bold",fontname="Times New Roman")
plt.axis('tight')
plt.ylim(390,435)
plt.show()

# to find the vicinity covered
columns['vicinity'] = 0
columns['vicinity'] = columns.Cluster.copy()

cluster_x = columns.CellCx[columns.Cluster == 1]
cluster_y = columns.CellCy[columns.Cluster == 1]
cluster_size = cluster_x.size
time = range(0,num_frames)

#cluster_vicinity = columns.vicinity[columns.Cluster == 0]

#vicinity =[]
for k in range(0, cluster_size):
    #for i in range(0,num_frames):
    x = cluster_x.iloc[k]
    y = cluster_y.iloc[k]
    x_range = np.round(np.max(cluster_x.iloc[k]) - np.min(cluster_x.iloc[k]))
    y_range = np.round(np.max(cluster_y.iloc[k]) - np.min(cluster_y.iloc[k]))
    #vicinity.append(x_range * y_range)
    columns.vicinity[columns.Cluster == 0].iloc[k] = x_range * y_range
plt.hist(vicinity, color = 'm')

# Trajectories title
plt.title('Cluster 3 - Trajectories',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.ylabel('y',fontweight="bold",fontname="Times New Roman")
plt.xlabel('t',fontweight="bold",fontname="Times New Roman")
plt.axis('tight')
plt.ylim(1980,2050)
plt.show()


# y vs t


for k in range(0, cluster_size):
    #for i in range(0,num_frames):
    x = cluster_x.iloc[k]
    y = cluster_y.iloc[k]
    plt.plot(x,y,linestyle='dotted', linewidth=2, marker='>', markersize=2, c = 'm')

plt.title('Cluster 3 - Trajectories',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.xlabel('x',fontweight="bold",fontname="Times New Roman")
plt.ylabel('y',fontweight="bold",fontname="Times New Roman")
plt.axis('tight')
plt.xlim(960,1100)
plt.ylim(250,500)



#to find the angle between the 2 position vectors - i.e. angle between (x1,y1) and (x2,y2)
cluster_x = columns.CellCx[columns.Cluster == 3]
cluster_y = columns.CellCy[columns.Cluster == 3]
cluster_size = cluster_x.size
time = range(0,num_frames)


rad_all = []; deg_all = []; 
for k in range(0, cluster_size):
    rad_i = []; deg_i = []; deg_sum = 0;
    for i in range(0,num_frames):
        x1 = cluster_x.iloc[k][i]
        y1 = cluster_y.iloc[k][i]
        x2 = cluster_x.iloc[k][i+1]
        y2 = cluster_y.iloc[k][i+1]
        
        pos1_vec = np.array([x1, y1])
        pos2_vec = np.array([x2, y2])
        
        inner = np.inner(pos1_vec,pos2_vec)
        norms = np.linalg.norm(pos1_vec) * np.linalg.norm(pos2_vec)
        
        cos = inner/norms
        
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        deg = np.rad2deg(rad)
        
        rad_i.append(rad)
        deg_i.append(deg)
        deg_sum = deg_sum + deg
    print(deg_sum)
    
    plt.plot(time,deg_i)
    
    rad_all.append(rad_i)
    deg_all.append(deg_i)
        
        
#plt.plot(time,deg_all)#,linestyle='dotted', linewidth=2, marker='>', markersize=2, c = 'b')
plt.title('Cluster 0 - Angle(deg) between 2 position vectors',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.ylabel('theta = arccos(dot(u.v)/norm(u)*norm(v))',fontweight="bold",fontname="Times New Roman")
plt.xlabel('t',fontweight="bold",fontname="Times New Roman")
plt.axis('tight')


ylabel = 'Frequency'; c1 = 'Cluster 1';c2 = 'Cluster 2';c3 = 'Cluster 3';c0 = 'Cluster 0';

deg_sum_3 = [4.958253075104496,
3.9135592707984057,
3.0519550199426964,
7.373771724215559,
6.876038318482647,
15.166683150342013,
14.398137771095488,
4.5366578412766545,
5.327437119687063,
7.0036669542737835,
7.523099912571069,
5.543223107922358]

deg_sum_2 = [2.5962808978597205,
5.402794113768709,
2.1085912061317273,
2.353943959082514,
3.2015521805749634,
3.62464807499712,
5.454610312555423,
10.61129521858229,
22.857642725088134,
3.6439834944508127,
5.345490271155996,
2.7820172158948187,
3.385582475644817,
2.471294719232832,
11.400528263584851,
11.268937574789467,
5.799563953012873,
3.053035836004317,
2.336499905731995,
10.802983016135304,
13.274699056860152,
1.210285630425038,
2.1561894527001844,
1.8709564341622884,
1.676354291864358,
9.916794762687257]

deg_sum_1 = [1.589036756991582,
2.0357818746477916,
1.2199805950869056,
5.028583891500847,
3.12934364319349,
0.9825302494016153,
3.86423433351335,
3.6921231322974863,
5.990160614963518,
4.38649748180211,
4.087833098756996,
2.318826801683121,
2.4856192915693036,
6.501302439869958,
3.299052476991249,
5.296684160512541,
5.258298478097933,
2.9862432732665356,
4.654839444618759,
3.9157647464162486,
8.225970047633949,
3.1408476189387464,
4.278753117375025,
2.5450504025985365,
3.828921874366302,
3.262005407534521,
8.517499899083061,
9.654191266147441,
11.61002661209208,
5.716422253822543,
3.1714942257738388,
2.3816932708287863,
2.6945431670732476]

deg_sum_0 = [2.461039142016602,
4.043672979218381,
2.5876383304456727,
6.354655916104524,
3.3769330345636708,
4.698960661088692,
7.248407616497502,
2.4070189774557487,
3.428057716314225,
3.838954998382096,
4.8028022892030835,
4.2671097916355905,
7.383086597206484,
8.812343489228535,
2.781459863529786,
5.287607514744376,
3.369164248845808,
4.674883371388118,
2.248092405100054,
18.796756376728492,
106.14529013645036,
6.782289232865178,
3.446931753893903,
9.026949819450591,
3.144069143389732,
17.565444803114417]

segPlots.histPlots(deg_sum_0,'Degree Sum',ylabel,c0,'r')



# sns.pairplot(columns)
# plt.show()

#random walk plots - The Durbin Watson Test

dw_num = np.array(range(0,cluster_size)); dw_den = np.array(range(0,cluster_size)); dw =np.array(range(0,cluster_size));
for i in range(0, cluster_size):
    for k in range(2, len(cluster_y.iloc[i])):
        dw_num[i] = dw_num[i] + (cluster_y.iloc[i][k] - cluster_y.iloc[i][k-1])**2

    for k in range(1, len(cluster_y.iloc[i])):
        dw_den[i] = dw_den[i] + cluster_x.iloc[i][k]**2

    dw[i] = dw_num[i]/dw_den[i]
    print(dw[i])

##GPR speed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


x = np.atleast_2d(range(1,num_frames+1)).T
dy0 = 1.1 + 1.0 * np.random.random(np.array(y0_spd).shape)
dy1 = 1.1 + 1.0 * np.random.random(np.array(y1_spd).shape)
dy2 = 1.1 + 1.0 * np.random.random(np.array(y2_spd).shape)
#dy3 = 1.1+ 1.0 * np.random.random(np.array(y3_spd).shape)

# Instantiate a Gaussian Process model -- Kernel parameters are estimated using maximum likelihood principle.
kernel = C(1.0, (1e-3, 1e3)) * RBF(240, (1e-2, 1e2))
gp0 = GaussianProcessRegressor(kernel=kernel, alpha=dy0 ** 2,
                              n_restarts_optimizer=10)
gp1 = GaussianProcessRegressor(kernel=kernel, alpha=dy1 ** 2,
                              n_restarts_optimizer=10)
gp2 = GaussianProcessRegressor(kernel=kernel, alpha=dy2 ** 2,
                              n_restarts_optimizer=10)
# gp3 = GaussianProcessRegressor(kernel=kernel, alpha=dy3 ** 2,
#                               n_restarts_optimizer=10)

xx = np.atleast_2d(np.linspace(1, (num_frames), (num_frames))).T
# Fit to data using Maximum Likelihood Estimation of the parameters
gp0.fit(x,y0_spd)
gp1.fit(x,y1_spd)
gp2.fit(x,y2_spd)
#gp3.fit(x,y3_spd)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred0, sigma0 = gp0.predict(xx, return_std=True)
y_pred1, sigma1 = gp1.predict(xx, return_std=True)
y_pred2, sigma2 = gp2.predict(xx, return_std=True)
#y_pred3, sigma3 = gp3.predict(xx, return_std=True)


plt.figure()
plt.plot(xx, y0_spd, 'r:', linewidth=2,label='Cluster 0')
plt.plot(xx, y1_spd, 'g:', linewidth=2,label='Cluster 1')
plt.plot(xx, y2_spd, 'b:', linewidth=2,label='Cluster 2')
#plt.plot(xx, y3_spd, 'm:', linewidth=2,label='Cluster 3')

#plt.errorbar(x, y0_spd, dy0, linestyle='',fmt='k.', markersize=5, label='Observations')
# plt.errorbar(x, y1_spd, dy1, fmt='g.', markersize=10, label='Observations')
# plt.errorbar(x, y2_spd, dy2, fmt='b.', markersize=10, label='Observations')
# plt.errorbar(x, y3_spd, dy3, fmt='m.', markersize=10, label='Observations')

#plt.plot(x, y3_spd, 'r.', markersize=10, label='Observations')
plt.plot(xx, y_pred0, 'r-')
plt.plot(xx, y_pred1, 'g-')
plt.plot(xx, y_pred2, 'b-')
#plt.plot(xx, y_pred3, 'm-')

#95% confidence interval.
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_pred0 - 1.9600 * sigma0,
                        (y_pred0 + 1.9600 * sigma0)[::-1]]),
         alpha=.5, fc='r', ec='None')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_pred1 - 1.9600 * sigma1,
                        (y_pred1 + 1.9600 * sigma1)[::-1]]),
         alpha=.5, fc='g', ec='None')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_pred2 - 1.9600 * sigma2,
                        (y_pred2 + 1.9600 * sigma2)[::-1]]),
         alpha=.5, fc='b', ec='None')
# plt.fill(np.concatenate([xx, xx[::-1]]),
#          np.concatenate([y_pred3 - 1.9600 * sigma3,
#                         (y_pred3 + 1.9600 * sigma3)[::-1]]),
#          alpha=.5, fc='m', ec='None')
#plt.xlabel('Time in frames(1hr-2hr)',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.xlabel('Time in frames',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.ylabel('Speed (px/frames)',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.title('Clusterwise comparision of Cell Speed',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.xlim(1,240)
#plt.xlim(12,24)
#plt.ylim(5,12)
plt.legend()

#### GPR for persistence
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


x = np.atleast_2d(range(1,num_frames+1)).T
dy0 = 0.03  * np.random.random(np.array(y0_pers).shape)
dy1 = 0.03  * np.random.random(np.array(y1_pers).shape)
dy2 = 0.03  * np.random.random(np.array(y2_pers).shape)
#dy3 = 0.03  * np.random.random(np.array(y3_pers).shape)

# Instantiate a Gaussian Process model -- Kernel parameters are estimated using maximum likelihood principle.
kernel = C(1.0, (1e-3, 1e3)) * RBF(36, (1e-2, 1e2))
gp0 = GaussianProcessRegressor(kernel=kernel, alpha=dy0 ** 2,
                              n_restarts_optimizer=10)
gp1 = GaussianProcessRegressor(kernel=kernel, alpha=dy1 ** 2,
                              n_restarts_optimizer=10)
gp2 = GaussianProcessRegressor(kernel=kernel, alpha=dy2 ** 2,
                              n_restarts_optimizer=10)
# gp3 = GaussianProcessRegressor(kernel=kernel, alpha=dy3 ** 2,
#                               n_restarts_optimizer=10)

xx = np.atleast_2d(np.linspace(1, (num_frames), (num_frames))).T
# Fit to data using Maximum Likelihood Estimation of the parameters
gp0.fit(x,y0_pers)
gp1.fit(x,y1_pers)
gp2.fit(x,y2_pers)
#gp3.fit(x,y3_pers)

# Make the prediction on the meshed x-axis (ask for MSE as wxxell)
y_pred0, sigma0 = gp0.predict(xx, return_std=True)
y_pred1, sigma1 = gp1.predict(xx, return_std=True)
y_pred2, sigma2 = gp2.predict(xx, return_std=True)
#y_pred3, sigma3 = gp3.predict(xx, return_std=True)


plt.figure()
plt.plot(xx, y0_pers, 'r:', linewidth=2,label='Cluster 0')
plt.plot(xx, y1_pers, 'g:', linewidth=2,label='Cluster 1')
plt.plot(xx, y2_pers, 'b:', linewidth=2,label='Cluster 2')
#plt.plot(xx, y3_pers, 'm:', linewidth=2,label='Cluster 3')

#plt.errorbar(x, y0_spd, dy0, linestyle='',fmt='k.', markersize=5, label='Observations')
# plt.errorbar(x, y1_spd, dy1, fmt='g.', markersize=10, label='Observations')
# plt.errorbar(x, y2_spd, dy2, fmt='b.', markersize=10, label='Observations')
# plt.errorbar(x, y3_spd, dy3, fmt='m.', markersize=10, label='Observations')

#plt.plot(x, y3_spd, 'r.', markersize=10, label='Observations')
plt.plot(xx, y_pred0, 'r-')
plt.plot(xx, y_pred1, 'g-')
plt.plot(xx, y_pred2, 'b-')
#plt.plot(xx, y_pred3, 'm-')

#95% confidence interval.
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_pred0 - 1.9600 * sigma0,
                        (y_pred0 + 1.9600 * sigma0)[::-1]]),
         alpha=.5, fc='r', ec='None')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_pred1 - 1.9600 * sigma1,
                        (y_pred1 + 1.9600 * sigma1)[::-1]]),
         alpha=.5, fc='g', ec='None')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_pred2 - 1.9600 * sigma2,
                        (y_pred2 + 1.9600 * sigma2)[::-1]]),
         alpha=.5, fc='b', ec='None')
# plt.fill(np.concatenate([xx, xx[::-1]]),
#          np.concatenate([y_pred3 - 1.9600 * sigma3,
#                         (y_pred3 + 1.9600 * sigma3)[::-1]]),
#          alpha=.5, fc='m', ec='None')
plt.xlabel('Time in frames',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.ylabel('Persistence',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.title('Clusterwise comparision of Cell Persistence',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.xlim(150,200)
#plt.xlim(12,24)  # between 12-24 frames i.e., 60 min-120 min == 1hr-2hr
plt.ylim(0,0.2)
plt.legend()

## to know the model's data fit: print(gp0.score(x,y0_pers))

##OTHER STATS:
np.std(y0_spd[12:24], ddof=1) / np.sqrt(np.size(y0_spd[12:24]))
np.std(y1_spd[12:24], ddof=1) / np.sqrt(np.size(y1_spd[12:24]))
np.std(y2_spd[12:24], ddof=1) / np.sqrt(np.size(y2_spd[12:24]))
np.std(y3_spd[12:24], ddof=1) / np.sqrt(np.size(y3_spd[12:24]))

np.std(y0_pers[12:24], ddof=1) / np.sqrt(np.size(y0_pers[12:24]))
np.std(y1_pers[12:24], ddof=1) / np.sqrt(np.size(y1_pers[12:24]))
np.std(y2_pers[12:24], ddof=1) / np.sqrt(np.size(y2_pers[12:24]))
np.std(y3_pers[12:24], ddof=1) / np.sqrt(np.size(y3_pers[12:24]))

np.mean(y0_pers[12:24]) # OTHER MEANS IN SIMILAR WAY
np.std(y0_pers[12:24])  # OTHER STANDARD DEVIATIONS IN SIMILAR WAY

####
plt.plot(x,y0_spd,color='r', linewidth=2, marker='>',label='Cluster 0')
plt.plot(x,y1_spd,color='g', linewidth=2, marker='>',label='Cluster 1')
plt.plot(x,y2_spd,color='b', linewidth=2, marker='>',label='Cluster 2')
plt.plot(x,y3_spd,color='m', linewidth=2, marker='>',label='Cluster 3')
sns.despine()     
plt.xlabel('Time in frames',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.ylabel('Speed (px/frames)',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.title('Clusterwise comparision of Cell Speed',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
#plt.legend()
plt.xlim(12,24)  # between 12-24 frames i.e., 60 min-120 min == 1hr-2hr
plt.ylim(5,10)
plt.show()

plt.plot(x, y0_pers, color='r', linewidth=2, label='Cluster 0')
plt.plot(x, y1_pers, color='g', linewidth=2, marker='>', label='Cluster 1')
plt.plot(x, y2_pers, color='b', linewidth=2, marker='>', label='Cluster 2')
plt.plot(x, y3_pers, color='m', linewidth=2, marker='>', label='Cluster 3')
sns.despine()     
plt.xlabel('Time in frames',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.ylabel('Persistence',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.title('Clusterwise comparision of Cell Persistence',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.legend()
plt.xlim(12,24)  # between 12-24 frames i.e., 60 min-120 min == 1hr-2hr
plt.ylim(0.1,0.35)
# plt.errorbar(x, y0_pers, yerr= np.mean(y0_pers), color='r',uplims = True,  
#              lolims = True, elinewidth=0.5, capsize=1)
# plt.errorbar(x, y1_pers, yerr= np.std(y0_pers, ddof=1) / np.sqrt(len(y0_pers)), color='g',uplims = True,  
#              lolims = True, elinewidth=0.5, capsize=2)
# plt.errorbar(x, y2_pers, xerr=np.std(y2_pers) * 2, color='b',uplims = True,  
#              lolims = True, elinewidth=0.5, capsize=2)
# plt.errorbar(x, y3_pers, xerr=np.std(y3_pers) * 2, color='m',uplims = True,  
#              lolims = True, elinewidth=0.5, capsize=2)

# plt.errorbar(x, y0_pers, yerr=0.01, fmt='o', color='black',
#              ecolor='lightgray', elinewidth=3, capsize=0);
plt.show()


plt.plot(x,y0_dist,color='r', linewidth=2, marker='>',label='Cluster 0')
plt.plot(x,y1_dist,color='g', linewidth=2, marker='>',label='Cluster 1')
plt.plot(x,y2_dist,color='b', linewidth=2, marker='>',label='Cluster 2')
plt.plot(x,y3_dist,color='m', linewidth=2, marker='>',label='Cluster 3')
sns.despine()     
plt.xlabel('Time in frames (1hr-2hr)',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.ylabel('Distance travelled (px)',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.title('Clusterwise comparision of Distance travelled',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.legend()
plt.xlim(12,24)  # between 12-24 frames i.e., 60 min-120 min == 1hr-2hr
# plt.ylim(0,0.8)
plt.show()




plt.plot(x,m0_per,color='r', linewidth=2, marker='>',label='M0')
plt.plot(x,m1_per,color='g', linewidth=2, marker='>',label='M1')
plt.plot(x,m2_per,color='b', linewidth=2, marker='>',label='M2')
sns.despine()     
plt.xlabel('Time in frames',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.ylabel('Persistence',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.title('Persistence of the cells in M0, M1 & M2 images',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.legend()
plt.xlim(2,36)
plt.ylim(0,0.8)
plt.show()


sns.regplot(x, y0_spd,ci=95, color='r',fit_reg=True,label='Cluster 0')
sns.regplot(x, y1_spd,ci=95, color='g',fit_reg=True,label='Cluster 1')
sns.regplot(x, y2_spd,ci=95, color='b',fit_reg=True,label='Cluster 2')
sns.regplot(x, y3_spd,ci=95, color='m',fit_reg=True,label='Cluster 3')
sns.despine()     
plt.xlabel('Time in frames',fontweight="bold",fontname="Times New Roman")
plt.ylabel('Speed (px/frames)',fontweight="bold",fontname="Times New Roman")
plt.title('Clusterwise comparision of Cell Speed',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.legend()
plt.xlim(12,24)
plt.ylim(0,18)
plt.show()


sns.regplot(x, y0_pers,ci=95, color='r',fit_reg=True,label='Cluster 0')
sns.regplot(x, y1_pers,ci=95, color='g',fit_reg=True,label='Cluster 1')
sns.regplot(x, y2_pers,ci=95, color='b',fit_reg=True,label='Cluster 2')
sns.regplot(x, y3_pers,ci=95, color='m',fit_reg=True,label='Cluster 3')
sns.despine()     
plt.xlabel('Time in frames',fontweight="bold",fontname="Times New Roman")
plt.ylabel('Persistence',fontweight="bold",fontname="Times New Roman")
plt.title('Clusterwise comparision of Cell Persistence',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.legend()
plt.xlim(12,24)
plt.ylim(0, 0.3)
plt.show()

from random import seed
from random import random
random_walk = list()
random_walk.append(-1 if random() < 0.5 else 1)
for i in range(1, 1000):
	movement = -1 if random() < 0.5 else 1
	value = random_walk[i-1] + movement
	random_walk.append(value)


seed(1)
from statsmodels.tsa.stattools import adfuller

autocorrelation_plot(y3_spd)
adfuller(y3_spd)
plt.title('cluster 3')
plt.show()







g = sns.lmplot('cell_num', 'VCL', col='Cluster', data=columns,
               markers=".", scatter_kws=dict(color='c'))
plt.xlim(1,75)
plt.ylim(1,1500)

g.map(plt.axhline, y=300, color="k", ls=":");

sns.regplot(range(0,num_frames), y0_pers,
                 x_estimator=np.mean, logx=True)

#### Single cells
singleCell_df = pd.read_csv("Round_Speed_allCells.txt") 
sin_speed_arr = singleCell_df.to_numpy()
singleCell_df['R_Speed'] =  sin_speed_arr.tolist()
singleCell_df['R_Per'] = pd.read_csv("Round_Persistence_cellWise.txt").to_numpy().tolist()


singleCell_df1 = pd.read_csv("Pro_Speed_allCells.txt")
sin_speed_arr = singleCell_df1.to_numpy()
singleCell_df1['P_Speed'] = sin_speed_arr.tolist()
singleCell_df1['P_Per'] = pd.read_csv("Pro_Persistence_cellWise.txt").to_numpy().tolist()



singleCell_df2 = pd.read_csv("Elon_Speed_allCells.txt")
sin_speed_arr = singleCell_df2.to_numpy()
singleCell_df2['E_Speed'] = sin_speed_arr.tolist()
singleCell_df2['E_Per'] = pd.read_csv("Elon_Persistence_cellWise.txt").to_numpy().tolist()


single_x =range(1,len(singleCell_df.R_Per[1])+1)
plt.plot(single_x, singleCell_df.R_Per[1],color='r', linewidth=2, label='Round')
plt.plot(single_x, singleCell_df1.P_Per[1],color='g', linewidth=2, label='Multipolar')
plt.plot(single_x, singleCell_df2.E_Per[0],color='b', linewidth=2, label='Elongated')
sns.despine()     
plt.xlabel('Time in frames',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.ylabel('Persistence',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.title('Persistence of the cells',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.legend()
plt.xlim(2,240)
plt.ylim(0,0.8)
plt.show()



plt.plot(single_x, singleCell_df.R_Speed[1],color='r', linewidth=2, label='Round')
plt.plot(single_x, singleCell_df1.P_Speed[1],color='g', linewidth=2, label='Multipolar')
plt.plot(single_x, singleCell_df2.E_Speed[0],color='b', linewidth=2, label='Elongated')
sns.despine()     
plt.xlabel('Time in frames',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.ylabel('Speed (px/frames)',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.title('Speed of the cells',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.legend()
plt.xlim(1,240)
# plt.ylim(0,0.8)
plt.show()








## box plots 


from numpy.random import default_rng
rng = default_rng(42)


fig = plt.figure()

ax = fig.add_axes([0, 0, 1, 1]) 

# to plot scatter points on the box plots
allData = np.array(data3)
xCenter = np.array(range(1,len(allData)+1))
spread = 0.5;
for i in range(0,np.size(allData)):
    ax.plot(rng.random(np.size(allData[i]))*spread -(spread/2) + xCenter[i], allData[i],'y.','linewidth', 0.01, marker='o', alpha=.5, ms=4)

bp = ax.boxplot(data3,notch=True,widths=0.15,labels=[c1,c2,c3,c4], patch_artist = True) 

for patch, color in zip(bp['boxes'], colors): 
    patch.set_facecolor(color) 
plt.ylabel('Compactness')
plt.title('Compactness')
plt.grid()
plt.show() 



### histograms
ylabel = 'Frequency'; c1 = 'Cluster 1';c2 = 'Cluster 2';c3 = 'Cluster 3';c4 = 'Cluster 4';

feq1 = segPlots.histPlots(columns.ecc[columns.Cluster == 0],'Eccentricity',ylabel,c1,'r')
feq2 = segPlots.histPlots(columns.sol[columns.Cluster == 0],'Solidity',ylabel,c1,'r')
segPlots.histPlots(columns.cmp[columns.Cluster == 0],'Compactness',ylabel,c1,'r')

segPlots.histPlots(columns.ecc[columns.Cluster == 1],'Eccentricity',ylabel,c2,'g')
segPlots.histPlots(columns.sol[columns.Cluster == 1],'Solidity',ylabel,c2,'g')
segPlots.histPlots(columns.cmp[columns.Cluster == 1],'Compactness',ylabel,c2,'g')

segPlots.histPlots(columns.ecc[columns.Cluster == 2],'Eccentricity',ylabel,c3,'b')
segPlots.histPlots(columns.sol[columns.Cluster == 2],'Solidity',ylabel,c3,'b')
segPlots.histPlots(columns.cmp[columns.Cluster == 2],'Compactness',ylabel,c3,'b')

segPlots.histPlots(columns.ecc[columns.Cluster == 3],'Eccentricity',ylabel,c4,'m')
segPlots.histPlots(columns.sol[columns.Cluster == 3],'Solidity',ylabel,c4,'m')
segPlots.histPlots(columns.cmp[columns.Cluster == 3],'Compactness',ylabel,c4,'m')


## to find the optimal k-value
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np

sil = []
kmax = 10

all_df = columns[['ecc','sol','cmp']]
# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(all_df)
  labels = kmeans.labels_
  sil.append(silhouette_score(all_df, labels, metric = 'euclidean'))
 

plt.plot(range(2, kmax+1), sil, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('The Elbow Method using Silhouette score')
plt.show()

##elbow method to get the optimal k value
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
for k in range(2, kmax+1):
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(all_df)
    kmeanModel.fit(all_df)
 
    distortions.append(sum(np.min(cdist(all_df, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / all_df.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(all_df, kmeanModel.cluster_centers_,
                                    'euclidean'), axis=1)) / all_df.shape[0]
    mapping2[k] = kmeanModel.inertia_

plt.plot(range(2, kmax+1), distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()



# #####histogram plot
# x = df['label']
# ecc = df['eccentricity']
# e0 = ecc[y==0]
# e1 = ecc[y==1]
# e2 = ecc[y==2]


# plt.hist(e0, bins=30, color='y',
#                             alpha=0.6, rwidth=0.5,label = 'Cluster 1')

# nall, binsAll, patchesAll = plt.hist(e1, bins=30, color='b',
#                             alpha=0.6, rwidth=0.5 , label='Cluster 2')
# nall1, binsAll1, patchesAll2 = plt.hist(e2, bins=30, color='r',
#                             alpha=0.6, rwidth=0.5, label='Cluster 3')

# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Eccentricity')
# plt.ylabel('Frequency')
# plt.title('Eccentricity values[0.5-0.85] clustered based on ecc,sol and compactness')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


## 3 cluster values  -- max(columns.compactness[columns.Cluster == 1])
## Eccentricity CLuster 1 range: [0.538764879510548,0.984371885114742]
## Eccentricity CLuster 2 range: [0.35401948158026003,0.995395274624147]
## Eccentricity CLuster 3 range: [0.244117682072647,0.8070170227851421]      -- Low Eccentricity -- circular


## Solidity CLuster 1 range: [0.739928909952607,0.9776264591439692]
## Solidity CLuster 2 range: [0.463364293085655,0.8581422464049749]         -- Low Solidity  -- less density
## Solidity CLuster 3 range: [0.7434697855750491,0.9840796019900501]

## Compactness CLuster 1 range: [0.3898676729622871,0.9497089110154879]
## Compactness CLuster 2 range: [0.13710462701382903,0.740236910935775]  -- Low compactness   --more roughness
## Compactness CLuster 3 range: [0.34883213624194803,1]
## Compactness CLuster 3 range: [0.141224329314772,0.5026794188062821]


