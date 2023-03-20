#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
"""
Referencias:
    
    Fuente primaria del reanálisis
    https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.pressure.html
    
    Altura geopotencial en niveles de presión
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=1498
    
    Temperatura en niveles de presión:
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=4237

    Temperatura en niveles de superficie:
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=1497
    
"""

import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
#from scipy.io import netcdf as nc
from sklearn.decomposition import PCA

workpath = "Archivos"

#import os
#workpath = "C:/NCEP"
#os.getcwd()
#os.chdir(workpath)
#files = os.listdir(workpath)


f = Dataset(workpath + "/air.2021.nc", "r", format="NETCDF4")
#f = Dataset(workpath + "/air.2m.gauss.2021.nc", "r", format="NETCDF4")
#f = nc.netcdf_file(workpath + "/air.2221.nc", 'r')

print(f.history)
print(f.dimensions)
print(f.variables)
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
level = f.variables['level'][:].copy()
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
air21 = f.variables['air'][:].copy()
air_units = f.variables['air'].units
#air_scale = f.variables['air'].scale_factor
#air_offset = f.variables['air'].add_offset
print(air21.shape)
f.close()


#Apartado 1
#Cargamos datos altura geopotencial 2021
f = Dataset(workpath + "/hgt.2021.nc", "r", format="NETCDF4")
print(f.history)
print(f.dimensions)
print(f.variables)
time21 = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
hgt21 = f.variables['hgt'][:].copy()
hgt_units = f.variables['hgt'].units
level21 = f.variables['level'][:].copy() 
#hgt_scale = f.variables['hgt'].scale_factor
#hgt_offset = f.variables['hgt'].add_offset
print(hgt21.shape)
f.close()

#Buscamos la posición donde están los 500hPa
pos500 = 0
for i in range(len(level21)):
    if level21[i] == 500.:
        pos500 = i
        break

hgtp500 = hgt21[:,pos500,:,:].reshape(len(time21),len(lats)*len(lons))

n_components=4


X = hgtp500
Y = hgtp500.transpose()
pca = PCA(n_components=n_components)
Element_pca0 = pca.fit_transform(Y)
Element_pca0 = Element_pca0.transpose(1,0).reshape(n_components,len(lats),len(lons))


#Varianza explicada
pca.fit(Y)
print("Varianza explicada:")
print(pca.explained_variance_ratio_)
out = pca.singular_values_

State_pca = pca.fit_transform(X)
Element_pca = pca.fit_transform(Y)
Element_pca = Element_pca.transpose(1,0).reshape(n_components,len(lats),len(lons))


print("Componentes principales:")
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca[i-1,:,:])
plt.show()

f = Dataset(workpath + "/hgt.2022.nc", "r", format="NETCDF4")
time22 = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
hgt22 = f.variables['hgt'][:].copy()
f.close()
#time_idx = 237  
# Python and the reanalaysis are slightly off in time so this fixes that problem
# offset = dt.timedelta(hours=0)
# List of all times in the file as datetime objects
dt_time21 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time21]
np.min(dt_time21)
np.max(dt_time21)

dt_time22 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time22]
np.min(dt_time22)
np.max(dt_time22)
dia0 = dt.date(2022, 1, 11)

#APARTADO 2
#Cargamos datos altura geopotencial 2022
f = Dataset(workpath + "/hgt.2022.nc", "r", format="NETCDF4")
#print(f.history)
#print(f.dimensions)
#print(f.variables)
time22 = f.variables['time'][:].copy()
time_bnds22 = f.variables['time_bnds'][:].copy()
time_units22 = f.variables['time'].units
hgt22 = f.variables['hgt'][:].copy()
hgt_units22 = f.variables['hgt'].units
level22 = f.variables['level'][:].copy() 
lat22 = f.variables['lat'][:].copy()
lon22 = f.variables['lon'][:].copy()
#hgt_scale = f.variables['hgt'].scale_factor
#hgt_offset = f.variables['hgt'].add_offset
#print(hgt22.shape)
f.close()

#Guardamos el índice del día 2022/01/11
index0 = dt_time22.index(dia0)

#Buscamos los índices de los límites para la latitud y longitud
#Como la longitud va de 0 a 360º, el intervalo (-20,20) es igual que (340,360)U(0,20) 
ini_lon = 0
fin_lon = 0
for i in range(144):
    if lon22[i] == 20.:
        ini_lon = i
    elif lon22[i] == 340.:
        fin_lon = i+1
        break

ini_lat = 0
fin_lat = 0
for i in range(73):
    if lat22[i] == 30.:
        fin_lat = i
        break
    elif lat22[i] == 50.:
        ini_lat = i+1
        
        
#Indices de presión en 500hPa y 1000hPa
p500 = 0
p1000 = 0
for i in range(len(level22)):
    if level22[i] == 500:
        p500 = i
        break
    if level22[i] == 1000:
        p1000 = i

#Lista de pares del día day y su distancia euclídea al día 2022/01/11
dist = []
for day in range(365):
    di = 0
    for p in range(17):
        if p == p500 or p == p1000:
            for lt in range(ini_lat,fin_lat):
                for ln in range (0,ini_lon):
                    di += 0.5*(hgt22[index0,p,lt,ln]-hgt21[day,p,lt,ln])**2
                for ln in range (fin_lon, 144):
                    di += 0.5*(hgt22[index0,p,lt,ln]-hgt21[day,p,lt,ln])**2
    dist.append((day,np.sqrt(di)))
    
#Ordenamos para obtener los 4 días más análogos 
dist.sort(key=lambda x: x[1])
print("Los 4 días más análogos son:")
for i in range(4):
    print(dt_time21[dist[i][0]])

#Cargamos air 2022
f = Dataset(workpath + "/air.2022.nc", "r", format="NETCDF4")
air22 = f.variables['air'][:].copy()
air_units22 = f.variables['air'].units
f.close()

#Presión 1000hPa
p1000 = 0
        
#Hacemos la media de los cuatro días más análogos a a0 para cada par latitud y longitud
media_analogos = np.zeros((73,144))
for lt in range(73):
    for ln in range(144):
        for d in range(4):
            media_analogos[lt, ln] += air21[dist[i][0],p1000,lt,ln]
    
# Media de los 4 días análogos
media_analogos /= 4

#Hallamos el error abosulto medio para el día a0
error_abs = np.average(np.absolute(np.subtract(media_analogos, air22[index0,p1000,:,:])))

print("El error absoluto medio para el día 2022/01/11: " + str(error_abs))

