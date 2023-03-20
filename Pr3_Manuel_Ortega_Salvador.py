#!/usr/bin/env python
# coding: utf-8


# -*- coding: utf-8 -*-

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
from scipy.spatial import ConvexHull, convex_hull_plot_2d, Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


# #############################################################################
# Aquí tenemos definido el sistema X de 1000 elementos de dos estados
# construido a partir de una muestra aleatoria entorno a unos centros:
centers = [[-0.5, 0.5], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,random_state=0)

def plot_clusters(X, n_clusters, algorithm, labels=None, centers=None, problem=None, voronoi_diag=False):
    """ Función auxiliar para pintar los puntos de los clusters y, optativamente, sus centros.
    :param X: array de puntos de dos dimensiones (array de array de enteros)
    :param labels: cluster al que pertenece cada punto (array de enteros)
    :param centers: coordenadas de los centroides de los clusters (array de array de enteros)
    """
    
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

    if voronoi_diag:
        vor = Voronoi(centers)
        voronoi_plot_2d(vor)
    
    plt.xlim(-2,2)
    plt.ylim(-2,1.5)
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    plt.title('Fixed number of ' + algorithm + ' clusters: %d' % n_clusters)
    
    # Pintar los centroides de los clusters
    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1], marker="x", color='k', s=150, linewidths = 5, zorder=10)
        
    if problem is not None:
        plt.plot(problem[:,0],problem[:,1],'o', markersize=12, markerfacecolor="red")
        
    plt.show()


#Pintamos la gráfica del KMeans para cada número de clusters (hasta 15)
K_MAX = 16
silhouette = np.zeros(K_MAX-2)
for k in range(2, K_MAX): 
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(X)
    plot_clusters(X, k, 'KMeans', km.labels_, km.cluster_centers_)
    silhouette[k-2] = metrics.silhouette_score(X, km.labels_)


#Gráfica de coeficiente de Silhouette en función de nº de clusters.
plt.plot(range(2, K_MAX), silhouette)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette value')
plt.title('Valor de Silhouette para diferentes k')
plt.show()

#Pintamos el caso óptimo que es k = 3
n_clusters_opt = 3
km = KMeans(n_clusters_opt, random_state=0)
km.fit(X)
plot_clusters(X, n_clusters_opt, 'KMeans', km.labels_, km.cluster_centers_, None, True)


#Aplicar el algoritmo DBSCAN a nuestro sistema X para un epsilon y métrica dados.
def dbscan(epsilon, metrica):

    db = DBSCAN(eps=epsilon, min_samples=10, metric=metrica).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    return metrics.silhouette_score(X, labels)

#Aplicamos DBSCAN para valores de epsilon entre 0.1 y 0.4
sil_eu = []
sil_man = []
for epsilon in np.arange(0.1, 0.4, 0.05):
    sil = dbscan(epsilon, 'euclidean')
    sil_eu.append([sil, epsilon])
    
    sil = dbscan(epsilon, 'manhattan')
    sil_man.append([sil, epsilon])

#Dibujar los clusters después de aplicar el DBSCAN.
def plot_dbscan(dbscan, epsilon):
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    plot_clusters(X, n_clusters_, 'DBSCAN', labels=labels)
    print("El número óptimo de clusters es " + str(n_clusters_) + ' para un umbral de distancia de ' + str(epsilon) + '.')


#Gráfica del coeficiente de Silhouette en función de epsilon (metrica euclidean)
plt.title('Valor de Silhouette en métrica euclidean para diferentes epsilon (umbral de distancia)')
plt.plot([sil_eu[i][1] for i in range(len(sil_eu))],[sil_eu[i][0] for i in range(len(sil_eu))])
plt.show()


#DBSCAN con métrica euclidean para el epsilon óptimo:
epsilon = max(sil_eu)[1]
metrica = 'euclidean'
db = DBSCAN(eps=epsilon, min_samples=10, metric=metrica).fit(X)
plot_dbscan(db, epsilon)


#Gráfica del coeficiente de Silhouette en función de epsilon (metrica manhattan)
plt.title('Valor de Silhouette en métrica manhattan para diferentes epsilon (umbral de distancia)')
plt.plot([sil_man[i][1] for i in range(len(sil_man))],[sil_man[i][0] for i in range(len(sil_man))])
plt.show()


#DBSCAN con métrica manhattan para el epsilon óptimo:
epsilon = max(sil_man)[1]
metrica = 'manhattan'
db = DBSCAN(eps=epsilon, min_samples=10, metric=metrica).fit(X)
plot_dbscan(db, epsilon)

#Clasificación de dos puntos
#Hacemos el caso óptimo que es k = 3
n_clusters_opt = 3
km = KMeans(n_clusters_opt, random_state=0)
km.fit(X)
# Predicción de elementos para pertenecer a una clase:
problem = np.array([[0, 0], [0,-1]])
clases_pred = km.predict(problem)
#Vemos qué cluster es según su centroide para poder comprender mejor la predicción
for i in range(n_clusters_opt):
    print("El cluster " + str(i) + " es el de centroide en " + str(km.cluster_centers_[i]) + '.')
for i in range(len(clases_pred)):
    print("El punto " + str(problem[i]) + " pertenece al cluster " + str(clases_pred[i]) + '.')
plot_clusters(X, n_clusters_opt, 'KMeans', km.labels_, km.cluster_centers_, problem, True)

