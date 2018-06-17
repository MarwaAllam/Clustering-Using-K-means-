# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 13:47:43 2018

@author: mn186035
"""

"""
=============================
Importing necessary libraries
=============================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style 
style.use("ggplot")

# to draw the spider chart

#import numpy as np
#import matplotlib.pyplot as plt

from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


# data and kmeans lib
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

"""
===================================
Some information about the features
===================================

if your data < 10,000 just use kmeans:
  if > 10,000 use min-batch kmeans
"""

#Iris data
iris = load_iris()
Data = iris.data
Target = iris.target


# to access a numpy array
np.array(Data)
Data[[0,1],[0,2]] 
Target


# Count for each categorical value
unique, counts = np.unique(Target, return_counts=True)
unique
counts

len(iris.data)  # 150
Target.tolist().count(0)  # 50

"""
===============================
Exploratory Data Analysis - EDA
===============================
"""

def Iris_EDW_fn(Target,Data):
    # scatter plt
    DT = Data
    Trgt = Target
    plt.subplot(221)
    plt.scatter(DT[:,0],Trgt)
    plt.subplot(222)
    plt.scatter(DT[:,1],Trgt)
    plt.subplot(223)
    plt.scatter(DT[:,2],Trgt)
    plt.subplot(224)
    plt.scatter(DT[:,3],Trgt)
    plt.show()
    #--------------------
    # Histogram for the 4 features of the iris data
    plt.subplot(221)
    plt.hist(DT[:,0],bins = 50)
    plt.subplot(222)
    plt.hist(DT[:,1],bins = 50)
    plt.subplot(223)
    plt.hist(DT[:,2],bins = 50)
    plt.subplot(224)
    plt.hist(DT[:,3],bins = 50)


Iris_EDW_fn(Target,Data)


"""
==================
Kmeans clustering 
==================
"""

def Kmeans_clustering_fn(K,rand_state,Traing_set):
    kmeans = KMeans(n_clusters=K, random_state=rand_state).fit(Traing_set)
    Lbls = kmeans.labels_
    clusters_result = kmeans.predict(Traing_set)
    centroids = kmeans.cluster_centers_
    print('Kmneans parameters : ', '\n', kmeans, '\n',
                 'Kmeans Labels for each points : ', '\n', Lbls ,'\n',
                 'prediction for the clustering : ', '\n', clusters_result, '\n',
                 'Clusters Centroids : ', '\n' ,centroids)

    return kmeans,Lbls, clusters_result, centroids

kmns_model, Lbls, clusters_result, centroid = Kmeans_clustering_fn(K=3,rand_state=0,Traing_set=Data)#Kmeans_clustering_fn(3,0,X)


"""
=================================
Plotting the data in each cluster
=================================
"""

def Plot_clusters_data(Data, Labels):
    DT = Data
    lbl = Labels
    colors = ["g.","r.","c.","y.","r.","c.","b."]
    for i in range(len(DT)):
        #print("coordinate:",Data[i],"lebel:",labels[i])
        plt.plot(DT[i][0],DT[i][1],colors[lbl[i]],markersize=10)
      
Plot_clusters_data(Data,Lbls)
    

#plt the clusters labels
#plt.scatter(kmns.cluster_centers_[:,0], kmns.cluster_centers_[:,1], marker = "x", s=150, linewidths = 5, zorder = 10)
#plt.show()

"""
==========================================
Cross tabular for the target and centroids
==========================================
"""

pd.crosstab(Target,Lbls)
pd.crosstab(Target,clusters_result)

## concate the data in one array

Iris_Data = np.concatenate((Data, labels_.reshape(-1,1)), axis=1)

Data.shape       ## (150, 4)
labels_.shape    ## (150,)
(labels_.reshape(-1,1)).shape    ## (150, 1)
Iris_Data.shape   ## (150, 5)


"""
======================================
Radar chart (aka spider or star chart)
======================================

This example creates a radar chart, also known as a spider or star chart.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
matplotlib.axis to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

"""

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta



def unit_poly_verts(theta):
    """
    Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts



def example_data():
    # The following data is the centroids from the kmeans result.
    data = [
        ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'],
        ('Setosa'    , [ centroids[0,:]]),
        ('Versicolor', [ centroids[1,:]]),
        ('Virginica' , [ centroids[2,:]])
        ]
    return data



if __name__ == '__main__':
    N = 4
    theta = radar_factory(N, frame='polygon')
    data = example_data()
    spoke_labels = data.pop(0)

    fig, axes = plt.subplots(figsize=(18, 18), nrows=2, ncols=2,
                             subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['r', 'b', 'g', 'm', 'y']
    
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axes.flatten(), data):
        ax.set_rgrids([1, 2, 3, 4, 5, 6, 7, 8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            #ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    ax = axes[0, 0]
    #labels = ('Setosa', 'Versicolor', 'Virginica')
    #legend = ax.legend( loc=(0.9, .95),
    #                  labelspacing=0.1, fontsize='small')

    fig.text(0.5, 0.965, '3 centroids for iris clustering',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.show()
    