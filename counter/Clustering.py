# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:00:02 2023

@author: adlm
"""

from numpy import column_stack, array, zeros, empty
from numpy import min as npmin
from math import sqrt, exp
import matplotlib
# matplotlib.use('agg')  # Use the agg backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import cdist
from pickle import dump, load
from scipy.signal import savgol_filter
from counter.Intersection import DeFishEye
from os import getcwd


colors = ['aqua','brown','burlywood','cadetblue','chartreuse','chocolate',
 'cornflowerblue','darkgreen','darkmagenta','darkorange','darksalmon',
 'darkseagreen','darkslategray','dimgrey','gold','lightcoral','lightsalmon',
 'lightseagreen','lightskyblue','magenta','mediumseagreen','sienna']

cwd = getcwd()

def gaussian_kernel(x, mu=1, sigma=0.3):
    return exp(-(x-mu)**2 / 2 / sigma**2)

def ClearTraj(Traj):
    CleanTraj = []
    for point in Traj:
        if point[0] == 1000000:
            pass
        else:
            CleanTraj.append(point)
            
    ShortTraj = [CleanTraj[0]]
    Minimum_Distance = 1
    for i in range(1, len(CleanTraj)):
        if sqrt((CleanTraj[i][0]-ShortTraj[-1][0])**2 + (CleanTraj[i][1]-ShortTraj[-1][1])**2) > Minimum_Distance:
            ShortTraj.append(CleanTraj[i])
    return(ShortTraj)

def smooth_trajectory(trajectory, window_length=11, polyorder=3):
    """
    Smooths a given trajectory using the Savitzky-Golay filter.

    Parameters:
        trajectory (numpy.ndarray): A 2D array of shape (n, 2) representing the trajectory.
        window_length (int): The length of the window used for filtering. Must be an odd integer.
        polyorder (int): The order of the polynomial used for fitting. Must be less than window_length.

    Returns:
        numpy.ndarray: A 2D array of shape (n, 2) representing the smoothed trajectory.
    """
    # Apply the Savitzky-Golay filter to each dimension separately
    x_smoothed = savgol_filter(trajectory[:, 0], window_length, polyorder)
    y_smoothed = savgol_filter(trajectory[:, 1], window_length, polyorder)

    # Combine the smoothed x and y coordinates into a 2D array
    smoothed_trajectory = column_stack((x_smoothed, y_smoothed))

    return smoothed_trajectory

def LCSS(traj1, traj2, eps):
    # traj1 and traj2 are two trajectories, each trajectory is a list of points
    # eps is the threshold of the distance between two points
    # state defines the type of output normalization
    n = len(traj1)
    m = len(traj2)
    matrix = zeros((n+1, m+1))
    # fill the matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            d = sqrt((traj1[i-1][0]-traj2[j-1][0])**2 + (traj1[i-1][1]-traj2[j-1][1])**2)
            if  d <= eps:
                matrix[i][j] = matrix[i-1][j-1] + gaussian_kernel(d/eps, mu=0, sigma=1)
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1])
    return matrix[n][m]/min(m, n) 


def TrajCluster(TrajList):

    def find_LCSS_matrix(trajectories):
        n = len(trajectories)
        LCSS_matrix = zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                LCSS_matrix[i][j] = 1- round(gaussian_kernel(LCSS(trajectories[i], trajectories[j], 2)), 2)
                LCSS_matrix[j][i] = LCSS_matrix[i][j] 
        return LCSS_matrix
     
    def cluster_trajectories(trajectories, similarity_matrix, k):
        """
        Cluster the trajectories into k clusters using the similarity matrix
        """
        diff = 1000
        Change = 1
        Cluster_num = 0
        distortions = [1000]
        while diff > 0.5 and Change > 0.2 and Cluster_num <= k:
            Cluster_num += 1
            kmeans = KMeans(n_clusters=Cluster_num, random_state=0).fit(similarity_matrix)
            distortions.append(sum(npmin(cdist(similarity_matrix, kmeans.cluster_centers_, 'euclidean'), axis=1)) / similarity_matrix.shape[0])
            Change = abs((distortions[-2] - distortions[-1])/distortions[-2])
            diff = distortions[-2] - distortions[-1] 

        k = Cluster_num - 1
        print("Cluster_num = ", k)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(similarity_matrix)
        cluster_labels = kmeans.labels_
        
        clusters = {}
        for i in range(k):
            clusters[i] = []

        for i in range(len(cluster_labels)):
            clusters[cluster_labels[i]].append(trajectories[i])

        return clusters
        
    def ClusterDBSCAN(trajectories, similarity_matrix):

        # instantiate the DBSCAN object
        min_samples= round(len(trajectories)/10) + 2
        dbscan = DBSCAN(eps=0.2, min_samples=min_samples, metric='precomputed')
        
        # fit the similarity matrix to the DBSCAN object
        dbscan.fit(similarity_matrix)
        
        # retrieve the cluster labels assigned by DBSCAN
        cluster_labels = dbscan.labels_
        print("cluster_num = ", max(cluster_labels) + 1)
        clusters = {}
        for i in range(max(cluster_labels) + 1):
            clusters[i] = []

        for i in range(len(cluster_labels)):
            if cluster_labels[i] >= 0:
                clusters[cluster_labels[i]].append(trajectories[i])

        return clusters


    def find_best_trajectory(trajectories):
        min_loss = float('inf')
        best_trajectory = None
        for i, t1 in enumerate(trajectories):
            loss = 0
            for j, t2 in enumerate(trajectories):
                if j == i:
                    continue
                loss += 1 - gaussian_kernel(LCSS(t1, t2, 2))
            if loss < min_loss:
                min_loss = loss
                best_trajectory = t1
        return best_trajectory


   
  
    AllTraj = []
    Zones = [1, 2, 3, 4]
    CZones = Zones.copy()
    fig = plt.figure(figsize=(10, 10))
    Ctype = 0
    n_clusters = 10
    TempTrajList = empty((4, 4), dtype=object)
    
    for i in range(4):
        for j in range(4):
            TempTrajList[i][j] = []
             
    for EnZ in Zones:
        CZones.append(CZones.pop(0))
        for ExZ in CZones[0:-1]:
            Track_List = []
            for track in TrajList[EnZ - 1][ExZ - 1]:
                index = track.Zones.index(0)              
                Traj = array(ClearTraj(array(track.Locations[index:]).tolist()))
                if len(Traj) > 5: 
                    smoothed_trajectory = smooth_trajectory(Traj, window_length=5, polyorder=3)
                    Track_List.append(smoothed_trajectory)
            print("Number of training tracks from zone ", EnZ, "to", ExZ, ": ", len(Track_List))
            if len(Track_List) > 0:   
                if len(Track_List) > 9:
                    LCSS_matrix = find_LCSS_matrix(Track_List)
                    #Clusters = cluster_trajectories(Track_List, LCSS_matrix, n_clusters)
                    Clusters = ClusterDBSCAN(Track_List, LCSS_matrix)
                else:
                    Clusters = {0:Track_List}
                
                AllTraj.append(Clusters) 
                for i in Clusters:
                    for track in Clusters[i]:
                        plt.plot([x[0] for x in  track], [x[1] for x in  track], color=colors[Ctype])  
                    Ctype = (Ctype + 1) % len(colors)
                    Best_trajectory = find_best_trajectory(Clusters[i]).tolist()
                    TempTrajList[EnZ - 1][ExZ - 1].append(Best_trajectory)
            else:
                print("Not enough data for vehicles")
    
    for tracklist in TempTrajList:
        for tracks in tracklist:
            for TempTraj in tracks:
                if TempTraj:
                    plt.plot([x[0] for x in  TempTraj], [x[1] for x in  TempTraj], color='k', linewidth=3)  
                        
    #plt.title("Movement Patterns")
    plt.axis('off')
    plt.savefig(cwd + "/Counter/Training/TrajClusters.jpg")
    
      
    # save the training data to a file
    with open(cwd + '/Counter/Training/TempTrajList.pkl', 'wb') as f:
        dump(TempTrajList, f)
    #return TempTrajList


class PTC:
    def __init__(self):
        with open(cwd + '/Counter/Training/TempTrajList.pkl', 'rb') as f:
            self.TempTrajList = load(f)
        
    def PredictTrajClass(self, track):
        def Predict(Trajectory, Temp):
            def Probablity(P):
                Q = [gaussian_kernel(x) for x in P]
                #Q = [x + 1 if x == 1 else x for x in Q]
                Sum = sum(Q)
                return [round(i/Sum, 2) for i in Q]
            fig = plt.figure(figsize=(10, 10))
            plt.plot([x[0] for x in  Trajectory], [x[1] for x in  Trajectory],'o', color='r', label='Broken track', markersize= 10) 
            Similarity = [0] * 4
            c = 0
            for i, TempTracks in enumerate(Temp):
                for TempTraj in TempTracks:
                    c = (c + 7) % len(colors)
                    Track_LCSS = LCSS(Trajectory, TempTraj, 2)
                    plt.plot([x[0] for x in  TempTraj], [x[1] for x in  TempTraj], label=f"Zone: {i+1} / Similarity: {Track_LCSS:.2f}", color= colors[c], linewidth=5)
                    Similarity[i] = Track_LCSS if Track_LCSS > Similarity[i] else Similarity[i]
            # print("Similarity = ", Similarity)
            Similarity_prob = Probablity(Similarity)
            # print("Similarity probablity = ", Similarity_prob)
            max_prob = max(Similarity_prob)
            if (max_prob > 0.5) and max(Similarity) > 0.4:
                return Similarity_prob.index(max_prob) + 1, max_prob
            else:
                max_prob = 0
                return "Unknown", max_prob 
        Trajectory = array(ClearTraj(DeFishEye(array(track.BBoxCenters)).tolist()))
        
        if track.Entrance_Zone in [1, 2, 3, 4]:
           track.Exit_Zone, prob = Predict(Trajectory, self.TempTrajList[track.Entrance_Zone - 1, :])   
           
        elif track.Exit_Zone in [1, 2, 3, 4]:
           track.Entrance_Zone, prob = Predict(Trajectory, self.TempTrajList[:, track.Exit_Zone - 1])
        
        else :
           prob = 0 
       
        if prob > 0.5:    
            track.ClassifyTrajectory()
            track.Type = "Classifiable_Broken"
            #print("broken track. ID:", int(track.ID), " Trajectory:", track.Trajectory, " Exit_Zone:", track.Exit_Zone, " First_Frame:", track.Frames[0], "Current frame", track.Frames[-1])
            plt.title("Trajectry: " + track.Trajectory +" / Probability: " + str(round(prob, 2)), fontsize=20)
            plt.xticks([])
            plt.yticks([])
            plt.legend(fontsize=20)
            plt.savefig(cwd + "/Counter/BrokenTracks/" + str(track.ID) + ".jpg")
            # print("Trajectry class = ", self.Str[track.Trajectory], "with calss probability of ", round(prob, 2))
        else:
            track.Type = "UnClassifiable_Broken"
            #print("*** Unclassified broken track. ID:", int(track.ID), " Exit_Zone:", track.Exit_Zone, " First_Frame:", track.Frames[0], "Current frame", track.Frames[-1])
        
