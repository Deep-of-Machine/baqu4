import pcl
import numpy as np
import random

cloud = pcl.load("/home/car/Downloads/rosbag_to_pcd/2022-09-25-15-49-54/778.812119100.pcd")

resolution = 0.2
octree = cloud.make_octreeSearch(resolution)
octree.add_points_from_input_cloud()


searchPoint = pcl.PointCloud()
searchPoints = np.zeros((1, 3), dtype=np.float32)
# searchPoints[0][0] = cloud[3000][0]
# searchPoints[0][1] = cloud[3000][1]
# searchPoints[0][2] = cloud[3000][2]


#searchPoints = (cloud[3000][0], cloud[3000][1], cloud[3000][2])

searchPoint.from_array(searchPoints)

ind = octree.VoxelSearch(searchPoint)

print('Neighbors within voxel search at (' + str(searchPoint[0][0]) + ' ' + str(
    searchPoint[0][1]) + ' ' + str(searchPoint[0][2]) + ')')

for i in range(0, 5):#range(0, ind.size):
    print('index = ' + str(ind[i]))
    print('(' + str(cloud[ind[i]][0]) + ' ' +
          str(cloud[ind[i]][1]) + ' ' + str(cloud[ind[i]][2]))

K = 10
print('K nearest neighbor search at (' + str(searchPoint[0][0]) + ' ' + str(searchPoint[0][1]) + ' ' + str(searchPoint[0][2]) + ') with K=' + str(K))

[ind, sqdist] = octree.nearest_k_search_for_cloud(searchPoint, K)

for i in range(0, ind.size):
    print('(' + str(cloud[ind[0][i]][0]) + ' ' + str(cloud[ind[0][i]][1]) + ' ' + str(
        cloud[ind[0][i]][2]) + ' (squared distance: ' + str(sqdist[0][i]) + ')')