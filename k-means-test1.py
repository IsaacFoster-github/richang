import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k,
                              1)) - centroids  # 相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2  # 平方
        squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
        distance = squaredDist ** 0.5  # 开根号
        clalist.append(distance)
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist


# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # print("clalist------------------",clalist)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()  # DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    # 计算变化量
    changed = newCentroids - centroids

    return changed, newCentroids


# 使用k-means分类
# def kmeans(dataSet, k):
#     # 随机取质心
#     centroids = random.sample(dataSet, k)
#     # 更新质心 直到变化量全为0
#     changed, newCentroids = classify(dataSet, centroids, k)
#     k_index=0
#     while np.any(changed != 0):
#         k_index += 1
#         changed, newCentroids = classify(dataSet, newCentroids, k)
#
#     centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序
#
#
#     # 根据质心计算每个集群
#     cluster = []
#     clalist = calcDis(dataSet, centroids, k)  # 调用欧拉距离
#     minDistIndices = np.argmin(clalist, axis=1)
#     for i in range(k):
#         cluster.append([])
#     for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
#         cluster[j].append(dataSet[i])
#
#     return centroids, cluster

def get_test_data():

    N = 1000

    # 产生点的区域
    #   area_1 = [0, N / 4, N / 4, N / 2]
    #   area_2 = [N / 2, 3 * N / 4, 0, N / 4]
    #   area_3 = [N / 4, N / 2, N / 2, 3 * N / 4]
    #   area_4 = [3 * N / 4, N, 3 * N / 4, N]
    #   area_5 = [3 * N / 4, N, N / 4, N / 2]
    area_1 = [0, 0, N / 4, N / 4]
    area_2 = [N / 4, 0, N / 2, N / 4]
    area_3 = [N / 4, N / 4, N / 2, N / 2]
    area_4 = [N / 2, 3 * N / 4, 3 * N / 4, N]
    area_5 = [3 * N / 4, N / 2, N, 3 * N / 4]
    area_6 = [3 * N / 4, 3 * N / 4, N, N]

    areas = [area_1, area_2, area_3, area_4, area_5, area_6]
    # k = len(areas)
    print(areas)
    # 在各个区域内，随机产生一些点
    points = []
    for area in areas:
        rnd_num_of_points = random.randint(1, 100)
        for r in range(0, rnd_num_of_points):
            rnd_add = random.randint(0, 100)
            rnd_add1 = random.randint(0, 100)
            rnd_x = random.randint(area[0] + rnd_add, area[2] - rnd_add)
            rnd_y = random.randint(area[1] + rnd_add1, area[3] - rnd_add1)
            points.append([rnd_x, rnd_y])
        # 自定义中心点，目标聚类个数为5，因此选定5个中心点
        #  center_points = [[0, 250], [500, 500], [500, 250], [500, 250], [500, 750]]
    # center_points = [[250, 750], [750, 250]]
    data = np.array(points)
    # data_x = [x[0] for x in data]
    # data_y = [x[1] for x in data]
    # plt.subplot()
    # plt.plot(data_x, data_y, '.')
    # plt.axis([0, 1000, 0, 1000])

    return data


    # for i in range(len(dataset)):
    #     plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='green', s=40, label='原始点')
    #     plt.show()
    #     #  记号形状       颜色      点的大小      设置标签
    #     for j in range(len(centroids)):
    #         plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50, label='质心')
    #         plt.show()

def get_scatter(cluster,centorid):
    label_1 = [0, 1, 2, 3]
    colors_1 = ['#00CED1', '#DC143C', '#FFD700', '#32CD32', '#CD853F']
    markers = ['x','+','o','>']
    c_mark=['1','2','3','4']
    for i in range(len(cluster)):
        x = [j[0] for j in cluster[i]]
        y = [j[1] for j in cluster[i]]
        plt.scatter(x, y, marker=markers[i], label=label_1[i],s=80)
    for i in range(len(centorid)):
        plt.scatter(centorid[i][0], centorid[i][1], marker=c_mark[i], label=label_1[i],s=500)
    plt.show()

if __name__ == '__main__':
    # dataset = np.random.uniform(0,10,size=[200,2])
    # dataset = np.round(dataset, 2)
    # dataSet = dataset.tolist()
    # dataset = set(dataset)
    dataset = get_test_data()
    dataSet = dataset.tolist()
    print("数据集dataset", dataset)
    k = 4
    # centroids, cluster = kmeans(dataset, 5)
    centroids = random.sample(dataSet, k)
    print('原始质心', centroids)
    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    newCentroids = np.round(newCentroids, 2)
    k_index=0
    # get_scatter(cluster)
    centroids = np.round(centroids, 2)
    while np.any(changed != 0):

        # 根据质心计算每个集群
        cluster = []
        clalist = calcDis(dataSet, centroids, k)  # 调用欧拉距离
        minDistIndices = np.argmin(clalist, axis=1)
        for i in range(k):
            cluster.append([])
        for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
            cluster[j].append(dataSet[i])
        print('第 %s ' %k_index +'次索引')
        print('质心为：%s' % centroids)
        print('集群为：%s' % cluster)
        print("cluster.size", len(cluster))
        cluster = np.array(cluster)
        get_scatter(cluster, centroids)
        changed, newCentroids = classify(dataSet, newCentroids, k)
        centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序
        centroids = np.round(centroids, 2)
        k_index += 1



