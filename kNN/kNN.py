#coding=utf-8
from numpy import *
import operator
from os import listdir

#创建数据集和标签函数
def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

#k临近值算法
def classify0(inX,dataSet,labels,k):
	#计算距离
	dataSetSize = dataSet.shape[0]#数据行数，也就是样本数
	diffMat = tile(inX,(dataSetSize,1))-dataSet#将输入向量“复制”N次，让其与样本数据集大小相同作减运算
	sqDiffMat = diffMat**2#减的平方
	sqDistances = sqDiffMat.sum(axis=1)#减的平方和
	distances = sqDistances**0.5#计算欧式距离
	sortedDistIndicies = distances.argsort()#排序，从小到大获得距离最短的样本数据编号
	classCount={}
	#确定前K个距离最小的元素所在的主要分类
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]#得到前K个距离最近样本的标签集
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#逆排序，并分解字典classCount。按从大到小的次数排序
	#classCount.iteritems分解字典为元组
	#itemgetter获取对象的（元组）的第一个域值，及类型
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	#f返回频数最高的元素标签
	return sortedClassCount[0][0]

#将文本记录转换为NumPy的解释程序，这部分代码回头再看一遍，为什么自己敲的时候老是出错
#这段代码可以经常复用，因为是读取文本数据的.readlines可以自动将文本分析诚一个列表
def file2matrix(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	fr = open(filename)#为什么这里还要打开一次？
	index = 0
	for line in fr.readlines():
		#整理格式：去掉空格,用/t分开元素列表
		line = line.strip()#去掉空格
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index +=1
	return returnMat,classLabelVector
	

#归一化特征值
def autoNorm(dataSet):
	#得到每一列的最小值和最大值
	minVals = dataSet.min(0)#1*3矩阵
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]#得到数据集行数，后面能复制成同样大小的矩阵方便计算
	normDataSet = dataSet- tile(minVals,(m,1))#tile用于复制举证
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals


