#PCA
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import datasets
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from   sklearn.model_selection  import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from matplotlib.colors import ListedColormap
#%% K折交叉验证
iris= load_iris()
train = iris.data
target =iris.target.reshape(150,1)
label=np.concatenate((train,target),axis=1)
'''
shuffle:选择是否在分割成批次之前对数据的每个分层进行打乱。
               供5次2折使用，这样每次的数据是进行打乱的，否则，每次取得的数据是相同的
random_state:控制随机状态，随机数生成器使用的种子
'''
kf = KFold(n_splits=2)
knn = KNeighborsClassifier(5)
aver_acc=[]
for train_index, test_index in kf.split(label):
    iris_train, iris_test=label[train_index],label[test_index]
    knn.fit(iris_train[:,:4],iris_train[:,4])
    y_pred=knn.predict(iris_test[:,:4])
    print('Accuracy: ' +str( metrics.accuracy_score(iris_test[:,4],y_pred)))
    aver_acc.append(metrics.accuracy_score(iris_test[:,4],y_pred))
    '''
    metrics是非常重要的类
    accuracy_socre:Accuracy classification score.
    '''
print('Accuracy prediction of Unkonwed dataset by K-Fold: '+ 
      str( np.average(aver_acc)))

#%% K临近实现+绘图

#首先分出训练集和验证集
iris_train, iris_test, target_train, target_test = train_test_split(train, target, test_size=0.5)

#创建绘图色彩RBG列表
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#找出各特征的最大最小值，用于创建网格
iris_fea1_min, iris_fea1_max = iris_test[:, 0].min() - 1, iris_test[:, 0].max() + 1
iris_fea2_min, iris_fea2_max = iris_test[:, 1].min() - 1, iris_test[:, 1].max() + 1
iris_fea3_min, iris_fea3_max = iris_test[:, 2].min() - 1, iris_test[:, 2].max() + 1
iris_fea4_min, iris_fea4_max = iris_test[:, 3].min() - 1, iris_test[:, 3].max() + 1

#网格创建输入的数据shape 需要相同，因此采用以下公式，保证其有相同的shape
step=[(iris_fea1_max-iris_fea1_min)/200,
                         (iris_fea2_max-iris_fea2_min)/200,
                         (iris_fea3_max-iris_fea3_min)/200,
                         (iris_fea4_max-iris_fea4_min)/200]

#创建各特征的网格


fea1, fea2= np.meshgrid(np.arange(iris_fea1_min, iris_fea1_max, step[0]),
                     np.arange(iris_fea2_min, iris_fea2_max, step[1]))

fea3,fea4=np.meshgrid(np.arange(iris_fea3_min, iris_fea3_max, step[2]),
                      np.arange(iris_fea4_min, iris_fea4_max, step[3]))
knn.fit(iris_train, target_train)
#ravel 用于产生一维的扁平矩阵
#np.c_用于按行连接矩阵(列数增加) ，对应的有np.r_
#下面的语句生成了对应每个网格值的
Z = knn.predict(np.c_[fea1.ravel(), fea2.ravel(),fea3.ravel(),fea4.ravel()])

#此处将Z (40000,1)转为(200,200) 每隔200个截取列转置为行（上述ravel()是取每行叠加到列上）
Z = Z.reshape(fea2.shape)

pre_t=knn.predict(iris_test)
plt.figure(figsize=(12,5))
'''
#绘制彩格，fea1,fea2，Z是标量二维矩阵，用于colorize 坐标系 ，三个矩阵则有三种颜色
#第一，第二顺位的变量会给定X,Y平面的网格，而第三顺位的变量Z会被投影到XY平面中，如下3D图可见
(类似于隐函数，详情见github中另一程序 隐函数作图2)
'''

#petal
plt.figure(figsize=(12,5))
plt.pcolormesh(fea3,fea4,Z, cmap=cmap_light)
plt.scatter( iris_test[:, 2],  iris_test[:, 3], c=pre_t, cmap=cmap_bold)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title("Iris K-means classfication" )
plt.savefig("petal.jpg")
plt.show()

#3D绘制
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(fea3, fea4, Z, rstride=1, cstride=1, cmap=cmap_bold,alpha=0.5)
ax.contourf(fea3, fea4, Z, zdir='z', offset=-2, cmap=cmap_bold)
ax.set_zlim(0,2)
ax.scatter(iris_test[:, 2],iris_test[:, 3],pre_t,cmap=plt.cm.hot)
plt.savefig('3D-KNN')
plt.show()
#sepal
plt.pcolormesh(fea1,fea2,Z, cmap=cmap_light,alpha=0.5)
#plt.grid(b=True)

#绘制散点图，变量分别为 sepal len，sepal wid，预测集的结果与color-maped
plt.scatter( iris_test[:, 0],  iris_test[:, 1], c=pre_t, cmap=cmap_bold)
plt.xlim(fea1.min(), fea1.max())
plt.ylim(fea2.min(), fea2.max())
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title("Iris K-means classfication" )
plt.savefig("sepal.jpg")
plt.show()


#%% 与PCA结合进行K邻近
#先使用PCA将特征进行降维
from sklearn.decomposition import PCA
y=iris.target
X1=iris.data
pca= PCA(n_components=2)
X_r=pca.fit(X1).transform(X1)
train=X_r

iris_train, iris_test, target_train, target_test = train_test_split(train, target, test_size=0.5)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
iris_fea1_min, iris_fea1_max = iris_test[:, 0].min() - 1, iris_test[:, 0].max() + 1
iris_fea2_min, iris_fea2_max = iris_test[:, 1].min() - 1, iris_test[:, 1].max() + 1
#iris_fea3_min, iris_fea3_max = iris_test[:, 2].min() - 1, iris_test[:, 2].max() + 1
#iris_fea4_min, iris_fea4_max = iris_test[:, 3].min() - 1, iris_test[:, 3].max() + 1
'''
step=[(iris_fea1_max-iris_fea1_min)/200,
                         (iris_fea2_max-iris_fea2_min)/200,
                         (iris_fea3_max-iris_fea3_min)/200,
                         (iris_fea4_max-iris_fea4_min)/200]
'''

step=[(iris_fea1_max-iris_fea1_min)/200,
                         (iris_fea2_max-iris_fea2_min)/200]
fea1, fea2= np.meshgrid(np.arange(iris_fea1_min, iris_fea1_max, step[0]),
                     np.arange(iris_fea2_min, iris_fea2_max, step[1]))
#fea3,fea4=np.meshgrid(np.arange(iris_fea3_min, iris_fea3_max, step[2]),
#                      np.arange(iris_fea4_min, iris_fea4_max, step[3]))
knn.fit(iris_train, target_train)

Z = knn.predict(np.c_[fea1.ravel(), fea2.ravel()])
#Z = knn.predict(np.c_[fea1.ravel(), fea2.ravel(),fea3.ravel(),fea4.ravel()])
Z = Z.reshape(fea2.shape)
knn.score(iris_test,target_test)
pre_t=knn.predict(iris_test)
plt.figure(3,figsize=(12,5))
plt.pcolormesh(fea1,fea2,Z, cmap=cmap_light)
plt.scatter( iris_test[:, 0],  iris_test[:, 1], c=pre_t, cmap=cmap_bold)
plt.xlim(fea1.min(), fea1.max())
plt.ylim(fea2.min(), fea2.max())
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.savefig("PCA-KNN.jpg")
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(fea1, fea2, Z, rstride=1, cstride=1, cmap=cmap_bold)
ax.contourf(fea1, fea2, Z, zdir='z', offset=-2, cmap=cmap_bold)
ax.set_zlim(0,2)
ax.scatter(iris_test[:, 0],iris_test[:,1],pre_t)
plt.savefig('3D-PCA-KNN')
plt.show()
'''
plt.figure()
plt.pcolormesh(fea3,fea4,Z, cmap=cmap_light)
plt.scatter( iris_test[:, 2],  iris_test[:, 3], c=pre_t, cmap=cmap_bold)

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title("3-Class classification (k = 15, weights = 'distance')" )

'''
'''
似乎PCA降维后，KNN分类的正确率更高
'''
#%%
#% 学习曲线 

train_sizes = [20, 50, 90, 110, 120]
for model, i in [( KNeighborsClassifier(5),1)]:
    plt.subplot(1,2,i)
    train_sizes, train_scores, test_scores=learning_curve(estimator=model, X=train,y=target, train_sizes=train_sizes)
plt.figure(figsize = (16,5))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
train_scores_mean = np.mean(train_scores, axis=1)    
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="red")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
plt.legend(loc='best')
plt.show()
'''
手动数据集分组
for i in range(n):
    label=pd.DataFrame(X1,iris.target)
    #X1=random.shuffle(X1)
    one_list = X1[math.floor(i / n * length):math.floor((i + 1) / n * length)]
    sep_data[i]=one_list
    label=
for x in range(n):
    train_data=sep_data[x]
    test_data={k:v for k,v in sep_data.items() if k!=x}
    for z in range(n):
        try:
            test_data1.append(test_data[z])
        except  KeyError as e:
            print('not including dataset '+str(z))
        else:
            print('...')
    test_data1=np.array(test_data1)[0,:,:]

'''

'''
参考：
https://zhuanlan.zhihu.com/p/37654241
https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics
https://blog.csdn.net/lllxxq141592654/article/details/81532855
https://www.matplotlib.org.cn/gallery/images_contours_and_fields/pcolormesh_levels.html
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolormesh.html#matplotlib-pyplot-pcolormesh
https://blog.csdn.net/mr_muli/article/details/84496351
https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics
https://blog.csdn.net/sunshunli/article/details/80639068
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
《深度学习》第五章 算法5.1
'''


