# coding=UTF-8
import tensorflow as tf
import numpy as np
import os

mat=np.zeros((1000,100))
data=np.loadtxt('data//buy_data.txt',dtype=str)
for i in range(data.shape[0]):
    x=int(data[i][0])
    y=int(data[i][1].replace('item',''))
    mat[x][y]=1                #将数据构建成矩阵，方便操作

def get_metrics(real_score, predict_score):     #计算指标的函数
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[(np.array([sorted_predict_score_num])*np.arange(1, 1000)/np.array([1000])).astype(int)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    TP = predict_score_matrix*real_score.T
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix=np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T

    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
    PR_dot_matrix[1,:] = -PR_dot_matrix[1,:]
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index, 0]
    accuracy = accuracy_list[max_index, 0]
    specificity = specificity_list[max_index, 0]
    recall = recall_list[max_index, 0]
    precision = precision_list[max_index, 0]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision] #jisu #


oneindex=np.where(mat==1)
oneindex=np.array(oneindex)
zeroindex=np.where(mat==0)
zeroindex=np.array(zeroindex)
np.random.seed(0)
np.random.shuffle(oneindex.T)
print(oneindex)
allresult=[]



#制作测试集，训练集，这里训练集和验证集一样,还有一个负样本集
with open('data//test.txt','w') as f:
    for i in range(oneindex.shape[1]):
        f.write(str(oneindex[0][i]) + '\t' + 'item' + str(oneindex[1][i]) + '\t' + 'buy' + '\n')  # 这里用所有正样本
with open('data//valid.txt','w') as f:
    for i in range(oneindex.shape[1]):
        f.write(str(oneindex[0][i]) + '\t' + 'item' + str(oneindex[1][i]) + '\t' + 'buy' + '\n')  # 这里用所有正样本

with open('data//train.txt','w') as f:
    for i in range(oneindex.shape[1]):
        f.write(str(oneindex[0][i]) + '\t' + 'item' + str(oneindex[1][i]) + '\t' + 'buy' + '\n')  #这里用所有正样本

os.system('bash run.sh')   #执行bash命令进行训练，windows下可直接在main.py中设置参数直接"python main.py"

entity = np.load('TransE_entity_emb.npy')
relation = np.load('TransE_relation_emb.npy')
entity = np.array(entity)
relation = np.array(relation)

with open('data//entities.txt') as f:
    entity2id = {}
    for line in f:
        eid, entity1 = line.strip().split('\t')
        entity2id[eid] = int(entity1)


with open('data//neg.txt', 'w') as f:
    for i in range(zeroindex.shape[1]):
        f.write(str(zeroindex[0][i]) + '\t' + 'item'+str(zeroindex[1][i]) + '\t' + 'buy' + '\n')
c = np.loadtxt('data//neg.txt', dtype=str)



negscore = []        #求所有负样本打分
aa = []
for i in range(c.shape[0]):
    aa.append(c[i][0])
bb = []
for i in range(c.shape[0]):
    bb.append(c[i][1])
aa = np.array(aa)
bb = np.array(bb)
cc = []
for i in range(aa.shape[0]):
    cc.append(entity2id[aa[i]])
entity_h = entity[cc]
cc = []
for i in range(bb.shape[0]):
    cc.append(entity2id[bb[i]])
entity_t = entity[cc]

score11=np.abs(entity_h+relation[0]-entity_t)#TransE

#score11 = np.abs(entity_h * np.matmul(relation[0], entity_t.T).T)  # rescal
score11 = -np.sum(score11, axis=1)
for i in range(len(c)):
    score1 = score11[[i]]
    negscore.append(score1)

#print(negscore)
negscore=np.array(negscore)
negscore=np.reshape(negscore,(negscore.shape[0]))
print(negscore)
index=np.argsort(negscore)
print(index)
index=index[0:100]#选出前100的打分

for i in range(100):
    print("用户"+str(zeroindex[0][index[i]])+"可能会购买商品"+"item"+str(zeroindex[1][index[i]]))





