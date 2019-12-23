import random


with open('data//buy_data.txt', 'w') as f:  #生成购买数据
    for i in range(1000):    #假设有1000个人
        for x in range(random.randint(0,30)):    #假设一人购买0~30件商品
            f.write(str(i) + '\t' +'item'+str(random.randint(0,99))  + '\t' + 'buy' + '\n')    #生成购买数据(假设有100件商品)


with open('data//entities.txt', 'w') as f:  #生成实体索引
    for i in range(1000):    #1000个人和100件物品
        f.write(str(i) + '\t' +str(i) +'\n')    #生成people索引
    for i in range(100):
        f.write('item'+str(i)+'\t'+str(1000+i)+'\n') #生成物品索引

with open('data//relations.txt', 'w') as f:  #生成关联索引
        f.write('buy' + '\t' +str(0) +'\n')    #还可扩展成加入购物车、购买、勾选喜爱类似的关系
