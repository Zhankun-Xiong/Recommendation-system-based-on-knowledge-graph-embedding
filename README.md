# knowledge-graph-embedding-
基于knowledge graph embedding的推荐系统

本系统是一个基于knowledge graph embedding的商品推荐系统，以下是对该系统的详细介绍，
基本代码都是自己所写，TransE和Rescal方法实现部分是照着论文与相关代码自己进行的复现
，并且相关代码中都有我写的一些注释。


1.generate_data.py是用于生成模拟数据，在进行真实使用时可以参照所生成模拟数据的格式进行数据录入

2.data文件夹下需要有entities.txt以及relations.txt两个数据，他们分别是实体（people和items）的名称以及索引号，以及关联的名称以及索引号，关联也可以有多种，
然后该文件夹下还应该有train.txt,valid.txt和test.txt，作为模型训练的依托，其中的neg.txt可要可不要，这个文件并不参与模型的训练过程

3.dataset.py文件主要是模型训练中处理数据的代码，model.py是我复现的两种knolwedge graph embedding的方法，在训练中进行调用

4.main.py是主函数，其中要修改成Rescal方法可以将TransE替换成Rescal，其他的地方可以设置训练所用的超参数

5.run.py是用于对模型的超参数进行设置，采用的是五折交叉验证，在求打分验证时可以采用不同的scoring 函数，注释后面写有“Rescal”的

6.use.py是用所有样本进行训练，对所有负样本地方打分并排序，取前100个打分进行输出（数量可选）

7.run.sh是在linux系统下使用的，如是windows系统，可以直接吧代码中os.system('bash run.sh')改为os.system('python main.py'),具体超参数在main中设置即可

8.如果在run.py中发现如何调超参数模型效果都不理想，可能是由于数据过于简单或者不真实所造成。



Recommendation system based on knowledge graph embedding
This system is a product recommendation system based on knowledge graph embedding. The following is a detailed introduction to the system.
The basic code is written by myself. The implementation part of the TransE and Rescal methods is based on the paper and the relevant code.
, And there are some comments I wrote in the relevant code.


1.generate_data.py is used to generate simulation data, and when it is used in real, you can refer to the format of the generated simulation data for data entry

2.data folder needs entities.txt and relationships.txt two data, they are the name of the entity (people and items) and the index number, and the name and index of the association, there can be multiple associations,
Then there should be train.txt, valid.txt, and test.txt in this folder as the basis for model training. Neg.txt is required or not. This file does not participate in the training process of the model.

3.dataset.py file is mainly the code for processing data in model training. Model.py is the two knolwedge graph embedding methods I reproduced, which are called during training.

4.main.py is the main function, where you need to modify the Rescal method to replace TransE with Rescal, and you can set the hyperparameters used for training elsewhere.

5.run.py is used to set the hyperparameters of the model. It uses 50% cross-validation. Different scoring functions can be used when scoring and verifying. The "Rescal"

6.use.py is training with all samples, scoring and sorting all negative samples, taking the first 100 scores for output (the number is optional)

7.run.sh is used in the Linux system. If it is a Windows system, you can directly change os.system ('bash run.sh') in the code to os.system ('python main.py'). The specific hyperparameters Just set it in main

8. If it is found in run.py how to adjust the hyperparameter model is not ideal, it may be caused by the data being too simple or unreal.
