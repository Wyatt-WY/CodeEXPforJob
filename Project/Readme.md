任务：利用RGCN模型和药物知识图谱做新冠肺炎的药物推荐。

#Input：DRKG dataset， 药物等医学方面的知识图谱，13种节点：化合物、疾病、副作用等。经过处理后成为模型需要的三元组的形式。
在list中的ID对应知识图谱中的ID。
(Dataset和处理文件没有包含在project中。)

#框架是Pytorch、DGL。
#model：RGCN。https://arxiv.org/abs/1703.06103
#loss:
#评分的triplet：[Compound，Compound::Disease, Disease],这种三元组是知识图谱中反映，“化合物治愈疾病”这类关系类型。
#其中Compounds：来自于原有的dataset并且去掉分子量小于250的compounds. 
#disease：用图谱里面基因序列和COVID-19相似的节点来代替COVID-19。

#Output：药物推荐清单。MMR9：23.6%  triplets排序见candidrugid.csv
