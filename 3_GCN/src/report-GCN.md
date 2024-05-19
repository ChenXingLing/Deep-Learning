## **【Report】图卷积神经网络GCN**

==肖羿 PB21010452==

### **【概述】**

Python 基于 Pytorch 库实现图卷积神经网络，在 *Cora 和 Citeseer 数据集* 上进行节点分类和链路预测。

### **【代码结构】**

#### **1.【main.py】**

```python
#————[模型搭建]————#

# 图卷积块
class GraphConv(nn.Module):
    def __init__(self,inputs,outputs,selfloops):
        ...

    def forward(self,x,edge_index):
		...
		
#图卷积神经网络
class GCN(nn.Module):
    def __init__(self,inputs,outputs,wide,blocks=2,drop=0.2,dropedge=0.2,norm=False,act=nn.ReLU(),selfloop=True):
        ...
        self.conv1=GraphConv(inputs,wide,selfloop)
        self.conv2=GraphConv(wide,wide,selfloop)
        self.conv3=GraphConv(wide,outputs,selfloop)

    def forward(self,x,edge_index):
        # edge_index传入边信息
        dropout_edge(edge_index) #随机丢弃一些边

        conv1 #图卷积块
        PairNorm #归一化
        act #激活函数
        dropout

        #若干组：
        conv2 #图卷积块
        PairNorm #归一化
        act #激活函数
        dropout
        ...

        conv3 #图卷积块
        PairNorm #归一化
```

使用 torch_geometric 库的 Planetoid 函数下载数据集。

进行节点分类任务时，直接使用划分好的数据集 `Data.train_mask` 。

进行链路预测任务时，使用 RandomLinkSplit 函数对边集进行随机划分。

```python
#————[生成数据集]————#
dataset=Planetoid(root='dataset/'+_dataset,name=_dataset,transform=NormalizeFeatures()) #NormalizeFeatures：标准化特征
data=dataset[0].to('cuda')
#【Cora】dataset[0]: Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
#【Cora】x: 2708个节点，每个节点1433个特征
#【Cora】edge_index: 10556条有向边
#【Cora】y: 每个节点的标签（共7类）

transform=RandomLinkSplit(num_val=0.1,num_test=0.1,is_undirected=True) #边集划分
train_data,val_data,test_data=transform(data) #数据集划分
#Data(edge_label=0/1:是否存在该边 edge_label_index:该边端点)
```

对比实验默认参数如下：

```python
#————[参数]————#
_wide=32 #隐藏层维度
_drop=0.2 #Dropout概率

_blocks=2 #层数
_dropedge=0.1 #DropEdge概率
_norm=False #是否使用PairNorm
_selfloop=True #是否添加自环
_act=nn.ReLU() #激活函数 ReLU、Tanh、Sigmoid
```

```python
#————[模型搭建]————#
#【节点分类】
#输入层维度inputs：节点特征数量
#输出层维度outputs：节点标签种类
model=GCN(inputs=in_,outputs=out_,wide=_wide,...) #使用GCN模型
model=model.to('cuda')
criterion=nn.CrossEntropyLoss().to('cuda') #交叉熵损失函数
optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4) #Adam优化器

#【链路预测】
#输入层维度inputs：节点特征数量
#输出层维度outputs：与隐藏层相同
model=GCN(inputs=in_,outputs=_wide,wide=_wide,...) #使用GCN模型
model=model.to('cuda')
criterion=nn.BCELoss() #二元交叉熵损失函数（需配合sigmoid使用）
optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4) #Adam优化器


#————[模型训练]————#
_epochs=500 #总训练次数上限
...

#————[调参分析]————#
...

#————[调参分析]————#
...


```

#### **2.【data.py】**

将验证集上调参对比所得数据绘制图表。纵坐标分别使用$\text{acc/auc}$ 和损失值 $\text{loss}$ 。

<div STYLE="page-break-after: always;"></div>
### **【参数对比 task=节点分类】**

#### **1.【网络深度 blocks】**

固定其他参数，分别取图卷积块个数 $\text{blocks}=2,3,4,5$。

对于两种数据集都为浅层网络最优，层数增加难以优化。

#####  **【Cora】**

<center class="half">
    <img src="./output/[ACC Cora block]_acc.png" width="280"/>
    <img src="./output/[ACC Cora block]_loss.png" width="280"/>
</center>

#####  **【Citeseer】**

<center class="half">
    <img src="./output/[ACC Citeseer block]_acc.png" width="280"/>
    <img src="./output/[ACC Citeseer block]_loss.png" width="280"/>
</center>

<div STYLE="page-break-after: always;"></div>
#### **2.【dropedge概率】**

固定其他参数，分别尝试 $0.0,0.1,0.2,0.5$ 四种 $\text{dropedge}$ 概率进行测试。

在浅层次网络中，对于两种数据集，作用效果都不明显，选取较低 $drop$ 概率或不使用时相对略好。

在深层次网络中，对于两种数据集都有提升，但 $drop$ 概率过高会降低性能。

#####  **【Cora】**

<center class="half">
    <img src="./output/[ACC Cora dropedge]_acc.png" width="280"/>
    <img src="./output/[ACC Cora dropedge]_loss.png" width="280"/>
</center>

<center class="half">
    <img src="./output/[ACC Cora dropedge4]_acc.png" width="280"/>
    <img src="./output/[ACC Cora dropedge4]_loss.png" width="280"/>
</center>

#####  **【Citeseer】**

<center class="half">
    <img src="./output/[ACC Citeseer dropedge]_acc.png" width="280"/>
    <img src="./output/[ACC Citeseer dropedge]_loss.png" width="280"/>
</center>

<center class="half">
    <img src="./output/[ACC Citeseer dropedge4]_acc.png" width="280"/>
    <img src="./output/[ACC Citeseer dropedge4]_loss.png" width="280"/>
</center>

<div STYLE="page-break-after: always;"></div>
#### **3.【归一化 pairnorm】**

固定其他参数，分别对使用/不使用 $\text{normalization}$ 进行测试。

使用 pairnorm 会显著降低网络性能。

#####  **【Cora】**

<center class="half">
    <img src="./output/[ACC Cora norm]_acc.png" width="280"/>
    <img src="./output/[ACC Cora norm]_loss.png" width="280"/>
</center>

#####  **【Citeseer】**

<center class="half">
    <img src="./output/[ACC Citeseer norm]_acc.png" width="280"/>
    <img src="./output/[ACC Citeseer norm]_loss.png" width="280"/>
</center>

<div STYLE="page-break-after: always;"></div>
#### **4.【添加自环 selfloop】**

固定其他参数，分别对不添加/添加自环（是否使用 renormalization trick）两种方法进行测试。

使用添加自环方法性能略有提升。

#####  **【Cora】**

<center class="half">
    <img src="./output/[ACC Cora selfloop]_acc.png" width="280"/>
    <img src="./output/[ACC Cora selfloop]_loss.png" width="280"/>
</center>

#####  **【Citeseer】**

<center class="half">
    <img src="./output/[ACC Citeseer selfloop]_acc.png" width="280"/>
    <img src="./output/[ACC Citeseer selfloop]_loss.png" width="280"/>
</center>

<div STYLE="page-break-after: always;"></div>
#### **5.【激活函数 activation】**

固定其他参数，分别取激活函数 $\text{act}=\text{ReLU},\text{Tanh},\text{Sigmoid}$ 进行测试。

对于两种数据集均为 $\text{Tanh}$ 略优于 $\text{ReLU}$。

$\text{Sigmoid}$ 在该模型中不适用，无法训练。

#####  **【Cora】**

<center class="half">
    <img src="./output/[ACC Cora act]_acc.png" width="280"/>
    <img src="./output/[ACC Cora act]_loss.png" width="280"/>
</center>

#####  **【Citeseer】**

<center class="half">
    <img src="./output/[ACC Citeseer act]_acc.png" width="280"/>
    <img src="./output/[ACC Citeseer act]_loss.png" width="280"/>
</center>

<div STYLE="page-break-after: always;"></div>
### **【参数对比 task=链路预测】**

#### **1.【网络深度 blocks】**

固定其他参数，分别取图卷积块个数 $\text{blocks}=2,3,4,5$。

对于两种数据集都为浅层网络最优，层数增加难以优化。

#####  **【Cora】**

<center class="half">
    <img src="./output/[AUC Cora block]_auc.png" width="280"/>
    <img src="./output/[AUC Cora block]_loss.png" width="280"/>
</center>

#####  **【Citeseer】**

<center class="half">
    <img src="./output/[AUC Citeseer block]_auc.png" width="280"/>
    <img src="./output/[AUC Citeseer block]_loss.png" width="280"/>
</center>

<div STYLE="page-break-after: always;"></div>
#### **2.【dropedge概率】**

固定其他参数，分别尝试 $0.0,0.1,0.2,0.5$ 四种 $\text{dropedge}$ 概率进行测试。

在浅层次网络中，对于两种数据集都有性能提升，$\text{Cora}$ 数据集选取 $\text{dropedge}=0.1$ 最优，$\text{Citeseer}$ 数据集选取 $\text{dropedge}=0.2$ 最优。

在深层次网络中，对于两种数据集都降低了性能。

#####  **【Cora】**

<center class="half">
    <img src="./output/[AUC Cora dropedge]_auc.png" width="280"/>
    <img src="./output/[AUC Cora dropedge]_loss.png" width="280"/>
</center>

<center class="half">
    <img src="./output/[AUC Cora dropedge4]_auc.png" width="280"/>
    <img src="./output/[AUC Cora dropedge4]_loss.png" width="280"/>
</center>

#####  **【Citeseer】**

<center class="half">
    <img src="./output/[AUC Citeseer dropedge]_auc.png" width="280"/>
    <img src="./output/[AUC Citeseer dropedge]_loss.png" width="280"/>
</center>

<center class="half">
    <img src="./output/[AUC Citeseer dropedge4]_auc.png" width="280"/>
    <img src="./output/[AUC Citeseer dropedge4]_loss.png" width="280"/>
</center>

<div STYLE="page-break-after: always;"></div>
#### **3.【归一化 pairnorm】**

固定其他参数，分别对使用/不使用 $\text{normalization}$ 进行测试。

使用 pairnorm 会降低网络性能。

#####  **【Cora】**

<center class="half">
    <img src="./output/[AUC Cora norm]_auc.png" width="280"/>
    <img src="./output/[AUC Cora norm]_loss.png" width="280"/>
</center>

#####  **【Citeseer】**

<center class="half">
    <img src="./output/[AUC Citeseer norm]_auc.png" width="280"/>
    <img src="./output/[AUC Citeseer norm]_loss.png" width="280"/>
</center>

<div STYLE="page-break-after: always;"></div>
#### **4.【添加自环 selfloop】**

固定其他参数，分别对不添加/添加自环（是否使用 renormalization trick）两种方法进行测试。

使用添加自环方法性能大幅提升。

#####  **【Cora】**

<center class="half">
    <img src="./output/[AUC Cora selfloop]_auc.png" width="280"/>
    <img src="./output/[AUC Cora selfloop]_loss.png" width="280"/>
</center>

#####  **【Citeseer】**

<center class="half">
    <img src="./output/[AUC Citeseer selfloop]_auc.png" width="280"/>
    <img src="./output/[AUC Citeseer selfloop]_loss.png" width="280"/>
</center>

<div STYLE="page-break-after: always;"></div>
#### **5.【激活函数 activation】**

固定其他参数，分别取激活函数 $\text{act}=\text{ReLU},\text{Tanh},\text{Sigmoid}$ 进行测试。

对于 $\text{Cora}$ 数据集，$\text{ReLU}$ 略优于 $\text{Tanh}$。

对于 $\text{Citeseer}$ 数据集，$\text{Tanh}$ 略优于 $\text{ReLU}$。

$\text{Sigmoid}$ 在该模型中不适用，无法训练。

#####  **【Cora】**

<center class="half">
    <img src="./output/[AUC Cora act]_auc.png" width="280"/>
    <img src="./output/[AUC Cora act]_loss.png" width="280"/>
</center>

#####  **【Citeseer】**

<center class="half">
    <img src="./output/[AUC Citeseer act]_auc.png" width="280"/>
    <img src="./output/[AUC Citeseer act]_loss.png" width="280"/>
</center>

<div STYLE="page-break-after: always;"></div>
### **【测试】**

在训练中增加早停判断，若验证集准确率达到要求且超过若干轮无变化则提前停止训练。

#### **1.【task=节点分类】**

对于两种数据集，最终使用参数相同：

```python
_blocks=2 #层数
_dropedge=0.1 #DropEdge概率
_norm=False #是否使用PairNorm
_selfloop=True #是否添加自环
_act=nn.Tanh() #激活函数 ReLU、Tanh、Sigmoid
```

##### **【Cora】**

最终测试结果（第 $54$ 轮停止）：损失值  $\text{loss=0.814000}$，$\text{acc=81.4}\%$ 。

#####  **【Citeseer】**

最终测试结果（第 $60$ 轮停止）：损失值  $\text{loss=1.163353}$，$\text{acc=71.1}\%$ 。

#### **2.【task=链路预测】**

##### **【Cora】**

最终使用参数：

```python
_blocks=2 #层数
_dropedge=0.1 #DropEdge概率
_norm=False #是否使用PairNorm
_selfloop=True #是否添加自环
_act=nn.ReLU() #激活函数 ReLU、Tanh、Sigmoid
```

最终测试结果（第 $200$ 轮停止）：损失值  $\text{loss=0.523941}$，$\text{auc=88.73}\%$ 。

##### **【Citeseer】**

最终使用参数：

```python
_blocks=2 #层数
_dropedge=0.2 #DropEdge概率
_norm=False #是否使用PairNorm
_selfloop=True #是否添加自环
_act=nn.Tanh() #激活函数 ReLU、Tanh、Sigmoid
```

最终测试结果（第 $152$ 轮停止）：损失值  $\text{loss=0.515354}$，$\text{auc=88.67}\%$ 。

### **【问题】**

- 链路预测任务中，由于给定边集大小远小于完全图中补集大小，只选取少量不足以概括整个补集的情况。
  
  > 解决方案：在每一轮训练中都对训练集重新负采样。
