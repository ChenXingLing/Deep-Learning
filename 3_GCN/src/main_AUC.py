
import sys
import time
import numpy
import torch
import sklearn
import torchvision
from torch import nn
import torch_geometric
from torch_geometric import nn as gnn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dropout_edge,add_self_loops
from matplotlib import pyplot as plot

#————[模型搭建]————#

# 图卷积块
class GraphConv(nn.Module):
    def __init__(self,inputs,outputs,selfloop=True):
        super(GraphConv,self).__init__()
        self.inputs,self.outputs,self.selfloop=inputs,outputs,selfloop
        self.weight=nn.Parameter(torch.randn(inputs,outputs)) #input*ouput大小的权重矩阵
        self.bias=nn.Parameter(torch.zeros(outputs)) #output大小的偏置向量
        # nn.init.xavier_uniform_(self.weight) #Xavier方法初始化，保持输入和输出的方差一致

    def forward(self,x,edge_index):
        # x [节点个数*每个节点特征个数]
        # edge_index [2*边个数]
        n=x.size(0) #节点个数
        adj=torch.zeros(n,n).to('cuda') #邻接矩阵
        adj[edge_index[0],edge_index[1]]=1 #无向图
        adj[edge_index[1],edge_index[0]]=1 #无向图

        if self.selfloop==False:
            deg=torch.sum(adj,dim=1) #节点度数
            deg=1.0/(torch.sqrt(deg)+1e-8) #加上一个小常数避免除零nan
            D=torch.diag(deg).to('cuda') # D^{-1/2}
            adj=torch.eye(n).to('cuda')+torch.mm(torch.mm(D,adj),D) # I + D^{-1/2} * Adj * D^{-1/2}
        else:
            adj=adj+torch.eye(n).to('cuda') #添加自环 A=A+I
            deg=torch.sum(adj,dim=1) #节点度数
            deg=1.0/torch.sqrt(deg)
            D=torch.diag(deg).to('cuda') # D^{-1/2}
            adj=torch.mm(torch.mm(D,adj),D) #D^{-1/2} * Adj * D^{-1/2}

        return torch.mm(adj,torch.mm(x,self.weight))+self.bias #D^{-1/2} * Adj * D^{-1/2} * (x*W) + bias

#图卷积神经网络
class GCN(nn.Module):
    def __init__(self,inputs,outputs,wide,blocks=2,drop=0.2,dropedge=0.2,norm=False,act=nn.ReLU(),selfloop=True):
        super(GCN,self).__init__()
        self.dropedge,self.drop=dropedge,drop
        self.norm,self.act=norm,act
        self.blocks,self.inputs,self.wide,self.outputs=blocks-2,inputs,wide,outputs
        self.conv1=GraphConv(inputs,wide,selfloop)
        self.conv2=GraphConv(wide,wide,selfloop)
        self.conv3=GraphConv(wide,outputs,selfloop)

    def forward(self,x,edge_index):
        # edge_index传入边信息
        edge_index,_=dropout_edge(edge_index,p=self.dropedge) #随机丢弃一些边
        x=self.conv1(x,edge_index)
        if self.norm:
            x=gnn.PairNorm()(x)
        x=self.act(x)
        x=nn.functional.dropout(x,p=self.drop,training=self.training)

        for i in range(self.blocks):
            x=self.conv2(x,edge_index)
            if self.norm:
                x=gnn.PairNorm()(x)
            x=self.act(x)
            x=nn.functional.dropout(x,p=self.drop,training=self.training)

        x=self.conv3(x,edge_index)
        if self.norm:
            x=gnn.PairNorm()(x)
        return x

# 对指定数据调用模型进行预测
def pred(model,criterion,data,edge_label):
    y_pred=model(data.x,data.edge_index)
    y_score=y_pred[data.edge_label_index[0],:]*y_pred[data.edge_label_index[1],:] #每条边每种特征将两端点分数相乘
    y_score=torch.sigmoid(torch.sum(y_score,dim=1))  #outputs个特征数值求和作为最终评估分数
    loss=criterion(y_score,edge_label)
    auc=roc_auc_score(edge_label.to('cpu'),y_score.to('cpu').detach().numpy())
    # tmp=y_score>0.5
    # acc=accuracy_score(edge_label.to('cpu'),tmp.to('cpu'))
    return loss,auc

if __name__=='__main__':

    #————[参数]————#

    _dataset="Cora"
    # _dataset="Citeseer"

    _wide=32 #隐藏层维度
    _drop=0.2 #Dropout概率

    _blocks=2 #层数
    _dropedge=0.1 #DropEdge概率
    _norm=False #是否使用PairNorm
    _selfloop=True #是否添加自环
    _act=nn.ReLU() #激活函数 ReLU、Tanh、Sigmoid

    print("[链路预测]\n[数据集 dataset={}]\n[卷积块个数 blocks={}]\n[DropEdge概率 dropedge={}]\n[是否使用PairNorm norm={}]\n[是否添加自环 selfloop={}]\n[激活函数 act={}]\n".format(_dataset,_blocks,_dropedge,_norm,_selfloop,_act))

    #————[生成数据集]————#
    dataset=Planetoid(root='dataset/'+_dataset,name=_dataset,transform=torch_geometric.transforms.NormalizeFeatures()) #NormalizeFeatures：标准化特征
    in_=dataset.num_features
    data=dataset[0].to('cuda')
    print(data)
    transform=torch_geometric.transforms.RandomLinkSplit(num_val=0.1,num_test=0.1,is_undirected=True) #边集划分
    train_data,val_data,test_data=transform(data) #数据集划分
    train_edge_label=train_data.edge_label#.to(torch.float)
    val_edge_label=val_data.edge_label#.to(torch.float)
    test_edge_label=test_data.edge_label#.to(torch.float)
    # train_data: DATA(edge_label=0/1:是否存在该边 edge_label_index:该边端点)

    #————[模型搭建]————#
    #输入层维度inputs：节点特征数量
    #输出层维度outputs：与隐藏层相同
    model=GCN(inputs=in_,outputs=_wide,wide=_wide,blocks=_blocks,drop=_drop,dropedge=_dropedge,norm=_norm,act=_act,selfloop=_selfloop) #使用GCN模型
    model=model.to('cuda')
    criterion=nn.BCELoss() #二元交叉熵损失函数（需配合sigmoid使用）
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4) #Adam优化器

    #————[模型训练]————#
    _time_ST=time.time() #记录开始时间
    _time_OLD=_time_ST
    _auc_best=[0,0]
    _epochs=500 #总训练次数上限
    _pat=0
    print("【开始训练】")
    for i in range(_epochs):
        model.train() #开启训练模式（启用Dropout层）
        y_pred=model(train_data.x,train_data.edge_index) #使用当前模型，预测训练数据x
        y_score=y_pred[train_data.edge_label_index[0],:]*y_pred[train_data.edge_label_index[1],:] #每条边每种特征将两端点分数相乘
        y_score=torch.sigmoid(torch.sum(y_score,dim=1))  #outputs个特征数值求和作为最终评估分数
        loss=criterion(y_score,train_edge_label)  #用损失函数计算预测值与真实值之间的误差
        train_loss=loss.item() #记录训练loss
        train_auc=roc_auc_score(train_edge_label.to('cpu'),y_score.to('cpu').detach().numpy()) #记录训练auc
        optimizer.zero_grad() #将梯度初始化清空
        loss.backward() #通过自动微分计算损失函数关于模型参数的梯度
        optimizer.step() #优化模型参数，减小损失值
        
        #————[调参分析]————#
        model.eval() #开启评估模式（不启用Dropout层）
        with torch.no_grad(): #禁用梯度计算
            loss_,auc_=pred(model,criterion,val_data,val_edge_label) #使用验证集
            print("[epoch={}/{}, train_loss={:.6f},train_auc={:.6f}, verify_loss={:.6f},verify_auc={:.6f}, time={:.6f} 秒]".format(i+1,_epochs,train_loss,train_auc,loss_,auc_,time.time()-_time_OLD))
            _time_OLD=time.time()
            _pat+=1
            if auc_>_auc_best[1]: #记录最佳auc
                _auc_best=[i+1,auc_]
                _pat=0
                print("  [最佳准确率更新！]")
            elif _auc_best[1]>0.87 and _pat>15: #超过若干无变化则早停
            # elif i>210: #超过若干无变化则早停
                break
    #————[数据测试]————#
    model.eval() #开启评估模式（不启用Dropout层）
    loss_,auc_=pred(model,criterion,test_data,test_edge_label) #使用测试集
    print("[测试集：test_loss={:.6f}, test_auc={:.6f}, time={:.6f} 秒]".format(loss_,auc_,time.time()-_time_OLD))
    
    print("运行总时间：{:.6f} 秒".format((time.time()-_time_ST)/1.0))