
import sys
import time
import numpy
import torch
import torchvision
from torch import nn
import torch_geometric
from torch_geometric import nn as gnn
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
        nn.init.xavier_uniform_(self.weight) #Xavier方法初始化，保持输入和输出的方差一致

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
def pred(model,criterion,data,mask):
    loss_,acc_=[],[]
    y_pred=model(data.x,data.edge_index)
    # for mask in mask_:
    loss=criterion(y_pred[mask],data.y[mask])
    loss_.append(loss.item())
    label_pred=y_pred[mask].argmax(dim=1) #预测类别
    acc=label_pred.eq(data.y[mask]).sum().item()/mask.sum().item()
    acc_.append(acc)

    return loss_,acc_

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
    _act=nn.Tanh() #激活函数 ReLU、Tanh、Sigmoid

    print("[节点分类]\n[数据集 dataset={}]\n[卷积块个数 blocks={}]\n[DropEdge概率 dropedge={}]\n[是否使用PairNorm norm={}]\n[是否添加自环 selfloop={}]\n[激活函数 act={}]\n".format(_dataset,_blocks,_dropedge,_norm,_selfloop,_act))

    #————[生成数据集]————#
    dataset=Planetoid(root='dataset/'+_dataset,name=_dataset,transform=torch_geometric.transforms.NormalizeFeatures()) #NormalizeFeatures：标准化特征
    # len(dataset) #只有一张图
    in_=dataset.num_features #每个节点有1433个特征
    out_=dataset.num_classes #节点一共7类
    #  【Cora】  dataset[0]: Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
    #【Citeseer】dataset[0]: Data(x=[3327, 3703], edge_index=[2, 9104 ], y=[3327], train_mask=[3327], val_mask=[3327], test_mask=[3327])
    #【Cora】x: 2708个节点，每个节点1433个特征
    #【Cora】edge_index: 10556条有向边
    #【Cora】y: 每个节点的标签（共7类）
    data=dataset[0].to('cuda')
    print(data)

    #————[模型搭建]————#
    #输入层维度inputs：节点特征数量
    #输出层维度outputs：节点标签种类
    model=GCN(inputs=in_,outputs=out_,wide=_wide,blocks=_blocks,drop=_drop,dropedge=_dropedge,norm=_norm,act=_act,selfloop=_selfloop) #使用GCN模型
    model=model.to('cuda')
    criterion=nn.CrossEntropyLoss().to('cuda') #交叉熵损失函数
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4) #Adam优化器

    #————[模型训练]————#
    _time_ST=time.time() #记录开始时间
    _time_OLD=_time_ST
    _acc_best=[0,0]
    _epochs=300 #总训练次数上限
    _pat=0
    print("【开始训练】")
    for i in range(_epochs):
        model.train() #开启训练模式（启用Dropout层）
        y_pred=model(data.x,data.edge_index) #使用当前模型，预测训练数据x
        loss=criterion(y_pred[data.train_mask],data.y[data.train_mask])  #用过损失函数计算预测值与真实值之间的误差
        optimizer.zero_grad() #将梯度初始化清空
        train_loss=loss.item() #记录训练loss
        tmp_y_pred=y_pred[data.train_mask]
        label_pred=tmp_y_pred.argmax(dim=1) #预测类别
        train_acc=label_pred.eq(data.y[data.train_mask]).sum().item()/data.train_mask.sum().item() #记录训练acc
        loss.backward() #通过自动微分计算损失函数关于模型参数的梯度
        optimizer.step() #优化模型参数，减小损失值
        
        #————[调参分析]————#
        model.eval() #开启评估模式（不启用Dropout层）
        with torch.no_grad(): #禁用梯度计算
            loss_,acc_=pred(model,criterion,data,data.val_mask) #使用训练集和验证集
            print("[epoch={}/{}, train_loss={:.6f},train_acc={:.6f}, verify_loss={:.6f},verify_acc={:.6f}, time={:.6f} 秒]".format(i+1,_epochs,train_loss,train_acc,loss_[0],acc_[0],time.time()-_time_OLD))
            _time_OLD=time.time()
            _pat+=1
            if acc_[0]>_acc_best[1]: #记录最佳acc
                _acc_best=[i+1,acc_[0]]
                _pat=0
                print("  [最佳准确率更新！]")
            elif (_acc_best[1]>0.80 or (_acc_best[1]>0.71 and _dataset=="Citeseer")) and _pat>15: #超过若干无变化则早停
            # elif i>=90:#_pat>30: #超过若干无变化则早停
                break
    #————[数据测试]————#
    model.eval() #开启评估模式（不启用Dropout层）
    loss_,acc_=pred(model,criterion,data,data.test_mask) #使用测试集
    print("[测试集：test_loss={:.6f}, test_acc={:.6f}, time={:.6f} 秒]".format(loss_[0],acc_[0],time.time()-_time_OLD))
    
    print("运行总时间：{:.6f} 秒".format((time.time()-_time_ST)/1.0))