
import sys
import time
import numpy
import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plot

#————[模型搭建]————#

# 卷积块
class CNN_block(nn.Module):
    def __init__(self,inputs,outputs,K,S,P,norm=True,drop=0.2,pool=True):
        # kerner_size 卷积核的尺寸 F
        # stride 卷积步长 S
        # padding 增加0边的层数 P
        super(CNN_block,self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(inputs,outputs,K,S,P), #通道数改变
            nn.BatchNorm2d(outputs) if norm else nn.Identity(), #归一化
            nn.Dropout(p=drop),
            nn.ReLU(),
            nn.Conv2d(outputs,outputs,K,S,P), #通道数不变
            nn.BatchNorm2d(outputs) if norm else nn.Identity(),
            nn.Dropout(p=drop),
            nn.ReLU(),
            nn.MaxPool2d(2) if pool else nn.Identity(),
        )

    def forward(self,x):
        x=self.block(x)
        return x

class CNN(nn.Module): #VGG结构
    def __init__(self,output,K=3,S=1,P=1,norm=False,drop=0.2,blocks=3): #block储存每个卷积块大小
        super(CNN,self).__init__()
        # 输入块（1个参数层）
        self.cn_st=64
        #16~512
        #32~1024
        #64~2048

        self.conv1=nn.Sequential(
            nn.Conv2d(3,self.cn_st,5,1,2), # (3*[32*32]) -> (16*[32*32])
            nn.BatchNorm2d(self.cn_st) if norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2) #(16*[32*32]) -> (16*[16*16])
        )

        # 卷积块 3+ 个（blocks*2个参数层）
        # 每块 2 个卷积层、0/1个汇聚层
        self.block1=CNN_block(self.cn_st,self.cn_st*2,K,S,P,norm,drop,True) #(16*[16*16]) -> (32*[8*8])
        self.block2=CNN_block(self.cn_st*2,self.cn_st*4,K,S,P,norm,drop,True) #(32*[8*8]) -> (64*[4*4])
        self.block3=CNN_block(self.cn_st*4,self.cn_st*8,K,S,P,norm,drop,True) #(64*[4*4]) -> (128*[2*2])
        
        self.add_blocks=blocks-3
        self.block_=CNN_block(self.cn_st*8,self.cn_st*8,K,S,P,norm,drop,False) #(128*[2*2]) -> (128*[2*2])
        
        # self.add_blocks=blocks-4
        # self.block4=CNN_block(128,256,3,1,1,norm,drop,True)  #(128*[2*2]) -> (256*[1*1])
        # self.block_=CNN_block(256,256,3,1,1,norm,drop,False)  #(256*[4*4]) -> (256*[1*1])

        # 全连接块 1 个（1个参数层）
        self.fc=nn.Sequential(
            nn.Linear(self.cn_st*8*4,self.cn_st*8*4), #(128*[2*2)] <=> (512*1)
            nn.BatchNorm1d(self.cn_st*8*4) if norm else nn.Identity(),
            nn.Dropout(p=drop),
            nn.ReLU(),
        )

        # 输出块（1个参数层）
        self.fc_out=nn.Linear(self.cn_st*8*4,output)

    def forward(self,x):
        x=self.conv1(x)
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        for i in range(self.add_blocks):
            x=self.block_(x)

        x=x.view(x.shape[0],-1) #二维像素铺平
        x=self.fc(x)
        
        x=self.fc_out(x)
        return x

# 对指定数据调用模型进行预测
def pred(_cuda,model,criterion,data):
    loss_=[]
    cor,tot=0,0
    for j, (x_, y_) in enumerate(data):
        if _cuda==True:  x_,y_=x_.to('cuda'),y_.to('cuda')
        y_pred=model(x_)
        loss=criterion(y_pred,y_)
        loss_.append(loss.item())
        label_pred=y_pred.argmax(dim=1)
        cor+=label_pred.eq(y_).sum().item()
        tot+=len(label_pred)
    # print("cor={}, tot={}".format(cor,tot))
    return numpy.mean(loss_),cor/tot

if __name__=='__main__':

    #————[参数]————#
    _cuda=True
    # _cuda=False

    _N=50000 #训练集总量
    _N_verify=5000 #验证集总量
    _N_test=10000 #测试集总量
    _picsize=(32,32) #输入图像大小
    _label_type=10 #输出标签种类

    _lr_st=0.01 #初始学习率
    _lr_ed=0.00001 #学习率下限
    _lr_decay=0.01 #学习率递减速度 0.01 0.1 0.9

    _blocks=3+1 #卷积块个数（4个:9层、5个:11层、6个:13层、7个:15层...）
    _ksp=[3,1,1] #[k,s,p]=[3,1,1] [5,1,2] [7,1,3]
    
    _drop=0.2 #dropout概率
    _norm=True #是否实用标准化层

    # _draw=True #是否绘图保存
    # _draw_show=True #是否展示
    # _picname="(lr_decay)={})".format(_lr_decay) #存图名称格式
    # _picname="Test" #存图名称格式

    print("[卷积块个数 blocks={}]\n[卷积核大小 kerner_size={}]\n[学习率递减速度 learning rate decay={}]\n[是否使用 normalization={}]\n[dropout概率 drop={}]\n".format(_blocks,_ksp[0],_lr_decay,_norm,_drop))

    #————[生成数据集]————#
    data_=torchvision.datasets.CIFAR10(root=".\dataset",train=True,transform=torchvision.transforms.ToTensor(),download=_cuda)
    data_train,data_verify=torch.utils.data.random_split(data_,[_N-_N_verify,_N_verify],generator=torch.Generator().manual_seed(233))
    data_test=torchvision.datasets.CIFAR10(root=".\dataset",train=False,transform=torchvision.transforms.ToTensor())

    #————[模型搭建]————#
    model=CNN(output=_label_type,K=_ksp[0],S=_ksp[1],P=_ksp[2],norm=_norm,drop=_drop,blocks=_blocks) #使用CNN模型
    if _cuda==True: model=model.to('cuda')
    criterion=nn.CrossEntropyLoss() #交叉熵损失函数
    optimizer=torch.optim.Adam(model.parameters(),lr=_lr_st) #Adam优化器
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        mode='max', #监测accurate指标
        factor=_lr_decay, #学习率下降速度
        patience=5, #能忍受多少个epoch指标不变
        threshold=0.0001, #判断指标变化的依据
        min_lr=_lr_ed) #学习率下限

    #————[模型训练]————#
    _time_ST=time.time() #记录开始时间
    _time_OLD=_time_ST
    _acc_best=[0,0]
    _epochs=75 #总训练次数
    _batch_size=256 #训练集分批大小
    dataloader=torch.utils.data.DataLoader(data_train,_batch_size,shuffle=True,generator=torch.Generator().manual_seed(233))
    dataloader_verify=torch.utils.data.DataLoader(data_verify,_batch_size,shuffle=True,generator=torch.Generator().manual_seed(233))
    dataloader_test=torch.utils.data.DataLoader(data_test,_batch_size,shuffle=True,generator=torch.Generator().manual_seed(233))
    print("【开始训练】")
    _pat=0
    for i in range(_epochs):
        for j, (x_, y_) in enumerate(dataloader):
            if _cuda==True:  x_,y_=x_.to('cuda'),y_.to('cuda')
            y_pred=model(x_) #使用当前模型，预测训练数据x
            loss=criterion(y_pred,y_)  #用过损失函数计算预测值与真实值之间的误差
            optimizer.zero_grad() #将梯度初始化清空
            loss.backward() #通过自动微分计算损失函数关于模型参数的梯度
            optimizer.step() #优化模型参数，减小损失值
        #————[调参分析]————#
        with torch.no_grad(): #禁用梯度计算
            verify_loss,verify_acc=pred(_cuda,model,criterion,dataloader_verify) #使用验证集
            scheduler.step(verify_acc) #监测acc指标调整学习率
            print("[epoch={}/{}, loss={:.6f}, acc={:.6f}, time={:.6f} 秒]".format(i+1,_epochs,verify_loss,verify_acc,time.time()-_time_OLD))
            _time_OLD=time.time()
            _pat+=1
            if verify_acc>_acc_best[1]: #记录最佳acc
                _acc_best=[i+1,verify_acc]
                _pat=0
                print("  [最佳准确率更新！]")
                # print("[最佳准确率更新为: epoch={}, acc={:.6f}]".format(_acc_best[0],_acc_best[1]))
            elif _pat>10: #超过十轮无变化则早停
                break
    #————[数据测试]————#
    test_loss,test_acc=pred(_cuda,model,criterion,dataloader_test) #使用测试集
    print("[测试集：loss={:.6f}, acc={:.6f}, time={:.6f} 秒]".format(test_loss,test_acc,time.time()-_time_OLD))
    
    print("运行总时间：{:.6f} 分".format((time.time()-_time_ST)/60.0))