import numpy
import torch
from torch import nn
from matplotlib import pyplot as plot

#————[模型搭建]————#
class FNN(nn.Module):
    def __init__(self,wide:list,act): #wide储存每层节点数
        super(FNN,self).__init__()
        self.act=act
        self.line_n=len(wide)-1
        self.line=nn.ModuleList([nn.Linear(wide[i],wide[i+1]) for i in range(self.line_n)])
            
    def forward(self,x):
        for i,line_f in enumerate(self.line):
            x=line_f(x) #线性函数
            if i<self.line_n-1:
                x=self.act(x) #激活函数
        return x

#————[自定义目标函数]————#
def _func(x):
    return numpy.log2(x)+numpy.cos(numpy.pi*x/2)

# 对指定数据调用模型进行预测
def pred(model,criterion,x_,y_,name="verify",draw=True,draw_show=True,picname="plot",picnum=0):
    y_pred=model(x_)
    loss=criterion(y_pred,y_)
    print("[{}_loss={:.6f}]\n".format(name,loss))
    
    if draw==True:
        # 绘图比较
        fig,aex=plot.subplots()
        aex.set_title("[N={} lr={} wide={} depth={} act={}\n{}_loss={:.6f}]".format(_N,_lr,_wide,_depth,_act,name,loss))
        aex.set_xlabel('x')
        aex.set_ylabel('y')
        # 绘散点 ( 'detach().numpy()': 张量转化为普通数据)
        aex.scatter(x_.detach().numpy(),y_.detach().numpy(),label="True")
        aex.scatter(x_.detach().numpy(),y_pred.detach().numpy(),label="Pred")
        aex.legend() #添加图例
        plot.savefig('{}_{}.png'.format(picname,picnum))
        #if draw_show==True:
            #plot.show()

    return loss

if __name__=='__main__':

    #————[参数]————#
    _N=2000 #数据集总量
    _lr=0.0025 #学习率
    _wide=128 #网络宽度(隐藏层每层节点数量)
    _depth=2 #网络深度(隐藏层数量)
    _act=nn.Sigmoid() #激活函数类型(Sigmoid,Tanh,ReLU,ELU,Softplus)

    _draw=True #是否绘图保存
    _draw_show=True #是否展示
    _picname="(lr)={})".format(_lr) #存图名称格式
    # _picname="Test" #存图名称格式

    print("[数据集总量 N={}]\n[网络宽度 wide={}]\n[网络深度 depth={}]\n[学习率 learning rate={}]\n[激活函数类型 activation={}]\n".format(_N,_wide,_depth,_lr,_act))

    #————[生成数据集]————#
    numpy.random.seed(233) #自定义随机数以固定数据集
    x=numpy.linspace(1,16,_N)
    # 按照 训练集：验证集：测试集=8：1：1 对数据集进行随机划分
    id=numpy.random.permutation(_N) #随机一个排列
    N1,N2=round(_N*0.8),round(_N*0.1)
    id1,id2,id3=id[0:N1],id[N1:N1+N2],id[N1+N2:_N]
    id2,id3=numpy.sort(id2),numpy.sort(id3)
    x1,x2,x3=x[id1].reshape(N1,1),x[id2].reshape(N2,1),x[id3].reshape(N2,1)
    y1,y2,y3=_func(x1),_func(x2),_func(x3)
    x_train,x_verify,x_test=torch.Tensor(x1),torch.Tensor(x2),torch.Tensor(x3) #数据转化成张量
    y_train,y_verify,y_test=torch.Tensor(y1),torch.Tensor(y2),torch.Tensor(y3) #数据转化成张量


    _times=3 #重复实验次数
    loss_=[] #网络性能
    for T in range(_times):
        print("第{}/{}次重复实验：".format(T+1,_times))

        #————[模型搭建]————#
        model=FNN(wide=[1,*[_wide]*_depth,1],act=_act) #使用FNN模型
        criterion=nn.MSELoss() #均方误差损失函数
        optimizer=torch.optim.Adam(model.parameters(),lr=_lr) #Adam优化器

        #————[模型训练]————#
        _epochs=round(75*10000/_N) #总训练次数（数据集较小时增加次数）
        _batch_size=64 #训练集分批大小
        dataset=torch.utils.data.TensorDataset(x_train,y_train) #使用训练集
        dataloader=torch.utils.data.DataLoader(dataset,_batch_size,shuffle=True) #将训练集分批次 (shuffle: 随机打乱数据)
        for i in range(_epochs):
            for j, (x_, y_) in enumerate(dataloader):
                y_pred=model(x_) #使用当前模型，预测训练数据x
                loss=criterion(y_pred,y_)  #用过损失函数计算预测值与真实值之间的误差
                optimizer.zero_grad() #将梯度初始化清空
                loss.backward() #通过自动微分计算损失函数关于模型参数的梯度
                optimizer.step() #优化模型参数，减小损失值
                #print("batch={}, loss={:.6f}".format(j+1,loss))
            if i%round(_epochs/20)==0:
                print("[batch_size={} ,epoch={}/{}, train_loss={:.6f}]".format(_batch_size,i+1,_epochs,loss))
        print("")

        #————[调参分析/数据测试]————#
        loss=0
        if _picname=="Test":
            loss=pred(model,criterion,x_test,y_test,"test",_draw,_draw_show,'{}'.format(_N)+_picname,picnum=T) #使用测试集
        else:
            loss=pred(model,criterion,x_verify,y_verify,"verify",_draw,_draw_show,_picname,picnum=T) #使用验证集
        loss_.append(loss.detach().numpy())
    
    #计算网络性能平均值
    print("loss:",['{:.6f}'.format(lo) for lo in loss_],"loss_average={:.6f}".format(numpy.mean(loss_)))

    if _draw_show==True:
        plot.show()