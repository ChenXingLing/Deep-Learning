
import sys
import time
import numpy
import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plot

def get(file):
    file=open(file,'r',encoding='utf-8')
    X,Y1_loss,Y1_acc=[],[],[]
    for line in file.readlines():
        if line[0:7]=='[epoch=':
            # print(line)
            tmp=line.split(',')
            tmp_=[p.split('=') for p in tmp]
            # print(tmp_)
            epoch=int((tmp_[0][1].split('/'))[0])
            loss=float(tmp_[3][1])
            acc=float(tmp_[4][1])*100.0
            # print("epoch={}, loss={:.6f}, acc={:.6f}".format(epoch,loss,acc))
            X.append(epoch)
            Y1_loss.append(loss)
            Y1_acc.append(acc)
            # break
    file.close()
    return X,Y1_acc,Y1_loss,

def plot_(title,task,dataset,label,mp,draw_show=True):
    cnt=len(mp)    
    _xy=[get('.\\{}\\{} {} {}={}.txt'.format(label,task,dataset,label,mp[i])) for i in range(cnt)]

    # print(_xy[0])
    # sys.exit()

    # 绘图比较
    fig,aex=plot.subplots()
    aex.set_title(title)
    aex.set_xlabel('epoch')
    aex.set_ylabel('{} (%)'.format(task))
    [aex.plot(x,y,label="{}={}".format(label,mp[i])) for i,(x,y,z) in enumerate(_xy)]
    aex.legend() #添加图例
    plot.savefig('[{} {} {}]_{}.png'.format(task,dataset,label,task))
    if draw_show==True:
        plot.show()

    # 绘图比较
    fig,aex=plot.subplots()
    aex.set_title(title)
    aex.set_xlabel('epoch')
    aex.set_ylabel('loss')
    if task=="AUC" and label=="act":
        plot.ylim(0,2)
    if task=="AUC" and label=="selfloop":
        plot.ylim(0,2)
    if task=="AUC" and (label=="dropedge" or label=="block") and dataset=="Cora":
        plot.ylim(0.4,2)
    if task=="AUC" and (label=="dropedge" or label=="block") and dataset=="CiteSeer":
        plot.ylim(0.4,1.2)
    [aex.plot(x,z,label="{}={}".format(label,mp[i])) for i,(x,y,z) in enumerate(_xy)]
    aex.legend() #添加图例
    plot.savefig('[{} {} {}]_loss.png'.format(task,dataset,label))
    if draw_show==True:
        plot.show()

def plot_dropedge4(title,task,dataset,label,mp,draw_show=True):
    cnt=len(mp)    
    _xy=[get('.\\dropedge4\\{} {} dropedge={}.txt'.format(task,dataset,mp[i])) for i in range(cnt)]

    # print(_xy[0])
    # sys.exit()

    # 绘图比较
    fig,aex=plot.subplots()
    aex.set_title(title)
    aex.set_xlabel('epoch')
    aex.set_ylabel('{} (%)'.format(task))
    [aex.plot(x,y,label="block=4 {}={}".format(label,mp[i])) for i,(x,y,z) in enumerate(_xy)]
    aex.legend() #添加图例
    plot.savefig('[{} {} {}4]_{}.png'.format(task,dataset,label,task))
    if draw_show==True:
        plot.show()

    # 绘图比较
    fig,aex=plot.subplots()
    aex.set_title(title)
    aex.set_xlabel('epoch')
    aex.set_ylabel('loss')
    if task=="AUC" and dataset=="Cora":
        plot.ylim(0.4,2)
    if task=="AUC" and dataset=="CiteSeer":
        plot.ylim(0.4,1.2)
    [aex.plot(x,z,label="block=4 {}={}".format(label,mp[i])) for i,(x,y,z) in enumerate(_xy)]
    aex.legend() #添加图例
    plot.savefig('[{} {} {}4]_loss.png'.format(task,dataset,label))
    if draw_show==True:
        plot.show()

def sakura(task="ACC",dataset="Cora"):
    name="Node Classification" if task=="ACC" else "Link Prediction"
    # plot_("[task=‘{}’ dataset={}]\n[block=x dropedge=0.1 norm=False selfloop=True act=ReLU]".format(name,dataset),task,dataset,"block",mp=[2,3,4,5],draw_show=True)
    # plot_("[task=‘{}’ dataset={}]\n[block=2 dropedge=x norm=False selfloop=True act=ReLU]".format(name,dataset),task,dataset,"dropedge",mp=['0.0','0.1','0.2','0.5'],draw_show=True)
    # plot_("[task=‘{}’ dataset={}]\n[block=2 dropedge=0.1 norm=x selfloop=True act=ReLU]".format(name,dataset),task,dataset,"norm",mp=['True','False'],draw_show=True)
    # plot_("[task=‘{}’ dataset={}]\n[block=2 dropedge=0.1 norm=False selfloop=x act=ReLU]".format(name,dataset),task,dataset,"selfloop",mp=['True','False'],draw_show=True)
    # plot_("[task=‘{}’ dataset={}]\n[block=2 dropedge=0.1 norm=False selfloop=True act=x]".format(name,dataset),task,dataset,"act",mp=['ReLU','Tanh','Sigmoid'],draw_show=True)

    plot_dropedge4("[task=‘{}’ dataset={}]\n[block=4 dropedge=x norm=False selfloop=True act=ReLU]".format(name,dataset),task,dataset,"dropedge",mp=['0.0','0.1','0.2','0.5'],draw_show=True)

if __name__=='__main__':
    # sakura("ACC","Cora")
    # sakura("ACC","CiteSeer")
    sakura("AUC","Cora")
    sakura("AUC","CiteSeer")
    