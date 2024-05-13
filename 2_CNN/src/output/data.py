
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
            tmp=line.split(', ')
            tmp_=[p.split('=') for p in tmp]
            # print(tmp_)
            epoch=int((tmp_[0][1].split('/'))[0])
            loss=float(tmp_[1][1])
            acc=float(tmp_[2][1])*100.0
            # print("epoch={}, loss={:.6f}, acc={:.6f}".format(epoch,loss,acc))
            X.append(epoch)
            Y1_loss.append(loss)
            Y1_acc.append(acc)
            # break
    file.close()
    return X,Y1_acc,Y1_loss,

def plot_(title,label,mp,draw_show=True):
    cnt=len(mp)
    _xy=[get('.\\{}\\{}={}.txt'.format(label,label,mp[i])) for i in range(cnt)]

    # title="[block=x K=3 lr_decay=0.1 norm=True drop=0.2]"
    # 绘图比较
    fig,aex=plot.subplots()
    aex.set_title(title)
    aex.set_xlabel('epoch')
    aex.set_ylabel('accurate (%)')
    [aex.plot(x,y,label="{}={}".format(label,mp[i])) for i,(x,y,z) in enumerate(_xy)]
    aex.legend() #添加图例
    plot.savefig('[{}]_acc.png'.format(label))
    if draw_show==True:
        plot.show()

    # 绘图比较
    fig,aex=plot.subplots()
    aex.set_title(title)
    aex.set_xlabel('epoch')
    aex.set_ylabel('loss')
    [aex.plot(x,z,label="{}={}".format(label,mp[i])) for i,(x,y,z) in enumerate(_xy)]
    aex.legend() #添加图例
    plot.savefig('[{}]_loss.png'.format(label))
    if draw_show==True:
        plot.show()

def plot_drop(draw_show=True):
    mp=['0.0','0.1','0.2','0.5']
    _xy=[get('.\\drop\\drop={}.txt'.format(mp[i])) for i in range(4)]

    title="[block=4 K=3 lr_decay=0.1 norm=True drop=x]"
    # 绘图比较
    fig,aex=plot.subplots()
    aex.set_title(title)
    aex.set_xlabel('epoch')
    aex.set_ylabel('accurate (%)')
    [aex.plot(x,y,label="drop={}".format(mp[i])) for i,(x,y,z) in enumerate(_xy)]
    aex.legend() #添加图例
    plot.savefig('[drop]_acc.png')
    if draw_show==True:
        plot.show()

    # 绘图比较
    fig,aex=plot.subplots()
    aex.set_title(title)
    aex.set_xlabel('epoch')
    aex.set_ylabel('loss')
    [aex.plot(x,z,label="drop={}".format(mp[i])) for i,(x,y,z) in enumerate(_xy)]
    aex.legend() #添加图例
    plot.savefig('[drop]_loss.png')
    if draw_show==True:
        plot.show()


if __name__=='__main__':
    plot_("[block=x K=3 lr_decay=0.1 norm=True drop=0.2]","block",mp=[3,4,5,6,7],draw_show=True)
    plot_("[block=4 K=3 lr_decay=0.1 norm=True drop=x]","drop",mp=['0.0','0.1','0.2','0.5'],draw_show=True)
    plot_("[block=4 K=3 lr_decay=0.1 norm=x drop=0.2]","norm",mp=['True','False'],draw_show=True)
    plot_("[block=4 K=3 lr_decay=x norm=True drop=0.2]","lr_decay",mp=['0.01','0.1','0.9'],draw_show=True)
    plot_("[block=4 K=x lr_decay=0.1 norm=True drop=0.2]","ksp",mp=['311','512'],draw_show=True)