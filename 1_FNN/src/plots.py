import numpy
from matplotlib import pyplot as plot

def work(picname,draw_show,title,xlb,ylb,x,y,xticks=None):
    fig,aex=plot.subplots()
    aex.set_title(title)
    aex.set_xlabel(xlb)
    aex.set_ylabel(ylb)
    if xticks!=None:
        plot.xticks(x,xticks)
    aex.scatter(x,y)
    #for a,b in zip(x,y):
    #    plot.text(a,b,b,ha='center')
    aex.plot(x,y,linestyle=':')
    plot.savefig('{}.png'.format(picname))
    if draw_show==True:
        plot.show()

if __name__=='__main__':

    #work(picname="Verify_N",draw_show=True,
    #    title="[N=x lr=0.01 wide=40 depth=4 act=ReLU]",xlb='N',ylb='log10(loss)',
    #    x=[200,2000,5000,10000,15000],
    #    y=numpy.log10([0.277195,0.392976,0.226068,0.005124,0.004184]))

    work(picname="Verify_act",draw_show=True,
        title="[N=2000 lr=0.0025 wide=128 depth=2 act=x]",xlb='activation',ylb='log10(loss)',
        x=[1,2,3,4,5],
        y=numpy.log10([0.000144,0.0005446,0.004816,0.009859,0.016910]),
        xticks=['Sigmoid','Tanh','ReLU','ELU','Softplus'])

    work(picname="Verify_lr",draw_show=True,
        title="[N=2000 lr=x wide=128 depth=2 act=Sigmoid]",xlb='learning rate',ylb='log10(loss)',
        x=[1,2,3,4,5,6,7,8],
        y=numpy.log10([0.228502,0.013815,0.014097,0.002007,0.001384,0.001231,0.000130,0.267277]),
        xticks=['0.075','0.05','0.025','0.01','0.0075','0.005','0.0025','0.001'])

    work(picname="Verify_depth",draw_show=True,
        title="[N=2000 lr=0.0025 wide=128 depth=x act=Sigmoid]",xlb='depth',ylb='log10(loss)',
        x=[1,2,3,4,5,6],
        y=numpy.log10([0.404436,0.000157,0.000304,0.000743,0.002417,0.027679]))

    work(picname="Verify_wide",draw_show=True,
        title="[N=2000 lr=0.0025 wide=x depth=2 act=Sigmoid]",xlb='wide',ylb='log10(loss)',
        x=[1,2,3,4,5],
        y=numpy.log10([0.275246,0.067299,0.000168,0.000459,0.000663]),
        xticks=['32','64','128','256','512'])

    