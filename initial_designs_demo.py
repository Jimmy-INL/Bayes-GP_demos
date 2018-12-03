import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import cartesian
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def factorial_design(n,d,plot_=False):
    # n is number of points in nth dimension
    # d is the number of factors
    # full factorial design is when the number of levels = number of factor
    #otmp=[]
    #if type(d) is int:
    otmp=d*[np.arange(n)]
    #else:
    #        for i in n:
    #        otmp.add(range(i))

    o=cartesian(otmp)

    D=(o-np.min(o))/(np.max(o)-np.min(o))
    if plot_:
        plt.close()
        fig1=plt.figure()
        ax=fig1.add_subplot(111)
        #fig,ax = plt.subplots()
        ax.scatter(D[:,0].reshape((D.shape[0],1)),D[:,1].reshape((D.shape[0],1)))
        ax.set_title('dim-1,dim-2 full factorial design')
        ax.set_xlabel('dim-1')
        ax.set_ylabel('dim-2')
        return(D,fig1)
        #fig.show()
    return(D)

def random_design(n,d,plot_=False):
    # n is number of points in nth dimension
    # d is the number of factors
    # full factorial design is when the number of levels = number of factor
    o=[]
    #if type(d) is int:
    o=np.random.rand(n,d)
    #else:
    #for i in n:
    #        o.add(range(i))
    fig=[]
    D=(o-np.min(o))/(np.max(o)-np.min(o))
    if plot_:
        fig2=plt.figure()
        ax=fig2.add_subplot(111)
        #fig,ax = plt.subplots()
        ax.scatter(D[:,0].reshape((D.shape[0],1)),D[:,1].reshape((D.shape[0],1)))
        ax.set_title('dim-1,dim-2 random design')
        ax.set_xlabel('dim-1')
        ax.set_ylabel('dim-2')
        ax.set_ylim((0,1))
        ax.set_xlim((0,1))
        return(D,fig2)
        #fig.show()
    return(D)

def latin_cube(n,d,plot_=False):
    # n is number of points in nth dimension
    # d is the number of factors
    # full factorial design is when the number of levels = number of factor
    o=[]
    x=np.arange(n)
    a=set(np.random.permutations(x))
    #R=np.array(list(a))[np.random.choice(n,d),:].transpose()
    R=[[]]*d
    #R=np.random.permutation(n).reshape(n,1)
    for i in range(d):
        R[i]=np.random.permutation(n).reshape(n,1)
    D=np.array(R).reshape(n,d)
    #D=cartesian(o)
    if plot_:
        #Dout=(D-np.min(D,0))/(np.max(D,0)-np.min(D,0))
        Dout=D+0.5
        fig3=plt.figure()
        ax=fig3.add_subplot(111)
        #fig,ax = plt.subplots()
        ax.scatter(Dout[:,0].reshape((Dout.shape[0],1)),Dout[:,1].reshape((Dout.shape[0],1)))
        #circles = [ax.Circle((xi,yi), radius=0.5, linewidth=0) for xi,yi in zip(Dout[:,0].reshape((Dout.shape[0],1)),Dout[:,1].reshape((Dout.shape[0],1)))]
        #c = matplotlib.collections.PatchCollection(circles)
        #ax.add_collection(c)
        ax.set_title('dim-1,dim-2 latin cube')
        ax.set_xlabel('dim-1')
        ax.set_ylabel('dim-2')
        ax.set_ylim((0,np.max(D)+1))
        ax.set_xlim((0,np.max(D)+1))
        return(Dout,fig3)
        #fig.show()
    return(D)

def LHD_design(n,d,plot_=False):
    # n is number of points in nth dimension
    # d is the number of factors
    # full factorial design is when the number of levels = number of factor
    Rij=latin_cube(n,d)+1
    R=Rij/np.max(Rij,0)
    rs=(Rij-1)/np.max(Rij,0)
    #rs.shape
    D=(R-rs)*np.random.rand(rs.shape[0],rs.shape[1])+rs
    if plot_:
        fig4=plt.figure()
        ax=fig4.add_subplot(111)
        #fig,ax = plt.subplots()
        ax.scatter(D[:,0].reshape((D.shape[0],1)),D[:,1].reshape((D.shape[0],1)))
        ax.set_title('dim-1,dim-2 latin cube random design')
        ax.set_xlabel('dim-1')
        ax.set_ylabel('dim-2')
        ax.set_ylim((0,1))
        ax.set_xlim((0,1))
        return(D,fig4)
        #fig.show()
    return(D)


def frankes_func(x,plot_=False):
    # Evaluate frankes function at x data points
#    f=function(x)
#{
    exp=np.exp
    x=np.atleast_2d(x)
    f1=3/4*exp(-.25*(9*x[:,0]-2)**2-.25*(9*x[:,1]-2)**2)
    f2=3/4*exp(-1/49*(9*x[:,0]+1)**2-1/10*(9*x[:,1]+1)**2)
    f3=1/2*exp(-.25*(9*x[:,0]-7)**2-.25*(9*x[:,1]-3)**2)
    f4=1/5*exp(-(9*x[:,0]-4)**2-(9*x[:,1]-7)**2)
    val=f1+f2+f3-f4

    if plot_:
        fig=plt.figure()
        ax=fig.gca(projection='3d')
        #fig,ax = plt.subplots()
        #ax.scatter(D[:,0].reshape((D.shape[0],1)),D[:,1].reshape((D.shape[0],1)))
        n=int(np.sqrt(len(x)))
        surf=ax.plot_surface(x[:,0].reshape(n,n),x[:,1].reshape(n,n),val.reshape(n,n),cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.set_title('dim-1,dim-2 latin cube random design')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        ax.set_zlim((0,1.4))
        ax.set_ylim((0,1))
        ax.set_xlim((0,1))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        return(val.reshape(x.shape[0],1),fig)

    return(val.reshape(x.shape[0],1))
#}

def parity_plot(ytest,ypred):
    fig5=plt.figure()
    ax=fig5.add_subplot(111)
    Y=np.vstack((ytest,ypred))
    #fig,ax = plt.subplots()
    ax.scatter(ytest,ypred)
    ax.set_title('Parity Plot')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Prediction')
    ax.set_ylim((np.min(Y),np.max(Y)))
    ax.set_xlim((np.min(Y),np.max(Y)))
    return(fig5)
