
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1802)

actfn = lambda x: np.tanh(x)
actfn2 = lambda x: x # Works just as well


def nonlin(x): # every other abs, e..g even version
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (np.abs(x2[:, 2 * i])**2).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (np.abs(x2[2 * i])**2).copy()
    return x2

N = 100 # 100 default
Ntrain = 50000
Ntest = 2000

rho_l = 28.0
sigma_l = 10.0
beta_l = 8.0 / 3.0
dim = 3
def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma_l * (y - x), x * (rho_l - z) - y, x * y - beta_l * z  # Derivatives

state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, (Ntrain+Ntest)/200 + .005, 0.005)

states = odeint(f, state0, t)
useFE = False # use manual forward euler if True
if (useFE):
    states[0,:] = state0 + .005*np.array(f(state0,0))
    for i in range(Ntrain + Ntest):
        states[i+1,:] = states[i,:] + .005*np.array(f(states[i,:],0))


# Center and Rescale
regularized = np.std(states,0)
states /= np.std(states,0)

useHenon = False
if useHenon:
    dim = 2  # Henon
    # Generate data - Using Henon Map
    alpha = 1.4
    b = .3
    # Xn+1 = 1 - alpha * xn**2 + yn
    # Yn+1 = b*xn
    states = np.zeros((dim, Ntrain + Ntest + 1))
    for i in range(1, Ntrain + Ntest + 1):
        states[:, i] = np.array(
            [1 - alpha * states[0, i - 1] * states[0, i - 1] + states[1, i - 1], b * states[0, i - 1]])

    states = np.transpose(states)/np.std(np.transpose(states), 0)


p = np.min([3/N,1.0]) # they use 3/N
rho = .1
A = np.random.rand(N,N)
Amask = np.random.rand(N,N)
A[Amask>p] = 0
[eigvals,eigvecs] = np.linalg.eig(A)
A/=(np.real(np.max(eigvals)))
A*=rho
A*=1

Win = np.random.rand(dim,N)*2 - 1 # dense


sigma = .5
Win = Win * sigma
# Generate network values - unroll output in time
rout = np.zeros((Ntrain,N))
# They initialize r0 = 0
r = np.zeros(N)
for i in range(Ntrain):
    r = actfn(A@r + (states[i,:]+(np.random.rand(3)*2-1)*0)@Win)
    rout[i,:] = r
trout = nonlin(rout)

# Train offline  - min error norm + l2 error of trout*Wout - states, where Wout = Nx3
Id_n = np.identity(N)
beta = .0001
U = np.dot(trout.transpose(),trout) + Id_n * beta
Uinv = np.linalg.inv(U)
Wout = np.dot(Uinv,np.dot(trout.transpose(),states[1:Ntrain+1,:]))

# Try to use a lower dim version of Wout?? Not sure how to test this
simplify = False
if (simplify):
    test = 1

r = np.zeros(N)
for j in range(Ntrain-50,Ntrain):
    r = actfn(A @ r + (states[j, :]) @ Win)

# Predictions - unroll reservoir for Ntest
rstore = np.copy(r)
rpred = np.zeros((Ntest,N))
rpred2 = np.zeros((Ntest,N))
r2 = np.copy(r)
for i in range(Ntest):
    r = actfn(A @ r + states[Ntrain+i, :] @ Win) # Teacher Forcing
    r3 = nonlin(r2)
    r2 = actfn(A @ r2 + r3 @ Wout @ Win)
    rpred[i,:] = r
    rpred2[i,:] = r2
trpred = nonlin(rpred)
trpred2 = nonlin(rpred2)

ResPred = trpred @ Wout
errors = np.sqrt(np.sum((ResPred - states[Ntrain+1:,:])**2,1))

ResPred2 = trpred2 @ Wout
errors2 = np.sqrt(np.sum((ResPred2 - states[Ntrain+1:,:])**2,1))

ResTrain = trout@Wout
errors_og = np.sqrt(np.sum((trout@Wout - states[1:Ntrain+1,:])**2,1))


plt.plot(errors2)
plt.show()

if (dim>2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(ResPred2[:, 0], ResPred2[:, 1], ResPred2[:, 2],'r')
    ax.plot(states[-Ntest:,0],states[-Ntest:,1],states[-Ntest:,2],'b')
    plt.draw()
    plt.show()

    if (False):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(ResPred[:, 0], ResPred[:, 1], ResPred[:, 2], 'r')
        ax.plot(states[-Ntest:, 0], states[-Ntest:, 1], states[-Ntest:, 2], 'b')
        plt.draw()
        plt.show()

# Calculate R^2 values for train,test
ybartrain = np.mean(states[:50000,:],0)
SSTtrain = np.sum((states[:50000,:] - ybartrain)**2)
SSEtrain = np.sum((trout@Wout - states[1:Ntrain+1,:])**2)
SSRegtrain = np.sum((ResTrain[:50000,:] - ybartrain)**2)
#Rsquaredtrain = SSRegtrain/SSTtrain
Rsquaredtrain = 1-SSEtrain/SSTtrain

ybartest = np.mean(states[50001:,:],0)
SSTtest = np.sum((states[50001:,:] - ybartest)**2)
SSEtest = np.sum((ResPred2 - states[Ntrain+1:,:])**2)
SSRegtest = np.sum((ResPred2 - ybartest)**2)
#Rsquaredtest = SSRegtest/SSTtest
Rsquaredtest = 1-SSEtest/SSTtest


if N>50:
    plotvals = np.array([0,1,5,9,24,50,-1])
else:
    plotvals = np.arange(N)
# At start
plt.figure(figsize=(12,6))
plt.plot(states[1:10001,0]-1,'k',linewidth = 4)
plt.plot(np.transpose(ResTrain[:10000,0]-1),'r',linewidth = 3)
plt.plot(trout[:10000,plotvals]+3)
plt.title('Reservoir Traces, Target (black) and Output (red), Train R^2 = '+"{:.5f}".format(Rsquaredtrain), fontsize=16)
plt.xlabel('Training Step', fontsize=14)
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off
plt.show()