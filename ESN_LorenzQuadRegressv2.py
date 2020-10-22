# V2 - Add in averaging over 100 trials, strip out extraneous precision stuff

# Try comparing ESN to a quadratic regression of lorenz features
# E.g., build x+1 from [1,x,x^2, cross terms]


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

# Get states

Ntrain = 50000
Ntest = 2000
Ntestrun = 100

rho_l = 28.0
sigma_l = 10.0
beta_l = 8.0 / 3.0
dim = 3
def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma_l * (y - x), x * (rho_l - z) - y, x * y - beta_l * z  # Derivatives

state0 = [1.0, 1.0, 1.0]
#state0 = [ 1.83843611, -2.47268234, 26.97935438]
t = np.arange(0.0, (Ntrain+Ntest*Ntestrun)/200 + .005, 0.005)

states = odeint(f, state0, t)
useFE = False # use manual forward euler if True
if (useFE):
    states[0,:] = state0 + .005*np.array(f(state0,0))
    for i in range(Ntrain + Ntest):
        states[i+1,:] = states[i,:] + .005*np.array(f(states[i,:],0))

regularized = np.std(states,0)
states /= np.std(states,0)

errorstore = np.zeros((2,Ntest,Ntestrun))

#states = states.astype(np.single)
#states = states.astype(np.half)

# Generate feature vector from states
nfeatures = 35 # 10 for quadratic, 20 for cubic, 35 for quartic
features = np.zeros((Ntrain,nfeatures))
features[:,0] = 1
features[:,1] = states[:Ntrain,0]
features[:,2] = states[:Ntrain,1]
features[:,3] = states[:Ntrain,2]
features[:,4] = states[:Ntrain,0] * states[:Ntrain,0]
features[:,5] = states[:Ntrain,0] * states[:Ntrain,1]
features[:,6] = states[:Ntrain,0] * states[:Ntrain,2]
features[:,7] = states[:Ntrain,1] * states[:Ntrain,1]
features[:,8] = states[:Ntrain,1] * states[:Ntrain,2]
features[:,9] = states[:Ntrain,2] * states[:Ntrain,2]
if (nfeatures == 20 or nfeatures == 35):
    features[:, 10] = states[:Ntrain, 0] * states[:Ntrain, 0] * states[:Ntrain, 0]
    features[:, 11] = states[:Ntrain, 0] * states[:Ntrain, 0] * states[:Ntrain, 1]
    features[:, 12] = states[:Ntrain, 0] * states[:Ntrain, 0] * states[:Ntrain, 2]
    features[:, 13] = states[:Ntrain, 0] * states[:Ntrain, 1] * states[:Ntrain, 1]
    features[:, 14] = states[:Ntrain, 0] * states[:Ntrain, 1] * states[:Ntrain, 2]
    features[:, 15] = states[:Ntrain, 0] * states[:Ntrain, 2] * states[:Ntrain, 2]
    features[:, 16] = states[:Ntrain, 1] * states[:Ntrain, 1] * states[:Ntrain, 1]
    features[:, 17] = states[:Ntrain, 1] * states[:Ntrain, 1] * states[:Ntrain, 2]
    features[:, 18] = states[:Ntrain, 1] * states[:Ntrain, 2] * states[:Ntrain, 2]
    features[:, 19] = states[:Ntrain, 2] * states[:Ntrain, 2] * states[:Ntrain, 2]
if (nfeatures == 35):
    features[:, 20] = states[:Ntrain, 0] * states[:Ntrain, 0] * states[:Ntrain, 0] * states[:Ntrain, 0]
    features[:, 21] = states[:Ntrain, 0] * states[:Ntrain, 0] * states[:Ntrain, 0] * states[:Ntrain, 1]
    features[:, 22] = states[:Ntrain, 0] * states[:Ntrain, 0] * states[:Ntrain, 0] * states[:Ntrain, 2]
    features[:, 23] = states[:Ntrain, 0] * states[:Ntrain, 0] * states[:Ntrain, 1] * states[:Ntrain, 1]
    features[:, 24] = states[:Ntrain, 0] * states[:Ntrain, 0] * states[:Ntrain, 1] * states[:Ntrain, 2]
    features[:, 25] = states[:Ntrain, 0] * states[:Ntrain, 0] * states[:Ntrain, 2] * states[:Ntrain, 2]
    features[:, 26] = states[:Ntrain, 0] * states[:Ntrain, 1] * states[:Ntrain, 1] * states[:Ntrain, 1]
    features[:, 27] = states[:Ntrain, 0] * states[:Ntrain, 1] * states[:Ntrain, 1] * states[:Ntrain, 2]
    features[:, 28] = states[:Ntrain, 0] * states[:Ntrain, 1] * states[:Ntrain, 2] * states[:Ntrain, 2]
    features[:, 29] = states[:Ntrain, 0] * states[:Ntrain, 2] * states[:Ntrain, 2] * states[:Ntrain, 2]
    features[:, 30] = states[:Ntrain, 1] * states[:Ntrain, 1] * states[:Ntrain, 1] * states[:Ntrain, 1]
    features[:, 31] = states[:Ntrain, 1] * states[:Ntrain, 1] * states[:Ntrain, 1] * states[:Ntrain, 2]
    features[:, 32] = states[:Ntrain, 1] * states[:Ntrain, 1] * states[:Ntrain, 2] * states[:Ntrain, 2]
    features[:, 33] = states[:Ntrain, 1] * states[:Ntrain, 2] * states[:Ntrain, 2] * states[:Ntrain, 2]
    features[:, 34] = states[:Ntrain, 2] * states[:Ntrain, 2] * states[:Ntrain, 2] * states[:Ntrain, 2]


# Train offline  - min error norm + l2 error of trout*Wout - states, where Wout = Nx3
Id_n = np.identity(nfeatures)
beta = .0001
U = np.dot(features.transpose(),features) + Id_n * beta
Uinv = np.linalg.inv(U)
Wout = np.dot(Uinv,np.dot(features.transpose(),states[1:Ntrain+1,:]))

# Test lower precision?
#Wout = Wout.astype(np.single)
#Wout = Wout.astype(np.half)
# Could also try lowering precision of inputs before training, or during testing...

for q in range(1,Ntestrun):
    spred2 = np.zeros((Ntest,nfeatures))
    cstate = states[Ntrain+q*Ntest,:]
    for i in range(Ntest):
        spred2[i, 0] = 1
        spred2[i, 1] = cstate[0]
        spred2[i, 2] = cstate[1]
        spred2[i, 3] = cstate[2]
        spred2[i, 4] = cstate[0] * cstate[0]
        spred2[i, 5] = cstate[0] * cstate[1]
        spred2[i, 6] = cstate[0] * cstate[2]
        spred2[i, 7] = cstate[1] * cstate[1]
        spred2[i, 8] = cstate[1] * cstate[2]
        spred2[i, 9] = cstate[2] * cstate[2]
        if (nfeatures == 20 or nfeatures == 35):
            spred2[i, 10] = cstate[0] * cstate[0] * cstate[0]
            spred2[i, 11] = cstate[0] * cstate[0] * cstate[1]
            spred2[i, 12] = cstate[0] * cstate[0] * cstate[2]
            spred2[i, 13] = cstate[0] * cstate[1] * cstate[1]
            spred2[i, 14] = cstate[0] * cstate[1] * cstate[2]
            spred2[i, 15] = cstate[0] * cstate[2] * cstate[2]
            spred2[i, 16] = cstate[1] * cstate[1] * cstate[1]
            spred2[i, 17] = cstate[1] * cstate[1] * cstate[2]
            spred2[i, 18] = cstate[1] * cstate[2] * cstate[2]
            spred2[i, 19] = cstate[2] * cstate[2] * cstate[2]
        if (nfeatures == 35):
            spred2[i, 20] = cstate[0] * cstate[0] * cstate[0] * cstate[0]
            spred2[i, 21] = cstate[0] * cstate[0] * cstate[0] * cstate[1]
            spred2[i, 22] = cstate[0] * cstate[0] * cstate[0] * cstate[2]
            spred2[i, 23] = cstate[0] * cstate[0] * cstate[1] * cstate[1]
            spred2[i, 24] = cstate[0] * cstate[0] * cstate[1] * cstate[2]
            spred2[i, 25] = cstate[0] * cstate[0] * cstate[2] * cstate[2]
            spred2[i, 26] = cstate[0] * cstate[1] * cstate[1] * cstate[1]
            spred2[i, 27] = cstate[0] * cstate[1] * cstate[1] * cstate[2]
            spred2[i, 28] = cstate[0] * cstate[1] * cstate[2] * cstate[2]
            spred2[i, 29] = cstate[0] * cstate[2] * cstate[2] * cstate[2]
            spred2[i, 30] = cstate[1] * cstate[1] * cstate[1] * cstate[1]
            spred2[i, 31] = cstate[1] * cstate[1] * cstate[1] * cstate[2]
            spred2[i, 32] = cstate[1] * cstate[1] * cstate[2] * cstate[2]
            spred2[i, 33] = cstate[1] * cstate[2] * cstate[2] * cstate[2]
            spred2[i, 34] = cstate[2] * cstate[2] * cstate[2] * cstate[2]

        cstate = spred2[i,:] @ Wout
    AutoPred = spred2 @ Wout
    autoerr = AutoPred - states[Ntrain + 1 + q*Ntest:Ntrain+1+(q+1)*Ntest, :]
    autoerre = np.linalg.norm(autoerr, axis=1) / np.linalg.norm(states[Ntrain + 1 + q*Ntest:Ntrain+1+(q+1)*Ntest, :], axis=1)
    errorstore[0,:,q] = autoerre


# Testing - plot D2R2 predictions in auto
#fig = plt.figure()
#ax = fig.add_subplot(311)
#ax.plot(AutoPred[:,0],'r')
#ax.plot(states[Ntrain+1:,0])
#ax2 = fig.add_subplot(312)
#ax2.plot(AutoPred[:,1],'r')
#ax2.plot(states[Ntrain+1:,1])
#ax3 = fig.add_subplot(313)
#ax3.plot(AutoPred[:,2],'r')
#ax3.plot(states[Ntrain+1:,2])
#plt.show()

# Same for 3D view
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot(AutoPred[:,0],AutoPred[:,1], AutoPred[:,2],'r')
#ax.plot(states[-2000:,0],states[-2000:,1],states[-2000:,2])
#plt.show()

np.random.seed(1802)

actfn = lambda x: np.tanh(x)
#actfn2 = lambda x: x # Works just as well

def nonlin(x): # making it cubed has it blow up instantly,**4 as well
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (x2[:, 2 * i] ** 2).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (x2[2 * i] ** 2).copy()
    return x2

Ntrain = 50000
Ntest = 2000
N = 100

p = np.min([3/N,1.0]) # they use 3/N
rho = .1 #.2 works well with N = 500 # default of .1 - rho as high as 1.25 works pretty well - mediocre results at intermidiate rho
# from .05 to .25 is similar to default. Middling ranges can give worse results. Results start to IMPROVE as rho above one, e.g. 1.15 is nearly 2x as good, continues to imrpve up to 1.75
A = np.random.randn(N,N)
# They use .rand? despite .randn giving better results... at their default rho. At extreme rho e.g. 2+, rand works better
A = np.random.rand(N,N)
Amask = np.random.rand(N,N)
A[Amask>p] = 0
[eigvals,eigvecs] = np.linalg.eig(A)
A/=(np.real(np.max(eigvals)))
A*=rho
A*=1

sigma = .5
Win = np.random.rand(dim,N)*2 - 1 # dense
# Experimental - in their version, they make Win black structure, e.g. each neuron only receives one input, not mixed
# Turns out to be very beneficial
#Winc = np.copy(Win)
#for i in range(dim):
#    Win[i,i*(N//dim):(i+1)*(N//dim)]*=2
#Win-=Winc
# New Ankit wants block Identity inputs?
Win = Win*0
Win[0,:N//3] = (np.random.rand(N//3)*2-1)*sigma
Win[1,N//3:2*N//3] = (np.random.rand(2*N//3 - N//3)*2-1)*sigma
Win[2,2*N//3:] = (np.random.rand(N - 2*N//3)*2-1)*sigma

sigma = .5 # .25 works quite well with N = 500 # from paper - scale of input. They use .5, although higher values appear to help (e.g. 1.5 > .5 performance) Very low (e.g. .05) hurt performance
Win = Win * sigma
# Generate network values - unroll output in time
rout = np.zeros((Ntrain,N))
# They initialize r0 = 0
r = np.zeros(N)
for i in range(Ntrain):
    r = actfn(A@r + (states[i,:]+(np.random.rand(3)*2-1)*0)@Win)
    rout[i,:] = r
    #  Add slight noise to either r or rout. Start with directly to r
    #r += (np.random.rand(N)*2 - 1)*(1e-3) # e-3 ish works ok, but no real benefit
#trout = np.copy(rout)
#for i in range(N//2):
#    trout[:,2*i] = (trout[:,2*i]**2).copy()
trout = nonlin(rout)
#trout += (np.random.rand(Ntrain,N)*2-1)*(1e-5) # works decently here too

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


# Predictions - unroll reservoir for Ntest
rstore = np.copy(r)
rpred = np.zeros((Ntest,N))
rpred2 = np.zeros((Ntest,N))
r2 = np.copy(r)
for i in range(Ntest):
    r = actfn(A @ r + states[Ntrain+i, :] @ Win) # Teacher Forcing
    #r3 = np.copy(r2)
    #for j in range(N // 2):
    #    r3[2 * j] = (r3[2 * j] ** 2).copy()
    r3 = nonlin(r2)
    r2 = actfn(A @ r2 + r3 @ Wout @ Win)
    rpred[i,:] = r
    rpred2[i,:] = r2
#trpred = np.copy(rpred)
#trpred2 = np.copy(rpred2)
#for i in range(N//2):
#    trpred[:,2*i] = (trpred[:,2*i]**2).copy()
#    trpred2[:, 2 * i] = (trpred2[:, 2 * i] ** 2).copy()
trpred = nonlin(rpred)
trpred2 = nonlin(rpred2)

ResPred = trpred @ Wout
errors = np.sqrt(np.sum((ResPred - states[Ntrain+1:Ntrain+1+Ntest,:])**2,1))

ResPred2 = trpred2 @ Wout
errors2 = np.sqrt(np.sum((ResPred2 - states[Ntrain+1:Ntrain+1+Ntest,:])**2,1))

ResTrain = trout@Wout
errors_og = np.sqrt(np.sum((trout@Wout - states[1:Ntrain+1,:])**2,1))

# Compute both errors in Devika metric, then compare results
resautoerr = ResPred2 - states[Ntrain+1:Ntrain+1+Ntest,:]
resautoerre = np.linalg.norm(resautoerr, axis=1)/np.linalg.norm(states[Ntrain+1:Ntrain+1+Ntest,:],axis = 1)
errorstore[1,:,0] = resautoerre
for q in range(1, Ntestrun):
    # Redo the next prediction for Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest
    # Need to do (50?) timesteps of true TF input in order to get reservoir warmed up
    r = np.zeros(N)
    rpred = np.zeros((Ntest, N))
    rpred2 = np.zeros((Ntest, N))
    for i in range(50):
        r = actfn(A @ r + states[Ntrain + (q) * Ntest - (50 - i), :] @ Win)  # Teacher Forcing
    r2 = np.copy(r)
    for i in range(Ntest):
        r = actfn(A @ r + states[Ntrain + (q) * Ntest + i, :] @ Win)
        # r3 = np.copy(r2)
        # for j in range(N // 2):
        #    r3[2 * j] = (r3[2 * j] ** 2).copy()
        r3 = nonlin(r2)
        r2 = actfn(A @ r2 + r3 @ Wout @ Win)
        rpred[i, :] = r
        rpred2[i, :] = r2
    trpred2 = nonlin(rpred2)
    ResPred2 = trpred2 @ Wout
    nerror = ResPred2 - states[Ntrain + 1 + (q) * Ntest:Ntrain + 1 + (q + 1) * Ntest, :]
    nerrore = np.linalg.norm(nerror, axis=1) / np.linalg.norm(
        states[Ntrain + 1 + (q) * Ntest:Ntrain + 1 + (q + 1) * Ntest, :], axis=1)
    errorstore[1, :, q] = nerrore

fig = plt.plot()
#plt.plot(autoerre)
plt.plot(np.mean(errorstore[0,:,:],1))
#plt.plot(resautoerre)
plt.plot(np.mean(errorstore[1,:,:],1))
plt.legend(['D2R2', 'LSR-ESN'])
plt.title('Lorenz63 Performance for ESN and D2R2')
plt.xlabel('Testing Step')
plt.ylabel('Normalized Error')
plt.show()
