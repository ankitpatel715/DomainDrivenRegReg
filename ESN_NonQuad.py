# ESN Implementing non-quadratic RHS function. In particular, interestedin (from https://en.m.wikipedia.org/wiki/Multiscroll_attractor)
# 1) Modified Chua attractor
# IC : 1,1,0
# x' = alpha*(y-h) = alpha*y + alpha*b*sin(x*pi/(2*a))
# y' = x-y+z
# z' = -beta*y
# h = -b*sin(pi*x/(2*a))
# alpha = 10.82, beta = 14.286, a = 1.3, b = .11.... may or may not be correct / complete. Check how it works

# 2) Modified Lu Chen Attractor (has a delay!)
# x' = a(y-x)
# y' = (c-a)x - x*f +cy
# z' = xy - bz
# f = d0 z + d1 z(t-tau) -d2 sin(z(t-tau))
# a = 35, c = 28, b = 3, d0 = 1, d1 = 1, d2 = -20.20, tau = .2
# init 1,1,14

# Could also try van der pol
# x' = y
# y' = u(1-xx)y - x
# u = 5?
# Arbitrary (smallish) IC?


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

Ntrain = 50000
Ntest = 2000
Ntestrun = 100
np.random.seed(1802)

# Modified Chua
state0 = [1.0, 1.0, 0.0]
alpha = 10.82
beta = 14.286
a = 1.3
b = .11
def f(state, t):
    x, y, z = state  # Unpack the state vector
    return alpha*(y-(-b*np.sin(np.pi*x/(2*a)))), x-y+z, -beta*y
t = np.arange(0.0, (Ntrain+Ntest*Ntestrun)/200 + .005, 0.005)
states = odeint(f, state0, t)

useFE = True # use manual forward euler if True
if (useFE):
    states[0,:] = state0 + .005*np.array(f(state0,0))
    for i in range(Ntrain + Ntest):
        states[i+1,:] = states[i,:] + .005*np.array(f(states[i,:],0))
# In FE, should be able to calculate the exact updates
# next_x = x + .005(alpha*y + b*np.sin(stuff)) = .005*b, 1, 0, .005*alpha, 0, 0
# next_y = y + .005(x-y+z) = 0,.005,0,.995,0,.005
# next_z = z + .005(-beta*y) = 0,0,0,-beta*.005,0,1
Wtrue = np.array([[.005*b,1,0,.005*alpha,0,0],[0,.005,0,.995,0,.005],[0,0,0,-beta*.005,0,1]])


regularized = np.std(states,0)
states /= regularized
states -= np.mean(states,0)

errorstore = np.zeros((3,Ntest,Ntestrun))

# Seems to work well enough
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot(states[:,0],AutoPred[:,1], AutoPred[:,2],'r')
#ax.plot(states[:,0],states[:,1],states[:,2])
#plt.show()

# Run ESN with either standardd odd square or odd sin nonlinearity - expect odd sin to work?

# Try with Pathak params


actfn = lambda x: np.tanh(x)
#actfn = lambda x: x # Works just as well
def nonlin(x): # odd square
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (x2[:, 2 * i] ** 2).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (x2[2 * i] ** 2).copy()
    return x2
def nonlin(x):
    return x
def nonlin(x): # odd square
    x2 = np.copy(x)
    if len(np.shape(x2)) == 2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = ((x2[:, 2 * i])**2).copy()
        return x2
    else:  # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = ((x2[2 * i])**2).copy()
    return x2
def nonlin2(x): # odd sine
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (np.sin(x2[:, 2 * i])).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (np.sin(x2[2 * i] )).copy()
    return x2
def nonlin2(x): # odd sine(pi*x/(2*a))
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (np.sin(np.pi*x2[:, 2 * i]/(2*a))).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (np.sin(np.pi*x2[2 * i]/(2*a))).copy()
    return x2
def nonlin2(x): # x,x^2, sin(x)
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1]//3):
            x2[:, 3 * i] = (np.sin(np.pi * x2[:, 2 * i] / (2 * a))).copy()
            #x2[:, 3 * i + 1] = (x2[:, 3 * i + 1]**2).copy()
            x2[:,3 * i + 1] = (np.sin(x2[:,3 * i + 1])).copy()
        return x2
    else:
        for i in range(len(x2)//3):
            x2[3 * i] = (np.sin(np.pi * x2[3 * i] / (2 * a))).copy()
            #x2[3 * i + 1] = (x2[3 * i + 1] ** 2).copy()
            x2[3 * i + 1] = (np.sin(x2[3*i+1])).copy()
    return x2


N = 100 # 100 default
dim = 3

p = np.min([3/N,1.0]) # they use 3/N
rho = .1 #.2 works well with N = 500 # default of .1 - rho as high as 1.25 works pretty well - mediocre results at intermidiate rho
#rho = 1.2 # For pathak, odd squaring, poor performance with 1.5 down to .8 - e.g. cannot get to work...
# Instead, turning sigma way down helps, although longer term performance is awful....
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

Win = np.random.rand(dim,N)*2 - 1 # dense
# Experimental - in their version, they make Win black structure, e.g. each neuron only receives one input, not mixed
# Turns out to be very beneficial
#Winc = np.copy(Win)
#for i in range(dim):
#    Win[i,i*(N//dim):(i+1)*(N//dim)]*=2
#Win-=Winc
# New Ankit wants block Identity inputs?
Win = Win*0
sigma = .5
Win[0,:N//3] = (np.random.rand(1,N//3)*2-1)*sigma
Win[1,N//3:2*(N//3)] = (np.random.rand(1,N//3)*2-1)*sigma
Win[2,2*(N//3):] = (np.random.rand(1,N - 2*(N//3))*2-1)*sigma

sigma = .5 # .25 works quite well with N = 500 # from paper - scale of input. They use .5, although higher values appear to help (e.g. 1.5 > .5 performance) Very low (e.g. .05) hurt performance
#Win = Win * sigma
# Generate network values - unroll output in time
rout = np.zeros((Ntrain,N))
# They initialize r0 = 0
r = np.zeros(N)
for i in range(Ntrain):
    r = actfn(A@r + states[i,:]@Win)
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
beta = .0001*.01
U = np.dot(trout.transpose(),trout) + Id_n * beta
Uinv = np.linalg.inv(U)
Wout = np.dot(Uinv,np.dot(trout.transpose(),states[1:Ntrain+1,:]))
# Exact least square solution
#Wout = np.linalg.lstsq(trout,states[1:Ntrain+1,:], rcond=-1)[0]
#from sklearn.linear_model import Ridge
#clf = Ridge(alpha=.0001*.00001)
#clf.fit(trout, states[1:Ntrain+1,:])
#Wout2 = clf.coef_
#Wout = np.transpose(Wout2)
#from sklearn.linear_model import Lasso # L1 reg - encourages sparsity
#clf = Lasso(alpha=.0001*.0001*0)
#clf.fit(trout, states[1:Ntrain+1,:])
#Wout2 = clf.coef_
#Wout = np.transpose(Wout2)


#Wout = Wtrue.transpose() * 2
# Not sure why 1) inferior performance and 2) requires *2 ????

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

#plt.plot(np.append(errors_og,errors2)) # not very interesting if errors_og is tiny
#plt.show()

#plt.plot(errors2)
#plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(311)
#ax.plot(ResPred2[:,0],'r')
#ax.plot(states[Ntrain+1:,0])
#ax2 = fig.add_subplot(312)
#ax2.plot(ResPred2[:,1],'r')
#ax2.plot(states[Ntrain+1:,1])
#ax3 = fig.add_subplot(313)
#ax3.plot(ResPred2[:,2],'r')
#ax3.plot(states[Ntrain+1:,2])
#plt.show()

# Same for 3D view
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot(ResPred2[:,0],ResPred2[:,1], ResPred2[:,2],'r')
#ax.plot(states[-2000:,0],states[-2000:,1],states[-2000:,2])
#plt.show()

# TF View
#fig = plt.figure()
#ax = fig.add_subplot(311)
#ax.plot(ResTrain[:,0],'r')
#ax.plot(states[:Ntrain,0])
#ax2 = fig.add_subplot(312)
#ax2.plot(ResTrain[:,1],'r')
#ax2.plot(states[:Ntrain,1])
#ax3 = fig.add_subplot(313)
#ax3.plot(ResTrain[:,2],'r')
#ax3.plot(states[:Ntrain,2])
#plt.show()

# Using paper error metric

errors2f = (ResPred2 - states[Ntrain+1:Ntrain+1+Ntest,:])
Resautoe = np.linalg.norm(errors2f, axis = 1) / np.linalg.norm(states[Ntrain+1:Ntrain+1+Ntest,:], axis = 1)
errorstore[0,:,0] = Resautoe
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
    errorstore[0, :, q] = nerrore


#plt.plot(Resautoe[:500])
#plt.xlabel('Time Step')
#plt.ylabel('Paper Error Metric')
#plt.show() # autoe hitting +/-.3 is theit failure case


# Not doing amazingly... What about true kernel regression?
# e.g. features are 1,x,y,z, sin(x),sin(y),sin(z) (possibly use sin(pi*x/(2*a)))
tfeatures = np.zeros((50000,7))
tfeatures[:, 0] = 1
tfeatures[:, 1] = states[:50000,0]
tfeatures[:, 2] = states[:50000,1]
tfeatures[:, 3] = states[:50000,2]
tfeatures[:, 4] = np.sin(states[:50000,0])
tfeatures[:, 5] = np.sin(states[:50000,1])
tfeatures[:, 6] = np.sin(states[:50000,2])
tfeatures[:, 4] = np.sin(np.pi*states[:50000,0]/(2*a))
tfeatures[:, 5] = np.sin(np.pi*states[:50000,1]/(2*a))
tfeatures[:, 6] = np.sin(np.pi*states[:50000,2]/(2*a))

Nf = 7
Id_nf = np.identity(Nf)
betaf = .0001*10
Uf = np.dot(tfeatures.transpose(),tfeatures) + Id_nf * betaf
Ufinv = np.linalg.inv(Uf)
Wfout = np.dot(Ufinv,np.dot(tfeatures.transpose(),states[1:Ntrain+1,:]))

# Run in auto mode, plot result, same as above
for q in range(0,Ntestrun):
    cstate = states[Ntrain+q*Ntest,:]
    spred2 = np.zeros((Ntest,7))
    for i in range(Ntest):
        spred2[i, 0] = 1
        spred2[i, 1] = cstate[0]
        spred2[i, 2] = cstate[1]
        spred2[i, 3] = cstate[2]
        #spred2[i, 4] = np.sin(cstate[0])
        #spred2[i, 5] = np.sin(cstate[1])
        #spred2[i, 6] = np.sin(cstate[2])
        spred2[i, 4] = np.sin(np.pi*cstate[0]/(2*a))
        spred2[i, 5] = np.sin(np.pi*cstate[1]/(2*a))
        spred2[i, 6] = np.sin(np.pi*cstate[2]/(2*a))

        cstate = spred2[i, :] @ Wfout

    AutoPred = spred2 @ Wfout
    autoerr = AutoPred - states[Ntrain + 1 + q*Ntest:Ntrain + 1 + (q+1)*Ntest, :]
    autoerre = np.linalg.norm(autoerr, axis=1) / np.linalg.norm(states[Ntrain + 1 + q*Ntest:Ntrain + 1 + (q+1)*Ntest, :], axis=1)
    errorstore[1,:,q] = autoerre

#fig = plt.figure()
#ax = fig.add_subplot(311)
#ax.plot(AutoPred[:,0],'r')
#ax.plot(states[Ntrain+1:Ntrain+1+Ntest,0])
#ax2 = fig.add_subplot(312)
#ax2.plot(AutoPred[:,1],'r')
#ax2.plot(states[Ntrain+1:Ntrain+1+Ntest,1])
#ax3 = fig.add_subplot(313)
#ax3.plot(AutoPred[:,2],'r')
#ax3.plot(states[Ntrain+1:Ntrain+1+Ntest,2])
#plt.show()

# Same for 3D view
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot(AutoPred[:,0],AutoPred[:,1], AutoPred[:,2],'r')
#ax.plot(states[Ntrain+1:Ntrain+1+Ntest,0],states[Ntrain+1:Ntrain+1+Ntest,1],states[Ntrain+1:Ntrain+1+Ntest,2])
#plt.show()


#fig = plt.figure()
#plt.plot(autoerre[:500])
#plt.xlabel('Testing Step')
#plt.ylabel('Relative (Norm) Error')
#plt.show()

# copy ESN (base) error, then re-run with ESN pathak
ResPred3 = np.copy(ResPred2)
errors3f = (ResPred3 - states[Ntrain+1:Ntrain+1+Ntest,:])
Res3autoe = np.linalg.norm(errors3f, axis = 1) / np.linalg.norm(states[Ntrain+1:Ntrain+1+Ntest,:], axis = 1)


actfn = lambda x: np.tanh(x)
#actfn = lambda x: x # Works just as well
def nonlin(x): # odd square
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (x2[:, 2 * i] ** 2).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (x2[2 * i] ** 2).copy()
    return x2
def nonlin(x):
    return x
def nonlin(x): # odd square
    x2 = np.copy(x)
    if len(np.shape(x2)) == 2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = ((x2[:, 2 * i])**2).copy()
        return x2
    else:  # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = ((x2[2 * i])**2).copy()
    return x2
def nonlin2(x): # odd sine
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (np.sin(x2[:, 2 * i])).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (np.sin(x2[2 * i] )).copy()
    return x2
def nonlin2(x): # odd sine(pi*x/(2*a))
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (np.sin(np.pi*x2[:, 2 * i]/(2*a))).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (np.sin(np.pi*x2[2 * i]/(2*a))).copy()
    return x2
def nonlin2(x): # x,x^2, sin(x)
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1]//3):
            x2[:, 3 * i] = (np.sin(np.pi * x2[:, 2 * i] / (2 * a))).copy()
            #x2[:, 3 * i + 1] = (x2[:, 3 * i + 1]**2).copy()
            x2[:,3 * i + 1] = (np.sin(x2[:,3 * i + 1])).copy()
        return x2
    else:
        for i in range(len(x2)//3):
            x2[3 * i] = (np.sin(np.pi * x2[3 * i] / (2 * a))).copy()
            #x2[3 * i + 1] = (x2[3 * i + 1] ** 2).copy()
            x2[3 * i + 1] = (np.sin(x2[3 * i + 1])).copy()
    return x2


N = 100 # 100 default
dim = 3

p = np.min([6/N,1.0]) # they use 3/N
rho = 1.1 #.2 works well with N = 500 # default of .1 - rho as high as 1.25 works pretty well - mediocre results at intermidiate rho
#rho = 1.2 # For pathak, odd squaring, poor performance with 1.5 down to .8 - e.g. cannot get to work...
# Instead, turning sigma way down helps, although longer term performance is awful....
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

Win = np.random.rand(dim,N)*2 - 1 # dense
# Experimental - in their version, they make Win black structure, e.g. each neuron only receives one input, not mixed
# Turns out to be very beneficial
#Winc = np.copy(Win)
#for i in range(dim):
#    Win[i,i*(N//dim):(i+1)*(N//dim)]*=2
#Win-=Winc
# New Ankit wants block Identity inputs?
Win = Win*0
sigma = .1
Win[0,:N//3] = (np.random.rand(1,N//3)*2-1)*sigma
Win[1,N//3:2*(N//3)] = (np.random.rand(1,N//3)*2-1)*sigma
Win[2,2*(N//3):] = (np.random.rand(1,N - 2*(N//3))*2-1)*sigma

sigma = .5 # .25 works quite well with N = 500 # from paper - scale of input. They use .5, although higher values appear to help (e.g. 1.5 > .5 performance) Very low (e.g. .05) hurt performance
#Win = Win * sigma
# Generate network values - unroll output in time
rout = np.zeros((Ntrain,N))
# They initialize r0 = 0
r = np.zeros(N)
for i in range(Ntrain):
    r = actfn(A@r + states[i,:]@Win)
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
beta = .0001*.01
U = np.dot(trout.transpose(),trout) + Id_n * beta
Uinv = np.linalg.inv(U)
Wout = np.dot(Uinv,np.dot(trout.transpose(),states[1:Ntrain+1,:]))
# Exact least square solution
#Wout = np.linalg.lstsq(trout,states[1:Ntrain+1,:], rcond=-1)[0]
#from sklearn.linear_model import Ridge
#clf = Ridge(alpha=.0001*.00001)
#clf.fit(trout, states[1:Ntrain+1,:])
#Wout2 = clf.coef_
#Wout = np.transpose(Wout2)
#from sklearn.linear_model import Lasso # L1 reg - encourages sparsity
#clf = Lasso(alpha=.0001*.0001*0)
#clf.fit(trout, states[1:Ntrain+1,:])
#Wout2 = clf.coef_
#Wout = np.transpose(Wout2)


#Wout = Wtrue.transpose() * 2
# Not sure why 1) inferior performance and 2) requires *2 ????

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

#plt.plot(np.append(errors_og,errors2)) # not very interesting if errors_og is tiny
#plt.show()

#plt.plot(errors2)
#plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(311)
#ax.plot(ResPred2[:,0],'r')
#ax.plot(states[Ntrain+1:Ntrain+1+Ntest,0])
#ax2 = fig.add_subplot(312)
#ax2.plot(ResPred2[:,1],'r')
#ax2.plot(states[Ntrain+1:Ntrain+1+Ntest,1])
#ax3 = fig.add_subplot(313)
#ax3.plot(ResPred2[:,2],'r')
#ax3.plot(states[Ntrain+1:Ntrain+1+Ntest,2])
#plt.show()

# Same for 3D view
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot(ResPred2[:,0],ResPred2[:,1], ResPred2[:,2],'r')
#ax.plot(states[Ntrain+1:Ntrain+1+Ntest,0],states[Ntrain+1:Ntrain+1+Ntest,1],states[Ntrain+1:Ntrain+1+Ntest,2])
#plt.show()

# TF View
#fig = plt.figure()
#ax = fig.add_subplot(311)
#ax.plot(ResTrain[:,0],'r')
#ax.plot(states[:Ntrain,0])
#ax2 = fig.add_subplot(312)
#ax2.plot(ResTrain[:,1],'r')
#ax2.plot(states[:Ntrain,1])
#ax3 = fig.add_subplot(313)
#ax3.plot(ResTrain[:,2],'r')
#ax3.plot(states[:Ntrain,2])
#plt.show()

# Using paper error metric

errors2f = (ResPred2 - states[Ntrain+1:Ntrain+1+Ntest,:])
Resautoe = np.linalg.norm(errors2f, axis = 1) / np.linalg.norm(states[Ntrain+1:Ntrain+1+Ntest,:], axis = 1)
errorstore[2,:,0] = Resautoe
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
    errorstore[2, :, q] = nerrore

#plt.plot(Resautoe[:500])
#plt.xlabel('Time Step')
#plt.ylabel('Paper Error Metric')
#plt.show() # autoe hitting +/-.3 is theit failure case

# Errorstore: Devika, D2R2, Pathak


fig = plt.figure()
#plt.plot(Resautoe)
plt.plot(np.mean(errorstore[2,:,:],1))
#plt.plot(Res3autoe)
plt.plot(np.mean(errorstore[0,:,:],1))
#plt.plot(autoerre)
plt.plot(np.mean(errorstore[1,:,:],1))
plt.legend(['HSR ESN', 'LSR ESN', 'D2R2'])
plt.title('Modified Chua Performance for varying algorithms')
plt.xlabel('Testing Step')
plt.ylabel('Relative (Norm) Error')
plt.show()

fig = plt.figure()
#plt.plot(Resautoe)
#plt.hlines(.3,0,500, linewidth = 1)
plt.semilogy(np.mean(errorstore[2,:,:],1))
#plt.plot(Res3autoe)
plt.semilogy(np.mean(errorstore[0,:,:],1))
#plt.plot(autoerre)
plt.semilogy(np.mean(errorstore[1,:,:],1))
plt.legend(['HSR ESN', 'LSR ESN', 'D2R2'])
plt.title('Modified Chua Performance for varying algorithms')
plt.xlabel('Testing Step')
plt.ylabel('Normalized Error')
plt.xlim(0,500)
plt.show()

