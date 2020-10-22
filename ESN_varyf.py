# ESN Plot - run standard ESN with varying f values (e.g. odd squaring)



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

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
Ntrainrun = 100
N = 100

rho_l = 28.0
#rho_l = 20
sigma_l = 10.0
beta_l = 8.0 / 3.0
dim = 3


def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma_l * (y - x), x * (rho_l - z) - y, x * y - beta_l * z  # Derivatives


state0 = [1.0, 1.0, 1.0]
# state0 = [ 1.83843611, -2.47268234, 26.97935438]
t = np.arange(0.0, (Ntrain + Ntest*Ntrainrun) / 200 + .005, 0.005)

states = odeint(f, state0, t)
useFE = False  # use manual forward euler if True
if (useFE):
    states[0, :] = state0 + .005 * np.array(f(state0, 0))
    for i in range(Ntrain + Ntest):
        states[i + 1, :] = states[i, :] + .005 * np.array(f(states[i, :], 0))

# Center and Rescale
# states -= np.mean(states,0)
# states /= np.max([[np.max(states,0)],[-np.min(states,0)]],0) # rescale to -1,1
# states /= np.std(states) # Rescale to std = 1
regularized = np.std(states, 0)
states /= np.std(states, 0)

errorstore = np.zeros((6,Ntest,Ntrainrun))

# Baseline - odd square

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
trout = nonlin(rout)


# Train offline  - min error norm + l2 error of trout*Wout - states, where Wout = Nx3
Id_n = np.identity(N)
beta = .0001
U = np.dot(trout.transpose(),trout) + Id_n * beta
Uinv = np.linalg.inv(U)
Wout = np.dot(Uinv,np.dot(trout.transpose(),states[1:Ntrain+1,:]))

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
ResPred2 = trpred2 @ Wout
nerror =  ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:]
nerrore_base = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+Ntest+1,:], axis = 1)
errorstore[0,:,0] = nerrore_base
for q in range(1,Ntrainrun):
    # Redo the next prediction for Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest
    # Need to do (50?) timesteps of true TF input in order to get reservoir warmed up
    r = np.zeros(N)
    rpred = np.zeros((Ntest, N))
    rpred2 = np.zeros((Ntest, N))
    for i in range(50):
        r = actfn(A @ r + states[Ntrain +(q)*Ntest-(50-i), :] @ Win)  # Teacher Forcing
    r2 = np.copy(r)
    for i in range(Ntest):
        r = actfn(A @ r + states[Ntrain+(q)*Ntest+i, :] @ Win)
        # r3 = np.copy(r2)
        # for j in range(N // 2):
        #    r3[2 * j] = (r3[2 * j] ** 2).copy()
        r3 = nonlin(r2)
        r2 = actfn(A @ r2 + r3 @ Wout @ Win)
        rpred[i, :] = r
        rpred2[i, :] = r2
    trpred2 = nonlin(rpred2)
    ResPred2 = trpred2 @ Wout
    nerror = ResPred2 - states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :]
    nerrore_base = np.linalg.norm(nerror, axis=1) / np.linalg.norm(states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :], axis=1)
    errorstore[0, :, q] = nerrore_base

# Repeat with new nonlin : 1.9, 2.1 power, tanh, exp, sin
def nonlin(x): # making it cubed has it blow up instantly,**4 as well
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (np.abs(x2[:, 2 * i]) ** 2.2).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (np.abs(x2[2 * i]) ** 2.2).copy()
    return x2

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
trout = nonlin(rout)


# Train offline  - min error norm + l2 error of trout*Wout - states, where Wout = Nx3
Id_n = np.identity(N)
beta = .0001
U = np.dot(trout.transpose(),trout) + Id_n * beta
Uinv = np.linalg.inv(U)
Wout = np.dot(Uinv,np.dot(trout.transpose(),states[1:Ntrain+1,:]))

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
ResPred2 = trpred2 @ Wout
nerror =  ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:]
nerrore_p21 = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+Ntest+1,:], axis = 1)
errorstore[1,:,0] = nerrore_p21
for q in range(1,Ntrainrun):
    # Redo the next prediction for Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest
    # Need to do (50?) timesteps of true TF input in order to get reservoir warmed up
    r = np.zeros(N)
    rpred = np.zeros((Ntest, N))
    rpred2 = np.zeros((Ntest, N))
    for i in range(50):
        r = actfn(A @ r + states[Ntrain +(q)*Ntest-(50-i), :] @ Win)  # Teacher Forcing
    r2 = np.copy(r)
    for i in range(Ntest):
        r = actfn(A @ r + states[Ntrain+(q)*Ntest+i, :] @ Win)
        # r3 = np.copy(r2)
        # for j in range(N // 2):
        #    r3[2 * j] = (r3[2 * j] ** 2).copy()
        r3 = nonlin(r2)
        r2 = actfn(A @ r2 + r3 @ Wout @ Win)
        rpred[i, :] = r
        rpred2[i, :] = r2
    trpred2 = nonlin(rpred2)
    ResPred2 = trpred2 @ Wout
    nerror = ResPred2 - states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :]
    nerrore_p21 = np.linalg.norm(nerror, axis=1) / np.linalg.norm(states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :], axis=1)
    errorstore[1, :, q] = nerrore_p21

def nonlin(x): # making it cubed has it blow up instantly,**4 as well
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (np.abs(x2[:, 2 * i]) ** 1.8).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (np.abs(x2[2 * i]) ** 1.8).copy()
    return x2

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
trout = nonlin(rout)


# Train offline  - min error norm + l2 error of trout*Wout - states, where Wout = Nx3
Id_n = np.identity(N)
beta = .0001
U = np.dot(trout.transpose(),trout) + Id_n * beta
Uinv = np.linalg.inv(U)
Wout = np.dot(Uinv,np.dot(trout.transpose(),states[1:Ntrain+1,:]))

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
ResPred2 = trpred2 @ Wout
nerror =  ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:]
nerrore_p19 = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+Ntest+1,:], axis = 1)
errorstore[2,:,0] = nerrore_p19
for q in range(1,Ntrainrun):
    # Redo the next prediction for Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest
    # Need to do (50?) timesteps of true TF input in order to get reservoir warmed up
    r = np.zeros(N)
    rpred = np.zeros((Ntest, N))
    rpred2 = np.zeros((Ntest, N))
    for i in range(50):
        r = actfn(A @ r + states[Ntrain +(q)*Ntest-(50-i), :] @ Win)  # Teacher Forcing
    r2 = np.copy(r)
    for i in range(Ntest):
        r = actfn(A @ r + states[Ntrain+(q)*Ntest+i, :] @ Win)
        # r3 = np.copy(r2)
        # for j in range(N // 2):
        #    r3[2 * j] = (r3[2 * j] ** 2).copy()
        r3 = nonlin(r2)
        r2 = actfn(A @ r2 + r3 @ Wout @ Win)
        rpred[i, :] = r
        rpred2[i, :] = r2
    trpred2 = nonlin(rpred2)
    ResPred2 = trpred2 @ Wout
    nerror = ResPred2 - states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :]
    nerrore_p19 = np.linalg.norm(nerror, axis=1) / np.linalg.norm(states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :], axis=1)
    errorstore[2, :, q] = nerrore_p19

def nonlin(x): # making it cubed has it blow up instantly,**4 as well
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (np.tanh(x2[:, 2 * i])).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (np.tanh(x2[2 * i])).copy()
    return x2

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
trout = nonlin(rout)


# Train offline  - min error norm + l2 error of trout*Wout - states, where Wout = Nx3
Id_n = np.identity(N)
beta = .0001
U = np.dot(trout.transpose(),trout) + Id_n * beta
Uinv = np.linalg.inv(U)
Wout = np.dot(Uinv,np.dot(trout.transpose(),states[1:Ntrain+1,:]))

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
ResPred2 = trpred2 @ Wout
nerror =  ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:]
nerrore_tanh = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+Ntest+1,:], axis = 1)
errorstore[3,:,0] = nerrore_tanh
for q in range(1,Ntrainrun):
    # Redo the next prediction for Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest
    # Need to do (50?) timesteps of true TF input in order to get reservoir warmed up
    r = np.zeros(N)
    rpred = np.zeros((Ntest, N))
    rpred2 = np.zeros((Ntest, N))
    for i in range(50):
        r = actfn(A @ r + states[Ntrain +(q)*Ntest-(50-i), :] @ Win)  # Teacher Forcing
    r2 = np.copy(r)
    for i in range(Ntest):
        r = actfn(A @ r + states[Ntrain+(q)*Ntest+i, :] @ Win)
        # r3 = np.copy(r2)
        # for j in range(N // 2):
        #    r3[2 * j] = (r3[2 * j] ** 2).copy()
        r3 = nonlin(r2)
        r2 = actfn(A @ r2 + r3 @ Wout @ Win)
        rpred[i, :] = r
        rpred2[i, :] = r2
    trpred2 = nonlin(rpred2)
    ResPred2 = trpred2 @ Wout
    nerror = ResPred2 - states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :]
    nerrore_tanh = np.linalg.norm(nerror, axis=1) / np.linalg.norm(states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :], axis=1)
    errorstore[3, :, q] = nerrore_tanh

def nonlin(x): # making it cubed has it blow up instantly,**4 as well
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (np.exp(x2[:, 2 * i])).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (np.exp(x2[2 * i])).copy()
    return x2

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
trout = nonlin(rout)


# Train offline  - min error norm + l2 error of trout*Wout - states, where Wout = Nx3
Id_n = np.identity(N)
beta = .0001
U = np.dot(trout.transpose(),trout) + Id_n * beta
Uinv = np.linalg.inv(U)
Wout = np.dot(Uinv,np.dot(trout.transpose(),states[1:Ntrain+1,:]))

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
ResPred2 = trpred2 @ Wout
nerror =  ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:]
nerrore_exp = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+Ntest+1,:], axis = 1)
errorstore[4,:,0] = nerrore_exp
for q in range(1,Ntrainrun):
    # Redo the next prediction for Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest
    # Need to do (50?) timesteps of true TF input in order to get reservoir warmed up
    r = np.zeros(N)
    rpred = np.zeros((Ntest, N))
    rpred2 = np.zeros((Ntest, N))
    for i in range(50):
        r = actfn(A @ r + states[Ntrain +(q)*Ntest-(50-i), :] @ Win)  # Teacher Forcing
    r2 = np.copy(r)
    for i in range(Ntest):
        r = actfn(A @ r + states[Ntrain+(q)*Ntest+i, :] @ Win)
        # r3 = np.copy(r2)
        # for j in range(N // 2):
        #    r3[2 * j] = (r3[2 * j] ** 2).copy()
        r3 = nonlin(r2)
        r2 = actfn(A @ r2 + r3 @ Wout @ Win)
        rpred[i, :] = r
        rpred2[i, :] = r2
    trpred2 = nonlin(rpred2)
    ResPred2 = trpred2 @ Wout
    nerror = ResPred2 - states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :]
    nerrore_exp = np.linalg.norm(nerror, axis=1) / np.linalg.norm(states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :], axis=1)
    errorstore[4, :, q] = nerrore_exp


def nonlin(x): # making it cubed has it blow up instantly,**4 as well
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (np.sin(x2[:, 2 * i])).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (np.sin(x2[2 * i])).copy()
    return x2

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
trout = nonlin(rout)


# Train offline  - min error norm + l2 error of trout*Wout - states, where Wout = Nx3
Id_n = np.identity(N)
beta = .0001
U = np.dot(trout.transpose(),trout) + Id_n * beta
Uinv = np.linalg.inv(U)
Wout = np.dot(Uinv,np.dot(trout.transpose(),states[1:Ntrain+1,:]))

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
ResPred2 = trpred2 @ Wout
nerror =  ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:]
nerrore_sine = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+Ntest+1,:], axis = 1)
errorstore[5,:,0] = nerrore_sine
for q in range(1,Ntrainrun):
    # Redo the next prediction for Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest
    # Need to do (50?) timesteps of true TF input in order to get reservoir warmed up
    r = np.zeros(N)
    rpred = np.zeros((Ntest, N))
    rpred2 = np.zeros((Ntest, N))
    for i in range(50):
        r = actfn(A @ r + states[Ntrain +(q)*Ntest-(50-i), :] @ Win)  # Teacher Forcing
    r2 = np.copy(r)
    for i in range(Ntest):
        r = actfn(A @ r + states[Ntrain+(q)*Ntest+i, :] @ Win)
        # r3 = np.copy(r2)
        # for j in range(N // 2):
        #    r3[2 * j] = (r3[2 * j] ** 2).copy()
        r3 = nonlin(r2)
        r2 = actfn(A @ r2 + r3 @ Wout @ Win)
        rpred[i, :] = r
        rpred2[i, :] = r2
    trpred2 = nonlin(rpred2)
    ResPred2 = trpred2 @ Wout
    nerror = ResPred2 - states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :]
    nerrore_sine = np.linalg.norm(nerror, axis=1) / np.linalg.norm(states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :], axis=1)
    errorstore[5, :, q] = nerrore_sine


fig = plt.figure()
#plt.plot(nerrore_base)
plt.plot(np.mean(errorstore[0,:,:],1))
#plt.plot(nerrore_p21)
plt.plot(np.mean(errorstore[1,:,:],1))
#plt.plot(nerrore_p19)
plt.plot(np.mean(errorstore[2,:,:],1))
#plt.plot(nerrore_tanh)
plt.plot(np.mean(errorstore[3,:,:],1))
#plt.plot(nerrore_exp)
plt.plot(np.mean(errorstore[4,:,:],1))
#plt.plot(nerrore_sine)
plt.plot(np.mean(errorstore[5,:,:],1))
plt.legend(['Odd Square', 'Odd (2.2) power', 'Odd (1.8) power', 'Odd Tanh', 'Odd Exp', 'Odd Sine'])
plt.ylabel('Normalized Error')
plt.xlabel('Autonomous Test Step')
plt.title('ESN Performance for Various Test Perturbations')
plt.show()

fig = plt.figure()
#plt.semilogy(nerrore_base)
plt.semilogy(np.mean(errorstore[0,:,:],1))
#plt.semilogy(nerrore_p21)
plt.semilogy(np.mean(errorstore[1,:,:],1))
#plt.semilogy(nerrore_p19)
plt.semilogy(np.mean(errorstore[2,:,:],1))
#plt.semilogy(nerrore_tanh)
plt.semilogy(np.mean(errorstore[3,:,:],1))
#plt.semilogy(nerrore_exp)
plt.semilogy(np.mean(errorstore[4,:,:],1))
#plt.semilogy(nerrore_sine)
plt.semilogy(np.mean(errorstore[5,:,:],1))
plt.legend(['Odd Square', 'Odd (2.2) power', 'Odd (1.8) power', 'Odd Tanh', 'Odd Exp', 'Odd Sine'])
plt.ylabel('Normalized Error')
plt.xlabel('Test Step')
plt.title('ESN with Various Nonlinearities')
plt.show()