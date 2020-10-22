# Test standard (devika) ESN for varying N on lorenz 63, plot results

# Upgrading to v2 - ensemble over multiple (100?) runs, change first plot to colormap
# When doing 100 runs - should they be over 100 different training runs? (e.g. same target, unique A)
# or should they be one run tested on 100 testing conditions (e.g. change autonomous testing time to 2000*100, test throughout?)


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
Ntrainrun = 100
Ntest = 2000


rho_l = 28.0
sigma_l = 10.0
beta_l = 8.0 / 3.0
dim = 3
def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma_l * (y - x), x * (rho_l - z) - y, x * y - beta_l * z  # Derivatives

state0 = [1.0, 1.0, 1.0]
#state0 = [ 1.83843611, -2.47268234, 26.97935438]
t = np.arange(0.0, (Ntrain+Ntest*Ntrainrun)/200 + .005, 0.005)

states = odeint(f, state0, t)
useFE = False # use manual forward euler if True
if (useFE):
    states[0,:] = state0 + .005*np.array(f(state0,0))
    for i in range(Ntrain + Ntest):
        states[i+1,:] = states[i,:] + .005*np.array(f(states[i,:],0))

# Center and Rescale
#states -= np.mean(states,0)
#states /= np.max([[np.max(states,0)],[-np.min(states,0)]],0) # rescale to -1,1
#states /= np.std(states) # Rescale to std = 1
regularized = np.std(states,0)
states /= np.std(states,0)


Nuse = np.array([10,20,30,40,50,75,100,150,200,250])
errorstore = np.zeros((len(Nuse),Ntest,Ntrainrun))
for k in range(len(Nuse)):
    print(k)
    N = Nuse[k]
    #N = 100 # 100 default

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

    #ResPred = trpred @ Wout
    #errors = np.sqrt(np.sum((ResPred - states[Ntrain+1:,:])**2,1))

    ResPred2 = trpred2 @ Wout
    #errors2 = np.sqrt(np.sum((ResPred2 - states[Ntrain+1:,:])**2,1))

    #ResTrain = trout@Wout
    #errors_og = np.sqrt(np.sum((trout@Wout - states[1:Ntrain+1,:])**2,1))

    #plt.plot(np.append(errors_og,errors2)) # not very interesting if errors_og is tiny
    #plt.show()

    #plt.plot(errors2)
    #plt.show()

    #if (dim>2):
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #ax.plot(ResPred2[:, 0], ResPred2[:, 1], ResPred2[:, 2],'r')
        #ax.plot(states[-Ntest:,0],states[-Ntest:,1],states[-Ntest:,2],'b')
        #plt.draw()
        #plt.show()

        #if (False):
            #fig = plt.figure()
            #ax = fig.gca(projection='3d')
            #ax.plot(ResPred[:, 0], ResPred[:, 1], ResPred[:, 2], 'r')
            #ax.plot(states[-Ntest:, 0], states[-Ntest:, 1], states[-Ntest:, 2], 'b')
            #plt.draw()
            #plt.show()

    # Calculate R^2 values for train,test
    #ybartrain = np.mean(states[:50000,:],0)
    #SSTtrain = np.sum((states[:50000,:] - ybartrain)**2)
    #SSEtrain = np.sum((trout@Wout - states[1:Ntrain+1,:])**2)
    #SSRegtrain = np.sum((ResTrain[:50000,:] - ybartrain)**2)
    ##Rsquaredtrain = SSRegtrain/SSTtrain
    #Rsquaredtrain = 1-SSEtrain/SSTtrain

    #ybartest = np.mean(states[50001:,:],0)
    #SSTtest = np.sum((states[50001:,:] - ybartest)**2)
    #SSEtest = np.sum((ResPred2 - states[Ntrain+1:,:])**2)
    #SSRegtest = np.sum((ResPred2 - ybartest)**2)
    ##Rsquaredtest = SSRegtest/SSTtest
    #Rsquaredtest = 1-SSEtest/SSTtest


    #if N>50:
    #    plotvals = np.array([0,1,5,9,24,50,-1])
    #else:
    #    plotvals = np.arange(N)
    # At start
    #plt.figure(figsize=(20,10))
    #plt.plot(states[1:10001,0]-1,'k',linewidth = 4)
    #plt.plot(np.transpose(ResTrain[:10000,0]-1),'r',linewidth = 3)
    #plt.plot(trout[:10000,plotvals]+3)
    #plt.title('Reservoir Traces, Target (black) and Output (red), Train R^2 = '+"{:.5f}".format(Rsquaredtrain))
    #plt.show()
    # At transition
    #plt.figure(figsize=(20,10))
    #plt.plot(states[-10000:,0]-1,'k',linewidth = 4)
    #plt.plot(np.transpose(np.append(ResTrain[-8000:,0],ResPred2[:2000,0]))-1,'r',linewidth = 3)
    #plt.plot(np.append(trout[-8000:,plotvals],trpred2[:2000,plotvals],0)+3)
    #plt.vlines(8000,-3,4)
    #plt.title('Reservoir Traces, Target (black) and Output (red), transition to test at veritcal line, Test R^2 = '+"{:.5f}".format(Rsquaredtest))
    #plt.xlim([8001,8300])
    #plt.ylim([np.min(trpred2[:250,plotvals]),np.max(trpred2[:250,plotvals])])
    #plt.show()
    # New plot - show +/-100 time steps from transition, x/y/z real and target with TEACHER FORCING (not true)
    #plt.figure(figsize = (20,10))
    #plt.plot(states[-2100:-1000,0]-2, 'k', linewidth = 4)
    #plt.plot(states[-2100:-1000,1], 'k', linewidth = 4)
    #plt.plot(states[-2100:-1000,2]+2, 'k', linewidth = 4)
    #plt.plot(np.transpose(np.append(ResTrain[-100:,0],ResPred[:1000,0]))-2,'r',linewidth = 3)
    #plt.plot(np.transpose(np.append(ResTrain[-100:,1],ResPred[:1000,1])),'r',linewidth = 3)
    #plt.plot(np.transpose(np.append(ResTrain[-100:,2],ResPred[:1000,2]))+2,'r',linewidth = 3)
    #plt.vlines(100,-4,4)
    #plt.title('Teacher Forcing, Targets (black) and Outputs (red), transition to test at veritcal line')
    #plt.show()

    # Standardized error: norm(error)/norm(true)
    nerror =  ResPred2 - states[Ntrain+1:Ntrain+1+Ntest,:]
    nerrore = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+1+Ntest,:], axis = 1)
    errorstore[k,:,0] = nerrore

    # Now do the remaining
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
        nerrore = np.linalg.norm(nerror, axis=1) / np.linalg.norm(states[Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest, :], axis=1)
        errorstore[k, :, q] = nerrore

plt.close()

colors = plt.cm.jet(np.linspace(0,1,len(Nuse)))

fig = plt.figure()
for i in range(len(Nuse)):
    plt.semilogy(np.mean(errorstore[i,:,:],1), label = "N = "+str(Nuse[i]), color =colors[i])
plt.legend()
plt.ylabel('Normalized Error')
plt.xlabel('Test Step')
plt.title('ESN Performance vs Reservoir size N')
plt.show()

print('vs N done')

# Compare (N = 100) tanh vs non tanh
errorstore2 = np.zeros((2,Ntest,Ntrainrun))
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

#ResPred = trpred @ Wout
#errors = np.sqrt(np.sum((ResPred - states[Ntrain+1:,:])**2,1))

ResPred2 = trpred2 @ Wout
errors2 = np.sqrt(np.sum((ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:])**2,1))

#ResTrain = trout@Wout
#errors_og = np.sqrt(np.sum((trout@Wout - states[1:Ntrain+1,:])**2,1))

nerror =  ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:]
nerrore = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+Ntest+1,:], axis = 1)
errorstore2[0, :, 0] = nerrore
for q in range(1, Ntrainrun):
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
    errorstore2[0, :, q] = nerrore



# Repeat, but with tanh nonlinearity removed
actfn = lambda x: x # Works just as well

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

#ResPred = trpred @ Wout
#errors = np.sqrt(np.sum((ResPred - states[Ntrain+1:,:])**2,1))

ResPred2 = trpred2 @ Wout
errors2 = np.sqrt(np.sum((ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:])**2,1))

#ResTrain = trout@Wout
#errors_og = np.sqrt(np.sum((trout@Wout - states[1:Ntrain+1,:])**2,1))

nerror =  ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:]
nerrore_notanh = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+Ntest+1,:], axis = 1)
errorstore2[1, :, 0] = nerrore_notanh
for q in range(1, Ntrainrun):
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
    nerrore_notanh = np.linalg.norm(nerror, axis=1) / np.linalg.norm(
        states[Ntrain + 1 + (q) * Ntest:Ntrain + 1 + (q + 1) * Ntest, :], axis=1)
    errorstore2[1, :, q] = nerrore_notanh

fig = plt.figure()
#plt.plot(nerrore)
plt.plot(np.mean(errorstore2[0,:,:],1))
#plt.plot(nerrore_notanh)
plt.plot(np.mean(errorstore2[1,:,:],1))
plt.legend(['Tanh Activation', 'Identity Activation'])
plt.ylabel('Normalized Error')
plt.xlabel('Test Step')
plt.title('ESN Performance vs Unit Activation Fn')
plt.show()

print('vs ActFn Done')


# Next - changing A/Wout/Win by +/- x% from true? Start with one percent.
# First, do training (up to getting Wout) correctly in normal case
errorstore3 = np.zeros((4,Ntest,Ntrainrun))
actfn = lambda x: np.tanh(x)

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

# Perturb each by one percent, then try each perturbed
Woutp = Wout*(np.random.rand(Wout.shape[0],Wout.shape[1])*.02 -.01 + 1)
Ap = A*(np.random.rand(A.shape[0],A.shape[1])*.02 -.01 + 1)
Winp = Win*(np.random.rand(Win.shape[0],Win.shape[1])*.02 -.01 + 1)
# First, do baseline
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
errorstore3[0, :, 0] = nerrore_base
for q in range(1, Ntrainrun):
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
    nerrore_base = np.linalg.norm(nerror, axis=1) / np.linalg.norm(
        states[Ntrain + 1 + (q) * Ntest:Ntrain + 1 + (q + 1) * Ntest, :], axis=1)
    errorstore3[0, :, q] = nerrore_base
# A Perturbed
r = np.copy(rstore)
rpred = np.zeros((Ntest,N))
rpred2 = np.zeros((Ntest,N))
r2 = np.copy(r)
for i in range(Ntest):
    r = actfn(Ap @ r + states[Ntrain+i, :] @ Win) # Teacher Forcing
    r3 = nonlin(r2)
    r2 = actfn(Ap @ r2 + r3 @ Wout @ Win)
    rpred[i,:] = r
    rpred2[i,:] = r2
trpred = nonlin(rpred)
trpred2 = nonlin(rpred2)
ResPred2 = trpred2 @ Wout
nerror =  ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:]
nerrore_Apert = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+Ntest+1,:], axis = 1)
errorstore3[1, :, 0] = nerrore_Apert
for q in range(1, Ntrainrun):
    # Redo the next prediction for Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest
    # Need to do (50?) timesteps of true TF input in order to get reservoir warmed up
    r = np.zeros(N)
    rpred = np.zeros((Ntest, N))
    rpred2 = np.zeros((Ntest, N))
    for i in range(50):
        r = actfn(Ap @ r + states[Ntrain + (q) * Ntest - (50 - i), :] @ Win)  # Teacher Forcing
    r2 = np.copy(r)
    for i in range(Ntest):
        r = actfn(Ap @ r + states[Ntrain + (q) * Ntest + i, :] @ Win)
        # r3 = np.copy(r2)
        # for j in range(N // 2):
        #    r3[2 * j] = (r3[2 * j] ** 2).copy()
        r3 = nonlin(r2)
        r2 = actfn(Ap @ r2 + r3 @ Wout @ Win)
        rpred[i, :] = r
        rpred2[i, :] = r2
    trpred2 = nonlin(rpred2)
    ResPred2 = trpred2 @ Wout
    nerror = ResPred2 - states[Ntrain + 1 + (q) * Ntest:Ntrain + 1 + (q + 1) * Ntest, :]
    nerrore_Apert = np.linalg.norm(nerror, axis=1) / np.linalg.norm(
        states[Ntrain + 1 + (q) * Ntest:Ntrain + 1 + (q + 1) * Ntest, :], axis=1)
    errorstore3[1, :, q] = nerrore_Apert
# Win perturbed
r = np.copy(rstore)
rpred = np.zeros((Ntest,N))
rpred2 = np.zeros((Ntest,N))
r2 = np.copy(r)
for i in range(Ntest):
    r = actfn(A @ r + states[Ntrain+i, :] @ Winp) # Teacher Forcing
    r3 = nonlin(r2)
    r2 = actfn(A @ r2 + r3 @ Wout @ Winp)
    rpred[i,:] = r
    rpred2[i,:] = r2
trpred = nonlin(rpred)
trpred2 = nonlin(rpred2)
ResPred2 = trpred2 @ Wout
nerror =  ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:]
nerrore_Winpert = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+Ntest+1,:], axis = 1)
errorstore3[2, :, 0] = nerrore_Winpert
for q in range(1, Ntrainrun):
    # Redo the next prediction for Ntrain+1+(q)*Ntest:Ntrain+1+(q+1)*Ntest
    # Need to do (50?) timesteps of true TF input in order to get reservoir warmed up
    r = np.zeros(N)
    rpred = np.zeros((Ntest, N))
    rpred2 = np.zeros((Ntest, N))
    for i in range(50):
        r = actfn(A @ r + states[Ntrain + (q) * Ntest - (50 - i), :] @ Winp)  # Teacher Forcing
    r2 = np.copy(r)
    for i in range(Ntest):
        r = actfn(A @ r + states[Ntrain + (q) * Ntest + i, :] @ Winp)
        # r3 = np.copy(r2)
        # for j in range(N // 2):
        #    r3[2 * j] = (r3[2 * j] ** 2).copy()
        r3 = nonlin(r2)
        r2 = actfn(A @ r2 + r3 @ Wout @ Winp)
        rpred[i, :] = r
        rpred2[i, :] = r2
    trpred2 = nonlin(rpred2)
    ResPred2 = trpred2 @ Wout
    nerror = ResPred2 - states[Ntrain + 1 + (q) * Ntest:Ntrain + 1 + (q + 1) * Ntest, :]
    nerrore_Winpert = np.linalg.norm(nerror, axis=1) / np.linalg.norm(
        states[Ntrain + 1 + (q) * Ntest:Ntrain + 1 + (q + 1) * Ntest, :], axis=1)
    errorstore3[2, :, q] = nerrore_Winpert
# Wout perturbed
r = np.copy(rstore)
rpred = np.zeros((Ntest,N))
rpred2 = np.zeros((Ntest,N))
r2 = np.copy(r)
for i in range(Ntest):
    r = actfn(A @ r + states[Ntrain+i, :] @ Win) # Teacher Forcing
    r3 = nonlin(r2)
    r2 = actfn(A @ r2 + r3 @ Woutp @ Win)
    rpred[i,:] = r
    rpred2[i,:] = r2
trpred = nonlin(rpred)
trpred2 = nonlin(rpred2)
ResPred2 = trpred2 @ Woutp
nerror =  ResPred2 - states[Ntrain+1:Ntrain+Ntest+1,:]
nerrore_Woutpert = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+Ntest+1,:], axis = 1)
errorstore3[3, :, 0] = nerrore_Woutpert
for q in range(1, Ntrainrun):
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
        r2 = actfn(A @ r2 + r3 @ Woutp @ Win)
        rpred[i, :] = r
        rpred2[i, :] = r2
    trpred2 = nonlin(rpred2)
    ResPred2 = trpred2 @ Woutp
    nerror = ResPred2 - states[Ntrain + 1 + (q) * Ntest:Ntrain + 1 + (q + 1) * Ntest, :]
    nerrore_Woutpert = np.linalg.norm(nerror, axis=1) / np.linalg.norm(
        states[Ntrain + 1 + (q) * Ntest:Ntrain + 1 + (q + 1) * Ntest, :], axis=1)
    errorstore3[3, :, q] = nerrore_Woutpert
# Plot each case
fig = plt.figure()
#plt.plot(nerrore_base)
plt.plot(np.mean(errorstore3[0,:,:],1))
#plt.plot(nerrore_Apert)
plt.plot(np.mean(errorstore3[1,:,:],1))
#plt.plot(nerrore_Winpert)
plt.plot(np.mean(errorstore3[2,:,:],1))
#plt.plot(nerrore_Woutpert)
plt.plot(np.mean(errorstore3[3,:,:],1))
plt.legend(['Base', 'A Perturbed', 'Win Perturbed', 'Wout Perturbed'])
plt.ylabel('Normalized Error')
plt.xlabel('Test Step')
plt.title('ESN Performance for Various Test Perturbations')
plt.show()

print('Perturbations Done')

# This one getting long - do the the 'vary f fn' figures in a new script

# Finish this off with the last case - base (already stored!) vs Wout full random, and A = 0, fn = I (not tanh)
errorstore4 = np.zeros((Ntest,Ntrainrun))
actfn = lambda x: x # Works just as well
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
A*=0

sigma = .5
Win = np.random.rand(dim,N)*2 - 1 # dense
# Experimental - in their version, they make Win black structure, e.g. each neuron only receives one input, not mixed
# Turns out to be very beneficial
#Winc = np.copy(Win)
#for i in range(dim):
#    Win[i,i*(N//dim):(i+1)*(N//dim)]*=2
#Win-=Winc
# New Ankit wants block Identity inputs?
#Win = Win*0
#Win[0,:N//3] = (np.random.rand(N//3)*2-1)*sigma
#Win[1,N//3:2*N//3] = (np.random.rand(2*N//3 - N//3)*2-1)*sigma
#Win[2,2*N//3:] = (np.random.rand(N - 2*N//3)*2-1)*sigma

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
nerrore_simple = np.linalg.norm(nerror, axis = 1)/np.linalg.norm(states[Ntrain+1:Ntrain+Ntest+1,:], axis = 1)
errorstore4[:, 0] = nerrore_simple
for q in range(1, Ntrainrun):
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
    nerrore_simple = np.linalg.norm(nerror, axis=1) / np.linalg.norm(
        states[Ntrain + 1 + (q) * Ntest:Ntrain + 1 + (q + 1) * Ntest, :], axis=1)
    errorstore4[:, q] = nerrore_simple

fig = plt.figure()
#plt.plot(nerrore_base)
plt.plot(np.mean(errorstore3[0,:,:],1))
#plt.plot(nerrore_simple)
plt.plot(np.mean(errorstore4,1))
plt.legend(['Base', 'Simplified'])
plt.ylabel('Normalized Error')
plt.xlabel('Test Step')
plt.title('Base ESN vs Simplified ESN')
plt.show()