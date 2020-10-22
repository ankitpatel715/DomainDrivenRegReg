import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures

# global variables
# This will change the initial condition used. Currently it starts from the first# value
shift_k = 0

# out of memory error with 5k. Try 2.5k
approx_res_size = 5000//2
#approx_res_size = 1500

model_params = {'tau': 0.25,
                'nstep': 1000,
                'N': 8,
                'd': 22}

lsr_res_params = {'radius': .1, #original: .1, pathak: 1.2
              'degree': 3, # original 3, pathak 6
              'sigma': 0.5, # original .5, pathak .1
              'train_length': 500000,
              'N': int(np.floor(approx_res_size / model_params['N']) * model_params['N']),
              'num_inputs': model_params['N'],
              'predict_length': 2000,
              'beta': 0.0001
              }

hsr_res_params = {'radius': 1.2, #original: .1, pathak: 1.2
              'degree': 6, # original 3, pathak 6
              'sigma': 0.1, # original .5, pathak .1
              'train_length': 500000,
              'N': int(np.floor(approx_res_size / model_params['N']) * model_params['N']),
              'num_inputs': model_params['N'],
              'predict_length': 2000,
              'beta': 0.0001
              }

# The ESN functions for training
def generate_reservoir(size, radius, degree):
    sparsity = degree / float(size);
    A = sparse.rand(size, size, density=sparsity).todense()
    vals = np.linalg.eigvals(A)
    e = np.max(np.abs(vals))
    A = (A / e) * radius
    return A


def reservoir_layer(A, Win, inp, res_params):
    states = np.zeros((res_params['N'], res_params['train_length']))
    for i in range(res_params['train_length'] - 1):
        states[:, i + 1] = np.tanh(np.dot(A, states[:, i]) + np.dot(Win, inp[:, i]))
    return states


def train_reservoir(res_params, data):
    A = generate_reservoir(res_params['N'], res_params['radius'], res_params['degree'])
    q = int(res_params['N'] / res_params['num_inputs'])
    Win = np.zeros((res_params['N'], res_params['num_inputs']))
    for i in range(res_params['num_inputs']):
        np.random.seed(seed=i)
        Win[i * q: (i + 1) * q, i] = res_params['sigma'] * (-1 + 2 * np.random.rand(1, q)[0])

    states = reservoir_layer(A, Win, data, res_params)
    Wout = train(res_params, states, data)
    x = states[:, -1]
    return x, Wout, A, Win, states


def train(res_params, states, data):
    beta = res_params['beta']
    idenmat = beta * sparse.identity(res_params['N'])
    states2 = states.copy()
    for j in range(2, np.shape(states2)[0] - 2):
        if (np.mod(j, 2) == 0):
            states2[j, :] = (states[j - 1, :] * states[j - 2, :]).copy()
    U = np.dot(states2, states2.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv, np.dot(states2, data.transpose()))
    return Wout.transpose()


def predict(A, Win, res_params, x, Wout):
    output = np.zeros((res_params['num_inputs'], res_params['predict_length']))
    for i in range(res_params['predict_length']):
        x_aug = x.copy()
        for j in range(2, np.shape(x_aug)[0] - 2):
            if (np.mod(j, 2) == 0):
                x_aug[j] = (x[j - 1] * x[j - 2]).copy()
        out = np.squeeze(np.asarray(np.dot(Wout, x_aug)))
        output[:, i] = out
        x1 = np.tanh(np.dot(A, x) + np.dot(Win, out))
        x = np.squeeze(np.asarray(x1))
    return output, x

# original predict function only works for AUTO, where we provide initial state x
# So we also need a TF predictor, and something to get initial state
# Actually, predictTF can be done for both - first provide previous~50 states (with xin = 0) and use x output...
def predictTF(A, Win, x, Wout, states):
    output = np.zeros((states.shape[0], states.shape[1]))
    for i in range(states.shape[1]):
        x_aug = x.copy()
        for j in range(2, np.shape(x_aug)[0] - 2):
            if (np.mod(j, 2) == 0):
                x_aug[j] = (x[j - 1] * x[j - 2]).copy()
        out = np.squeeze(np.asarray(np.dot(Wout, x_aug)))
        output[:, i] = out
        x1 = np.tanh(np.dot(A, x) + np.dot(Win, states[:,i]))
        x = np.squeeze(np.asarray(x1))
    return output, x

##### New poly functions below

def polypred(state, order): # Given a state vector, output its polynomial expansion 1,x,x^2, order must be 2,3,or 4
    N = len(state)
    size = 1+N # 0 order will have 1 term, 1 order will have N terms
    for i in range(2,order+1):
        comb = combinations_with_replacement(np.arange(N), i)
        combos = list(comb)
        size+=len(combos)
    polyexp = np.zeros(size)
    polyexp[0] = 1
    polyexp[1:N+1] = state[:]
    comb = combinations_with_replacement(np.arange(N), 2)
    combos = list(comb)
    for i in range(len(combos)):
        polyexp[N+1+i] = state[combos[i][0]] * state[combos[i][1]]
    if order > 2:
        comb3 = combinations_with_replacement(np.arange(N), 3)
        combos3 = list(comb3)
        for j in range(len(combos3)):
            polyexp[N+2+i+j] = state[combos3[j][0]] * state[combos3[j][1]] * state[combos3[j][2]]
    if order > 3:
        comb4 = combinations_with_replacement(np.arange(N), 4)
        combos4 = list(comb4)
        for k in range(len(combos4)):
            polyexp[N + 3 + i + j + k] = state[combos4[k][0]] * state[combos4[k][1]] * state[combos4[k][2]] * state[combos4[k][3]]
    return polyexp

def polyfeat(states, order): # Given state vector, turn it into feature vector to fit linear regression with
    # We are using their convention that features[0] = 0, features[1] = polypred(states[0]) etc
    N = np.shape(states)[0]
    size = 1 + N  # 0 order will have 1 term, 1 order will have N terms
    for i in range(2, order + 1):
        comb = combinations_with_replacement(np.arange(N), i)
        combos = list(comb)
        size += len(combos)
    polyfeatures = np.zeros((size,np.shape(states)[1]))
    #for i in range(np.shape(states)[1]-1):
    #    polyfeatures[:,i+1] = polypred(states[:,i],order)
    #return polyfeatures

    #polyfeatures[:,i] = pp.fit_transform(states[:,i].reshape(1, -1)) # no longer off by one - target must now be off by one
    #return polyfeatures
    pp = PolynomialFeatures(degree=order)
    #polyfeatures2 = (pp.fit_transform(states.transpose())).transpose()
    #return polyfeatures2
    for i in range(np.shape(states)[1]):
        polyfeatures[:, i] = pp.fit_transform(states[:, i].reshape(1,-1))
    return polyfeatures

def polytrain(features, beta, size, data):
    idenmat = beta * sparse.identity(size)
    U = np.dot(features, features.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv, np.dot(features, data.transpose()))
    return Wout.transpose()

def polyauto(startstate,Wout,order, autotime):
    N = len(startstate)
    state = startstate
    predictions = np.zeros((N, autotime))
    for i in range(autotime):
        polyfeatures = polypred(state,order)
        state = np.array(np.dot(Wout,polyfeatures)).reshape(N)
        predictions[:,i] = state
    return predictions

def polytf(states,Wout, order):
    N = states.shape[0]
    tftime = states.shape[1]
    predictions = np.zeros((N,tftime))
    for i in range(tftime):
        polyfeatures = polypred(states[:,i], order)
        state = np.array(np.dot(Wout, polyfeatures)).reshape(N)
        predictions[:, i] = state
    return predictions


def polyrun(data,res_params,dw_pct,drop_weights=False):

    offset = 1 # set to 1 if using Devika new formulation
    order = 4  # 2 to 4 for quadratic to quartic
    size = 1+8

    for i in range(2, order + 1):
        comb = combinations_with_replacement(np.arange(8), i)
        combos = list(comb)
        size += len(combos)

    polyfeatures = polyfeat(data[:, shift_k:shift_k + res_params['train_length']],order)
    print('Poly Feature Done')

    Wout = polytrain(polyfeatures,res_params['beta'],size,data[:, shift_k+offset:shift_k+offset + res_params['train_length']])
    print('Poly Wout Done')

    # Apply lower (bit) precision if interested. Note that lower bit precision only used during prediction, not training

    # Wout precision
    W32prec = np.single
    W16prec = np.half
    # Poly Features
    pf32prec = np.single
    pf16prec = np.half
    # Start Data
    sd32prec = np.single
    sd16prec = np.half


    # do all the combos: all_squash, all_but_Wout, only_Wout

    for mode in ['all_squash','only_Wout','all_but_Wout']:
        if mode=='all_squash':
            Wout32 = np.copy(Wout).astype(W32prec)
            Wout16 = np.copy(Wout).astype(W16prec)
            polyfeatures32 = np.copy(polyfeatures).astype(pf32prec)
            polyfeatures16 = np.copy(polyfeatures).astype(pf16prec)

        if mode=='only_Wout':
            Wout32 = np.copy(Wout).astype(W32prec)
            Wout16 = np.copy(Wout).astype(W16prec)
            polyfeatures32 = np.copy(polyfeatures)
            polyfeatures16 = np.copy(polyfeatures)
            sd32prec = np.double
            sd16prec = np.double
            
        if mode=='all_but_Wout':
            Wout32 = np.copy(Wout)
            Wout16 = np.copy(Wout)
            polyfeatures32 = np.copy(polyfeatures).astype(pf32prec)
            polyfeatures16 = np.copy(polyfeatures).astype(pf16prec)


        # New - Krishna wants dropping terms or orders
        if drop_weights:
            Wout32 = np.copy(Wout)
            Wout32[np.abs(Wout32)<np.percentile(np.abs(Wout32), dw_pct)] = 0 # bottom 15% set to 0
            Wout16 = np.copy(Wout)
            Wout16[np.abs(Wout16)<np.percentile(np.abs(Wout16), dw_pct)] = 0 # bottom 20% set to 0

        trainout = Wout @ polyfeatures
        trainerr = trainout - data[:,shift_k+offset:shift_k+offset+res_params['train_length']]
        trainerre = np.linalg.norm(trainerr,axis=0) / np.linalg.norm(data[:,shift_k+offset:shift_k+offset+res_params['train_length']], axis = 0)

        trainout32 = Wout32 @ polyfeatures32
        trainerr32 = trainout32 - data[:, shift_k + offset:shift_k + offset + res_params['train_length']]
        trainerre32 = np.linalg.norm(trainerr32, axis=0) / np.linalg.norm(
            data[:, shift_k + offset:shift_k + offset + res_params['train_length']], axis=0)

        trainout16 = Wout16 @ polyfeatures16
        trainerr16 = trainout16 - data[:, shift_k + offset:shift_k + offset + res_params['train_length']]
        trainerre16 = np.linalg.norm(trainerr16, axis=0) / np.linalg.norm(
            data[:, shift_k + offset:shift_k + offset + res_params['train_length']], axis=0)

        # For each of 100 different starting positions, do a TF and AUTO run, and store final (mean) normed errors

        offsets = np.arange(0,200000,2000)
        tabledat = np.zeros((3,len(offsets)))
        tabledat32 = np.zeros((3, len(offsets)))
        tabledat16 = np.zeros((3, len(offsets)))

        # added data structure for maintaining errors -- devika
    
        poly_tf_errors = np.zeros((2000,len(offsets)))
        poly_auto_errors = np.zeros((2000,len(offsets)))

        poly_tf_errors32 = np.zeros((2000,len(offsets)))
        poly_auto_errors32 = np.zeros((2000,len(offsets)))

        poly_tf_errors16 = np.zeros((2000,len(offsets)))
        poly_auto_errors16 = np.zeros((2000,len(offsets)))


        # do the 100 prediction runs with different starting points (after training run)
        for i in range(len(offsets)):
            starttime = shift_k + res_params['train_length']+offsets[i]

            startdat = np.copy(data[:,starttime:starttime+2000])
            startdat32 = np.copy(startdat).astype(sd32prec)
            startdat16 = np.copy(startdat).astype(sd16prec)

            polytfpred = polytf(startdat,Wout,order)
            polyautopred = polyauto(startdat[:,0],Wout,order,2000)

            polytferr = polytfpred - data[:,starttime+1:starttime+2001]
            polyautoerr = polyautopred - data[:,starttime+1:starttime+2001]
            polytferre = np.linalg.norm(polytferr,axis = 0) / np.linalg.norm(data[:,starttime+1:starttime+2001], axis = 0)
            polyautoerre = np.linalg.norm(polyautoerr, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)

            tabledat[0,i] = np.mean(polytferre)
            tabledat[1,i]   = np.mean(polyautoerre)
            tabledat[2,i] = np.where(polyautoerre>.3)[0][0] # raw number of timesteps - not divided by 200 yet

            # keep the error curves -- devika
            poly_tf_errors[:,i] = polytferre
            poly_auto_errors[:,i] = polyautoerre
        
            # 32 bit version
            polytfpred32 = polytf(startdat32, Wout32, order)
            polyautopred32 = polyauto(startdat32[:, 0], Wout32, order, 2000)

            polytferr32 = polytfpred32 - data[:, starttime + 1:starttime + 2001]
            polyautoerr32 = polyautopred32 - data[:, starttime + 1:starttime + 2001]
            polytferre32 = np.linalg.norm(polytferr32, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
            polyautoerre32 = np.linalg.norm(polyautoerr32, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)

            tabledat32[0, i] = np.mean(polytferre32)
            tabledat32[1, i] = np.mean(polyautoerre32)
            tabledat32[2, i] = np.where(polyautoerre32 > .3)[0][0]  # raw number of timesteps - not divided by 200 yet

            # keep the error curves -- devika
            poly_tf_errors32[:,i] = polytferre32
            poly_auto_errors32[:,i] = polyautoerre32
            
            # 16 bit version
            polytfpred16 = polytf(startdat16, Wout16, order)
            polyautopred16 = polyauto(startdat16[:, 0], Wout16, order, 2000)

            polytferr16 = polytfpred16 - data[:, starttime + 1:starttime + 2001]
            polyautoerr16 = polyautopred16 - data[:, starttime + 1:starttime + 2001]
            polytferre16 = np.linalg.norm(polytferr16, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
            polyautoerre16 = np.linalg.norm(polyautoerr16, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)

            tabledat16[0, i] = np.mean(polytferre16)
            tabledat16[1, i] = np.mean(polyautoerre16)
            tabledat16[2, i] = np.where(polyautoerre16 > .3)[0][0]  # raw number of timesteps - not divided by 200 yet

            # keep the error curves -- devika
            poly_tf_errors16[:,i] = polytferre16
            poly_auto_errors16[:,i] = polyautoerre16

        # plot the normalized error over 100 runs
        fig, ax = plt.subplots()
        ax.plot(np.mean(poly_auto_errors,axis=1),label='64 bit')
        ax.plot(np.mean(poly_auto_errors32,axis=1), label='32 bit')
        ax.plot(np.mean(poly_auto_errors16,axis=1), label = '16 bit')
        ax.legend(loc='upper left')
        plt.xlabel('Testing Step')
        plt.ylabel('Relative (Norm) Error')
        plt.title('Lorenz96 for varying precision')
        plt.xlim([0,500])
        plt.ylim([0,2.0])
        plt.savefig("./experiments/D2R2/"+mode+"/d2r2.png")

        # save all the computed relative errors as CSV files


        np.savetxt("./experiments/D2R2/"+mode+"/tabledat.csv",tabledat,delimiter=",")
        np.savetxt("./experiments/D2R2/"+mode+"/tabledat32.csv",tabledat32,delimiter=",")
        np.savetxt("./experiments/D2R2/"+mode+"/tabledat16.csv",tabledat16,delimiter=",")

        np.savetxt("./experiments/D2R2/"+mode+"/poly_auto_errors.csv",poly_auto_errors,delimiter=",")
        np.savetxt("./experiments/D2R2/"+mode+"/poly_auto_errors32.csv",poly_auto_errors32,delimiter=",")
        np.savetxt("./experiments/D2R2/"+mode+"/poly_auto_errors16.csv",poly_auto_errors16,delimiter=",")
        np.savetxt("./experiments/D2R2/"+mode+"/poly_tf_errors.csv",poly_tf_errors,delimiter=",")
        np.savetxt("./experiments/D2R2/"+mode+"/poly_tf_errors32.csv",poly_tf_errors32,delimiter=",")
        np.savetxt("./experiments/D2R2/"+mode+"/poly_tf_errors16.csv",poly_tf_errors16,delimiter=",")

        np.savetxt("./experiments/D2R2/"+mode+"/poly_train_err.csv",trainerre,delimiter=",")
        np.savetxt("./experiments/D2R2/"+mode+"/poly_train_err32.csv",trainerre32,delimiter=",")
        np.savetxt("./experiments/D2R2/"+mode+"/poly_train_err16.csv",trainerre16,delimiter=",")



    


# run experiments on LSR-ESN

def esnrun(data,esn_type,res_params,dw_pct,drop_weights=False):

    x, Wout, A, Win, trainstates = train_reservoir(res_params, data[:, shift_k:shift_k + res_params['train_length']])
    print("Training Done")

    # Get training err
    trainstates2 = np.copy(trainstates)
    for j in range(2, np.shape(trainstates2)[0] - 2):
        if (np.mod(j, 2) == 0):
            trainstates2[j, :] = (trainstates[j - 1, :] * trainstates[j - 2, :]).copy()
    del trainstates


    W32prec = np.single
    W16prec = np.half
    Win32prec = np.single
    Win16prec = np.half
    A32prec = np.single
    A16prec = np.half
    
    # do all the combos: all_squash, all_but_Wout, only_Wout

    for mode in ['all_squash','only_Wout','all_but_Wout']:
        if mode=='all_squash':
            Wout32 = np.copy(Wout).astype(W32prec)
            Wout16 = np.copy(Wout).astype(W16prec)
            Win32 = np.copy(Win).astype(Win32prec)
            Win16 = np.copy(Win).astype(Win16prec)
            A32 = np.copy(A).astype(A32prec)
            A16 = np.copy(A).astype(A16prec)

            ef32prec = np.single
            ef16prec = np.half
            ss32prec = np.single
            ss16prec = np.half
            sd32prec = np.single
            sd16prec = np.half
            

        if mode=='only_Wout':
            Wout32 = np.copy(Wout).astype(W32prec)
            Wout16 = np.copy(Wout).astype(W16prec)
            Win32 = np.copy(Win)
            Win16 = np.copy(Win)
            A32 = np.copy(A)
            A16 = np.copy(A)

            ef32prec = np.double
            ef16prec = np.double
            ss32prec = np.double
            ss16prec = np.double
            sd32prec = np.double
            sd16prec = np.double


        if mode=='all_but_Wout':
            Wout32 = np.copy(Wout)
            Wout16 = np.copy(Wout)
            Win32 = np.copy(Win).astype(Win32prec)
            Win16 = np.copy(Win).astype(Win16prec)
            A32 = np.copy(A).astype(A32prec)
            A16 = np.copy(A).astype(A16prec)

            ef32prec = np.single
            ef16prec = np.half
            ss32prec = np.single
            ss16prec = np.half
            sd32prec = np.single
            sd16prec = np.half

        trainout = Wout @ trainstates2
        trainerr = trainout - data[:, shift_k:shift_k + res_params['train_length']] # no need to increment by one here
        trainerre = np.linalg.norm(trainerr, axis=0) / np.linalg.norm(data[:, shift_k:shift_k + res_params['train_length']], axis=0)

        trainstates2_32 = np.copy(trainstates2).astype(ef32prec)
        trainout32 = Wout32 @ trainstates2_32
        trainerr32 = trainout32 - data[:, shift_k:shift_k + res_params['train_length']]  # no need to increment by one here
        trainerre32 = np.linalg.norm(trainerr32, axis=0) / np.linalg.norm(data[:, shift_k:shift_k + res_params['train_length']], axis=0)
        del trainstates2_32

        trainstates2_16 = np.copy(trainstates2).astype(ef16prec)
        trainout16 = Wout16 @ trainstates2_16
        trainerr16 = trainout16 - data[:, shift_k:shift_k + res_params['train_length']]  # no need to increment by one here
        trainerre16 = np.linalg.norm(trainerr16, axis=0) / np.linalg.norm(data[:, shift_k:shift_k + res_params['train_length']], axis=0)
        del trainstates2_16

        # Prediction
        offsets = np.arange(0,200000,2000)
        tabledat = np.zeros((3, len(offsets)))
        tabledat32 = np.zeros((3, len(offsets)))
        tabledat16 = np.zeros((3, len(offsets)))

        # added data structure for maintaining errors -- devika
        esn_tf_errors = np.zeros((2000,len(offsets)))
        esn_auto_errors = np.zeros((2000,len(offsets)))

        esn_tf_errors32 = np.zeros((2000,len(offsets)))
        esn_auto_errors32 = np.zeros((2000,len(offsets)))

        esn_tf_errors16 = np.zeros((2000,len(offsets)))
        esn_auto_errors16 = np.zeros((2000,len(offsets)))

        for i in range(len(offsets)):
            print('Starting condition ', i)
            starttime = shift_k + res_params['train_length'] + offsets[i]
            priordat = np.copy(data[:,starttime-50:starttime])

            _, startstate = predictTF(A,Win, x*0,Wout,data[:,starttime-50:starttime])
            startdat = np.copy(data[:, starttime:starttime+res_params['predict_length']])

            output, _ = predict(A, Win, res_params, startstate, Wout)
            output2, _ = predictTF(A, Win, startstate, Wout, startdat)
            tferr = output2 - data[:, starttime + 1:starttime + 2001]
            autoerr = output - data[:, starttime + 1:starttime + 2001]
            tferre = np.linalg.norm(tferr, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
            autoerre = np.linalg.norm(autoerr, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)

            tabledat[0, i] = np.mean(tferre)
            tabledat[1, i] = np.mean(autoerre)
            tabledat[2, i] = np.where(autoerre > .3)[0][0]  # raw number of timesteps - not divided by 200 yet

            #32 version

            startstate32 = np.copy(startstate).astype(ss32prec)
            startdat32 = np.copy(startdat).astype(sd32prec) # Data - should this be full?
            output32, _ = predict(A32, Win32, res_params, startstate32, Wout32)
            output2_32, _ = predictTF(A32, Win32, startstate32, Wout32, startdat32)
            tferr32 = output2_32 - data[:, starttime + 1:starttime + 2001]
            autoerr32 = output32 - data[:, starttime + 1:starttime + 2001]
            tferre32 = np.linalg.norm(tferr32, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
            autoerre32 = np.linalg.norm(autoerr32, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)

            tabledat32[0, i] = np.mean(tferre32)
            tabledat32[1, i] = np.mean(autoerre32)
            tabledat32[2, i] = np.where(autoerre32 > .3)[0][0]  # raw number of timesteps - not divided by 200 yet

            # 16 version

            startstate16 = np.copy(startstate).astype(ss16prec)
            startdat16 = np.copy(startdat).astype(sd16prec)
            output16, _ = predict(A16, Win16, res_params, startstate16, Wout16)
            output2_16, _ = predictTF(A16, Win16, startstate16, Wout16, startdat16)
            tferr16 = output2_16 - data[:, starttime + 1:starttime + 2001]
            autoerr16 = output16 - data[:, starttime + 1:starttime + 2001]
            tferre16 = np.linalg.norm(tferr16, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
            autoerre16 = np.linalg.norm(autoerr16, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)

            tabledat16[0, i] = np.mean(tferre16)
            tabledat16[1, i] = np.mean(autoerre16)
            tabledat16[2, i] = np.where(autoerre16 > .3)[0][0]  # raw number of timesteps - not divided by 200 yet

            # keep the error curves
            esn_tf_errors[:,i] = tferre
            esn_auto_errors[:,i] = autoerre

            esn_tf_errors32[:,i] = tferre32
            esn_auto_errors32[:,i] = autoerre32

            esn_tf_errors16[:,i] = tferre16
            esn_auto_errors16[:,i] = autoerre16


        # plot the normalized error over 100 runs
        fig, ax = plt.subplots()
        ax.plot(np.mean(esn_auto_errors,axis=1),label='64 bit')
        ax.plot(np.mean(esn_auto_errors32,axis=1), label='32 bit')
        ax.plot(np.mean(esn_auto_errors16,axis=1), label = '16 bit')
        ax.legend(loc='upper left')
        plt.xlabel('Testing Step')
        plt.ylabel('Relative (Norm) Error')
        plt.title('Lorenz96 for varying precision')
        plt.xlim([0,500])
        plt.ylim([0,2.0])
        plt.savefig("./experiments/" + esn_type + "/" + mode + '/esn.png')

        # save all the computed relative errors as CSV files


        np.savetxt("./experiments/" + esn_type +"/"+mode+"/tabledat.csv",tabledat,delimiter=",")
        np.savetxt("./experiments/"+ esn_type +"/"+mode+"/tabledat32.csv",tabledat32,delimiter=",")
        np.savetxt("./experiments/"+ esn_type +"/"+mode+"/tabledat16.csv",tabledat16,delimiter=",")

        np.savetxt("./experiments/"+ esn_type +"/"+mode+"/esn_auto_errors.csv",esn_auto_errors,delimiter=",")
        np.savetxt("./experiments/"+ esn_type +"/"+mode+"/esn_auto_errors32.csv",esn_auto_errors32,delimiter=",")
        np.savetxt("./experiments/"+ esn_type +"/"+mode+"/esn_auto_errors16.csv",esn_auto_errors16,delimiter=",")
        np.savetxt("./experiments/"+ esn_type +"/"+mode+"/esn_tf_errors.csv",esn_tf_errors,delimiter=",")
        np.savetxt("./experiments/"+ esn_type +"/"+mode+"/esn_tf_errors32.csv",esn_tf_errors32,delimiter=",")
        np.savetxt("./experiments/"+ esn_type +"/"+mode+"/esn_tf_errors16.csv",esn_tf_errors16,delimiter=",")

        np.savetxt("./experiments/"+ esn_type +"/"+mode+"/esn_train_err.csv",trainerre,delimiter=",")
        np.savetxt("./experiments/"+ esn_type +"/"+mode+"/esn_train_err32.csv",trainerre32,delimiter=",")
        np.savetxt("./experiments/"+ esn_type +"/"+mode+"/esn_train_err16.csv",trainerre16,delimiter=",")
            



dataf = pd.read_csv('3tier_lorenz_v3.csv', header=None)
data = np.transpose(np.array(dataf)) # data is 8x1M


# run experiments on D2R2 (all modes)

#polyrun(data,lsr_res_params,0,drop_weights=False)

#esnrun(data,'ESN_HSR',hsr_res_params,0,drop_weights=False)

esnrun(data,'ESN_LSR',lsr_res_params,0,drop_weights=False)
