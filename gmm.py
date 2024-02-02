#from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.ones((M,1))
        self.mu = np.ones((M,d))
        self.Sigma = np.ones((M,d))


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout
    
        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    #getting dimensions from data
    d = myTheta.mu.shape[1]
    M = myTheta.mu.shape[0]

    result = np.float32(0)
    bleh = np.float32(1)

    if preComputedForM == []: #no precomputed terms, do full computation
        for n in range(0, d):
            bleh *= myTheta.Sigma[m, n] ** 2
            result -= (((x[n] - myTheta.mu[m, n])**2) / (2* (myTheta.Sigma[m, n]**2)))
        result -= (d/2)*np.log(2*np.pi)
        result -= (1/2)*np.log(bleh)

    else: #use precomputed float term
        for n in range(0, d):
            result -= 1/2*(x[n]**2)*(myTheta.Sigma[m,n]**-2)  - (myTheta.mu[m, n]*x[n]*(myTheta.Sigma[m, n]**-2))
        result -= preComputedForM 
        
    return result



    
def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    #log(wm) + log(bmx) - logsumexp(bkx, wkx)
    d = myTheta.mu.shape[1]
    M = myTheta.mu.shape[0]

    lgbmx = log_b_m_x(m, x, myTheta)
    lgw = np.log(myTheta.omega[m])
    lgbmxs = np.zeros((M))

    for i in range(0, M):
        lgbmxs[i] =  log_b_m_x(i, x, myTheta)
    sumwkbk = logsumexp(lgbmxs, b= myTheta.omega) #not sure why axis is needed... check this later

    return lgbmxs + lgw - sumwkbk


    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    #need axis, sum along different t's
    return np.sum(logsumexp(log_Bs, b= myTheta.omega, axis = 0)) #sum per xt, then gather probabilities and return

    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)
    Note for self: X dim = (T,D)'''
    D = X.shape[1]
    myTheta = theta( speaker, M, X.shape[1] )
    T = X.shape[0]
    improvement = np.inf
    prev_L = -1*np.inf
    iteration = 0
    while iteration <= maxIter and improvement > epsilon:
        logbmxs = np.zeros((M, T))
        logpmxs = np.zeros((M, T))
        for i in range(M):
            for j in range(T): #populate the (m, t)
                logbmxs[i,j] = log_b_m_x(i, X[j], myTheta)
                print("askjdhaksjdsahkj")
                print(log_p_m_x(i, X[j], myTheta))
                logpmxs[i,j] =log_p_m_x(i, X[j], myTheta)

        #exponentiate the arrays
        bmxs = np.exp(logbmxs)
        pmxs = np.exp(logpmxs)
        #get current L
        L = loglik(logbmxs, myTheta) 
        #update theta, updating each m'th term
        for i in range(M):
            pmx_sum = np.sum(pmxs[i])
            myTheta.omega[i][:] = np.divide(pmx_sum, T)

            muNum = np.zeros((D))
            for j in range(T):
                muNum += np.dot(pmxs[i][j], X[j])
            myTheta.mu[i] = np.divide(muNum, pmx_sum)


            sigNum = np.zeros((D))
            for j in range(T):
                sigNum += np.dot(pmxs[i][j], np.multiply(X[j], X[j]))
            myTheta.Sigma[i] = np.divide(sigNum, pmx_sum) - np.multiply(myTheta.mu[i], myTheta.mu[i])

        #recalc improvement
        improvement = L - prev_L
        prev_L = L        
    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    oustlst = []
    bestModel = -1
    bestProb = -1*np.inf
    for model in models:
        logbmxs = np.zeros((M, T))
        for i in range(M):
            for j in range(T): #populate the (m, t)
                logpmxs[i,j] =log_p_m_x(i, x[j], model)
        prob = loglik(logpmxs, model)
        #stuff to write to file
        outlst.append(model.name, prob)
        if prob > bestProb:
            bestProb = prob
            bestModel = model.name
    sorted = outlst.sort(key = lambda tup: tup[1])

    with open(f"{os.getcwd()}/gmmLiks.txt", "w") as outf:
        outf.write(f'[{correctID}]\n')
        for i in range(k):
            print("alllaaaaahh")
            outf.write(f'[{outlst[-1-i][0]}] [{outlst[-1-i][1]}]:\n')
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)

