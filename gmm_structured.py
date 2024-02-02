#from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp
dataDir = "/u/cs401/A3/data/"


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        #returns float

        sum_term = np.sum((self.mu[m]**2)/ (2*self.Sigma[m]))
        product_term = np.prod(self.Sigma[m])
        sum_term += (self._d/2)*np.log(2*np.pi)
        sum_term += 0.5 * np.log( product_term)
        return sum_term

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    d = myTheta.mu.shape[1]
    M = myTheta.mu.shape[0]
    result = np.zeros(0)
    bleh = np.float64(1)
    if len(x.shape) == 1:
        result = -1* np.sum(1/2*(x**2) /  np.maximum(1e-9, myTheta.Sigma[m]) - (myTheta.mu[m]*x) /  np.maximum(1e-9, myTheta.Sigma[m]))
    else:
        T = x.shape[0]
        result =  -1*np.sum((1/2*(x**2)) / np.maximum(1e-9, myTheta.Sigma[m]) - (myTheta.mu[m]*x) /  np.maximum(1e-9, myTheta.Sigma[m]), axis = 1)
        

    #print(myTheta.precomputedForM(m))
    return result - myTheta.precomputedForM(m)
    


def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)
    
    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout
    z
    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """

    lower = logsumexp(log_Bs, axis = 0, b = myTheta.omega)
    T = log_Bs.shape[1]
    M = log_Bs.shape[0]
    cpy = np.copy(log_Bs)
    #print(cpy + np.log(np.maximum(1e-8, myTheta.omega)) - lower.T)

    stuff = cpy + np.log(np.maximum(1e-8, myTheta.omega)) - lower.T

    stoppls = stuff > 0
    stuff[stoppls] = 0
    return stuff
    


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    return np.sum(logsumexp(log_Bs, b=myTheta.omega, axis=0))



def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    # perform initialization (Slide 32) 
    
    # for ex.,
    # myTheta.reset_omega(omegas_with_constraints)
    # myTheta.reset_mu(mu_computed_using_data) 
    # myTheta.reset_Sigma(some_appropriate_sigma)
    #Using the recommendations on slide 19 here : 
    d = X.shape[1]
    muFromData = np.zeros((M, d))
    Xcopy = np.copy(X) #not sure if aliasing will be problem
    indices = [x for x in range(Xcopy.shape[0])]
    random.shuffle(indices)
    for i in range(M):
        muFromData[i] = Xcopy[indices.pop()] 
    #sigma starting as ID matrices()
    appropSigma = np.ones((M, d))

    omegaWeights = random.sample(range(0,100), M)
    omegas_with_constraints = np.float64(omegaWeights)
    omegas_with_constraints /= np.sum(omegas_with_constraints)

    myTheta.reset_omega(omegas_with_constraints)
    myTheta.reset_mu(muFromData) 
    myTheta.reset_Sigma(appropSigma)

    D = X.shape[1]
    T = X.shape[0]
    improvement = np.inf
    prev_L = -1*np.inf
    iteration = 0
    while iteration < maxIter and improvement >= epsilon:
        iteration += 1
        logbmxs = np.zeros((M, T))
        for i in range(M):
            logbmxs[i] = log_b_m_x(i, X, myTheta)   
        logpmxs = log_p_m_x(logbmxs, myTheta)
        #print(np.amax(logpmxs))
        #exponentiate the arrays
        pmxs = np.exp(logpmxs)
        #get current L
        L = logLik(logbmxs, myTheta) 
        #print(L)
        #update theta, updating each m'th term

        #pmx_sums = np.sum(pmxs, axis = 1) #summed per m
        #myTheta.omega = pmx_sums / T
        for i in range(M):
            pmx_sum = np.sum(pmxs[i])
            myTheta.omega[i] = np.divide(pmx_sum, T) #checked

            myTheta.mu[i] = np.sum(X.T*pmxs[i], axis = 1) / np.maximum(1e-8, pmx_sum) #checked

            myTheta.Sigma[i] = np.sum((X*X).T*pmxs[i], axis = 1) / np.maximum(1e-8, pmx_sum) - np.multiply(myTheta.mu[i], myTheta.mu[i])

        #recalc improvement
        improvement = L - prev_L
        prev_L = L        

    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]  
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    T = mfcc.shape[0]
    outlst = []
    bestModel = -1
    bestProb = -1*np.inf
    count = 0
    for model in models:
        logbmxs = np.zeros((M, T))
        for m in range(M):
            logbmxs[m] = log_b_m_x(m, mfcc, model)
        logpmxs = log_p_m_x(logbmxs, model)
        prob = logLik(logpmxs, model)
        #stuff to write to file
        outlst.append((model.name, prob))
        if prob < bestProb:
            bestProb = prob
            bestModel = count
        count += 1
    outlst.sort(key = lambda tup: tup[1])
    with open(f"{os.getcwd()}/gmmLiks.txt", "a") as outf:
        outf.write(f'[{correctID}]\n')
        for i in range(k): #K CANT BE BIGGER OR EQUAL TO NUM SPEAKERS OUT OF RANGE
            outf.write(f'[{outlst[-1-i][0]}] [{outlst[-1-i][1]}]:\n')
    #print(str(bestModel) + " AND CORRECT ANSWER WAS " +str(correctID))
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 5
    epsilon = 0.0
    maxIter = 15
    maxSpeakers = 32

    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            maxSpeakers -= 1
            if maxSpeakers <= 0:
                break
            #print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print( "Correct#: " + str(numCorrect) + " Accuracy: " + str(accuracy))
