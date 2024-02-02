import os, fnmatch
import numpy as np
import string

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> Levenshtein("who is there".split(), "is there".split())
    0.333 0 0 1                                                                           
    >>> Levenshtein("who is there".split(), "".split())
    1.0 0 0 3                                                                           
    >>> Levenshtein("".split(), "who is there".split())
    Inf 0 3 0                                                                           
    """
    r.insert(0, "<s>")
    h.insert(0, "<s>")
    r.append("</s>")
    h.append("</s>")
    R = len(r)
    H = len(h)
    lev_matrix = np.zeros((R, H))
    backtrace = np.zeros((R, H)) #store direction of prev min state: 0=diagonal-left, 1=left, 2=up 
    backtrace[1:, 0] = 2
    backtrace[0, 1:] = 1
    lev_matrix[0] = np.array([x for x in range(H)])
    lev_matrix[:, 0] = np.array([x for x in range(R)])

    for i in range(1, R):
        for j in range(1, H):
            up = lev_matrix[i-1, j] + 1
            left = lev_matrix[i, j-1] + 1
            diag = 0
            if r[i] == h[j]:
                diag = lev_matrix[i-1, j-1] 
            else:
                diag = lev_matrix[i-1, j-1] + 1
            best = np.amin([up, left, diag], axis=0)
            lev_matrix[i, j] = best

            if up == best:
                backtrace[i, j] = 2
            if left == best:
                backtrace[i, j] = 1
            if diag == best:
                backtrace[i, j] = 0

    #follow the cookie crumbs back to origin
    del_count = 0
    ins_count = 0
    sub_count = 0
    cords = [R-1, H-1]

    while cords != [0, 0]:
        if backtrace[cords[0], cords[1]] == 2.0: #up/delete
            del_count += 1
            cords = [cords[0]-1, cords[1]]
        elif backtrace[cords[0], cords[1]] == 1.0: #left/insert
            ins_count += 1
            cords = [cords[0], cords[1]-1]
        elif backtrace[cords[0], cords[1]] == 0.0:
            if r[cords[0]] != h[cords[1]]:
                sub_count += 1 #tks for sub
            cords = [cords[0]-1, cords[1]-1]
    WER = 0
    if R-2 == 0:
        WER = np.inf
    else:   
        WER = 1.0 * lev_matrix[R-1, H-1] / (R-2)

    return (WER, sub_count, ins_count, del_count)

def de_punc(a):
    #a quick helper function that removes all punctuation from a string
    b = ""
    for char in a:
        if char not in string.punctuation:  
            b += char        
    return b

if __name__ == "__main__":
    GWER_sum = np.zeros((0))
    KWER_sum = np.zeros((0))
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:


            #get the file path for each of the three transcripts
            Google = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*Google.txt")
            Kaldi = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*Kaldi.txt")
            Transcript = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*transcripts.txt")
            
            Gfile = open(os.path.join(dataDir, speaker, Google[0]), 'r')
            Tfile = open(os.path.join(dataDir, speaker, Transcript[0]), 'r')
            Kfile = open(os.path.join(dataDir, speaker, Kaldi[0]), 'r')

            Glines = Gfile.readlines()
            Tlines = Tfile.readlines()
            Klines = Kfile.readlines()

            Glst = []
            Tlst = []
            Klst = []

            #lines ready to check
            for line in Glines:
                Glst.append(de_punc(str(line)).split())

            for line in Tlines:
                Tlst.append(de_punc(str(line)).split())

            for line in Klines:
                Klst.append(de_punc(str(line)).split())

            #lets do the google ones first:
            for i in range(len(Tlst)):
                WER, sub, ins, deletes = Levenshtein(Tlst[i], Glst[i])
                #PIPE THIS INTO A FILE DURING RUNTIME!!!!!!!!!!!!!!!!!!!!!!!!
                print("[" + str(speaker) +  "] [Google] [" + str(i) + "] [" + str(WER) + "] S: [" + str(sub) + "], I: [" + str(ins) + "], D: [" + str(deletes) +"]")
                #if np.isfinite(WER) and 0 <= WER <= 5:
                    #GWER_sum = np.append(GWER_sum, WER)
            for i in range(len(Tlst)):
                WER, sub, ins, deletes = Levenshtein(Tlst[i], Klst[i])
                #PIPE THIS INTO A FILE DURING RUNTIME!!!!!!!!!!!!!!!!!!!!!!!!
                print("[" + str(speaker) +  "] [Kaldi] [" + str(i) + "] [" + str(WER) + "] S: [" + str(sub) + "], I: [" + str(ins) + "], D: [" + str(deletes) +"]")
                #if np.isfinite(WER) and 0<= WER <= 5:
                    #KWER_sum = np.append(KWER_sum, WER)


    #Gavg = np.sum(GWER_sum)/len(GWER_sum)
    #Kavg = np.sum(KWER_sum)/len(KWER_sum)

    #Gsd = np.std(GWER_sum)
    #Ksd = np.std(KWER_sum)

    #print("GOOGLE avg: " + str(Gavg) + " std: " + str(Gsd))
    #print("Kaldi avg: " + str(Kavg) + " std: " + str(Ksd))