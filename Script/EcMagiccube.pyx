import numpy as np
import copy
cimport numpy as np

def CoverDepth(list Q, list R):
    cdef int L = R[1] - R[0] + 1
    cdef np.ndarray[np.int64_t, ndim=1] N = np.zeros(L, dtype=np.int64)
    cdef list i, j
    cdef float C, D

    for i in Q:
        for j in i :
            N[j[0]-R[0]:j[1]-R[0]+1] += 1

    C = N[N>0].shape[0]/len(N)
    D = sum(N)/len(N)
    return [C, D]

def InterV2(list intvs):
    """
    :param intvs: List[List[int]]
    :return: List[List[int]]
    """
    intvs = sorted(intvs)
    if  intvs[0][1] < intvs[1][0]:
        return intvs
    else:
        return [[ intvs[0][0], max(intvs[0][1], intvs[1][1])]]

def InterVs(list intvs):
    """
    :param intvs: List[List[int]]
    :return: List[List[int]]
    """
    cdef list Inters, merged, intv
    Inters = sorted(intvs, key=lambda x:x[0])
    merged = [ copy.deepcopy(Inters[0]) ]
    for intv in Inters[1:]:
        if  merged[-1][-1] < intv[0]:
            merged.append(intv)
        else:
            merged[-1][-1] = max(merged[-1][-1], intv[-1])
    return merged

def InterSm(list intvs):
    return sum(map(lambda x:x[1]-x[0]+1, intvs))

def MaxBetween(np.ndarray inmap, int maxd):
    '''
    #row columns ['#chrom', 'start', 'end', 'forword', 'start_n', 'end_n']
    # input numpy must be sorted: inmap= inmap[np.lexsort((inmap[:,2], inmap[:,1]))]
    '''
    cdef np.ndarray l, B
    for l in inmap:
        B = (inmap[:,-2] >= l[1] - maxd) & (inmap[:,-2] <= l[1] + maxd) & \
            (inmap[:,-1] >= l[2] - maxd) & (inmap[:,-1] <= l[2] + maxd)
        if inmap[B].size!=0:
            inmap[B, -2] = inmap[B, -2].min()
            inmap[B, -1] = inmap[B, -1].max()
    return inmap

def OrderLinks(np.ndarray _G):
    ''''
    #row columns ['#chrom', 'start_n', 'end_n', 'length_n', 'forword', 'raw_order', 'query_name', 'SID']
    #add columns ['forword_n', 'Order', 'Link', 'LINKS']
    # input numpy must be sorted by: raw_order
    '''
    cdef np.ndarray _O
    cdef int _S = np.lexsort( (-_G[:,2],  _G[:,1], _G[:,0], -_G[:,3]) )[0]
    cdef int _R = (<object> _G).shape[0]

    if _G[_S, 4] == '+':
        _O = np.r_[ _G[_S:], _G[:_S] ]
        _O = np.c_[_O, _O[:,4]]   #forword_n
    else:
        _O = np.r_[ _G[_S::-1], _G[:_S:-1] ] 
        _O = np.c_[_O, np.vectorize({'+':'-','-':'+'}.get)(_O[:,4])] #forword_n
    del _G

    cdef list _T = [ '{0}:{1}-{2}'.format(*x[:3]) if x[-1] =='+' 
                        else '{0}:{2}-{1}'.format(*x[:3])
                    for x in _O[:,[0,1,2,-1]] ]  #apply_along_axis have a bug for str

    return np.column_stack(( _O,
                np.arange(1, _R+1), #Order
                np.array(_T),     #Link
                np.repeat( ';'.join(_T) , _R) )) #LINKS

def EcRPM(np.ndarray _G, long int perdep=1000000):
    ''''
    #columns ['support_ID_num', 'mapped_reads']
    #mapped_reads also can be instead by sum of support_ID_num
    '''
    return perdep*_G[:,0]/_G[:,1]

def EcRPKM(np.ndarray _G, long int perdep=1000000, long int perlen=100000):
    ''''
    #columns ['support_ID_num', 'mapped_reads', 'LINKSLen']
    #mapped_reads also can be instead by sum of support_ID_num
    '''
    return perlen*perdep*_G[:,0]/(_G[:,1]*_G[:,2])

def EcTPM(np.ndarray _G, long int perdep=1000000, long int perlen=100000):
    ''''
    #columns ['support_ID_num', 'LINKSLen']
    '''
    cdef np.ndarray A = perlen*_G[:,0]/_G[:,1]
    return (perdep*A)/sum(A)

def GroupBY(np.ndarray _K, str _M='EcTPM', long int col=0, ):
    cdef int _R = (<object> _K).shape[0]
    cdef np.ndarray[np.double_t, ndim=1]  _T = np.zeros(_R)
    cdef np.ndarray idx
    cdef str i
    for i in np.unique( _K[:,col] ):
        idx = _K[:,col] == i
        _T[idx] = eval( _M, globals())(_K[idx, 1:])
    return _T