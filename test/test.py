def _splitmap(_inmap, maxdistance):
    inmap  = _inmap.copy()
    for _n, _l in inmap.iterrows():
        S = inmap.start_n.between(_l.start - maxdistance, _l.start + maxdistance, inclusive=True)
        E = inmap.end_n.between(  _l.end   - maxdistance, _l.end   + maxdistance, inclusive=True)
        inmap.loc[(S & E),'start_n'] = inmap[(S & E)]['start'].min()
        inmap.loc[(S & E),  'end_n'] = inmap[(S & E)]['end'  ].max()
    return inmap


'''
F='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3-P1/EcMINIMAPont/04.EcRegion/All.circle.region.Keep'
K=pd.read_csv(F, sep='\t')
D=K[['#chrom','start', 'end', 'forword']]
D[['start_n', 'end_n']]=D[['start', 'end']]


sortN = ['#chrom', 'start', 'end', 'forword']
mapsN = ['#chrom', 'start', 'end', 'forword', 'start_n', 'end_n']
grpby = ['#chrom', 'forword']

indf  = K.copy().sort_values(by=sortN)
indf[['start_n', 'end_n']] = indf[['start', 'end']]
inmap = indf[mapsN].drop_duplicates(keep='first')

T=inmap.groupby(by=grpby, sort=False)
T=inmap[((inmap['#chrom']=='1') & (inmap['forword']=='+'))]


timeit.timeit(stmt= '_splitmap(T, 500)', globals=globals(), number=1)
timeit.timeit(stmt= '_splitmap(T, 500)', globals=globals(), number=1)

#T.values[:,4]

EcMagiccube.maxbetween(T.values, 500)

'''

def orderlinks1(_G): #check
    _G = _G.reset_index(drop=True)
    _S = _G.sort_values(by= ['length_n', '#chrom', 'start_n', 'end_n'], ascending=[0, 1, 1, 0]).iloc[0].name
    if _G.loc[_S,'forword'] == '+':
        _O = _G.index.tolist()[_S:] +  _G.index.tolist()[:_S]
        _G['forword_n'] = _G['forword']
    else:
        _O = _G.index.tolist()[_S::-1] + _G.index.tolist()[:_S:-1]
        _G['forword_n'] = _G['forword'].replace({'+':'-','-':'+'})
    _G = _G.loc[_O]
    _G['Link'] = _G[['#chrom', 'start_n', 'end_n', 'forword_n']]\
                    .apply(lambda x: '{0}:{1}-{2}'.format(*x[:3]) if x[3] =='+'
                                else '{0}:{2}-{1}'.format(*x[:3]), axis=1)
    _G['LINKS'] = _G.Link.str.cat(sep=';')
    _G['Order'] = range(1, _G.shape[0] + 1)
    return _G


def OrderLinks1(_G):
    ''''
    #row columns ['#chrom', 'start_n', 'end_n', 'length_n', 'forword', 'query_name', 'SID']
    #add columns ['forword_n', 'Link', 'LINKS', 'Order' ]
    # input numpy must be sorted by: raw_order
    '''
    _S = np.lexsort( (-_G[:,2],  _G[:,1], _G[:,0], -_G[:,3]) )[0]
    if _G[_S, 4] == '+':
        _O = np.r_[ _G[_S:], _G[:_S] ] 
        _O = np.c_[_O, _O[:,4]]   #forword_n
    else:
        _O = np.r_[ _G[_S::-1], _G[:_S:-1] ] 
        _O = np.c_[_O, np.vectorize({'+':'-','-':'+'}.get)(_O[:,4])] #forword_n
    del _G
    _O = np.c_[ _O, 
                np.array([ '{0}:{1}-{2}'.format(*x[:3]) if x[-1] =='+' 
                                else '{0}:{2}-{1}'.format(*x[:3])
                            for x in _O[:,[0,1,2,-1]] ]), #Link #apply_along_axis have a bug for str
                np.arange(1,_O.shape[0]+1), #Order
                ]  
    _O = np.insert(_O, -1, ';'.join(_O[:,-2]), axis=1)
    return _O

import pandas as pd
import numpy as np
import timeit 
from EcMagiccube import OrderLinks
COL = ['#chrom', 'start_n', 'end_n', 'length_n', 'forword', 'query_name', 'SID']
L='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3-P1/EcMINIMAPont/04.EcRegion/All.circle.region.Links'
L=pd.read_csv(L, sep='\t')[COL]
L=L[(L['#chrom']=='A10-pg-p221')]
N=L.to_numpy()

orderlinks1(L.head(5))
pd.DataFrame(OrderLinks1(L.head(5).values))
pd.DataFrame(OrderLinks(L.head(5).values))


K=orderlinks1(L)
M=pd.DataFrame(OrderLinks1(N))
J=pd.DataFrame(OrderLinks(N))

K[K[:, -3] != M[:, -3]]


timeit.timeit(stmt= 'orderlinks1(L)', globals=globals(), number=3)
timeit.timeit(stmt= 'OrderLinks(N)', globals=globals(), number=3)
timeit.timeit(stmt= 'OrderLinks1(N)', globals=globals(), number=3)


import numba
import numpy as np
@numba.vectorize
def Rmerge(interVs):
    merged = []
    for iv in sorted(interVs, key=lambda x:x[0]):
        if (not merged) or (merged[-1][-1] < iv[0]):
            merged.append(iv)
        else:
            merged[-1][-1] = max(merged[-1][-1], iv[-1])
    merged = np.sum([i[1]-i[0] + 1 for i in merged ])
    return merged

@numba.jit(debug=True, nopython=True, parallel=True)
def getsum(d):
    return np.array(d)

def Rmerge1(interVs):
    """
    :param interVs: List[List[int]]
    :return: List[List[int]]
    """
    merged = []
    for iv in sorted(interVs, key=lambda x:x[0]):
        if (not merged) or (merged[-1][-1] < iv[0]):
            merged.append(iv)
        else:
            merged[-1][-1] = max(merged[-1][-1], iv[-1])
    merged = sum([i[1]-i[0] + 1 for i in merged ])
    return merged


a=[[1,10000],[354634,5555555],[231133,2455555]]