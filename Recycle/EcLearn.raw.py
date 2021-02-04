import os
import pandas as pd
import numpy  as np
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit, ShuffleSplit, 
                                     LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold, RepeatedKFold)
from sklearn.feature_selection import f_regression
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             classification_report, make_scorer, balanced_accuracy_score,
                             precision_recall_curve, mean_squared_error, roc_auc_score, 
                             roc_curve, auc, r2_score, mean_absolute_error,
                             average_precision_score, explained_variance_score)

from scipy.stats import pearsonr, stats, linregress, t
from scipy.sparse import hstack, vstack
import statsmodels.api as sm

import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype']= 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
#from adjustText import adjust_text

CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
        '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
        'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
        'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
        'SunTag-CRISPRi', 'V7-MC-HG-FA']
CELLS = [ 'BC%s'%i for i in range(1,13) ]
MARKGENE = ['EGFR', 'CDK6', 'SEPTIN14', 'MYC', 'DENND3', 
            'PCAT1', 'BAP1', 'SOX2', 'MUC4', 'MECOM', 'PIK3CA', 
            'CCND1', 'MYCN', 'TERT', 'RPS6', 'SMARCA4', 'WDR60', 
            'AC019257.8', 'DLG1', 'WNK1', 'MUC2', 'AHRR']

def SMols(X,y):
    #statsmodels.regression.linear.OLS
    import statsmodels.api as sm
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    y_pre= est2.fittedvalues
    #print(est2.summary())
    return {'R2' : est2.rsquared,
            'R2_adj' : est2.rsquared_adj,
            'p_fv' : est2.f_pvalue,
            'intcoef' : est2.tvalues,
            'clf'  : est2,
            'p_tv' : est2.pvalues,
            'func' : "y={:.4f}x{:+.4f}".format(est2.params[X.columns[0]], est2.params['const']),
            'matrx' : pd.DataFrame( np.c_[ X, y, est2.fittedvalues], columns=['X','y', 'y_pre'])
    }

def SMols2(X,y):
    #statsmodels.regression.linear.OLS
    import statsmodels.api as sm
    X1 = np.log(X+1)
    X2 = sm.add_constant(X1)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    y_pre= est2.fittedvalues
    #print(est2.summary())

    return {'R2' : est2.rsquared,
            'R2_adj' : est2.rsquared_adj,
            'p_fv' : est2.f_pvalue,
            'intcoef' : est2.tvalues,
            'p_tv' : est2.pvalues,
            'func' : "y={:.4f}ln(x+1){:+.4f}".format(est2.tvalues['ecDNAcounts'], est2.tvalues['const']),
            'matrx' : pd.DataFrame( np.c_[ X, y, y_pre], columns=['X','y', 'y_pre'])
    }

def SCI(X, y):
    import scipy
    return scipy.stats.linregress(X, y)

def F_reg_pvalue(y, y_pre):
    return f_regression(y.values.reshape(-1, 1), y_pre)

def t_pvalue(X, y, y_pre, coef_):
    import scipy
    sse = np.sum((y_pre - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
    se  = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
    t = coef_ / se
    p = np.squeeze(2 * (1 - scipy.stats.t.cdf(np.abs(t), y.shape[0] - X.shape[1])))
    return [t, p]

def T_pvalue(X, y, y_pre, clf):
    import scipy
    X2  = np.append(np.ones((X.shape[0],1)), X, axis=1)
    MSE = np.sum((y-y_pre)**2)/float(X2.shape[0] -X2.shape[1])
    SE  = np.sqrt(MSE*(np.linalg.inv(np.dot(X2.T,X2)).diagonal()))
    T   = np.append(clf.intercept_, clf.coef_)/SE
    P   = np.squeeze(2*(1-scipy.stats.t.cdf(np.abs(T),(X2.shape[0] -X2.shape[1]))) )
    return [T, P]

def SKLR(X, y):
    import scipy
    clf = LinearRegression()
    clf.fit(X, y)
    y_pre= clf.predict(X)
    R2 = clf.score(X, y)
    R2 = r2_score(y, y_pre)
    R2_adj = 1 - (1-R2)*(len(y)-1)/(len(y)-X.shape[1]-1)
    intercept_ = clf.intercept_
    coef_ = clf.coef_

    p_Tv = T_pvalue(X, y, y_pre, clf)[1][1]
    p_fv = F_reg_pvalue(y, y_pre)[1][0]

    return {'R2' : R2,
            'R2_adj' : R2_adj,
            'p_fv' : p_fv,
            'intcoef' : coef_,
            'clf'  : clf,
            'p_tv' : p_Tv,
            'func' : "y={:.4f}x{:+.4f}".format(coef_[0], intercept_),
            'matrx' : pd.DataFrame( np.c_[ X, y, y_pre], columns=['X','y', 'y_pre'])
    }

def GridS(M='enet'):
    from sklearn.model_selection import GridSearchCV, LeaveOneOut
    G = {'enet': {'estimator':ElasticNet(max_iter=1000, random_state=None),
              'parameters' : { 'alpha'  : [0.5,  1, 2, 5],
                                'l1_ratio': [.01, .05, .1, .2, .3, .4, 0.5],
                                    'tol' : [1e-3, 1e-4]}},
        'Ridge' : {'estimator' : Ridge(),
                    'parameters' : {'alpha'  : [ 1, 2, 5, 7, 10, 20,30, 100],
                                    'tol' : [1e-3, 1e-4]}},
        }
    
    clf = GridSearchCV(G[M]['estimator'], G[M]['parameters'],
                    n_jobs=-2,
                    cv= ShuffleSplit(4) ,#LeaveOneOut(),
                    error_score = np.nan,
                    return_train_score=True,
                    refit = True)
    
    return clf

def SKEnet(X0, y):

    import scipy
    clf = GridS()

    X = np.log( X0+1 )
    #X = X0
    clf.fit(X, y)
    y_pre= clf.predict(X)
    R2 = clf.score(X, y)
    R2 = r2_score(y, y_pre)
    R2_adj = 1 - (1-R2)*(len(y)-1)/(len(y)-X.shape[1]-1)
    intercept_ = clf.best_estimator_.intercept_
    coef_ = clf.best_estimator_.coef_
    p_Tv = T_pvalue(X, y, y_pre, clf.best_estimator_)[1][1]
    p_fv = F_reg_pvalue(y, y_pre)[1][0]

    return {'R2' : R2,
            'R2_adj' : R2_adj,
            'p_fv' : p_fv,
            'intcoef' : coef_,
            'p_tv' : p_Tv,
            'func' : "Enet: l1_ratio(%s) alpha:(%s)"%(clf.best_params_['l1_ratio'], clf.best_params_['alpha']),
            'matrx' : pd.DataFrame( np.c_[ X0, y, y_pre], columns=['X','y', 'y_pre'])
    }

def catdf():
    TV='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3_Colon_theoretical_value.txt'
    TherVu =pd.read_csv(TV, sep='\t')
    Tcol =  TherVu.columns.drop('#chrom')
    TVmelt = pd.melt(TherVu, id_vars=['#chrom'], value_vars=Tcol,  var_name='Therical', value_name='Thervalue')
    TVmelt['Cellline'] = TVmelt.Therical.str.split('_').str[0]

    IN='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore'
    ACounts = []
    for i in ['Colon-P1', 'Colon-P2', 'PC3-P1', 'PC3-P2']:
        INput='%s/%s/EcMINIMAPont/05.CheakBP/BPState/All.plasmid.Keep.matrix'%(IN, i)
        INdata=pd.read_csv(INput, sep='\t')
        INdata.drop('Links', axis=1, inplace=True)
        Vcol = INdata.columns.drop(['#chrom'])
        INmelt = pd.melt(INdata, id_vars=['#chrom'], value_vars=Vcol,  var_name='Cells', value_name='ecDNAcounts')
        INmelt['Cellline'] = i
        ACounts.append(INmelt)
    ACounts = pd.concat(ACounts, axis=0, sort=False)
    xyData = ACounts.merge(TVmelt, on=['#chrom','Cellline'], how='outer')
    return xyData

def RegPlot(*args, **kwargs):
    _d = kwargs.pop('data')
    Stat = SMols(_d[['ecDNAcounts']], _d['Thervalue'])
    rp = sns.regplot(x='X', y='y', data=Stat['matrx'],
                line_kws={'label': "%s\n$R^2$:%.4f $R^2(adj)$:%.4f p:%.4f"%(Stat['func'], Stat['R2'], Stat['R2_adj'], Stat['p_fv'])},
                scatter_kws={"s":4},
                )

def PltPlot(*args, **kwargs):
    Model = args[0]
    _d = kwargs.pop('data')
    Stat = Model(_d[['ecDNAcounts']], _d['Thervalue'])
    
    label1 =  Stat['func']
    label2 = "$R^2$:%.4f $R^2(adj)$:%.4f p:%.4f"%(Stat['R2'], Stat['R2_adj'], Stat['p_fv'])

    plt.plot(Stat['matrx'].X, Stat['matrx'].y_pre,'ro-', label=label1)
    plt.plot(Stat['matrx'].X, Stat['matrx'].y,     'bo', label=label2)

    plt.legend(loc='upper left')

def FGirid(xyData, plotM, OUT):
    pal = dict(TRA='Set1', TRB='Set2', IGH='Set3', IGL='cool', IGK='hot' )
    g = sns.FacetGrid(xyData, 
                    row='Therical',
                    col="Cells",
                    sharey=False,
                    sharex=False,
                    palette='Set1',
                    #style='dark',
                    aspect=1.5,
                    legend_out=False,
                    #height=10,
                    #col_order=['TRA','TRB','IGH','IGL','IGK'],
        )
    if plotM == 'lr':
        Model = SMols
        g.map_dataframe(RegPlot)
    elif  plotM == 'loglr':
        Model = SMols2
        g.map_dataframe(PltPlot, Model)
    elif plotM == 'enet':
        Model = SKEnet
        g.map_dataframe(PltPlot, Model)

    for ax in g.axes.ravel():
        ax.legend(loc='upper left')

    g.savefig('%s.%s.pdf'%(OUT, plotM))
    plt.close()

    Stat = []
    for (_t,_c,_l), _g in  xyData.groupby(by=['Therical', 'Cells', 'Cellline'], sort=False):
        _S = Model(_g[['ecDNAcounts']], _g['Thervalue'])
        Stat.append( [_t,_c,_l, _S['R2'], _S['R2_adj'], _S['p_fv']] )
    Stat = pd.DataFrame(Stat, columns=['Therical', 'Cells', 'Cellline', 'R2', 'R2_adj', 'p_fv'])
    Stat.to_csv('%s.%s.score.xls'%(OUT, plotM), sep='\t', index=False) 

    n = sns.relplot(x="Cells", y="R2", hue="Therical", style="Cellline", kind="line", palette='cool', data=Stat)
    n.set_xticklabels(rotation=270)
    n.savefig('%s.%s.score.R2.pdf'%(OUT, plotM))

def barP(K):
    K = K[(K.Cells != 'support_num')]
    K.set_index('Cells').T.plot(kind='bar', stacked=True)

#fig, ax = plt.subplots()
def chrec(C):
    plt.figure(figsize=(13,10))
    #C['ecDNAcounts'] = np.log2(C['ecDNAcounts'] +1)
    gc = sns.boxplot(x="#chrom", y="ecDNAcounts", hue="type", meanprops={'linestyle':'-.'},
                    data=C, palette="Set3",  fliersize=3, linewidth=1.5)
    plt.xticks(rotation='270')
    plt.savefig('./CellFit//AB.compare.pdf')

def cellec(C):
    plt.figure(figsize=(13,10))
    #C['ecDNAcounts'] = np.log2(C['ecDNAcounts'] +1)
    gc = sns.boxplot(x="Cells", y="ecDNAcounts", hue="type", meanprops={'linestyle':'-.'},
                    data=C, palette="Set3",  fliersize=3, linewidth=1.5)
    plt.xticks(rotation='270')
    plt.savefig('./CellFit//AB.compare.cell.pdf')

def ODtest(X):
    from sklearn import svm
    from sklearn.datasets import make_moons, make_blobs
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from pyod.models.knn import KNN
    import time

    # Example settings
    n_samples = 12
    outliers_fraction = 0.1
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    # define outlier/anomaly detection methods to be compared
    anomaly_algorithms = [
        ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)),
        ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=None)),
        ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=8, contamination=outliers_fraction))]


    print(X)
    for name, algorithm in anomaly_algorithms:
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()

        # fit the data and tag outliers
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)
        print(name, y_pred)

def CellRegres(xyData, _M):
    if _M == 'lr':
        Model = SMols
    elif  _M == 'loglr':
        Model = SMols2
    elif _M == 'enet':
        Model = SKEnet

    Stat = []
    for (_t,_c,_l), _g in  xyData.groupby(by=['Therical', 'Cells', 'Cellline'], sort=False):
        _S = Model(_g[['ecDNAcounts']], _g['Thervalue'])
        print(_S)
        K =(_S['matrx'].y- _S['matrx'].y_pre).abs().to_frame()
        print(K)
        ODtest(K)
        break

        #Stat.append( [_t,_c,_l, _S['R2'], _S['R2_adj'], _S['p_fv']] )
    #Stat = pd.DataFrame(Stat, columns=['Therical', 'Cells', 'Cellline', 'R2', 'R2_adj', 'p_fv'])
    #Stat.to_csv('%s.%s.score.xls'%(OUT, plotM), sep='\t', index=False) 

def CMD():
    opre='B.line'
    OUT='./CellFit/' + opre

    '''
    xyData = catdf()
    xyData.to_csv('%s.Plasmid_Col_PC3.xls'%OUT, sep='\t', index=False)
    FGirid(xyData, 'lr', OUT)
    '''

    A=pd.read_csv( './CellFit//A.line.Plasmid_Col_PC3.xls', sep='\t')
    B=pd.read_csv( './CellFit//B.line.Plasmid_Col_PC3.xls', sep='\t')
    '''
    R =  pd.read_csv( './CellFit//B.line.lr.score.xls', sep='\t')
    Rt = R.pivot(index='Therical', columns='Cells', values='R2')
    Rt.to_csv('./CellFit//B.line.lr.score.R2.t.xls', sep='\t')

    CellRegres(B, 'lr')
    '''


    A['type'] = 'A'
    B['type'] = 'B'
    C=pd.concat((A,B), axis=0)
    C = C[(C.Cells != 'support_num')]
    chrec(C)
    cellec(C)

###################################second time################
def getcounts(dfmrx):
    dfmrx.loc[(dfmrx['#chrom'].isin(PLMD)), 'gene_name'] = dfmrx.loc[(dfmrx['#chrom'].isin(PLMD)), '#chrom']

    countdict={}
    for _, _l in dfmrx.iterrows():
        if _l.gene_name=='.':
            continue
        _G = _l.gene_name.split(';')
        _B = _l.gene_biotype.split(';')
        _S = dict(zip( _l.support_IDs.split(';'), map( int,_l.support_read_num.split(';')) ))
        for _i in list(zip(_G, _B)):
            if _i[0] !='.':
                countdict.setdefault(_i, []).append(_S)
    countlist = []
    for k, v in countdict.items():
        genedict ={'#chrom': k[0], 'gene_biotype': k[1]}
        for _d in v:
            for _id, _count in _d.items():
                if _id in genedict.keys():
                    genedict[_id] += _count
                else:
                    genedict[_id] = _count
        genedict = pd.Series(genedict)
        countlist.append( genedict )
    countlist = pd.concat(countlist, axis=1).T
    countlist = countlist[(countlist.gene_biotype.isin(['protein_coding', '.']))]
    CCol = countlist.columns.drop(['#chrom', 'gene_biotype'])
    countlist = pd.melt(countlist, id_vars=['#chrom'], value_vars=CCol, var_name='Cells', value_name='ECfiltcounts' )
    return countlist

def getdf():
    TV='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3_Colon_theoretical_value.txt'
    TherVu =pd.read_csv(TV, sep='\t')
    Tcol =  TherVu.columns.drop('#chrom')
    TVmelt = pd.melt(TherVu, id_vars=['#chrom'], value_vars=Tcol,  var_name='Therical', value_name='Thervalue')
    TVmelt['Cellline'] = TVmelt.Therical.str.split('_').str[0]
    print(TVmelt)

    IN='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/CellFit2/'
    ACounts = []
    for i in ['Colon-P1', 'Colon-P2', 'PC3-P1', 'PC3-P2']:
        INput='%s/%s/EcMINIMAPont/05.CheakBP/BPState/All.plasmid.Keep.matrix'%(IN, i)
        INdata=pd.read_csv(INput, sep='\t')
        INdata.drop('Links', axis=1, inplace=True)
        Vcol = INdata.columns.drop(['#chrom'])
        INmelt = pd.melt(INdata, id_vars=['#chrom'], value_vars=Vcol,  var_name='Cells', value_name='BPcounts')
        INmelt['Cellline'] = i
        #INmelt['Datatype'] = 'BPcount'
        ACounts.append(INmelt)
    ACounts = pd.concat(ACounts, axis=0, sort=False)
    #print(ACounts)

    BCounts = []
    for i in ['Colon-P1', 'Colon-P2', 'PC3-P1', 'PC3-P2']:
        UpFilter=t='%s/%s/EcMINIMAPont/04.EcRegion/All.circle.region.UpFilter'%(IN, i)
        UpFilter=pd.read_csv(UpFilter, sep='\t')
        UpFilter=UpFilter.loc[ (UpFilter.groupby(by='LINKS')['length'].idxmax()) ] # Tpye='maxlen'
        #UpFilter=UpFilter.loc[ (UpFilter.Type==1) ] # Tpye='type1'
        UpFilter=getcounts(UpFilter)
        UpFilter['Cellline'] = i
        #UpFilter['Datatype'] = 'ECfilt'
        BCounts.append(UpFilter)
    BCounts = pd.concat(BCounts, axis=0, sort=False)

    #CCounts = pd.concat(ACounts + BCounts, axis=0, sort=False)
    xyData = BCounts\
                .merge(ACounts, on=['#chrom','Cellline', 'Cells'], how='outer')\
                .merge(TVmelt,  on=['#chrom','Cellline'], how='outer')

    xyData.to_csv('%s/EcDNA_Plasmid_Col_PC_maxlen.xls'%OU, sep='\t', index=False)
    print(xyData)

def FGPlot(*args, **kwargs):
    _d = kwargs.pop('data')
    M = kwargs.pop('M')
    P = kwargs.pop('P')
    R = kwargs.pop('R')
    C = kwargs.pop('C')

    _d.columns = _d.columns.tolist()[:-2] + ['X', 'y']
    Stat = M(_d[['X']], _d['y'])

    label1 =  Stat['func']
    label2 = "$R^2$:%.4f p:%.4f"%(Stat['R2'], Stat['p_fv'])
    
    plt.plot(Stat['matrx'].X, Stat['matrx'].y_pre,'ro-', label=label1)
    plt.plot(Stat['matrx'].X, Stat['matrx'].y,    'bo', label=label2)
    if not P.empty:
        P = P[( (P[R].isin(_d[R])) & (P[C].isin(_d[C])) )]
        for _, _l in P.iterrows():
            plt.plot(_l[-2], _l[-1],'c*')
            plt.text(_l[-2], _l[-1], _l[0])
    plt.legend(loc='upper left')

def linearReg(xyData, OUT, PD = pd.DataFrame(), _M='sklr', R='Therical', C='Cells', R2=False):
    if _M == 'lr':
        Model = SMols
    elif  _M == 'loglr':
        Model = SMols2
    elif _M == 'enet':
        Model = SKEnet
    elif _M == 'sklr':
        Model = SKLR
    #print(locals())
    pal = dict(TRA='Set1', TRB='Set2', IGH='Set3', IGL='cool', IGK='hot' )
    g = sns.FacetGrid(xyData, 
                    row=R,
                    col=C,
                    sharey=False,
                    sharex=False,
                    palette='Set1',
                    #style='dark',
                    aspect=1.5,
                    legend_out=False,
                    #height=10,
                    col_order=CELLS,
    )

    g.map_dataframe(FGPlot, M=Model, P=PD, R=R, C=C)
    g.set_axis_labels(C, R)
    for ax in g.axes.ravel():
        ax.legend(loc='upper left')
        #ax.set_yscale('log')
        #ax.set_xscale('log')
    g.tight_layout()
    g.savefig(OUT + '.pdf')
    plt.close()

    Stat = []
    for (_c,_l), _g in  xyData.groupby(by=[R, C], sort=False):
        _S = Model(_g.iloc[:, [-2]], _g.iloc[:, -1])
        Stat.append( [_c,_l, _S['R2'], _S['R2_adj'], _S['p_fv']] )
    Stat = pd.DataFrame(Stat, columns=[R, C, 'R2', 'R2_adj', 'p_fv'])
    Stat['Cells'] = pd.Categorical(Stat['Cells'], CELLS)
    Stat.sort_values(by=['Cells', R], inplace=True)
    Stat.to_csv(OUT + '.xls', sep='\t', index=False) 
    
    if R2:
        n = sns.relplot(x=C, y="R2", hue=R, style=R, kind="line", palette='tab10', data=Stat)
        n.set_xticklabels(rotation=270)
        n.set(ylim=(0, 1))
        n.savefig(OUT + '.R2.pdf')

def linearRegP(T, P, OUT, R='Therical', C='Cells', Xc=['ECfiltcounts'], yc='Thervalue'):
    def mstat(X, y, Xpre, _M='sklr'):
        if _M == 'lr':
            M = SMols
        elif  _M == 'loglr':
            M = SMols2
        elif _M == 'enet':
            M = SKEnet
        elif _M == 'sklr':
            M = SKLR
        S = M(X, y)
        l1 =  S['func']
        l2 = "$R^2$:%.4f p:%.4f"%(S['R2'], S['p_fv'])
        ypre = S['clf'].predict(Xpre) if len(Xpre)>0 else []
        return (l1, l2, S, ypre)

    rowl = T[R].unique()
    coll = T[C].unique()
    P = P.copy()
    P[Xc] = P[Xc].astype(int)

    fig, axs = plt.subplots(len(rowl), len(coll), figsize=( 60, 18)) 
    fig.set_alpha(0.0)
    #, figsize=(, 17), frameon=False ,  facecolor='w', edgecolor='k'
    for _r, _rr in enumerate(rowl):
        for _c, _cc in enumerate(coll):
            _tt = T[( (T[R]==_rr) & (T[C]==_cc) )]
            _bb = ((P[R]==_rr) & (P[C]==_cc))
            _pp = P[_bb]
            l1, l2, S, ypre = mstat(_tt[Xc], _tt[yc], _pp[Xc])
            P.loc[_bb, yc] = ypre

            axs[_r, _c].plot(S['matrx'].X, S['matrx'].y_pre, 'ro-', label=l1)
            axs[_r, _c].plot(S['matrx'].X, S['matrx'].y,    'bo'  , label=l2)
            axs[_r, _c].legend(loc='upper left')
            axs[_r, _c].title.set_text('y: %s | x: %s'%(_rr, _cc))

            if _bb.any():
                axins = axs[_r, _c].inset_axes([0.57, 0.1, 0.4, 0.4]) #[left, bottom, width, height]
                axins.plot(_pp[Xc], ypre, 'r*-.')
                for _xx, _l in _pp.groupby(by=Xc):
                    _ttxt = _l['#chrom'].str.cat(sep='\n')
                    axins.text(_xx, _l[yc].iloc[0], _ttxt, fontsize='x-small')
                axs[_r, _c].indicate_inset_zoom(axins)
                '''
                #axins.set_xlim(x1, x2)
                #axins.set_ylim(y1, y2)
                texts = []
                for _xx, _l in _pp.groupby(by=Xc):
                    _ttxt = _l['#chrom'].str.cat(sep='\n')
                    texts.append( axins.text(_xx, _l[yc].iloc[0], _ttxt, fontsize='x-small') )
                adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='c', lw=0.5))
                axs[_r, _c].indicate_inset_zoom(axins)
                '''
    fig.savefig(OUT+'.pdf',  bbox_inches='tight')


def linearRegC(xyData, OUT, _M='sklr', R='Therical', xl = 'BPcounts', yl='ECfiltcounts', R2=False):
    if _M == 'lr':
        Model = SMols
    elif  _M == 'loglr':
        Model = SMols2
    elif _M == 'enet':
        Model = SKEnet
    elif _M == 'sklr':
        Model = SKLR

    pal = dict(TRA='Set1', TRB='Set2', IGH='Set3', IGL='cool', IGK='hot' )
    g = sns.FacetGrid(xyData, 
                    col=R,
                    sharey=False,
                    sharex=False,
                    palette='Set1',
                    #style='dark',
                    aspect=1.5,
                    legend_out=False,
                    #height=10,
                    #col_order=CELLS,
    )
    g.map_dataframe(FGPlot, M=Model)
    g.set_axis_labels(xl, yl)
    for ax in g.axes.ravel():
        ax.legend(loc='upper left')
    g.tight_layout()
    g.savefig(OUT + '.pdf')
    plt.close()

def predictCN(T, P, _M ='sklr',
              plasther={'Colon-P1':'Colon-P1_pikein-100', 
                        'Colon-P2':'Colon-P2_100', 
                        'PC3-P1'  :'PC3-P1_spikein-100',
                        'PC3-P2'  :'PC3-P2_P2-100'}):
    MODLES = {}
    if _M == 'lr':
        Model = SMols
    elif  _M == 'loglr':
        Model = SMols2
    elif _M == 'enet':
        Model = SKEnet
    elif _M == 'sklr':
        Model = SKLR

    T = T[( (T.Therical.isin(plasther.values())) & (T.Cellline.isin(plasther.keys())) )].copy()
    #T[['ECfiltcounts', 'Thervalue']] = T[['ECfiltcounts', 'Thervalue']].fillna(0)

    P = P.copy()
    P['ECfiltcounts'] = P['ECfiltcounts'].fillna(0)
    P['Therical'] = P.Cellline.map(plasther)

    for (_c, _l, _t), _g in  T.groupby(by=['Cells', 'Cellline', 'Therical']):
        _B = ((P.Cells==_c) & (P.Cellline==_l) & (P.Therical==_t))
        if _B.any():
            Stat = Model(_g[['ECfiltcounts']], _g['Thervalue'])
            P.loc[_B, 'Thervalue'] = Stat['clf'].predict(P.loc[_B, ['ECfiltcounts']] )
    return T, P

def CMD2(M = 'sklr', Type='maxlen'):
    IN='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/CellFit2'
    #getdf()

    if Type == 'maxlen':
        A=pd.read_csv( '%s/EcDNA_Plasmid_Col_PC_maxlen.xls'%OU, sep='\t')
    elif Type == 'type1':
        A=pd.read_csv( '%s/EcDNA_Plasmid_Col_PC_type1.xls'%OU, sep='\t')
    
    A = A[(~A.ECfiltcounts.isna())]
    A['Cells'] = pd.Categorical(A['Cells'], CELLS+['support_num'])
    A.sort_values(by=['Cells', '#chrom'], inplace=True)


    T  = A[( A['#chrom'].isin(PLMD) & (A.Cells.isin(CELLS)) & (~ A.Thervalue.isna()) )]
    P  = A[( A['#chrom'].isin(MARKGENE) & (A.Cells.isin(CELLS)) & (A.ECfiltcounts>0) )]
    T, P = predictCN(T, P, _M=M)
    T = T[['#chrom', 'Cells', 'Cellline', 'Therical', 'ECfiltcounts', 'Thervalue']]
    P = P[['#chrom', 'Cells', 'Cellline', 'Therical', 'ECfiltcounts', 'Thervalue']]
    H = '%s/EcDNA_ECfiltvsThervalue_predict_%s_%s'%(OU, M, Type)
    pd.concat([T, P], axis=0).to_csv( H + '.xls', sep='\t', index=False)
    linearRegP(T, P, H)
    #linearReg(T, '%s/EcDNA_ECfiltvsThervalue_predict_%s_%s'%(OU, M, Type), PD=P, R='Therical', C='Cells', R2=True)

    '''
    ####BPcount vs ECfilt
    B = A[(A['#chrom'].isin(PLMD) & (A.Cells.isin(CELLS)) & (~A.BPcounts.isna()) )].copy()
    B = B[['#chrom', 'Cells', 'Cellline', 'ECfiltcounts', 'BPcounts']].drop_duplicates(keep='first')
    linearReg(B, '%s/EcDNA_BPvsEcFilter_%s_%s'%(OU, M, Type), _M =M, R='Cellline', C='Cells')


    ####BPcount vs Thervalue
    C = A[(A['#chrom'].isin(PLMD) & (A.Cells.isin(CELLS)) & (~ A.BPcounts.isna()) )]
    C = C[['#chrom', 'Cells', 'Cellline', 'Therical', 'BPcounts', 'Thervalue']].drop_duplicates(keep='first')
    linearReg(C, '%s/EcDNA_BPvsThervalue_%s_%s'%(OU, M, Type), R='Therical', C='Cells', R2=True)

    ####ECfilt vs Thervalue
    D = A[(A['#chrom'].isin(PLMD) & (A.Cells.isin(CELLS)) & (~ A.Thervalue.isna()) )]
    D = D[['#chrom', 'Cells', 'Cellline', 'Therical', 'ECfiltcounts', 'Thervalue']].drop_duplicates(keep='first')
    linearReg(D, '%s/EcDNA_ECfiltvsThervalue_%s_%s'%(OU, M, Type), R='Therical', C='Cells', R2=True)
    linearRegC(D, '%s/EcDNA_ECfiltvsThervalueC_%s_%s'%(OU, M, Type), R='Therical', R2=True)
    '''

CMD2()
    