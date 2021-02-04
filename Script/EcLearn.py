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

import glob
import pysam
from EcBammanage import BamFilter
from concurrent import futures
from joblib import Parallel, delayed, dump, load

class STATE:
    def SMols(self, X,y):
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

    def SMols2(self, X,y):
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

    def SCI(self, X, y):
        import scipy
        return scipy.stats.linregress(X, y)

    def F_reg_pvalue(self, y, y_pre):
        return f_regression(y.values.reshape(-1, 1), y_pre)

    def t_pvalue(self, X, y, y_pre, coef_):
        import scipy
        sse = np.sum((y_pre - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1]).astype(np.float)
        se  = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
        t = coef_ / se
        p = np.squeeze(2 * (1 - scipy.stats.t.cdf(np.abs(t), y.shape[0] - X.shape[1])))
        return [t, p]

    def T_pvalue(self, X, y, y_pre, clf):
        import scipy
        X2  = np.append(np.ones((X.shape[0],1)), X, axis=1).astype(np.float)
        MSE = np.sum((y-y_pre)**2)/float(X2.shape[0] -X2.shape[1])
        SE  = np.sqrt(MSE*(np.linalg.inv(np.dot(X2.T,X2)).diagonal()))
        T   = np.append(clf.intercept_, clf.coef_)/SE
        P   = np.squeeze(2*(1-scipy.stats.t.cdf(np.abs(T),(X2.shape[0] -X2.shape[1]))) )
        return [T, P]

    def SKLR(self, X, y):
        import scipy
        clf = LinearRegression()
        clf.fit(X, y)
        y_pre= clf.predict(X)
        R2 = clf.score(X, y)
        R2 = r2_score(y, y_pre)
        R2_adj = 1 - (1-R2)*(len(y)-1)/(len(y)-X.shape[1]-1)
        intercept_ = clf.intercept_
        coef_ = clf.coef_

        p_Tv = self.T_pvalue(X, y, y_pre, clf)[1][1]
        p_fv = self.F_reg_pvalue(y, y_pre)[1][0]

        return {'R2' : R2,
                'R2_adj' : R2_adj,
                'p_fv' : p_fv,
                'intcoef' : coef_,
                'clf'  : clf,
                'p_tv' : p_Tv,
                'func' : "y={:.4f}x{:+.2f}".format(coef_[0], intercept_),
                'matrx' : pd.DataFrame( np.c_[ X, y, y_pre], columns=['X','y', 'y_pre'])
        }

    def GridS(self, M='enet'):
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

    def SKEnet(self, X0, y):

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

    def ODtest(self, X):
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

class PLOT:
    def __init__(self, out):
        self.out=out
        self.color_ = [ '#00DE9F', '#FF33CC', '#16E6FF', '#E7A72D', '#8B00CC', '#EE7AE9',
                        '#B2DF8A', '#CAB2D6', '#B97B3D', '#0072B2', '#FFCC00', '#0000FF',
                        '#FF2121', '#8E8E38', '#6187C4', '#FDBF6F', '#666666', '#33A02C',
                        '#FB9A99', '#D9D9D9', '#FF7F00', '#1F78B4', '#FFFFB3', '#5DADE2',
                        '#95A5A6', '#FCCDE5', '#FB8072', '#B3DE69', '#F0ECD7', '#CC66CC',
                        '#A473AE', '#FF0000', '#EE7777', '#009E73', '#ED5401', '#CC0073',]

    def barP(self, K):
        K = K[(K.Cells != 'support_num')]
        K.set_index('Cells').T.plot(kind='bar', stacked=True)

    def RegPlot(self, *args, **kwargs):
        _d = kwargs.pop('data')
        Stat = SMols(_d[['ecDNAcounts']], _d['Thervalue'])
        rp = sns.regplot(x='X', y='y', data=Stat['matrx'],
                    line_kws={'label': "%s\n$R^2$:%.4f $R^2(adj)$:%.4f p:%.4f"%(Stat['func'], Stat['R2'], Stat['R2_adj'], Stat['p_fv'])},
                    scatter_kws={"s":4},
        )

    def PltPlot(self, *args, **kwargs):
        Model = args[0]
        _d = kwargs.pop('data')
        Stat = Model(_d[['ecDNAcounts']], _d['Thervalue'])
        
        label1 =  Stat['func']
        label2 = "$R^2$:%.4f $R^2(adj)$:%.4f p:%.4f"%(Stat['R2'], Stat['R2_adj'], Stat['p_fv'])

        plt.plot(Stat['matrx'].X, Stat['matrx'].y_pre,'ro-', label=label1)
        plt.plot(Stat['matrx'].X, Stat['matrx'].y,     'bo', label=label2)

        plt.legend(loc='upper left')

    def FGirid(self, xyData, plotM, OUT):
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

    def chrec(self, C):
        plt.figure(figsize=(13,10))
        #C['ecDNAcounts'] = np.log2(C['ecDNAcounts'] +1)
        gc = sns.boxplot(x="#chrom", y="ecDNAcounts", hue="type", meanprops={'linestyle':'-.'},
                        data=C, palette="Set3",  fliersize=3, linewidth=1.5)
        plt.xticks(rotation='270')
        plt.savefig('./CellFit//AB.compare.pdf')

    def cellBox(self, xyData):
        xyR2   = xyData[['Cells', 'Rename', 'Group', 'Cellline', 'Platform', 'R2', 'R2_adj']]\
                    .drop_duplicates(keep='first').copy()
        #plt.figure(figsize=(13,10))
        #C['ecDNAcounts'] = np.log2(C['ecDNAcounts'] +1)
        xyR2.sort_values(by='Group',inplace=True)
        fig, ax = plt.subplots()
        Col = sns.set_palette(sns.color_palette(self.color_[:8]))
        sns.boxplot(x="Group", y="R2",  meanprops={'linestyle':'-.'}, width=0.45,  
                        data=xyR2, palette='Set3',  fliersize=3, linewidth=0.8, ax=ax)
        sns.swarmplot(x="Group", y="R2", data=xyR2, palette=Col, linestyles='--', size=2.5, linewidth=.3, ax=ax)

        plt.xticks(rotation='270')
        plt.ylim(0, 1)
        #leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1)
        #plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.savefig( self.out, bbox_inches='tight')
        plt.close()

    def cellBox1(self, xyData):
        xyR2   = xyData[['Cells', 'Rename', 'Group', 'Cellline', 'Platform', 'R2', 'R2_adj', 'p_fv', 'p_tv', 'func']]\
                    .drop_duplicates(keep='first').copy()

        xyR2.sort_values(by='Group',inplace=True)
        xyR2.to_csv(self.out +'.xls', sep='\t', index=False)
        import ptitprince as pt

        fig, ax = plt.subplots(figsize=(5,5))
        labels= sorted(xyR2.Cellline.unique())
        ort = 'v'
        pal = 'Set2'
        Col = sns.set_palette(sns.color_palette(self.color_[:8]))
        ax=pt.half_violinplot( x="Cellline", y="R2", data = xyR2, palette = Col,
                                bw = .15, cut = 0.,scale = "area", width = .28, offset=0.12, 
                                linewidth = 0.8, 
                                inner = None, orient = ort)
        ax=sns.swarmplot(x="Cellline", y="R2",  data=xyR2, palette='Pastel1', 
                        linestyles='-', size=2.5, linewidth=.3)
        #ax=sns.stripplot( x="Cellline", y="R2", data = xyR2, palette = Col,
        #                    edgecolor = "white",size = 2.5, jitter = 0.04,
        #                    linestyles='--', linewidth=.3,
        #                    orient = ort)
        ax=sns.boxplot( x="Cellline", y="R2", data=xyR2, orient = ort, notch=False,
                        meanprops={'linestyle':'-.'}, width=0.2,
                        flierprops={'markersize':3, 'marker':'*', 'linestyle':'--', },
                        palette=Col, fliersize=3, linewidth=0.7)

        plt.xticks(rotation='270')
        plt.ylim(0.5, 0.75)
        #leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1)
        #plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.savefig( self.out, bbox_inches='tight')
        plt.close()

    def FGPlot(self, *args, **kwargs):
        _d = kwargs.pop('data')
        X = kwargs.pop('X')
        yt= kwargs.pop('yt')
        yp= kwargs.pop('yp')

        P = _d[~(_d[yt].isna())] #plasmid
        G = _d[ (_d[yt].isna())] #gene

        label1 =  _d['func'].iloc[0]
        label2 = "$R^2$:%.4f p:%.4f"%(_d['R2'].iloc[0], _d['p_fv'].iloc[0])
        
        '''
        plt.plot( _d[X], _d[yp], linestyle='--', linewidth=2, color='fuchsia',
                    marker="*", markeredgecolor='#DA3B95', markerfacecolor='magenta',
                    markeredgewidth=0.2, markersize=8.5, label=label1)
        plt.scatter( P[X],  P[yt], linewidths=0.2,
                     s=35, marker='o' , edgecolors='#009BD2', color= 'cyan', label=label2)
        '''

        plt.plot( _d[X], _d[yp], linestyle='-', linewidth=2, color='black',
                    markersize=0, label=label1)
        plt.scatter( P[X],  P[yt], linewidths=0.2,
                     s=35, marker='o' , edgecolors='red', color= 'red', label=label2)

        if not G.empty:
            for _, _l in G.iterrows():
                plt.plot(_l[X], _l[yp],'c+')
                #plt.text(_l[X], _l[yp], _l[0])
        plt.legend(loc='lower right')

    def linepre(self, xyData,  R='Group', C='CELLs', X='ECfiltcounts', 
                yt='Thervalue', yp='Predict', Order=range(1,13), **kwargs):
        pal = dict(TRA='Set1', TRB='Set2', IGH='Set3', IGL='cool', IGK='hot' )
        g = sns.FacetGrid(xyData, 
                        row=R,
                        col=C,
                        sharey=False,
                        sharex=False,
                        palette='Set1',
                        #style='dark',
                        aspect=1.2,
                        legend_out=False,
                        #height=10,
                        col_order=Order,
                        despine=False,
        )
        g.map_dataframe(self.FGPlot, X=X, yt=yt, yp=yp)
        #g.set_axis_labels(C, R)
        for ax in g.axes.ravel():
            ax.legend(loc='lower right')
            #ax.set_yscale('log')
            #ax.set_xscale('log')
        g.tight_layout()
        g.savefig(self.out)
        plt.close()

    def linelm(self, xyData):
        g=sns.lmplot(x="ECfiltcounts", y="Thervalue", hue="Rename", data=xyData)
        g.savefig(self.out)
        plt.close()

    def linearRegP(self, xyData, R='Group', C='CELLs', X='ECfiltcounts', yt='Thervalue', yp='Predict', Mk='gene'):
        rowl = sorted(xyData[R].unique())
        coll = sorted(xyData[C].unique())

        fig, axs = plt.subplots(len(rowl), len(coll), figsize=( 60, 35)) 
        fig.set_alpha(0.0)
        #, figsize=(, 17), frameon=False ,  facecolor='w', edgecolor='k'
        for _r, _rr in enumerate(rowl):
            for _c, _cc in enumerate(coll):
                rcD = xyData[( (xyData[R]==_rr) & (xyData[C]==_cc) )]
                P = rcD[~(rcD[yt].isna())].copy() #plasmid
                G = rcD[ (rcD[yt].isna())].copy() #gene
                
                if P.empty: #same sample have no the number
                    continue
                l1 =  P['func'].iloc[0]
                l2 = "$R^2$:%.4f p:%.4f"%(P['R2'].iloc[0], P['p_fv'].iloc[0])
            
                axs[_r, _c].plot(  P[X],   P[yt], 'bo' , label=l1)
                axs[_r, _c].plot(rcD[X], rcD[yp], 'ro-', label=l2)

                axs[_r, _c].legend(loc='upper left')
                axs[_r, _c].title.set_text('y: %s | x: BC%s'%(_rr, _cc))

                if not G.empty:
                    axins = axs[_r, _c].inset_axes([0.6, 0.1, 0.38, 0.39]) #[left, bottom, width, height]
                    axins.plot(G[X], G[yp], 'c*-.')
                    for _xx, _l in G.groupby(by=X):
                        _ttxt = _l[Mk].str.cat(sep='\n')
                        axins.text(_xx, _l[yp].iloc[0], _ttxt, fontsize='x-small')
                    axs[_r, _c].indicate_inset_zoom(axins)
        fig.savefig(self.out,  bbox_inches='tight')
        plt.close()

    def Heatgene(self, xyData):
        g = sns.clustermap(xyData, row_cluster=False)
        g.savefig(self.out)
        plt.close()

    def ClustMap(self, xyData, _colm):
        figsize = (20,20)
        colm = _colm.copy()

        cor1 = colm.Platform.unique()
        cor1 = dict(zip(cor1, plt.cm.Set3(range(len(cor1)))))

        cor2 = colm.Cellline.unique()
        cor2 = dict(zip(cor2, self.color_[:len(cor2)]))

        colm.Platform = colm.Platform.map(cor1)
        colm.Cellline = colm.Cellline.map(cor2)

        hm = sns.clustermap(xyData,
                            method='complete',
                            metric='euclidean',
                            z_score=None,
                            figsize=figsize,
                            linewidths=0.001,
                            cmap="coolwarm",
                            center=0,
                            #fmt='.2f',
                            #square=True, 
                            #cbar=True,
                            yticklabels=True,
                            xticklabels=True,
                            vmin=-1.1,
                            vmax=1.1,
                            annot=False,
                            row_colors=colm,
                            )
        hm.savefig(self.out)
        #hm.fig.subplots_adjust(right=.2, top=.3, bottom=.2)
        plt.close()

    def Cnvbox(self, xyData, x='Cellline',  y='CV'):
        #xyData = xyData[['Cellline', 'CV', 'Gini']].copy()
        import ptitprince as pt
        fig, ax = plt.subplots(figsize=(5,5))
        labels= sorted(xyData[x].unique())
        ort = 'v'
        pal = 'Set2'
        Col = sns.set_palette(sns.color_palette(self.color_[:8]))
        ax=pt.half_violinplot( x=x, y=y, data = xyData, palette = Col,
                                bw = .15, cut = 0.,scale = "area", width = .28, offset=0.17, 
                                linewidth = 0.8, 
                                inner = None, orient = ort)
        ax=sns.swarmplot(x=x, y=y,  data=xyData, palette='Pastel1', 
                        linestyles='-', size=2.5, linewidth=.3)
        ax=sns.boxplot( x=x, y=y, data=xyData, orient = ort, notch=False,
                        meanprops={'linestyle':'-.'}, width=0.3,
                        flierprops={'markersize':3, 'marker':'*', 'linestyle':'--', },
                        palette=Col, fliersize=3, linewidth=0.7)

        plt.xticks(rotation='270')
        plt.savefig( self.out, bbox_inches='tight')
        plt.close()

###################################first time################
class CellFit1:
    def catdf(self):
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

    def CellRegres(self, xyData, _M):
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

    def CMD(self):
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
class CellFit2:
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
    def getcounts(self, dfmrx):
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

    def getdf(self):
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

    def FGPlot(self, *args, **kwargs):
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

    def linearReg(self, xyData, OUT, PD = pd.DataFrame(), _M='sklr', R='Therical', C='Cells', R2=False):
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

    def linearRegP(self, T, P, OUT, R='Therical', C='Cells', Xc=['ECfiltcounts'], yc='Thervalue'):
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

    def linearRegC(self, xyData, OUT, _M='sklr', R='Therical', xl = 'BPcounts', yl='ECfiltcounts', R2=False):
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

    def predictCN(self, T, P, _M ='sklr',
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

    def CMD2(self, M = 'sklr', Type='maxlen'):
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

###################################third time################
class CellFit3:
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']
        self.MARKGENE = ['EGFR', 'CDK6', 'SEPTIN14', 'MYC', 'DENND3', 
                        'PCAT1', 'BAP1', 'SOX2', 'MUC4', 'MECOM', 'PIK3CA', 
                        'CCND1', 'MYCN', 'TERT', 'RPS6', 'SMARCA4', 'WDR60', 
                        'AC019257.8', 'DLG1', 'WNK1', 'MUC2', 'AHRR']

    def _getinfo(self, INF, TherVu, OU):
        INdf   = pd.read_csv(INF, sep='\t')
        self.INdf   = INdf[(INdf.Filter == 'Keep')].copy()
        self.INdf.rename(columns={"DNA": "Cells"}, inplace=True)
        self.TVmelt = TherVu
        self.outdir = OU
        self.outhead= 'allsamples.TRF.'
        self.outpre = self.outdir + '/' + self.outhead
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def getcounts(self, dfmrx):
        countdict={}
        for _, _l in dfmrx.iterrows():
            if _l.gene_name=='.':
                continue
            _G = _l.gene_name.split(';')
            _B = _l.gene_biotype.split(';')
            _S = dict(zip( _l.support_IDs.split(';'), map( int,_l.support_read_num.split(';')) ))
            for _i in list(zip(_G, _B)):
                #if _i[0] !='.':
                countdict.setdefault(_i, []).append(_S)
        countlist = []
        for k, v in countdict.items():
            genedict ={'gene': k[0], 'gene_biotype': k[1]}
            for _d in v:
                for _id, _count in _d.items():
                    if _id in genedict.keys():
                        genedict[_id] += _count
                    else:
                        genedict[_id] = _count
            genedict = pd.Series(genedict)
            countlist.append( genedict )
        countlist = pd.concat(countlist, axis=1).T
        return countlist

    def ecRegion(self, INs):
        BCounts = pd.DataFrame(columns=['gene', 'gene_biotype'])
        for IN in INs:
            UpFilter=pd.read_csv(IN, sep='\t')

            PLLinks = UpFilter.loc[(UpFilter['#chrom'].isin(self.PLMD)), 'LINKS'].unique()
            UpChr = UpFilter[~(UpFilter.LINKS.isin(PLLinks))].copy()
            UpPLD = UpFilter[(UpFilter.LINKS.isin(PLLinks))].copy()
            UpPLD = UpPLD.loc[ (UpPLD.groupby(by='LINKS')['length'].idxmax()) ] # Plasmid only keep the Type='maxlen'
            UpPLD.loc[(UpPLD['#chrom'].isin(self.PLMD)), 'gene_name'] = UpPLD.loc[(UpPLD['#chrom'].isin(self.PLMD)), '#chrom']
            UpFilter= pd.concat([UpChr, UpPLD], axis=0)

            UpFilter= self.getcounts(UpFilter)
            BCounts = BCounts.merge(UpFilter, on=['gene', 'gene_biotype'], how='outer')
        return BCounts

    def getdf(self, INs):
        #ACounts = self.CheakBPMT()
        BCounts = self.ecRegion(INs)
        BCounts.to_csv(self.outpre + 'gene.counts.txt', sep='\t', index=False)
        #CCounts = pd.concat(ACounts + BCounts, axis=0, sort=False)

        BCounts= pd.read_csv( self.outpre + 'gene.counts.txt', sep='\t' )
        xyData = BCounts[(BCounts.gene_biotype.isin(['protein_coding', '.']))]
        CCol   = xyData.columns.drop(['gene', 'gene_biotype'])
        xyData = pd.melt(xyData, id_vars=['gene'], value_vars=CCol, var_name='Cells', value_name='ECfiltcounts')
        xyData = xyData.merge(self.TVmelt, on='gene', how='outer')
        xyData = xyData.merge(self.INdf[['Cells', 'Rename', 'Group', 'Cellline', 'Platform']], on='Cells', how='outer')
        xyData.to_csv(self.outpre + 'trans.gene.plasmid.counts.txt', sep='\t', index=False)
        return xyData

    def getstat(self, xyData, GRP='Cells'):
        xyStat = []
        for _c, _g in xyData.groupby(GRP, sort=False):
            Train = _g[ (_g.gene.isin(self.PLMD) & (~ _g.Thervalue.isna()) & (~ _g.ECfiltcounts.isna()) )].copy()
            Pred  = _g[ (_g.gene.isin(self.MARKGENE) & (~ _g.ECfiltcounts.isna()) )].copy()
            if Train.shape[0] <3:
                continue
            State = self.TrainPre( Train[['ECfiltcounts']], Train['Thervalue'], Pred[['ECfiltcounts']])
            xyTP  = pd.concat( [Train, Pred], axis=0 )
            xyTP['Predict'] = np.r_[State['matrx']['y_pre'].values, State['predy']]
            xyTP['R2']      = State['R2']
            xyTP['R2_adj']  = State['R2_adj']
            xyTP['p_fv']    = State['p_fv']
            xyTP['p_tv']    = State['p_tv']
            xyTP['func']    = State['func']
            xyStat.append(xyTP)
        xyStat = pd.concat(xyStat,axis=0)
        xyStat.to_csv(self.outpre + 'linear.gene.plasmid.counts.txt', sep='\t', index=False)
        return xyStat

    def TrainPre(self, Xtr, ytr, Xpr, _M='sklr'):
        if _M == 'lr':
            Model = STATE().SMols
        elif  _M == 'loglr':
            Model = STATE().SMols2
        elif _M == 'enet':
            Model = STATE().SKEnet
        elif _M == 'sklr':
            Model = STATE().SKLR
        Stat = Model(Xtr, ytr)
        
        if Xpr.shape[0] >0:
            Stat['predy'] = Stat['clf'].predict(Xpr)
        else:
            Stat['predy'] = np.array([])
        return Stat

    def CMD3(self, INs, INF, TherVu, OU):
        self._getinfo(INF, TherVu, OU)
        '''
        xyData = self.getdf(INs)
        xyStat = self.getstat(xyData)
        #xyData = pd.read_csv(self.outpre + 'trans.gene.plasmid.counts.txt', sep='\t', low_memory=False)
        #xyStat = pd.read_csv(self.outpre + 'linear.gene.plasmid.counts.txt', sep='\t')
        xyStat['CELLs'] = xyStat.Cells.str.split('_BC').str[1].astype(int)
        xyStat.sort_values(by=['Group','CELLs'], inplace=True)
        PLOT(self.outpre + 'linear.gene.plasmid.counts.pdf').linepre(xyStat)
        PLOT(self.outpre + 'linear.gene.plasmid.counts2.pdf').linearRegP(xyStat)
        PLOT(self.outpre + 'linear.gene.plasmid.R2.pdf').cellBox(xyStat)
        '''
        geneMT = pd.read_csv(self.outpre + 'gene.counts.txt', sep='\t', low_memory=False)
        geneMT = geneMT[ (geneMT.gene_biotype.isin(['protein_coding', '.']))].fillna(0)
        geneMT.drop('gene_biotype', axis=1, inplace=True)
        geneMT.set_index('gene', inplace=True)
        #geneMT = geneMT/geneMT.sum(0)
        geneMT = geneMT[~(geneMT.index.isin(self.PLMD))]
        geneMT = geneMT[~(geneMT.index.str.contains('MT'))]
        print(geneMT.max(1).sort_values())
        PLOT(self.outpre + 'gene.counts.pdf').Heatgene(geneMT)

def Start3():
    THEORETV = [['2x35S-eYGFPuv-T878-p73', 10000],
                ['2x35S-LbCpf1-pQD', 8000],
                ['380B-eYGFPuv-d11-d15', 2000],
                ['5P2T-pKGW7', 1200],
                ['A10-pg-p221', 9000],
                ['Cas9-U6-sgRNA-pQD', 1500],
                ['HD-T878-UBQ10', 100],
                ['Lat52-grim-TE-MC9-prk6-pKGW7', 3000],
                ['Lat52-RG-HTR10-1-GFP-pBGW7', 5500],
                ['myb98-genomic-nsc-TOPO', 4000],
                ['pB2CGW', 7500],
                ['pHDzCGW', 800],
                ['pQD-in', 400],
                ['pro18-Mal480-d1S-E9t', 500],
                ['SunTag-CRISPRi', 200],
                ['V7-MC-HG-FA', 4000]]
    THEORETV = pd.DataFrame(THEORETV, columns=['gene', 'Thervalue'])
    #CELLS = [ 'BC%s'%i for i in range(1,13) ]

    IN=['/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PDAC/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/Nanopore/U2OS/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/U2OS/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/HEK293T/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',]
    INF='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210128.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/CellFit/20210128'
    CellFit3().CMD3(IN, INF, THEORETV, OU)

###################################fourth time################
class CellFit4:
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']
        self.MARKGENE = ['EGFR', 'CDK6', 'SEPTIN14', 'MYC', 'DENND3', 
                        'PCAT1', 'BAP1', 'SOX2', 'MUC4', 'MECOM', 'PIK3CA', 
                        'CCND1', 'MYCN', 'TERT', 'RPS6', 'SMARCA4', 'WDR60', 
                        'AC019257.8', 'DLG1', 'WNK1', 'MUC2', 'AHRR']

    def _getinfo(self, INF, TherVu, OU):
        INdf   = pd.read_csv(INF, sep='\t')
        self.INdf   = INdf[(INdf.Filter == 'Keep')].copy()
        self.INdf.rename(columns={"DNA": "Cells"}, inplace=True)
        self.TVmelt = TherVu
        self.outdir = OU
        self.outhead= 'allsamples.TRF.'
        self.outpre = self.outdir + '/' + self.outhead
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def getcounts(self, dfmrx):
        countdict={}
        for _, _l in dfmrx.iterrows():
            if _l.gene_name=='.':
                continue
            _G = _l.gene_name.split(';')
            _B = _l.gene_biotype.split(';')
            _S = dict(zip( _l.support_IDs.split(';'), map( int,_l.support_read_num.split(';')) ))
            for _i in list(zip(_G, _B)):
                #if _i[0] !='.':
                countdict.setdefault(_i, []).append(_S)
        countlist = []
        for k, v in countdict.items():
            genedict ={'gene': k[0], 'gene_biotype': k[1]}
            for _d in v:
                for _id, _count in _d.items():
                    if _id in genedict.keys():
                        genedict[_id] += _count
                    else:
                        genedict[_id] = _count
            genedict = pd.Series(genedict)
            countlist.append( genedict )
        countlist = pd.concat(countlist, axis=1).T
        return countlist

    def ecRegion(self, INs):
        BCounts = pd.DataFrame(columns=['gene', 'gene_biotype'])
        for IN in INs:
            UpFilter=pd.read_csv(IN, sep='\t')

            PLLinks = UpFilter.loc[(UpFilter['#chrom'].isin(self.PLMD)), 'LINKS'].unique()
            UpChr = UpFilter[~(UpFilter.LINKS.isin(PLLinks))].copy()
            UpPLD = UpFilter[(UpFilter.LINKS.isin(PLLinks))].copy()
            UpPLD = UpPLD.loc[ (UpPLD.groupby(by='LINKS')['length'].idxmax()) ] # Plasmid only keep the Type='maxlen'
            UpPLD.loc[(UpPLD['#chrom'].isin(self.PLMD)), 'gene_name'] = UpPLD.loc[(UpPLD['#chrom'].isin(self.PLMD)), '#chrom']
            UpFilter= pd.concat([UpChr, UpPLD], axis=0)

            UpFilter= self.getcounts(UpFilter)
            BCounts = BCounts.merge(UpFilter, on=['gene', 'gene_biotype'], how='outer')
        BCounts.to_csv(self.outpre + 'gene.counts.txt', sep='\t', index=False)
        return BCounts

    def getdf(self, INs):
        #ACounts = self.CheakBPMT()
        BCounts = self.ecRegion(INs)
        #CCounts = pd.concat(ACounts + BCounts, axis=0, sort=False)
        BCounts= pd.read_csv( self.outpre + 'gene.counts.txt', sep='\t' )
        xyData = BCounts[(BCounts.gene_biotype.isin(['protein_coding', '.']))]
        CCol   = xyData.columns.drop(['gene', 'gene_biotype'])
        xyData = pd.melt(xyData, id_vars=['gene'], value_vars=CCol, var_name='Cells', value_name='ECfiltcounts')
        xyData = xyData.merge(self.TVmelt, on='gene', how='outer')
        xyData = xyData.merge(self.INdf[['Cells', 'Rename', 'Group', 'Cellline', 'Platform']], on='Cells', how='outer')
        xyData.to_csv(self.outpre + 'trans.gene.plasmid.counts.txt', sep='\t', index=False)
        return xyData

    def TrainPre(self, Xtr, ytr, Xpr, _M='sklr'):
        if _M == 'lr':
            Model = STATE().SMols
        elif  _M == 'loglr':
            Model = STATE().SMols2
        elif _M == 'enet':
            Model = STATE().SKEnet
        elif _M == 'sklr':
            Model = STATE().SKLR

        Stat = Model(Xtr, ytr)
        
        if Xpr.shape[0] >0:
            Stat['predy'] = Stat['clf'].predict(Xpr)
        else:
            Stat['predy'] = np.array([])
        return Stat

    def getstat(self, xyData, GRP='Cells'):
        xyStat = []
        for _c, _g in xyData.groupby(GRP, sort=False):
            Train = _g[ (_g.gene.isin(self.PLMD) & (~ _g.Thervalue.isna()) & (~ _g.ECfiltcounts.isna()) )].copy()
            Pred  = _g[ (_g.gene.isin(self.MARKGENE) & (~ _g.ECfiltcounts.isna()) )].copy()
            if Train.shape[0] <3:
                continue
            State = self.TrainPre( Train[['ECfiltcounts']], Train['Thervalue'], Pred[['ECfiltcounts']])
            xyTP  = pd.concat( [Train, Pred], axis=0 )
            xyTP['Predict'] = np.r_[State['matrx']['y_pre'].values, State['predy']]
            xyTP['R2']      = State['R2']
            xyTP['R2_adj']  = State['R2_adj']
            xyTP['p_fv']    = State['p_fv']
            xyTP['p_tv']    = State['p_tv']
            xyTP['func']    = State['func']
            xyStat.append(xyTP)
        xyStat = pd.concat(xyStat,axis=0)
        xyStat.to_csv(self.outpre + 'linear.gene.plasmid.counts.txt', sep='\t', index=False)
        return xyStat

    def getdf1(self, INs):
        BCounts = self.ecRegion(INs)

        xyData = BCounts[(BCounts.gene_biotype.isin(['protein_coding', '.']))]
        CCol   = xyData.columns.drop(['gene', 'gene_biotype'])
        xyData = pd.melt(xyData, id_vars=['gene'], value_vars=CCol, var_name='Cells', value_name='ECfiltcounts')
        xyData = xyData.merge(self.INdf[['Cells', 'Rename', 'Group', 'Cellline', 'Platform']], on='Cells', how='outer')
        xyData = xyData.merge(self.TVmelt, on=['gene','Group'], how='right')
        xyData.to_csv(self.outpre + 'trans.gene.plasmid.counts.txt', sep='\t', index=False)

        return xyData

    def CMD4(self, INs, INF, TherVu, OU):
        TherVu = pd.read_csv(TherVu, sep='\t')
        self._getinfo(INF, TherVu, OU)
        self.Tcol   = TherVu.columns.drop('gene')
        self.TVmelt = pd.melt(TherVu, id_vars=['gene'], value_vars=self.Tcol, var_name='Group', value_name='Thervalue')
        xyData = self.getdf1(INs)
        xyStat = self.getstat(xyData)
        #xyData = pd.read_csv(self.outpre + 'trans.gene.plasmid.counts.txt', sep='\t', low_memory=False)

        xyStat = pd.read_csv(self.outpre + 'linear.gene.plasmid.counts.txt', sep='\t')
        xyStat['CELLs'] = xyStat.Cells.str.split('_BC').str[1].astype(int)
        xyStat.sort_values(by=['Group','CELLs'], inplace=True)
        PLOT(self.outpre + 'linear.gene.plasmid.R2.pdf').cellBox1(xyStat)
        PLOT(self.outpre + 'linear.gene.plasmid.counts.pdf').linepre(xyStat, R='Group',Order=None )

        SelectC = xyStat[['Cells', 'Cellline', 'R2']]\
                    .drop_duplicates(keep='first')\
                    .sort_values(by='R2', ascending=False)\
                    .groupby('Cellline', sort=False)\
                    .head(6)
        SelectC['neworder']=1
        SelectC['neworder'] = SelectC.groupby(by='Cellline', sort=False)['neworder'].apply(np.cumsum)
        xyStat= xyStat.merge(SelectC, on=['Cells', 'Cellline','R2'], how='right')
        xyStat.to_csv(self.outpre + 'linear.3cellline.plasmid.counts.xls', sep='\t', index=False)
        PLOT(self.outpre + 'linear.3cellline.plasmid.counts.pdf').linepre(xyStat, R='Cellline', C='neworder', Order=None )

        xyStat = xyStat[(xyStat.Cells.isin(['Colon-P2_BC5', 'PC3-P2_BC5', 'PDACDNA2_BC5']))]
        xyStat.to_csv(self.outpre + 'linear.3cells.plasmid.counts.xls', sep='\t', index=False)
        PLOT(self.outpre + 'linear.3cells.plasmid.counts.pdf').linepre(xyStat, R=None, C='Cells', Order=None )

def Start4():
    THEORETV = '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/Colon-P2_PC3-P2_PDAC_theoretical_value.txt'
    IN=['/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PDAC/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/Nanopore/U2OS/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/U2OS/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/HEK293T/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',]
    INF='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210128.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/CellFit/20210131'
    CellFit4().CMD4(IN, INF, THEORETV, OU)

###################################Nomalize ecDNA################
class NomalCounts:
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']

    def bamfilter(self, inbam):
        Counts  = 0
        samfile = pysam.AlignmentFile(inbam, "rb")
        def filter_read(read):            
            return ((read.flag in [0, 1, 16]) & 
                    (read.reference_name in self.CHRS))

        for read in samfile.fetch():
            if filter_read(read):
                Counts += 1
        samfile.close()
        return Counts

    def _mapreadA(self, inbam):
        f = open(inbam + '.st.stat', 'r')
        mapreads = [i for i in f.readlines() if 'reads mapped' in i]
        f.close()
        if mapreads:
            return int(mapreads[0].split('\t')[-1])
        else:
            #raise ValueError('cannot find the reads mapped line in the file: %s'%Bstat)
            print('cannot find the reads mapped line in the file: %s'%Bstat)
            return 0

    def _mapreads(self, infile):
        readsdict = {}
        for _, _l in infile.iterrows():
            Fstat= '{bamfile}/*/{sid}/{sid}.sorted.bam'.format(bamfile=_l.DNAWorkpath , sid=_l.DNA)
            Bstat= glob.glob(Fstat)
            if Bstat:
                Bstat = Bstat[0]
            else:
                raise ValueError('cannot find the file: %s'%Fstat)
            readsdict[_l.DNA] = Bstat
        KEYS = list(readsdict.keys())
        VULS = list(readsdict.values())

        with futures.ProcessPoolExecutor() as executor: #ThreadPoolExecutor/ProcessPoolExecutor
            CouBase = executor.map(self._mapreadA, VULS)
            CouBase = list(CouBase)
            K = [KEYS, VULS, CouBase]
            dump(K, self.outpre + 'readsdict.pkl')
        return readsdict

    def getcounts(self, dfmrx):
        countdict={}
        for _, _l in dfmrx.iterrows():
            if _l.gene_name=='.':
                continue
            _G = _l.gene_name.split(';')
            _B = _l.gene_biotype.split(';')
            _S = dict(zip( _l.support_IDs.split(';'), map( int,_l.support_read_num.split(';')) ))
            for _i in list(zip(_G, _B)):
                #if _i[0] !='.':
                countdict.setdefault(_i, []).append(_S)
        countlist = []
        for k, v in countdict.items():
            genedict ={'gene': k[0], 'gene_biotype': k[1]}
            for _d in v:
                for _id, _count in _d.items():
                    if _id in genedict.keys():
                        genedict[_id] += _count
                    else:
                        genedict[_id] = _count
            genedict = pd.Series(genedict)
            countlist.append( genedict )
        countlist = pd.concat(countlist, axis=1).T
        return countlist

    def ecRegion(self, INs):
        BCounts = pd.DataFrame(columns=['gene', 'gene_biotype'])
        for IN in INs:
            UpFilter=pd.read_csv(IN, sep='\t')

            PLLinks = UpFilter.loc[(UpFilter['#chrom'].isin(self.PLMD)), 'LINKS'].unique()
            UpChr = UpFilter[~(UpFilter.LINKS.isin(PLLinks))].copy()
            UpPLD = UpFilter[(UpFilter.LINKS.isin(PLLinks))].copy()
            UpPLD = UpPLD.loc[ (UpPLD.groupby(by='LINKS')['length'].idxmax()) ] # Plasmid only keep the Type='maxlen'
            UpPLD.loc[(UpPLD['#chrom'].isin(self.PLMD)), 'gene_name'] = UpPLD.loc[(UpPLD['#chrom'].isin(self.PLMD)), '#chrom']
            UpFilter= pd.concat([UpChr, UpPLD], axis=0)

            UpFilter= self.getcounts(UpFilter)
            BCounts = BCounts.merge(UpFilter, on=['gene', 'gene_biotype'], how='outer')
        BCounts.to_csv(self.outpre + 'gene.counts.txt', sep='\t', index=False)
        return BCounts

    def _getdb(self, INf):
        infile = pd.read_csv(INf, sep='\t')
        infile = infile[(infile.Filter == 'Keep')]
        #self._mapreads(infile)
        MCounts =  pd.DataFrame(load(self.outpre + 'readsdict.pkl'),index=['SID', 'INbam', 'Counts']).T
        MCounts.to_csv(self.outpre + 'readsdict.xls', sep='\t', index=False)

        #INFs =  (infile.DNAWorkpath + '/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF').unique()
        #BCounts= self.ecRegion(INFs)
    
        BCounts = pd.read_csv(self.outpre + 'gene.counts.txt', sep='\t')
        MCounts = dict(zip(MCounts['SID'], MCounts['Counts']))

        BCounts = BCounts.apply(lambda x: x*10e6/MCounts[x.name] if x.name in MCounts else x, axis=0)
        BCounts.to_csv(self.outpre + 'gene.counts.RPM.txt', sep='\t', index=False)

    def _getinfo(self, INf, OUT, Head ):
        self.outdir= OUT
        self.Head  = Head
        self.outpre= OUT + '/' + Head
        os.makedirs(self.outdir, exist_ok=True)
        return self
    
    def CMD(self, INf, OUT, Head):
        self._getinfo(INf, OUT, Head)
        self._getdb(INf)
        'Rscript /share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/genecounts2searatcounts.R'

def Start5():
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210128.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210131/GeneCount'
    Head='Nonalize.ecgeneCounts.'
    NomalCounts().CMD(INfile, OU, Head)
#samtools  view -@ 20 -bS -F  260  Colon-P1_BC6.sorted.bam | samtools  sort -@ 20 - -o $OU/${ID}.bam 
###################################CNV#######################
class CNV:
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']

    def _getinfo(self, INf, OUT, Head ):
        self.infile = pd.read_csv(INf, sep='\t')
        self.infile = self.infile[(self.infile.Filter == 'Keep')].rename(columns={'DNA': 'SID'})
        self.outdir= OUT
        self.Head  = Head
        self.outpre= OUT + '/' + Head
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getio(self, ftail='.all.cnv.xls'):
        CNVs = []
        for _l in self.infile.DNAWorkpath.unique():
            CNVfs = '{DNAWorkpath}/CNV/20210121/*{tail}'.format(DNAWorkpath=_l, tail=ftail)
            CNVf  = glob.glob(CNVfs)
            if CNVf:
                CNVf = CNVf[0]
            else:
                raise ValueError('cannot find the file: %s'%CNVfs)
            CNVf = pd.read_csv(CNVf, sep='\t')
            CNVs.append(CNVf)
        CNVs = pd.concat(CNVs, axis=0)
        return CNVs

    def ginilike(self, lcnv):
        return ( np.power(2, lcnv)*lcnv ).sum(skipna=True, axis=0)

    def cv(self, lcnv):
        return lcnv.std(skipna=True, axis=0)/lcnv.mean(skipna=True, axis=0)

    def cnvpearson(self, lcnv):
        _lcnv = lcnv.copy().dropna(axis=0, how='any')
        return pd.DataFrame( np.corrcoef(_lcnv, rowvar=False), columns=_lcnv.columns, index=_lcnv.columns)

    def copymatrix(self, vcol='logcopy'):
        logcopy = self.CNVs.pivot(index=['chrom', 'start', 'end', 'length', 'gc', 'rmsk', 'bins' ],
                                    columns=["SID"], values=vcol)
        logcopy.to_csv(self.outpre + 'all.sample.logcnv.txt', sep='\t', index=True)

        Score = pd.concat([self.cv(logcopy), self.ginilike(logcopy)], axis=1)
        Score.columns = ['CV', 'Gini']
        Score.reset_index(inplace=True)
        Score = self.infile[['SID', 'Rename', 'RNA', 'Group', 'Cellline', 'Platform']]\
                    .merge(Score, on=['SID'], how='right')
        Score.to_csv(self.outpre + 'all.sample.logcnv.cvgini.txt', sep='\t', index=False)
        PLOT(self.outpre + 'all.sample.logcnv.cv.pdf').Cnvbox(Score)
        PLOT(self.outpre + 'all.sample.logcnv.gini.pdf').Cnvbox(Score, y='Gini')

        Cor  = self.cnvpearson(logcopy)
        Cor.to_csv(self.outpre + 'all.sample.logcnv.pearson.txt', sep='\t', index=True)
        PLOT(self.outpre + 'all.sample.logcnv.pearson.pdf')\
            .ClustMap(Cor, self.infile[['SID', 'Platform', 'Cellline']].set_index('SID') )

    def CMD(self, INfile, OU, Head):
        self._getinfo(INfile, OU, Head)
        self.CNVs = self._getio()
        self.CNVs = self.CNVs[(self.CNVs.SID.isin(self.infile.SID))]
        self.CNVs.to_csv(self.outpre + 'all.sample.txt', sep='\t', index=False)
        self.copymatrix()

def Start6():
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210128.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210131/CNV'
    Head='CNV.Analyze.'
    CNV().CMD(INfile, OU, Head)

###################################RNA#######################
class RNA():
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']

    def _getinfo(self, INf, OUT, Head ):
        self.infile = pd.read_csv(INf, sep='\t')
        self.infile = self.infile[(self.infile.Filter == 'Keep')].rename(columns={'DNA': 'SID'})
        self.genbed = '/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf.gene.bed'
        self.genbed = pd.read_csv(self.genbed, sep='\t')[['#chrom', 'start', 'end', 'gene_name', 'gene_id', 'gene_biotype']]\
                        .rename(columns={'gene_name' : 'gene', 'gene_id' : 'gene_ID'}).copy()
        self.outdir= OUT
        self.Head  = Head
        self.outpre= OUT + '/' + Head
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getio(self, ftail='.rsemce.genes.results'):
        RNAs = []
        for _, _l in self.infile.iterrows():
            RNVfs= '{RNA}/{RNAID}/SS2/RSEM/{RNAID}*{tail}'.format(RNA=_l.RNAWorkpath , RNAID=_l.RNAID, tail=ftail)
            RNVf = glob.glob(RNVfs)
            if RNVf:
                RNVf = RNVf[0]
            else:
                raise ValueError('cannot find the file: %s'%RNVf)
            RNVf = pd.read_csv(RNVf, sep='\t')
            RNVf.insert(0, 'RNAID', _l.RNAID)
            RNVf[['gene_ID','gene']] = RNVf['gene_id'].str.split('_', expand=True)
            RNAs.append(RNVf[['RNAID', 'gene', 'gene_ID', 'TPM', 'FPKM']])
        RNAs = pd.concat(RNAs, axis=0)
        return RNAs

    def pearsonMT(self, rdf):
        rdf = rdf[ (rdf.sum(1)>2) ].copy()
        return pd.DataFrame( np.corrcoef(rdf, rowvar=False), columns=rdf.columns, index=rdf.columns)

    def expreseq(self, eqtype='TPM'):
        self.EQ = self.RNAs.pivot(index=['gene', 'gene_ID'], columns='RNA', values=eqtype)
        self.EQ.to_csv( '%sall.sample.%s.txt'%(self.outpre, eqtype), sep='\t', index=True)

    def RNAmtr(self):
        self.RNAs = self._getio()
        self.RNAs = self.infile[['Rename','RNA', 'RNA', 'Group', 'Cellline', 'Platform', 'RNAID']]\
                        .merge(self.RNAs, on='RNAID', how='right')
        self.RNAs.to_csv(self.outpre + 'all.sample.txt', sep='\t', index=False)

    def meanEQ(self, RNAs, eqtype='TPM'):
        for (_p, _c), _g in RNAs.groupby(by=['Platform', 'Cellline']):
            _G = _g.groupby(by=['gene_ID', 'gene'])[eqtype].mean()\
                    .to_frame('mean'+eqtype).reset_index()\
                    .merge(self.genbed, on=['gene_ID', 'gene'], how='left')
            _G = _G[['#chrom', 'start', 'end', 'gene_ID', 'gene', 'gene_biotype', 'meanTPM']]\
                    .sort_values(by=['#chrom', 'start', 'end'])
            _G = _G[((_G['#chrom'].isin(self.CHRS)) & (_G['gene_biotype']=='protein_coding'))]
            _G.to_csv('%sall.sample.mean%s_%s_%s.txt'%(self.outpre, eqtype, _p, _c), sep='\t', index=False)

            #_K = _G.loc[(_G.meanTPM>0), ['#chrom', 'start', 'end','meanTPM']].copy()
            #_K['meanTPM'] = np.log2(_K['meanTPM'])
            #_K['#chrom']  = 'hs' + _K['#chrom']
            #_K.to_csv('%sall.sample.logmean%s_%s_%s.txt'%(self.outpre, eqtype, _p, _c), sep='\t', index=False)

    def CMD(self, INfile, OU, Head):
        self._getinfo(INfile, OU, Head)
        self.infile = self.infile[(~self.infile.RNAID.isna())]

        #self.RNAmtr()
        self.RNAs = pd.read_csv(self.outpre + 'all.sample.txt', sep='\t')
        self.meanEQ(self.RNAs)
        #self.expreseq()
        #self.expreseq(eqtype='FPKM')

        '''
        self.Cor  = self.pearsonMT(self.EQ)
        self.Cor.to_csv(self.outpre + 'all.sample.pearson.txt', sep='\t', index=True)
        PLOT(self.outpre + 'all.sample.pearson.pdf')\
            .ClustMap(self.Cor, self.infile[['RNA', 'Platform', 'Cellline']].set_index('RNA'))
        '''

def Start7():
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210128.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210131/RNA'
    Head='RNA.Analyze.'
    RNA().CMD(INfile, OU, Head)
Start7()
