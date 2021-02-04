#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy  as np
from joblib import Parallel, delayed
import copy

#import coverdepth
from EcMagiccube import InterV2, InterVs, InterSm, OrderLinks, GroupBY
from .EcUtilities import Utilities
from .EcVisual import Visal

class MergeReads():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.maxreadstwoends = self.arg.maxreadstwoends
        self.readsmergeways  = self.arg.readsmergeways

    def _getinfo(self, _info):
        self.info = _info
        self.inid = _info.sampleid
        self.outdir= '%s/%s'%(self.arg.Merge, self.inid)
        self.arg.outpre= '%s/%s'%(self.outdir, self.inid)
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getkeep(self):
        self._inbed= '{0}/{1}/{1}.Keep'.format(self.arg.Search, self.inid)
        if not os.path.exists(self._inbed):
            self.inbed =pd.DataFrame()
            self.log.CW('cannot find the file: ' + self._inbed)
        else:
            self.inbed = pd.read_csv( self._inbed, sep='\t', dtype={'#chrom':str}, low_memory=False)
            self.inbed[['start', 'end']]  = self.inbed[['start', 'end']].astype(int)
            self.inbed['#chrom']  = self.inbed['#chrom'].astype(str)
            self.inbed['HTSites']  = self.inbed['HTSites'].map(eval)
        return self

    def orderlinks(self, _G): #check
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

    def updataLinks(self, inbed):
        sortN = ['SID', 'query_name', 'raw_order']
        gropN = ['SID', 'query_name']
        ColmR = ['#chrom', 'start_n', 'end_n', 'length_n', 'forword', 'raw_order','query_name', 'SID']
        ColmA = ['forword_n', 'Order', 'Link', 'LINKS']

        inbed = inbed.copy()
        inbed['raw_order']   = inbed['raw_order'].astype(int)
        inbed = inbed.merge(inbed.groupby(by=gropN).size().to_frame(name='Type').reset_index(), on=gropN)
        inbed = inbed.sort_values(by=sortN) #keep raw order right

        #Reduce compution time
        inbed1 = inbed[(inbed.Type <=1)].copy()
        inbed1['forword_n'] = '+'
        inbed1['Order'] = 1
        inbed1['Link'] = inbed1[['#chrom', 'start_n', 'end_n', 'forword']]\
                                    .apply(lambda x: '{0}:{1}-{2}'.format(*x[:3]), axis=1)
        inbed1['LINKS'] = inbed1['Link']

        inbed2 = inbed[(inbed.Type > 1)]
        if inbed2.shape[0] >0:
            outlink = Parallel( n_jobs=-1, backend='loky')( delayed( OrderLinks)(_g[ColmR].to_numpy()) 
                                for _l, _g in inbed2.groupby(by=gropN, sort=False))
            outlink = pd.DataFrame(np.vstack(outlink), columns=ColmR+ColmA)
            inbed2  = inbed2.merge(outlink, on=ColmR)
        return pd.concat([inbed1, inbed2], axis=0, sort=False)

    def supportParall(self, _g):
        COLS = ['#chrom', 'start', 'end', 'LINKS', 'length', 'forword', 'Type', 'Order',
                'support_num', 'support_read_num', 'support_ID_num', 'support_IDs']
        _G = _g.copy()
        _K = _g[(_g.Order == 1)]\
                .groupby(by='SID',sort=False)['query_name'].size().to_frame(name='SUP').reset_index()

        _G['support_num'] = _g['query_name'].unique().size
        _G['support_IDs']    =  _K['SID'].str.cat(sep=';')
        _G['support_ID_num']   =  _K['SID'].size
        _G['support_read_num'] =  _K['SUP'].astype(str).str.cat(sep=';')
        _G = _G[COLS].drop_duplicates(keep='first')
        return _G

    def statCircle(self, _G):
        BPHTNum = _G.Order.unique().size
        if BPHTNum == 1:
            _Q = _G.HTSites.map(InterV2).sum()
            _C = InterSm(InterVs(_Q)) # reduce time
            _D = InterSm(_Q)               # reduce time
            _L = _G.length_n.values[0]
            #Cover, Depth = coverdepth.coverdepth(_Q, _R)
            Cover, Depth = _C/_L, _D/_L
        else:
            Cover = 1
            Depth = _G.shape[0]

        return pd.Series({
            'LINKS': _G.iloc[0]['LINKS'],
            'LINKSLen' : _G.length_n.sum(),
            'SID'  : _G.iloc[0]['SID'],
            'mapped_reads' : _G.iloc[0]['mapped_reads'],
            'Cover': round(Cover, 3),
            'Depth': round(Depth, 3),
            'BPHTNum': BPHTNum,
            'support_ID_num' : _G.shape[0],
            'reads' : _G.query_name.str.cat(sep=';')
            })

    def mergeLinks(self, _inbed):
        GRPA  = ['LINKS']
        GRPE  = ['LINKS', 'SID']

        COL1  = ['#chrom', 'start_n', 'end_n', 'length_n', 'Order', 'fflag', 'HTSites', 'query_name', 'mapped_reads']
        COL2  = ['#chrom', 'start_n', 'end_n', 'Type', 'length_n', 'forword_n', 'LINKS', 'Order']

        #Support = _inbed.loc[(_inbed.fflag.str.contains(';HTBREAKP')), COL1 + GRPE]\
        #            .groupby(by=GRPE, sort=False)\
        #            .apply(lambda x:self.statCircle(x)).reset_index()
        # reduce time
        Support = _inbed.loc[(_inbed.fflag.str.contains(';HTBREAKP')), COL1 + GRPE].groupby(by=GRPE, sort=False)
        Support = Parallel( n_jobs=-1, backend='loky')( delayed( self.statCircle )(_g) for _l, _g in Support)
        Support = pd.concat(Support, axis=1).T.infer_objects()
        Support['EcRPM'] = GroupBY( Support[['SID','support_ID_num', 'mapped_reads']].values, _M='EcRPM' )
        #Visal().Lmplot('./aa.pdf', x='support_ID_num', y='EcRPM', hue='SID', data=Support)

        Supgrpb = Support.groupby(by=['LINKS'], sort=True)
        Suplist = [ Supgrpb['support_ID_num'].sum().to_frame('support_num'),
                    Supgrpb['SID'].size().to_frame('support_ID_num'),
                    Supgrpb['Cover'].mean().to_frame('Mean_Cover'),
                    Supgrpb['Depth'].mean().to_frame('Mean_Depth'),
                    Supgrpb['BPHTNum'].mean().to_frame('Mean_BPHTNum'),
                    Supgrpb['SID'].apply(lambda x: x.str.cat(sep=';')).to_frame('support_IDs'),
                    Supgrpb['support_ID_num'].apply(lambda x: x.astype(str).str.cat(sep=';')).to_frame('support_read_num'),
                    Supgrpb['Cover'].apply(lambda x: x.astype(str).str.cat(sep=';')).to_frame('Covers'),
                    Supgrpb['Depth'].apply(lambda x: x.astype(str).str.cat(sep=';')).to_frame('Depths'),
                    Supgrpb['BPHTNum'].apply(lambda x: x.astype(str).str.cat(sep=';')).to_frame('BPHTNums') ]
        Suplist = pd.concat( Suplist, ignore_index=False, join = 'outer', sort = False, axis=1).reset_index()
        del Supgrpb

        inbed = _inbed[COL2].drop_duplicates(keep='first').copy()
        inbed.rename(columns={'start_n': 'start', 'end_n': 'end', 'length_n': 'length', 'forword_n': 'forword'}, inplace=True)
        inbed = inbed.merge(Suplist, on='LINKS', how='outer')
        return inbed, Support

    def mergeReads(self, _inbed, Lplot=True, Hplot=False):
        inbed = Utilities(self.arg, self.log)\
                    .mapanytwo(_inbed, maxdistance = self.maxreadstwoends, maxreg = self.readsmergeways)

        # merge breakpoint
        inbed = MergeReads(self.arg, self.log).updataLinks(inbed)
        inbed.to_csv(self.arg.outpre+'.Links', sep='\t', index=False)

        inbed, Support = MergeReads(self.arg, self.log).mergeLinks(inbed)
        inbed.to_csv(self.arg.outpre+'.LinksUp', sep='\t', index=False)
        Support.to_csv(self.arg.outpre+'.Support', sep='\t', index=False)

        KEYS    = ['#chrom', 'start', 'end', 'length']
        BedAnno = inbed[KEYS].drop_duplicates(keep='first').sort_values(by=KEYS)
        BedAnno = Utilities(self.arg, self.log).annogene(BedAnno, self.arg.outpre)

        inbed = inbed.merge(BedAnno, on=KEYS, how='left')\
                    .sort_values(by=['Type', 'LINKS', 'Order', '#chrom', 'start', 'end'])
        inbed.to_csv(self.arg.outpre+'.UpMerge', sep='\t', index=False)

        inbed = inbed.sort_values(by=['Type', 'Order', '#chrom', 'start', 'end', 'LINKS'])
        inbed.to_csv(self.arg.outpre+'.UpMerge_sort', sep='\t', index=False)

        if Lplot:
            Visal().query_length(inbed, self.arg.outpre+'.UpMerge.length.pdf', X='length', Dup='', log=True, title='breakpoint length' )
            #Visal().clustmap(pvot, self.outpre+'.Keep.matrix.pdf')
            #Visal().clustmap(np.log2(pvot+1), self.outpre+'.Keep.matrix.log2.pdf')

    def EachEcDNA(self, _info):
        self._getinfo(_info)
        self._getkeep()
        self.log.CI('start merging breakpoin region: ' + self.inid)
        if not self.inbed.empty:
            self.mergeReads( self.inbed )
        else:
            self.log.CW('cannot find any circle region singal: ' + self.inid)
        self.log.CI('finish merging breakpoin region: ' + self.inid)
