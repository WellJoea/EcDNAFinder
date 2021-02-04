#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy  as np
from joblib import Parallel, delayed
import pybedtools as bt
import re

from .EcUtilities import Utilities
from .EcVisual import Visal


class FilterLinks():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.chrs=[str(i) for i in range(1,23)] + ['MT','X','Y'] \
                    + ['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                        '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                        'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                        'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                        'SunTag-CRISPRi', 'V7-MC-HG-FA']

    def _getinfo(self):
        self.outdir= self.arg.Region
        self.arg.outpre = '%s/%s'%(self.arg.Region, self.arg.regionpre)
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getupmerge(self):
        self._getinfo()
        if self.arg.upmerge:
            self.UpMerge = self.arg.upmerge
        else:
            self.UpMerge = self.arg.outpre + '.UpMerge'
        return self
    
    def AnnotRepeat(self, _inbed, minover=1, trfdistance=500):
        minover     = self.arg.minover #1
        trfdistance = self.arg.trfdistance

        repeat= pd.read_csv(self.arg.simplerepeat, sep='\t', header=None, names=['#chrom_trf', 'start_trf', 'end_trf', 'trf'])
        repeat.drop('trf', axis=1, inplace=True)

        repeat['#chrom_trf'] = repeat['#chrom_trf'].astype(str)
        _inbed['#chrom'] = _inbed['#chrom'].astype(str)

        COL1 = _inbed.columns.tolist()
        COLs = COL1 + repeat.columns.tolist()
        repeat = repeat[(repeat['#chrom_trf'].str.len()<6)]
        repeat = bt.BedTool.from_dataframe(repeat)
        repeat = repeat.sort().merge()
        inbed  = bt.BedTool.from_dataframe(_inbed)
        inbed  = inbed.intersect(repeat, s=False, S=False, loj=True)\
                        .to_dataframe(disable_auto_names=True,  header=None, names=COLs)

        inbed = inbed.infer_objects()
        inbed['#chrom'] = inbed['#chrom'].astype(str)
        inbed['TRF']    = 'Out'
        inbed['len_m']  = inbed[['end', 'end_trf']].min(1) - inbed[['start', 'start_trf']].max(1) + 1
        inbed.loc[( (inbed.start_trf != -1) &(inbed.len_m >= minover)), 'TRF']='TRF'

        inbed['start_trf']  -= trfdistance
        inbed['end_trf']    += trfdistance
        inbed.loc[( (inbed.TRF.str.contains('TRF')) & inbed.start.between(inbed.start_trf, inbed.end_trf) ), 'TRF'] +=';Head'
        inbed.loc[( (inbed.TRF.str.contains('TRF')) & inbed.end.between(inbed.start_trf, inbed.end_trf) ), 'TRF'] +=';Tail'

        inbed = inbed[COL1 + ['TRF']]\
                    .groupby(by=COL1, sort=False)['TRF']\
                    .apply(lambda x: 'TRFBP' if re.search(r'Head|Tail', x.str.cat(sep=';')) else 'NOTRF')\
                    .to_frame('TRF').reset_index()
        return inbed

    def FormatLink(self):
        self._getupmerge()
        UpMerge =pd.read_csv(self.UpMerge, sep='\t', dtype={'#chrom':str}, low_memory=False)
        UpMerge['MaxCovers']   = UpMerge.Covers.apply(lambda x:max(map( float, str(x).split(';'))))
        UpMerge['MaxDepths']   = UpMerge.Depths.apply(lambda x:max(map( float, str(x).split(';'))))
        UpMerge['MaxBPHTNums'] = UpMerge.BPHTNums.apply(lambda x:max(map( int, str(x).split(';'))))
        UpMerge['LINKSLen']    = UpMerge.groupby('LINKS')['length'].transform('sum')
        UpMerge.insert(8, 'LinkScore', '')

        UpMerge.loc[(UpMerge.MaxBPHTNums>= self.arg.breakpiontnum), 'LinkScore'] += 'MultiBP;'
        UpMerge.loc[(UpMerge.MaxCovers>= self.arg.maxcoverage), 'LinkScore'] += 'Cover;'
        UpMerge.loc[(UpMerge.MaxDepths>= self.arg.maxdepth), 'LinkScore']    += 'Depth;'
        UpMerge.loc[(UpMerge.support_num>= self.arg.minsupportnum), 'LinkScore'] += 'Support;'
        UpMerge.loc[(UpMerge.support_ID_num>= self.arg.minsupportidnum), 'LinkScore'] += 'SupportID;'
        UpMerge.loc[(UpMerge.LINKSLen<= self.arg.maxlinklen), 'LinkScore'] += 'LenKeep;'
        UpMerge.loc[(UpMerge['#chrom'].isin(self.chrs)), 'LinkScore'] += 'Chr;'

        Drop = UpMerge[~(UpMerge.LinkScore.str.contains('Support;SupportID;LenKeep;Chr;'))]['LINKS']
        UpFilter = UpMerge[~(UpMerge.LINKS.isin(Drop))]

        Keys     = ['#chrom', 'start', 'end']
        UpFilter = UpFilter.merge( self.AnnotRepeat(UpFilter[Keys].drop_duplicates(keep='first')), on=Keys, how='outer')
        TRF   = UpFilter[(UpFilter.TRF=='TRFBP')]['LINKS']
        UpTRF = UpFilter[~(UpFilter.LINKS.isin(TRF))]

        UpFilter.to_csv(self.arg.outpre + '.UpFilter', sep='\t',index=False)
        UpTRF.to_csv(self.arg.outpre + '.UpFilterTRF', sep='\t',index=False)
