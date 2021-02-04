#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : 03.bamtrans.py
* @Author  : Zhou Wei                                     *
* @Date    : 2020/10/10 11:41:07                          *
* @E-mail  : welljoea@gmail.com                           *
* @Version : --                                           *
* You are using the program scripted by Zhou Wei.         *
* If you find some bugs, please                           *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''
import pysam
import pybedtools as bt
from Bio.Seq import Seq
import os
from joblib import Parallel, delayed
import glob
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', 10000000)
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns

class GetSimu():
    def __init__(self):
        pass
    def simuget(self, reffa, region, ratio, overlap):
        Type = region[-1]
        if Type=='a':
            regions= reffa.fetch(*region[0:3])
            regionn= '%s:%s_%s-r%s-v%s-t%s'%( tuple(region[0:3] + [ratio, overlap, Type]) )
        elif Type=='b':
            regions= reffa.fetch(*region[0:3]) + reffa.fetch(*region[3:6])
            regionn= '%s:%s_%s_%s:%s_%s-r%s-v%s-t%s'%( tuple(region[0:6] + [ratio, overlap, Type]) )
        elif Type=='c':
            regionf= reffa.fetch(*region[0:3])
            regionr= str(Seq(regionf).reverse_complement())[: int((region[2]-region[1])*region[3]) ]
            regions= regionf + regionr
            regionn= '%s:%s_%s_R%s-r%s-v%s-t%s'%( tuple(region[0:4] + [ratio, overlap, Type]) )

        regionb= int(len(regions)*ratio)
        regionq= regions[regionb-overlap:] + regions[:regionb+overlap]
        return '@%s\n%s\n+\n%s'%(regionn, regionq, '~'*len(regionq) )

    def simufa(self):
        hg38_ens='/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
        hg38_ens=pysam.FastaFile(hg38_ens)
        simatx=[
            ['10', 13070000, 13090000, 'a'], 
            ['11', 13800000, 13830000, 'a'],
            ['1' , 13610000, 13615000, '3', 13710000, 13715000, 'b'],
            ['2' , 67250000, 67251000, '2', 67800000, 67810000, 'b'], 
            ['3' , 80950000, 80970000, 0.2, 'c'],
            ['3' , 80950000, 80970000, 0.5, 'c'],
            ['3' , 80950000, 80970000, 0.7, 'c'],
            ['3' , 80950000, 80970000, 1.0, 'c'],
            ['4' , 21500000, 21520000, 0.2, 'c'],
            ['4' , 21500000, 21520000, 0.5, 'c'],
            ['4' , 21500000, 21520000, 0.7, 'c'],
            ['4' , 21500000, 21520000, 1.0, 'c'],
            #['12', 13830000, 13800000, 'a1'],
            #['13', 13830000, 13800000, 'a2'],
            #['3' , 67250000, 67251000, '2', 67800000, 67810000, 'b1'], 
            #['4' , 67250000, 67251000, '2', 67800000, 67810000, 'b2'], 
        ]
        ratios=[0.1, 0.3, 0.5, 0.7, 0.9]
        ovlaps=[0, 100, -100, -500]
        allfas=[ self.simuget(hg38_ens, _s, _r, _o) for _s in simatx for _r in ratios for _o in ovlaps]
        allfas='\n'.join(allfas)
        f=open('./simulate.fq','w')
        f.write(allfas)
        f.close()
#GetSimu().simufa()

class Visal():
    def __init__(self):
        pass
    def query_length(self, _indf, out):
        if not _indf.empty:
            indef = _indf[['query_name', 'query_length']].drop_duplicates(keep='first')
            indef['query_length'] = indef['query_length'].astype(int)
            dp = sns.displot(data=indef, x="query_length", kde=True)
            #hm.savefig(self.out)
            #dp.despine(right=False, top=False)
            dp.set_xticklabels(rotation=270)
            plt.tight_layout()
            plt.savefig( out )
            plt.close()

class Utilities():
    def __init__(self ):
        pass
    def readgtf(self, gtffile):
        def splitinfo(_l):
            _l = _l.strip(';').split('; ')
            _k = ['']*3
            for _ll in _l:
                if  _ll.split(' ')[0]=='gene_id':
                    _k[0] = _ll.split(' ')[1].strip('"')
                if  _ll.split(' ')[0]=='gene_name':
                    _k[1] = _ll.split(' ')[1].strip('"')
                if  _ll.split(' ')[0]=='gene_biotype':
                    _k[2] = _ll.split(' ')[1].strip('"')
            return _k
        with open(gtffile, 'r') as  f:
            gtf = [ i.strip().split('\t')[:-1] + splitinfo(i.strip().split('\t')[-1]) 
                    for i in f.readlines() 
                    if (not i.startswith('#')) and (i.split('\t')[2]=='gene') ]
        gtf = pd.DataFrame(gtf, columns=['#chrom', 'db', 'gtype', 'start', 'end', 'U1', 'forword',
                                            'U2', 'gene_id', 'gene_name', 'gene_biotype'])
        gtf['length'] = gtf.end.astype(int) - gtf.start.astype(int) + 1
        gtf = gtf[['#chrom', 'start', 'end', 'gtype', 'length', 'forword', 'gene_name', 'gene_id', 'gene_biotype']]
        gtf.to_csv('./Homo_sapiens.GRCh38.100.gtf.gene.bed', sep='\t',index=False)

    def maxbeddistance(self, _inbed):
        inbed = _inbed.reset_index(drop=True)
        k=[]
        if not inbed.empty:
            for _n, _l in inbed.iterrows() :
                for _m, _j in inbed.loc[_n+1:].iterrows():
                    if (_l['#chrom'] == _j['#chrom']) and (_l['forword'] == _j['forword']):
                        k.append( [_n ,_m, _m - _n, inbed.loc[_n:_m,'length'].sum() ])
        if k:
            k = pd.DataFrame(k, columns=['s','e','d','l'])\
                    .sort_values(by=['l','d'],ascending=[False, False])\
                    .iloc[0,:]
            return inbed.loc[k.s : k.e, :]
        else:
            return inbed

class Mapping():
    def __init__(self, fq1, inid, outdir):
        self.fq1 = fq1
        self.inid = inid
        self.outdir=outdir + '/' + inid
        os.makedirs(self.outdir, exist_ok=True)

    def emptyfile(_func):
        def wrapper(self, *args, **kargs):
            file = self.fq1
            if os.path.exists(file):
                sz = os.path.getsize(file)
                if not sz:
                    print(file, " is empty!")
                else:
                    _func(self, *args, **kargs)
            else:
                print(file, " is not exists!")
        return wrapper

    @emptyfile
    def SoftMINI(self,
                 samtls='/share/home/share/software/samtools-1.10/bin/samtools',
                 minip2='/share/home/share/software/minimap2-2.17_x64-linux/minimap2',
                 REF='/share/home/share/Repository/GenomeDB/Index/Homo_Sapiens/MinimapDB/ENSEMBL_GRch38_PRIM_MINIM/Homo_sapiens.GRCh38.dna.primary_assembly.mmi'):
        cmd = '''
        mkdir -p {outdir}
        {minimap2} \
            -ax asm20 \
            -t 20 \
            -k 19  -w 10 -H -O 5,56 -E 4,1 -A 2 -B 5 -z 400,50 -r 2000 -g 5000 --lj-min-ratio 0.5 \
            --cs --MD -Y \
            --secondary no \
            {REF} \
            {fq1} | \
            {samtools} view -bS -@ 10  - \
            > {outdir}/{ID}.bam
        {samtools} sort -@ 20 -o {outdir}/{ID}.sorted.bam {outdir}/{ID}.bam
        {samtools} index -@ 20  {outdir}/{ID}.sorted.bam
        '''.format(minimap2=minip2, REF=REF, fq1=self.fq1, samtools=samtls,outdir=self.outdir,ID=self.inid)
        print(cmd)
        os.system(cmd)

class SoftCMD():
    def __init__(self):
        self.annotatepeak='/share/home/share/software/Homer/bin/annotatePeaks.pl'
        self.hg38gtf='/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf'

    def emptyfile(_func):
        def wrapper(self, *args, **kargs):
            file = args[0]
            if os.path.exists(file):
                sz = os.path.getsize(file)
                if not sz:
                    print(file, " is empty!")
                else:
                    _func(self, *args, **kargs)
            else:
                print(file, " is not exists!")
        return wrapper

    def annotatebed(self, inbed):
        cmd = '''
        INBED={inbed}
        GTF={hg38gtf}
        {annotatepeak} \
            $INBED \
            hg38 \
            -gtf $GTF \
            -gid \
            -cpu 10 \
            -go $INBED.go \
            -genomeOntology  $INBED.go.detail \
            1> $INBED.annotate.txt
            2> $INBED.annotate.log'''.format(inbed=inbed, hg38gtf=self.hg38gtf, annotatepeak=self.annotatepeak)
        print(cmd)
        os.system(cmd)

class SoftFetch():
    def __init__(self, inbam, inid, outdir):
        self.inbam=inbam
        self.inid = inid
        self.outdir=outdir + '/' + inid
        self.chrs=[str(i) for i in range(1,23)] + ['MT','X','Y'] #remove
        self.minsoflt = 5
        os.makedirs(self.outdir, exist_ok=True)

    def cigarmerge(self, ct, indel=100000, skip=1000000000, hard=100000, pad=1000000000, match='Q'):
        ctN = ct[:1]
        for i in ct[1:]:
            if match=='Q':
                if ( i[1] <=indel and i[0] ==1 ):
                    i = (0, i[1])
                elif ( i[1] <=indel and i[0] ==2 ):
                    i = (0, 0)
                elif ( i[1] <=skip and i[0] ==3 ):
                    i = (0, 0)
                elif ( i[1] <=hard and i[0] ==5 ):
                    i = (0, 0)
                elif ( i[1] <=pad and i[0] ==6 ):
                    i = (0, 0)
                else:
                    i = i

            elif match=='R':
                if ( i[1] <=indel and i[0] ==1 ):
                    i = (0, 0)
                elif ( i[1] <=indel and i[0] ==2 ):
                    i = (0, i[1])
                elif ( i[1] <=skip and i[0] ==3 ):
                    i = (0, i[1])
                elif ( i[1] <=hard and i[0] ==5 ):
                    i = (0, 0)
                elif ( i[1] <=pad and i[0] ==6 ):
                    i = (0, 0)
                else:
                    i = i

            if ( ctN[-1][0]==0 and i[0]==0 ):
                ctN[-1] = (ctN[-1][0] + i[0], ctN[-1][1] + i[1])
            else:
                ctN.append(i)
        return ctN

    def bamcigarsoft(self):
        samfile = pysam.AlignmentFile(self.inbam, "rb")
        #samidx = pysam.IndexedReads(samfile)
        #samidx.build()
        sampd = []
        for read in samfile.fetch():
            tags = dict(read.tags)
            raw_cigarstring = read.cigarstring
            raw_cigartuples = read.cigartuples
            reference_name = read.reference_name
            softlens = [ i[1] for i in raw_cigartuples if (i[0]==4) and (i[1] >self.minsoflt) ]

            if (('SA' in tags) \
                    and ('S' in raw_cigarstring) \
                    and len(softlens) >0):
                    #and (reference_name in self.chrs) \
                TP = tags['tp']
                SA = tags['SA']
                query_name = read.query_name
                flag = read.flag
                reference_start  = read.reference_start
                read.cigartuples = self.cigarmerge(raw_cigartuples, match='Q')
                cigartuples_ref  = self.cigarmerge(raw_cigartuples, match='R')
                query_sequence   = read.query_sequence
                query_qualities  = pysam.array_to_qualitystring(read.query_qualities)
                is_reverse = '-' if read.is_reverse else '+'
                sampd.append([self.inid, query_name, flag, reference_name, reference_start, read.cigartuples, cigartuples_ref,
                              is_reverse, len(query_sequence), TP, SA, read.cigarstring ,raw_cigarstring, raw_cigartuples, query_sequence, query_qualities])
                if len( [i[0] for i in raw_cigartuples if i[0] not in [0, 1, 2, 4]] ) >0:
                    print(query_name, raw_cigartuples)

        sampd = pd.DataFrame(sampd, 
                            columns=['SID', 'query_name', 'flag', 'reference_name', 'reference_start', 'cigartuples', 'cigartuples_ref', 'is_reverse',
                                     'query_length', 'TP', 'SA', 'cigarstring', 'raw_cigar', 'raw_cigartuples', 'query_sequence', 'query_qualities'])
        sampd = sampd.merge(sampd.groupby('query_name')['query_name'].size().reset_index(name='query_counts'),
                            on='query_name', how='outer')
        sampd = sampd[(sampd.query_counts>1)]

        sampd.sort_values(by=['query_name', 'reference_name', 'reference_start', 'flag'],
                          ascending=[True, True, True, True], inplace=True)
        sampd.to_csv('%s/%s.chimeric.txt'%(self.outdir, self.inid), sep='\t', index=False)
        return sampd

    def getsoftfq(self, _l, softminlen=100):
        cigt  = eval(_l.cigartuples) if type(_l.cigartuples)==str else _l.cigartuples
        start = 0
        SEQs  = []
        for  n,i in enumerate(cigt):
            end = int(start) + int(i[1])
            if (i[0] ==4 and i[1]>=softminlen):
                seq = _l.query_sequence[start:end]
                qul = _l.query_qualities[start:end]
                name= '@%s_soft%s-%s_%s_%s'%(_l.query_name, _l.reference_name, _l.reference_start, n, i[1])
                SEQs.append('%s\n%s\n+\n%s'%(name, seq, qul) )
            start += int(i[1])
        return '\n'.join(SEQs)

    def getsoftregion(self, _g):
        start  = int(_g.reference_start)
        cigref = eval(_g.cigartuples_ref) if type(_g.cigartuples_ref)==str else _g.cigartuples_ref
        cigartuples = eval(_g.cigartuples) if type(_g.cigartuples)==str else _g.cigartuples
        match_type  = ''.join([ 'S' if i[0]==4 else 'M' for i in cigartuples ])
        cigarpos = [(0, cigartuples[0][1])]
        for i in cigartuples[1:]:
            cigarpos.append( (cigarpos[-1][1], cigarpos[-1][1]+i[1]) )

        Regs  = []
        for  n,i in enumerate(cigref):
            if (i[0] !=4 ):
                bed = [ _g.reference_name, start, start + int(i[1]-1), _g.SID, i[1], _g.is_reverse, _g.query_name, 
                        _g.flag, _g.cigarstring, cigarpos, cigarpos[n], _g.query_length, cigref, match_type, n, _g.query_counts]
                Regs.append(bed)
                start += int(i[1]-1)
        return Regs

    def getsoft(self, softfq=True, researh=True):
        print('start finding soft signal: ' + self.inid)
        if researh:
            INsoft = self.bamcigarsoft()
        else:
            INsoft = pd.read_csv('%s/%s.chimeric.txt'%(self.outdir, self.inid), sep='\t')

        if INsoft.empty:
            print(self.inid +' donot find soft signal...')
        
        BEDS = Parallel( n_jobs=-1 )( delayed( self.getsoftregion )(_l) for _n, _l in INsoft.iterrows() )
        colm = ['#chrom', 'start', 'end',  'SID', 'length', 'forword', 'query_name', 'flag', 
                'cigarstring', 'cigarpos',  'cigarreg' , 'query_length', 'cigarreffilt', 'match_type', 'match_order', 'query_counts']
        BEDS = pd.DataFrame( np.reshape( np.array(BEDS),(-1,len(colm))), columns=colm)
        BEDS.sort_values(by=['#chrom', 'start', 'end', 'query_name', 'cigarreg'],
                            ascending=[True]*5, inplace=True)
        BEDS.to_csv('%s/%s.chimeric.bed'%(self.outdir, self.inid), header=True, index=False, sep='\t')
        Visal().query_length(BEDS , '%s/%s.chimeric.query_length.pdf'%(self.outdir, self.inid))

        if softfq:
            SEQS = Parallel( n_jobs=-1 )( delayed( self.getsoftfq )(_l) for _n, _l in INsoft.iterrows() )
            f=open( '%s/%s.chimeric.fq'%(self.outdir, self.inid),'w')
            f.write('\n'.join(SEQS))
            f.close()
        print('finish finding soft signal: ' + self.inid)

class SearchType():
    def __init__(self, inbed, outpre):
        self._inbed= inbed
        self.outpre= outpre
        self.inbed = pd.read_csv( self._inbed, sep='\t' )
        if self.inbed.empty:
            self.inbed=pd.DataFrame( columns = self.inbed.columns.tolist() + ['beds', 'region_order', 'links'])
        else:
            #self.inbed[['Type', 'beds', 'links']] = pd.DataFrame([['',[],[]]]*self.inbed.shape[0])
            self.inbed['beds'] = self.inbed[['#chrom', 'start', 'end']].apply(lambda x: [tuple(x)], axis=1)
            self.inbed[['start', 'end']]  = self.inbed[['start', 'end']].astype(int)
            self.inbed['cigarreg'] = self.inbed.cigarreg.apply(lambda x: tuple([int(i) for i in eval(x)]) )
            self.inbed['region_order'] = 'region_order_0'
            self.inbed['links'] = ''
            #self.inbed['cigarreg_s'] = self.inbed.cigarreg.apply(lambda x: int(x[0]))
            #self.inbed['cigarreg_e'] = self.inbed.cigarreg.apply(lambda x: int(x[1]))
            self.inbed.sort_values(by=['query_name', 'cigarreg', '#chrom', 'start', 'end' ], ascending=[True]*5, inplace=True)
        self.COLs = ['#chrom', 'start', 'end',  'SID', 'length', 'forword', 'query_name',
                     'query_length', 'region_order','query_counts', 'cigarstring',  'cigarreg' , 'beds', 'links']
    def BTmerge(self, _Beddf):
        Beddf = bt.BedTool.from_dataframe( _Beddf )
        #Bedsort = bt.BedTool(self.bedfiles[0])
        #Bedsort = Bedsort.cat(*self.bedfiles[1:], postmerge=False)
        Beddf = Beddf.sort()\
                    .merge( c=','.join([ str(i) for i in range(2,12)]),
                            o=','.join(['collapse']*10))\
                    .to_dataframe(disable_auto_names=True, header=None,names='merg_n')
    def typeEB(self):
        #Typeds, Typebs, Typecs = [], [], []
        Bsearch, Btypes = [], []

        for _q, _g in self.inbed.groupby(by=['query_name'], sort=False):
            if  sorted(_g.match_type.unique()) == ['MS', 'SM', 'SMS']:
                _SMS = _g[((_g.match_type)=='SMS')]
                _MSSM= _g[((_g.match_type).isin(['MS','SM']))]
                _MSSM= self.typeBase(_MSSM, typex='typeX')
                if len(_MSSM)>0 and (_SMS.shape[0] >1):
                    _SMS = self.mergeBED(_SMS)
                    VersC, VersR, NVers = self.mergecheck(_MSSM, _SMS, same=False, opposite=True)
                    OverC, OverR, NOver = self.mergecheck(_MSSM, NVers, same=True)
                    Bsearch.append(_MSSM)
                    if not VersC.empty:
                        typec = VersC
                        typec['Type']  = 'typeC'
                        typec['links'] = [VersR[['#chrom','start', 'end']].to_numpy().tolist()]*typec.shape[0]
                        tmerge = VersR.merge(typec, on='query_name',how='outer',suffixes=('','_l') )
                        Btypes.append(tmerge)
                        Bsearch.append(typec)
                    if not OverC.empty:
                        typed = OverC
                        typed['Type']  = 'typeD'
                        typed['links'] = [OverR[['#chrom','start', 'end']].to_numpy().tolist()]*typed.shape[0]
                        tmerge = OverR.merge(typed, on='query_name',how='outer',suffixes=('','_l') )
                        Btypes.append(tmerge)
                        Bsearch.append(typed)
                    if not NOver.empty:
                        typeb = NOver
                        typeb['Type']  = 'typeB'
                        typeb['links'] = [_MSSM[['#chrom','start', 'end']].to_numpy().tolist()]*typeb.shape[0]
                        tmerge = _MSSM.merge(typeb, on='query_name',how='outer',suffixes=('','_l') )
                        Btypes.append(tmerge)
                        Bsearch.append(typeb)

        if len(Bsearch) >0:
            Bsearch = pd.concat(Bsearch,axis=0)
            Btypes  = pd.concat(Btypes,axis=0)
            Btypes.sort_values(by=['#chrom', 'start', 'end', 'query_name'], ascending=[True]*4, inplace=True)
        else:
            Bsearch = pd.DataFrame(columns=self.COLs)
            Btypes  = pd.DataFrame(columns=self.COLs)

        Btypes.to_csv(self.outpre+'.link.typeeB', sep='\t', index=False)
        Bsearch.to_csv(self.outpre+'.typeeB', sep='\t', index=False)

    def mergecheck1(self, _RefBED, _cheakBED, minoverR=0.1, minoverC=0.8, same=True, opposite=False):
        cheakBED = bt.BedTool.from_dataframe( pd.DataFrame(_cheakBED) )
        RefBED   = bt.BedTool.from_dataframe( pd.DataFrame(_RefBED) )
        OverC    = cheakBED.intersect(RefBED, s=same, S=opposite, f=minoverC, wa=True)\
                    .to_dataframe(disable_auto_names=True,names=self.COLs )
        OverR    = RefBED.intersect(cheakBED, s=same, S=opposite, F=minoverC, wa=True)\
                    .to_dataframe(disable_auto_names=True,names=self.COLs )
        NOver    = cheakBED.intersect(RefBED, s=same, S=opposite, f=minoverC, wa=True, v=True)\
                    .to_dataframe(disable_auto_names=True,names=self.COLs )
        #return {'Region': OverL, 'Bool': bool(OverL)}
        return [OverC, OverR, NOver]

    def mergeBED(self, _inbed, oriant=True, maxover=0.8):
        if oriant:
            inbed = _inbed[self.COLs].sort_values(by=['#chrom', 'forword', 'start', 'end' ], ascending=[True]*4)
        else:
            inbed = _inbed[self.COLs].sort_values(by=['#chrom', 'start', 'end', 'forword' ], ascending=[True]*4)
        if inbed.shape[0] <2:
            return inbed
        else:
            overm = [ inbed.iloc[0,:] ]
            for _n, _l in inbed.iloc[1:,:].iterrows():
                if  _l['#chrom']!=overm[-1]['#chrom']:
                    overm.append(_l)
                else:
                    fronL = overm[-1]['end'] - overm[-1]['start'] + 1
                    overL = overm[-1]['end'] - _l['start'] + 1
                    backL = _l['end'] - _l['start'] + 1
                    _L = overm[-1].copy()
                    _L['start'] = min([_L['start'], _l['start']])
                    _L['end']   = max([_L['end'],   _l['end']]  )
                    _L['length']= _L['end'] - _L['start'] + 1 
                    _L['flag']  = '%s;%s'%(_L['flag'], _l['flag'] )
                    _L['cigarstring'] = '%s;%s'%(_L['cigarstring'], _l['cigarstring'] )
                    _L['Type'] = 'typeb'
                    _L['beds'].append((_l['#chrom'], _l['start'], _l['end']) )
                    if oriant \
                        and (_L['forword'] ==_l['forword']) \
                        and max( [overL/fronL, overL/backL] ) >=0.8 :
                        overm[-1] = _L
                    elif  (not oriant) and (max( [overL/fronL, overL/backL] ) >=0.8) :
                        _L['forword'] = '*'
                        overm[-1] = _L
                    else:
                        overm.append(_l)
            overm = pd.concat(overm, axis=1).T
            return overm  
    
    def maxbeddistance(self, _inbed):
        inbed = _inbed.reset_index(drop=True)
        k=[]
        if not inbed.empty:
            for _n, _l in inbed.iterrows() :
                for _m, _j in inbed.loc[_n+1:].iterrows():
                    if (_l['#chrom'] == _j['#chrom']) and (_l['forword'] == _j['forword']):
                        k.append( [_n ,_m, _m - _n, inbed.loc[_n:_m,'length'].sum() ])
        if k:
            k = pd.DataFrame(k, columns=['s','e','d','l'])\
                    .sort_values(by=['l','d'],ascending=[False, False])\
                    .iloc[0,:]
            return inbed.loc[k.s : k.e, :]
        else:
            return inbed
    
    def mergeHeadTail(self, _inbed, oriant=True, maxhtdistance=10000000):
        inbed = _inbed.reset_index(drop=True).copy()
        if inbed.shape[0] >=2:
            _H = inbed.iloc[0,:]
            _T = inbed.iloc[-1,:]
            _S = min([_H['start'], _T['start']])
            _E = max([_H['end'],   _T['end']])
            _L = _E - _S + 1
            if (_H['#chrom'] == _T['#chrom']) \
                and (_H['forword'] == _T['forword']):
                #if _L > maxhtdistance:
                #    print('warning: the ecDNA breakpoint lenght is large than %s:\n%s'%(_L, inbed))
                _H['start']  = _S
                _H['end']    = _E
                _H['length'] = _L
                _H['beds'].append((_T['#chrom'], _T['start'], _T['end']) )
                inbed.iloc[0,:] = _H
                inbed = inbed.iloc[:-1,:].reset_index(drop=True)

        inbed['region_order'] = 'region_order_' + inbed.index.astype(str)
        inbed['links']= [inbed[['#chrom','start', 'end']].to_numpy().tolist()]*inbed.shape[0]
        return inbed

    def mergeNeigh(self, _inbed, maxdistance=500, maxsimilar=50, maxreg=True, oriant=True):
        SortL = ['query_name', 'cigarreg', '#chrom', 'start', 'end' ]
        inbed = _inbed.sort_values(by=SortL, ascending=[True]*len(SortL))
        inbed['cigarreg'] = inbed['cigarreg'].apply(lambda x: [x])
        inbed = inbed[self.COLs]

        if inbed.shape[0] <2:
            return inbed
        else:
            overm = [ inbed.iloc[0,:].copy() ]
            for _n, _l in inbed.iloc[1:,:].iterrows():
                _L = overm[-1].copy()
                R  = _l['#chrom'] == _L['#chrom']
                S1 = (_L['start'] - maxdistance) <= _l['start'] <= (_L['start'] + maxdistance)
                E1 = (_L['end'] - maxdistance)   <= _l['end']   <= (_L['end'] + maxdistance)
                S2 = (_L['start'] - maxsimilar)  <= _l['start'] <= (_L['start'] + maxsimilar)
                E2 = (_L['end'] - maxsimilar)    <= _l['end']   <= (_L['end'] + maxsimilar)
                if R \
                    and ( not(oriant and (_L['forword'] !=_l['forword'])) ) \
                    and ( (S1 and E1) or (S2 or E2) ):
                    if maxreg:
                        _L['start'] = min([_L['start'], _l['start']])
                        _L['end']   = max([_L['end'],   _l['end']]  )
                    else:
                        _L['start'] = max([_L['start'], _l['start']])
                        _L['end']   = min([_L['end'],   _l['end']]  )
                    _L['length']     = _L['end'] - _L['start'] + 1
                    _L['forword']    = ''.join(sorted( set(list(_L['forword']) + list(_l['forword'])) ))
                    _L['cigarreg']   = _L['cigarreg']  + _l['cigarreg']
                    _L['cigarstring'] = '%s;%s'%(_L['cigarstring'], _l['cigarstring'] )
                    _L['beds'].append((_l['#chrom'], _l['start'], _l['end']) )
                    overm[-1] = _L
                else:
                    overm.append(_l)

            overm = pd.concat(overm, axis=1).T
            overm = self.mergeHeadTail(overm)
            return overm

    def typeBase(self, indf, maxmis=0.1, maxoverlap=100):
        BaseKeep, BaseDiss, BaseKMerge = [], [], []

        for (_s, _q,), _g in indf\
                        .sort_values(by=['query_name', 'cigarreg', '#chrom', 'start', 'end' ],
                                    ascending=[True]*5)\
                        .groupby(by=['SID', 'query_name'], sort=False):
            _g = self.maxbeddistance(_g)
            if not _g.empty:
                BreakF = _g.iloc[0,:]
                BreakL = _g.iloc[-1,:]
                C  = BreakF['#chrom'] == BreakL['#chrom']
                F  = BreakF.forword == BreakL.forword
                QF = BreakF.cigarreg[0] <= BreakF.query_length*maxmis
                QL = BreakL.cigarreg[1] >= BreakF.query_length*(1-maxmis)
                S1 = BreakF.start > BreakL.start
                E1 = BreakF.end   > BreakL.end
                X1 = BreakF.start > BreakL.end - maxoverlap

                S2 = BreakF.start < BreakL.start
                E2 = BreakF.end   < BreakL.end

                if (C and F and QF and QL) :
                    if ( (S1 and E1 and BreakF.forword=='+')
                         or(S2 and E2 and BreakF.forword=='-') ):
                        BaseKMerge.append(self.mergeNeigh(_g))
                        BaseKeep.append(_g)
                    elif ( (S1 and E1 and BreakF.forword=='-')
                        or(S2 and E2 and BreakF.forword=='+') ):
                        BaseDiss.append(_g)
        if BaseKeep:
            BaseKeep = pd.concat(BaseKeep,axis=0, sort=False)
        else:
            BaseKeep   = indf
        if BaseKMerge:
            BaseKMerge = pd.concat(BaseKMerge,axis=0, sort=False)
        else:
            BaseKMerge = indf
        if BaseDiss:
            BaseDiss = pd.concat(BaseDiss,axis=0, sort=False)
        else:
            BaseDiss = indf

        BaseDrop = pd.concat([indf,BaseKeep], axis=0, sort=False)\
                        .reset_index(drop=True)\
                        .drop_duplicates(subset=['#chrom', 'start', 'end', 'query_name', 'cigarstring'], keep=False)\
                        .sort_values(by=['#chrom', 'start', 'end', 'query_name', 'cigarreg'], ascending=[True]*5)

        BaseKeep.to_csv(self.outpre+'.Keep', sep='\t', index=False)
        BaseDiss.to_csv(self.outpre+'.Diss', sep='\t', index=False)
        BaseDrop.to_csv(self.outpre+'.Drop', sep='\t', index=False)
        BaseKMerge.to_csv(self.outpre+'.KMerge', sep='\t', index=False)
    
    def mergeReads(self, _inbed, maxdistance=500, maxreg=True, oriant=False):
        COL=['#chrom', 'start', 'end', 'SID', 'support_reads', 'forword',  'length', 'region_order', 'query_name',  'ReadReg'] #, 'links'
        if oriant:
            inbed=_inbed.sort_values(by=['#chrom', 'forword', 'start', 'end'], ascending=[True]*4)
        else:
            inbed=_inbed.sort_values(by=['#chrom', 'start', 'end', 'forword'], ascending=[True]*4)
        inbed['ReadReg'] = inbed[['#chrom', 'start', 'end']].apply(lambda x : [tuple(x)],axis=1)
        inbed['support_reads'] = inbed.ReadReg.apply(lambda x:len(x))
        inbed = inbed[COL]

        if inbed.shape[0] <2:
            return inbed
        else:

            overm = [ inbed.iloc[0,:] ]
            for _n, _l in inbed.iloc[1:,:].iterrows():
                _L = overm[-1].copy()
                R = _l['#chrom'] == _L['#chrom']
                S = (_L['start'] - maxdistance) <= _l['start'] <= (_L['start'] + maxdistance)
                E = (_L['end'] - maxdistance)   <= _l['end']   <= (_L['end'] + maxdistance)
                if (R and S and E) and ( not(oriant and (_L['forword'] !=_l['forword'])) ):
                    if maxreg:
                        _L['start'] = min([_L['start'], _l['start']])
                        _L['end']   = max([_L['end'],   _l['end']]  )
                    else:
                        _L['start'] = max([_L['start'], _l['start']])
                        _L['end']   = min([_L['end'],   _l['end']]  )
                    _L['forword']   = ''.join(sorted( set(list(_L['forword']) + list(_l['forword'])) ))
                    _L['region_order'] = '%s;%s'%(_L['region_order'], _l['region_order'] )
                    _L['query_name'] = '%s;%s'%(_L['query_name'], _l['query_name'] )
                    _L['length']     = _L['end'] - _L['start'] + 1 
                    _L['ReadReg'].append((_l['#chrom'], _l['start'], _l['end']) )
                    _L['support_reads'] = len(_L['ReadReg'])
                    overm[-1] = _L
                else:
                    overm.append(_l)

            overm = pd.concat(overm, axis=1).T
            return overm


    def typeEA(self):
        print('start searching breakpoin region: ' + self._inbed)
        #self.typeBase( self.inbed )
        kmerge = pd.read_csv(self.outpre+'.KMerge', sep='\t')
        rmerge = self.mergeReads(kmerge)
        rmerge.to_csv(self.outpre+'.KMerge.reads', sep='\t', index=False)
        print('finish searching breakpoin region: ' + self._inbed)

class AnnotateBed():
    def __init__(self):
        self.GTFBED = '/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf.gene.bed'
    def annogene(self, inbed):
        annobed = bt.BedTool(self.GTFBED)
        if type(inbed) == str:
            inbed = bt.BedTool( inbed )
        elif type(inbed) == pd.core.frame.DataFrame:
            inbed = bt.BedTool.from_dataframe( inbed )
        
        OverR = inbed.intersect(annobed, s=False, S=False, loj=True)\
                    .to_dataframe(disable_auto_names=True )
        OverR.to_csv('./bb.txt',sep='\t',index=False)

class ecDNA():
    def __init__(self, infile, indir, outdir):
        self.infile=infile
        self.indir=indir
        self.outdir=outdir
        self.SoftFetchDir =outdir + '/03.SoftMap1'
        self.SearchTypeDir=outdir + '/03.SoftMap1'

    def eachwork(self, _l ):
        #SoftFetch(  '%s/02.MiniMap/%s.sorted.bam'%(self.indir, _l.sampleid),
        #            _l.sampleid,
        #            self.SoftFetchDir).getsoft()

        SearchType('{0}/{1}/{1}.chimeric.bed'.format(self.SoftFetchDir , _l.sampleid),
                   '{0}/{1}/{1}.chimeric.bed'.format(self.SearchTypeDir, _l.sampleid),
                ).typeEA()

        '''
        Mapping( '{0}/03.SoftMap/{1}/{1}.chimeric.fq'.format(OUTdir, _l.sampleid),
                _l.sampleid,
                OUTdir+'/03.SoftMap').SoftMINI()
        '''
  
    def mergeATA(self, _inbed, maxdistance=500, maxreg=True):
        COL=['#chrom', 'start', 'end', 'SID', 'support_reads', 'forword',  'length', 'support_SID_num',  'support_SID_read', 'support_region', 'query_name', 'Type', 'ReadReg']
        inbed=_inbed.sort_values(by=['#chrom', 'start', 'end', 'SID'], ascending=[True]*4)
        inbed['support_SID_num'] = 1
        inbed['forword'] = '+'
        inbed['SID'] = inbed['SID'].apply(lambda x : [x])
        inbed['support_SID_read'] = inbed['support_reads'].apply(lambda x : [x])
        inbed['support_region']   = inbed[['#chrom', 'start', 'end']].apply(lambda x : [tuple(x)], axis=1)
        inbed = inbed[COL]
        if inbed.shape[0] <2:
            return inbed
        else:
            overm = [ inbed.iloc[0,:] ]
            for _n, _l in inbed.iloc[1:,:].iterrows():
                _L = overm[-1].copy()
                R = _l['#chrom'] == _L['#chrom']
                S = (_L['start'] - maxdistance) <= _l['start'] <= (_L['start'] + maxdistance)
                E = (_L['end'] - maxdistance)   <= _l['end']   <= (_L['end'] + maxdistance)
                if (R and S and E):
                    if maxreg:
                        _L['start'] = min([_L['start'], _l['start']])
                        _L['end']   = max([_L['end'],   _l['end']]  )
                    else:
                        _L['start'] = max([_L['start'], _l['start']])
                        _L['end']   = min([_L['end'],   _l['end']]  )
                    _L['length']     = _L['end'] - _L['start'] + 1 
                    
                    if _L['SID'][-1] != _l['SID']:
                        _L['SID'] += _l['SID']
                        _L['support_SID_read'].append(_l['support_reads'])
                    else:
                        _L['support_SID_read'][-1] = int(_L['support_SID_read'][-1]) + int(_l['support_reads'])
                
                    _L['support_reads'] = int(_L['support_reads']) + int(_l['support_reads'])
                    _L['support_SID_num'] = len( _L['SID'])
                    _L['support_region'].append(tuple(_l[['#chrom', 'start', 'end']]))
                    _L['query_name'] = '%s;%s'%(_L['query_name'], _l['query_name'] )
                    _L['ReadReg'] += _l['ReadReg']
                    #_L['support_reads'] = len(_L['ReadReg'])
                    overm[-1] = _L
                else:
                    overm.append(_l)

            overm = pd.concat(overm, axis=1).T
            return overm

    def TypeASum(self, INdf):
        Typeall = []
        for _n, _l in INdf.iterrows():
            File = '{0}/03.SoftMap/{1}/{1}.chimeric.bed.typeeA.merge'.format(self.outdir, _l.sampleid)
            Typeall.append(pd.read_csv(File, sep='\t',header=0))
        Typeall = pd.concat(Typeall, axis=0)
        self.mergeATA(Typeall).to_csv('./all.merge.xls', sep='\t', index=False)

    def Pipeline(self):
        INdf = pd.read_csv(self.infile, sep='\t', comment='#')
        #Parallel( n_jobs=5 )( delayed( self.eachwork )(_l) for _n, _l in INdf.iterrows() )

        for _n, _l in INdf.iterrows():
            if _l.sampleid in ['simulate', 'HEK293T.CCS.lima.BC5--BC5']:
                self.eachwork(_l)

        #self.TypeASum(INdf)
        #AnnotateBed().annogene('./all.merge.xls')

INfile='/data/zhouwei/01Projects/03ecDNA/sample.info.txt'
INdir= '/data/zhouwei/01Projects/03ecDNA'
OUTdir='/data/zhouwei/01Projects/03ecDNA'

GTF='/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf'
GTFBED='/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf.gene.bed'
ecDNA(INfile, INdir, OUTdir).Pipeline()
