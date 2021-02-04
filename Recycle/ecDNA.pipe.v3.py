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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
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
    def query_length(self, _indf, out, X='query_length', Dup='query_name', log=False, title=''):
        if not _indf.empty:
            indef = _indf.copy()
            if Dup:
                indef = indef[[Dup, X]].drop_duplicates(keep='first')

            indef[X] = indef[X].astype(int)
            dp = sns.displot(data=indef, x=X, kde=True, log_scale=log)
            dp.set_xticklabels(rotation=270)

            if title:
                plt.title(title)

            plt.tight_layout()
            plt.savefig( out )
            plt.close()

class Utilities():
    def __init__(self ):
        self.GTFBED = '/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf.gene.bed'
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

    def annogene(self, _inbed, minoverA=30, minoverB=30, 
                    biotype=['miRNA','lncRNA', 'protein_coding', 'IG_C_gene', 'IG_D_gene', 
                            'IG_J_gene', 'IG_V_gene', 'TR_C_gene', 'TR_D_gene', 'TR_J_gene']
                            #'TR_V_gene', 'rRNA', 'scaRNA', 'scRNA', 'snoRNA', 'snRNA', 'sRNA'] ):
                ):
                            
        GTFBED = pd.read_csv(self.GTFBED, sep='\t', header=0)
        GTFBED = GTFBED[(GTFBED.gene_biotype.isin(biotype))]
        annobed = bt.BedTool.from_dataframe(GTFBED)
        Rcol = ['#chrom', 'start', 'end', 'length', 'support_sum',
                '#chrom_g', 'start_g', 'end_g', 'gtype', 'length_g', 'forword_g',  'gene_name', 'gene_id', 'gene_biotype']
        Fcol = ['#chrom', 'start', 'end', 'length', 'support_sum',
                'head_dist', 'tail_dist', 'gene_name', 'gene_id', 'gene_biotype', 'overlen', 'overfre']

        def eachinter(_df):
            inbt = bt.BedTool.from_dataframe( _df )
            _minoverA = minoverA/_df.length.max()

            over = inbt.intersect(annobed, s=False, S=False, loj=True, f=_minoverA )\
                        .to_dataframe(disable_auto_names=True,  header=None, names=Rcol)
            over['start_m']   = over[['start', 'start_g']].max(1)
            over['end_m']     = over[['end'  , 'end_g']].min(1)
            over['overlen']   = over['end_m'] - over['start_m']
            over['overfre']   = over['overlen']/over['length'].astype(int)

            df=_df.copy()
            if over.iloc[0]['gene_name'] != '.':
                df['head_dist'] = over.iloc[0]['start_g'] - over.iloc[0]['start']
                df['tail_dist'] = over.iloc[-1]['end']    - over.iloc[-1]['end_g']
                df['gene_name'] = over.gene_name.str.cat(sep=';')
                df['gene_id']   = over.gene_id.str.cat(sep=';')
                df['gene_biotype'] = over.gene_biotype.str.cat(sep=';')
                df['overlen'] = over.overlen.astype(str).str.cat(sep=';')
                df['overfre'] = over.overfre.round(4).astype(str).str.cat(sep=';')   
            else:
                df['head_dist'] = -1
                df['tail_dist'] = -1
                df['gene_name'] = '.'
                df['gene_id']   = '.'
                df['gene_biotype'] = '.'
                df['overlen'] = -1
                df['overfre'] = -1
            return df[Fcol]

        inbed = Parallel( n_jobs=-1, backend='loky')( delayed( eachinter )(_g) 
                    for _r, _g in _inbed.groupby(by=['#chrom', 'start', 'end'],sort=True) )
        inbed = pd.concat(inbed, axis=0,sort=False)
        return inbed

    @emptyfile
    def Peakannotate(self, inbed):
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

class SoftFetch():
    def __init__(self, inbam, inid, outdir):
        self.inbam=inbam
        self.inid = inid
        self.outdir=outdir + '/' + inid
        self.chrs=[str(i) for i in range(1,23)] + ['MT','X','Y'] #remove
        self.minsoflt = 5
        self.mapq     = 5
        os.makedirs(self.outdir, exist_ok=True)

    def cigarmerge(self, _ct, indel=100000, skip=1000000000, hard=100000, pad=1000000000, match='Q'):
        ct  = [ (0, i[1]) if (i[0] in [7,8]) else i for i in _ct ]
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
                elif ( i[0] in [7, 8] ):
                    i = (0, i[1])
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
                elif ( i[0] in [7, 8] ):
                    i = (0, i[1])
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
                    and len(softlens) >0) \
                    and read.mapping_quality >=self.mapq :
                    #and (reference_name in self.chrs) \
                #TP = tags['tp'] if 'tp' in tags else ''
                SA = tags['SA']
                query_name = read.query_name
                flag = read.flag
                mapping_quality  = read.mapping_quality 
                reference_start  = read.reference_start
                read.cigartuples = self.cigarmerge(raw_cigartuples, match='Q')
                cigartuples_ref  = self.cigarmerge(raw_cigartuples, match='R')
                query_sequence   = read.query_sequence
                #query_qualities  = pysam.array_to_qualitystring(read.query_qualities)
                is_reverse = '-' if read.is_reverse else '+'
                sampd.append([self.inid, query_name, flag, reference_name, reference_start, mapping_quality, read.cigartuples, cigartuples_ref,
                              is_reverse, len(query_sequence), SA, read.cigarstring ,raw_cigarstring, query_sequence])

        sampd = pd.DataFrame(sampd, columns=['SID', 'query_name', 'flag', 'reference_name', 'reference_start', 'mapping_quality',
                                            'cigartuples', 'cigartuples_ref', 'is_reverse', 'query_length', 'SA',
                                            'cigarstring', 'raw_cigar', 'query_sequence'])
        sampd = sampd.merge(sampd.groupby('query_name')['query_name'].size().reset_index(name='query_counts'),
                            on='query_name', how='outer')
        sampd = sampd[(sampd.query_counts>1)]

        #sampd.sort_values(by=['query_name', 'reference_name', 'reference_start', 'flag'], ascending=[True, True, True, True], inplace=True)
        sampd.to_csv('%s/%s.chimeric.txt'%(self.outdir, self.inid), sep='\t', index=False)
        return sampd

    def getsoftfq(self, INsoft, softminlen=100):
        f=open( '%s/%s.chimeric.fq'%(self.outdir, self.inid),'w')
        for _n, _l in INsoft.iterrows():
            cigt  = eval(_l.cigartuples) if type(_l.cigartuples)==str else _l.cigartuples
            start = 0
            SEQs  = []
            for  n,i in enumerate(cigt):
                end = int(start) + int(i[1])
                if (i[0] ==4 and i[1]>=softminlen):
                    seq = _l.query_sequence[start:end]
                    #qul = _l.query_qualities[start:end]
                    qul = '~'*len(seq)
                    name= '@%s_soft%s-%s_%s_%s'%(_l.query_name, _l.reference_name, _l.reference_start, n, i[1])
                    SEQs.append('%s\n%s\n+\n%s'%(name, seq, qul) )
                start += int(i[1])
            f.write('\n'.join(SEQs))
        f.close()

    def getsoftregion(self, _g):
        start  = int(_g.reference_start)
        cigref = eval(_g.cigartuples_ref) if type(_g.cigartuples_ref)==str else _g.cigartuples_ref
        cigartuples = eval(_g.cigartuples) if type(_g.cigartuples)==str else _g.cigartuples
        match_type  = ''.join([ 'S' if i[0]==4 else 'M' for i in cigartuples ])

        if _g.is_reverse=='-':
            cigartuples = cigartuples[::-1]

        cigarpos = [(0, cigartuples[0][1])]
        for i in cigartuples[1:]:
            cigarpos.append( (cigarpos[-1][1], cigarpos[-1][1]+i[1]) )
        if _g.is_reverse=='-':
            cigarpos = cigarpos[::-1]

        Regs  = []
        for  n,i in enumerate(cigref):
            if (i[0] !=4 ):
                bed = [ _g.reference_name, start, start + int(i[1]-1), _g.SID, i[1], _g.is_reverse, _g.query_name, 
                        _g.flag, _g.cigarstring, cigarpos, cigarpos[n], _g.query_length, cigref, _g.query_counts]
                Regs.append(bed)
                start += int(i[1]-1)
        return Regs

    def getsoft(self, softfq=False, researh=True):
        print('start finding soft signal: ' + self.inid)
        if researh:
            INsoft = self.bamcigarsoft()
        else:
            INsoft = pd.read_csv('%s/%s.chimeric.txt'%(self.outdir, self.inid), sep='\t', low_memory=False)

        if INsoft.empty:
            print(self.inid +' donot find soft signal...')

        BEDS = Parallel( n_jobs=-1, backend='loky')( delayed( self.getsoftregion )(_l) for _n, _l in INsoft.iterrows() )

        if softfq:
            self.getsoftfq(INsoft)
        del(INsoft)

        colm = ['#chrom', 'start', 'end',  'SID', 'length', 'forword', 'query_name', 'flag', 
                'cigarstring', 'cigarpos',  'cigarreg' , 'query_length', 'cigarreffilt', 'query_counts']
        BEDS = pd.DataFrame( np.reshape( np.array(BEDS),(-1,len(colm))), columns=colm)
        BEDS.sort_values(by=['#chrom', 'start', 'end', 'query_name', 'cigarreg'],
                            ascending=[True]*5, inplace=True)
        BEDS.to_csv('%s/%s.chimeric.bed'%(self.outdir, self.inid), header=True, index=False, sep='\t')
        Visal().query_length(BEDS , '%s/%s.chimeric.query_length.pdf'%(self.outdir, self.inid))

        print('finish finding soft signal: ' + self.inid)

class SearchType():
    def __init__(self, inbed, outpre, Chrom=False ):
        self._inbed= inbed
        self.outpre= outpre
        self.Chrom = Chrom
        self.chrs=[str(i) for i in range(1,23)] + ['MT','X','Y'] \
                    + ['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                        '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                        'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                        'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                        'SunTag-CRISPRi', 'V7-MC-HG-FA']
        if self.Chrom:
            self.outpre += '.OnlyChrom'
    
    def _getbeddb(self):
        self.inbed = pd.read_csv( self._inbed, sep='\t')
        if self.inbed.empty:
            self.inbed=pd.DataFrame( columns = self.inbed.columns.tolist() + ['beds', 'raw_order', 'links', 'type'])
        else:
            #self.inbed[['Type', 'beds', 'links']] = pd.DataFrame([['',[],[]]]*self.inbed.shape[0])
            self.inbed['beds'] = self.inbed[['#chrom', 'start', 'end']].apply(lambda x: [tuple(x)], axis=1)
            self.inbed[['start', 'end']]  = self.inbed[['start', 'end']].astype(int)
            self.inbed['#chrom']  = self.inbed['#chrom'].astype(str)
            self.inbed['cigarreg'] = self.inbed.cigarreg.apply(lambda x: tuple([int(i) for i in eval(x)]) )
            self.inbed['raw_order'] = 'raw_order_0'
            self.inbed['links'] = ''
            self.inbed['type']  = 'DROP'
            self.inbed.sort_values(by=['query_name', 'cigarreg', '#chrom', 'start', 'end' ], ascending=[True]*5, inplace=True)
        self.COLs = ['#chrom', 'start', 'end',  'SID', 'length', 'forword', 'query_name',
                     'query_length', 'type', 'raw_order','query_counts', 'cigarstring',  'cigarreg' , 'beds', 'links']
        self.inbed = self.inbed[self.COLs]
        if self.Chrom:
            self.inbed  = self.inbed[ (self.inbed['#chrom'].isin(self.chrs) )]
        return self

    def maxbeddistance(self, _inbed):
        inbed = _inbed.copy().reset_index(drop=True)
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
    
    def dropcigarinclude(self, _inbed, errors=30, dropcigarover=True):
        if (not dropcigarover) or (_inbed.shape[0]<2):
            return _inbed, pd.DataFrame(columns=_inbed.columns)
        else:
            inbed = _inbed.copy().reset_index(drop=True)
            for _n, _l in inbed.iterrows():
                for _m, _k in inbed.loc[_n+1 :,:].iterrows():
                    _s1 = _l.cigarreg[0]
                    _e1 = _l.cigarreg[1]
                    _f1 = _l.forword
                    _s2 = _k.cigarreg[0]
                    _e2 = _k.cigarreg[1]
                    _f2 = _k.forword
                    if (_s1 <= _s2) and (_e1 >= _e2 ) and (_f1==_f2):
                        inbed.loc[_m, 'type'] ='OVER'
                    elif  (_s1 >= _s2 ) and (_e1 <= _e2 ) and (_f1==_f2):
                        inbed.loc[_n, 'type'] ='OVER'
            return inbed[(inbed.type != 'OVER')], inbed[(inbed.type == 'OVER')]
                
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
                _H['cigarreg']   = _H['cigarreg']  + _T['cigarreg']
                _H['cigarstring'] = '%s;%s'%(_H['cigarstring'], _T['cigarstring'] )
                inbed.iloc[0,:] = _H
                inbed = inbed.iloc[:-1,:].reset_index(drop=True)

        inbed['raw_order'] = 'raw_order_' + inbed.index.astype(str)
        inbed['links']= [inbed[['#chrom','start', 'end']].to_numpy().tolist()]*inbed.shape[0]
        return inbed

    def mergeNeighb(self, _inbed, maxdistance=200, maxsimilar=50, maxreg=True, oriant=True, dropcigarover=False):
        SortL = ['query_name', 'cigarreg', '#chrom', 'start', 'end' ]
        inbed = _inbed.sort_values(by=SortL, ascending=[True]*len(SortL))
        inbed['cigarreg'] = inbed['cigarreg'].apply(lambda x: [x])

        if dropcigarover:
            inbed = inbed[(inbed.type!='OVER')]
            overd = inbed[(inbed.type=='OVER')]

        if inbed.shape[0] <2:
            overm = inbed
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

        if dropcigarover:
            overm = pd.concat([overm, overd],axis=0, sort=False)
        return overm

    def typeBase1(self, indf, maxmis=0.1, maxoverlap=300):
        BaseMark, BaseKeep = [],[]

        for (_s, _q,), _g in indf\
                        .sort_values(by=['query_name', 'cigarreg', '#chrom', 'start', 'end' ],
                                    ascending=[True]*5)\
                        .groupby(by=['SID', 'query_name'], sort=False):

            _g = self.maxbeddistance(_g)
            _g, _d = self.dropcigarinclude(_g)
            _g = self.mergeNeighb(_g)

            if _g.shape[0]>=2:
                BreakF = _g.iloc[ 0,:]
                BreakL = _g.iloc[-1,:]
                C  = BreakF['#chrom'] == BreakL['#chrom']
                F  = BreakF.forword == BreakL.forword

                QF = BreakF.cigarreg[ 0][0] <= BreakF.query_length*maxmis
                QL = BreakL.cigarreg[-1][1] >= BreakF.query_length*(1-maxmis)
                S1 = BreakF.start > BreakL.start
                E1 = BreakF.end   > BreakL.end
                X1 = BreakF.start > BreakL.end - maxoverlap

                S2 = BreakF.start < BreakL.start
                E2 = BreakF.end   < BreakL.end
                X2 = BreakF.start < BreakL.end - maxoverlap

                if (C and F and QF and QL):
                    if (S1 and E1 and X1) :
                        _g['type'] ='KEEP'
                        BaseKeep.append(self.mergeHeadTail(_g))
                    else:
                        _g['type'] ='DISS'
            BaseMark.append(_g)
            BaseMark.append(_d)

        if BaseMark:
            BaseMark = pd.concat(BaseMark,axis=0, sort=False)
        else:
            BaseMark = pd.DataFrame(columns=indf.columns)
        if BaseKeep:
            BaseKeep = pd.concat(BaseKeep,axis=0, sort=False)
        else:
            BaseKeep = pd.DataFrame(columns=indf.columns)
        BaseMark.to_csv(self.outpre+'.Mark', sep='\t', index=False)
        BaseKeep.to_csv(self.outpre+'.Keep', sep='\t', index=False)
        return BaseKeep

    def typemark(self, _G, maxmis=0.1, maxoverlap=300):
        _g = self.maxbeddistance(_G)
        _g, _d = self.dropcigarinclude(_g)
        _g = self.mergeNeighb(_g)
        _k = pd.DataFrame()
        if _g.shape[0]>=2:
            BreakF = _g.iloc[ 0,:]
            BreakL = _g.iloc[-1,:]
            C  = BreakF['#chrom'] == BreakL['#chrom']
            F  = BreakF.forword == BreakL.forword

            QF = BreakF.cigarreg[ 0][0] <= BreakF.query_length*maxmis
            QL = BreakL.cigarreg[-1][1] >= BreakF.query_length*(1-maxmis)
            S1 = BreakF.start > BreakL.start
            E1 = BreakF.end   > BreakL.end
            X1 = BreakF.start > BreakL.end - maxoverlap

            S2 = BreakF.start < BreakL.start
            E2 = BreakF.end   < BreakL.end
            X2 = BreakL.start > BreakF.end - maxoverlap

            if (C and F and QF and QL):
                if (S1 and E1 and X1 and BreakF.forword=='+' ) \
                    or (S2 and E2 and X2 and BreakF.forword=='-' ):
                    _g['type'] ='KEEP'
                    _k = self.mergeHeadTail(_g)
                else:
                    _g['type'] ='DISS'
        _d['cigarreg'] = _d['cigarreg'].apply(lambda x:[x])
        _g = pd.concat( [_g,_d], axis=0, sort=False )
        return (_g, _k)

    def typeBase(self, indf, maxmis=0.1, maxoverlap=300):
        SortN = ['query_name', 'cigarreg', '#chrom', 'start', 'end' ]
        BaseCat = Parallel( n_jobs= -1, backend='loky')\
                    ( delayed( self.typemark )(_g)
                        for _, _g in indf\
                            .sort_values(by=SortN, ascending=[True]*len(SortN))\
                            .groupby(by=['SID', 'query_name'], sort=False))
        BaseMark = [ i[0] for i in BaseCat ]
        BaseKeep = [ i[1] for i in BaseCat if not i[1].empty]
        del(BaseCat)

        if BaseMark:
            BaseMark = pd.concat(BaseMark,axis=0, sort=False)
        else:
            BaseMark = pd.DataFrame(columns=indf.columns)
        BaseMark.to_csv(self.outpre+'.Mark', sep='\t', index=False)
        del(BaseMark)

        if BaseKeep:
            BaseKeep = pd.concat(BaseKeep,axis=0, sort=False)
        else:
            BaseKeep = pd.DataFrame(columns=indf.columns)
        BaseKeep.to_csv(self.outpre+'.Keep', sep='\t', index=False)                        

        return BaseKeep

    def mergeMap(self, _inbed, maxdistance=500, maxreg=True, oriant=False):
        _mergeB= _inbed[['#chrom', 'start', 'end', 'SID', 'query_name', 'forword']].drop_duplicates(keep='first').copy()
        _mergeB[['start', 'end']]  = _mergeB[['start', 'end']].astype(int)
        _mergeB['Beds']       = _mergeB[['#chrom', 'start', 'end']].apply(lambda x: [tuple(x)], axis=1)
        _mergeB['query_name'] = _mergeB['query_name'].apply(lambda x: [x])
        _mergeB['SID']        = _mergeB['SID'].apply(lambda x: [x])
        _mergeB[['#chrom_', 'start_', 'end_']] = _mergeB[['#chrom', 'start', 'end']]

        if oriant:
            _mergeB.sort_values(by=['#chrom', 'forword', 'start', 'end'], ascending=[True]*4, inplace=True)
        else:
            _mergeB.sort_values(by=['#chrom', 'start', 'end', 'forword'], ascending=[True]*4, inplace=True)

        if _mergeB.shape[0] <2:
            mergeE = _mergeB
        else:
            mergeB = [_mergeB.iloc[0,:] ]
            for _n, _l in _mergeB.iloc[1:,:].iterrows():
                _L = mergeB[-1].copy()
                R = _l['#chrom_'] == _L['#chrom_']
                S = (_L['start_'] - maxdistance) <= _l['start_'] <= (_L['start_'] + maxdistance)
                E = (_L['end_'] - maxdistance)   <= _l['end_']   <= (_L['end_'] + maxdistance)
                if (R and S and E) and ( not(oriant and (_L['forword'] !=_l['forword'])) ):
                    if maxreg:
                        _L['start_'] = min([_L['start_'], _l['start_']])
                        _L['end_']   = max([_L['end_'],   _l['end_']]  )
                    else:
                        _L['start_'] = max([_L['start_'], _l['start_']])
                        _L['end_']   = min([_L['end_'],   _l['end_']]  )
                    _L['forword']   = _L['forword'] + _l['forword']
                    _L['query_name']= _L['query_name'] + _l['query_name']
                    _L['SID']       = _L['SID']        + _l['SID']
                    _L['Beds'].append((_l['#chrom_'], _l['start_'], _l['end_']) )
                    mergeB[-1] = _L
                else:
                    mergeB.append(_l)
            mergeE = []
            for _m in mergeB:
                for _n, _r in enumerate(_m.Beds):
                    _x = _m.copy() 
                    _x['query_name'] = _m['query_name'][_n]
                    _x['SID']        = _m['SID'][_n]
                    _x['forword']    = _m['forword'][_n]
                    _x[['#chrom', 'start', 'end']]= _r
                    mergeE.append(_x)
            mergeE = pd.concat(mergeE, axis=1, sort=False).T
        return mergeE[['#chrom', 'start', 'end', 'SID', 'query_name', 'forword', '#chrom_', 'start_', 'end_']]

    def mergeLinks(self, _inbed):
        mergeE = self.mergeMap(_inbed)
        inbed  = _inbed.merge(mergeE, on=[ '#chrom', 'start', 'end', 'SID', 'query_name', 'forword'], how='left')
        inbed[['#chrom_raw', 'start_raw', 'end_raw']] = inbed[['#chrom', 'start', 'end']]
        inbed[['#chrom', 'start', 'end']] = inbed[['#chrom_', 'start_', 'end_']]
        inbed['length'] = inbed['end'] - inbed['start'] + 1
        inbed['links_sorted'] = inbed['links'] 
        inbed['Type']  = 'Mono'
        inbed['new_order']  = 'new_order_0'
        inbed.drop(['#chrom_', 'start_', 'end_'], axis=1,inplace=True)

        uplinks=[]
        for _r, _g in inbed\
                        .sort_values(by=['SID', 'query_name', 'raw_order'], ascending=[True, True, True])\
                        .groupby( by= ['SID', 'query_name'], sort=False):
            _p = _g.copy()
            _p['links'] = [_g[['#chrom','start', 'end']].apply(lambda x: '%s:%s-%s'%tuple(x),axis=1).str.cat(sep=';')]*_g.shape[0]
            if _p.shape[0] >=2:
                _p['Type']  = 'Multi'
                #_S = _p.sort_values(by=['#chrom','start', 'end']).iloc[0].name
                #_p = pd.concat((_p.loc[_S:], _p.loc[:_S]),axis=0, sort=False).iloc[:-1]
                # need update order
                _p = _p.sort_values(by=['#chrom','start', 'end'])
                _p['new_order'] = ['new_order_%s'%i for i in range(_p.shape[0])]
            _p['links_sorted']  = [_p[['#chrom','start', 'end']].apply(lambda x: '%s:%s-%s'%tuple(x),axis=1).str.cat(sep=';')]*_p.shape[0]
            uplinks.append(_p)

        uplinks= pd.concat(uplinks, axis=0, sort=False)
        uplinks.sort_values(by=['links', 'SID', 'query_name', 'raw_order'],inplace=True)

        return uplinks
    
    def mergeReads(self, _inbed):
        uplinks = self.mergeLinks(_inbed)
        uplinks.to_csv(self.outpre+'.Uplinks', sep='\t', index=False)

        uplinks = uplinks.sort_values(by=['links_sorted', 'new_order'])
        uplinks = uplinks.merge(  
                    uplinks.groupby(by=['SID', 'links_sorted'], sort=False)['query_name'].unique().to_frame('support_reads').reset_index(), 
                    on=['SID', 'links_sorted'], how='outer')
        uplinks['forword'] = '*'
        uplinks['support_number'] = uplinks.support_reads.apply(lambda x: len(x))
        uplinks['support_reads']  = uplinks.support_reads.apply(lambda x: ';'.join(x))
        COLS = ['#chrom', 'start', 'end', 'SID', 'length', 'forword', 'support_number', 'Type', 'links_sorted', 'new_order', 'support_reads']
        uplinks = uplinks[COLS]\
                    .drop_duplicates(keep='first')\
                    .sort_values(by=['Type','links_sorted', 'new_order', '#chrom', 'start', 'end','SID'])

        Bedregn = uplinks[['#chrom', 'start', 'end', 'support_number', 'length' ]].copy()
        Bedregn = Bedregn.merge(
                        Bedregn.groupby(by=['#chrom', 'start', 'end'])['support_number'].sum().to_frame('support_sum').reset_index(),
                        on=['#chrom', 'start', 'end'], how='outer')\
                    .drop('support_number',axis=1).drop_duplicates(keep='first')
        Bedregn = Utilities().annogene(Bedregn)
        Bedregn.to_csv(self.outpre+'.BedAnnotate', sep='\t', index=False)

        uplinks = uplinks\
                    .merge(Bedregn, on=['#chrom',  'start',  'end', 'length'], how='outer')\
                    .sort_values(by=['Type', 'links_sorted', 'new_order', '#chrom', 'start', 'end',  'SID'])
        uplinks.to_csv(self.outpre+'.UpMerge', sep='\t', index=False)

        uplinks =uplinks.sort_values(by=['Type', 'new_order',  '#chrom', 'start', 'end', 'links_sorted', 'SID'])
        uplinks.to_csv(self.outpre+'.UpMerge_sort', sep='\t', index=False)
        Visal().query_length(uplinks, self.outpre+'.UpMerge.length.pdf', X='length', Dup='', log=True, title='breakpoint length' )

    def EachEcDNA(self):
        print('start searching breakpoin region: ' + self._inbed)
        self._getbeddb()
        if not self.inbed.empty:
            kmerge = self.typeBase( self.inbed )
            #kmerge = pd.read_csv(self.outpre+'.Keep', sep='\t', low_memory=False)
        else:
            kmerge = pd.DataFrame()
            print('cannot find any circle region singal: ' + self._inbed)

        if not kmerge.empty:
            self.mergeReads(kmerge)
        else:
            print('cannot find any circle region type: ' + self._inbed)
        print('finish searching breakpoin region: ' + self._inbed)

    def linkParall(self, _g):
        COLS = ['#chrom', 'start', 'end', 'Type', 'length', 'forword', 'support_number', 'links_sorted', 
                    'new_order', 'support_read_num', 'support_ID_num', 'support_IDs']
        _G = _g.copy()
        _K = _g[(_g['new_order'] =='new_order_0')]\
                .groupby(by='SID',sort=False)['query_name'].size().to_frame(name='SUP').reset_index()
        _G['forword']        = '*'
        _G['support_number'] = _g['query_name'].unique().size
        _G['support_reads']  = ';'.join( _g['query_name'].unique() )
        _G['support_IDs']    =  _K['SID'].str.cat(sep=';')
        _G['support_ID_num']   =  _K['SID'].size
        _G['support_read_num'] =  _K['SUP'].astype(str).str.cat(sep=';')
        _G = _G[COLS].drop_duplicates(keep='first')
        return _G

    def mergeSamples(self, _inbed):
        uplinks = self.mergeLinks(_inbed)
        uplinks.to_csv(self.outpre+'.Uplinks', sep='\t', index=False)

        #uplinks = pd.read_csv(self.outpre+'.Uplinks', sep='\t')
        UpLinks = Parallel( n_jobs=-1)( delayed( self.linkParall )(_g) for _l, _g in uplinks.groupby(by='links_sorted', sort=False) )
        UpLinks = pd.concat(UpLinks, axis=0, sort=False)

        Bedregn = UpLinks[['#chrom', 'start', 'end', 'support_number', 'length' ]].copy()
        Bedregn = Bedregn.merge(
                        Bedregn.groupby(by=['#chrom', 'start', 'end'])['support_number'].sum().to_frame('support_sum').reset_index(),
                        on=['#chrom', 'start', 'end'], how='outer')\
                    .drop('support_number',axis=1).drop_duplicates(keep='first')
        Bedregn = Utilities().annogene(Bedregn)
        Bedregn.to_csv(self.outpre+'.BedAnnotate', sep='\t', index=False)

        UpLinks = UpLinks\
                    .merge(Bedregn, on=['#chrom',  'start',  'end', 'length'], how='outer')\
                    .sort_values(by=['Type', 'links_sorted', 'new_order', '#chrom', 'start', 'end'])
        UpLinks.to_csv(self.outpre+'.UpMerge', sep='\t', index=False)

        UpLinks =UpLinks.sort_values(by=['Type', 'new_order',  '#chrom', 'start', 'end', 'links_sorted'])
        UpLinks.to_csv(self.outpre+'.UpMerge_sort', sep='\t', index=False)
        Visal().query_length(UpLinks, self.outpre+'.UpMerge.length.pdf', X='length', Dup='', log=True, title='breakpoint length' )

    def AllEcDNA(self):
        self.mergeSamples(self._inbed)

class ecDNA():
    'The pipeline used for machine learning models'
    def __init__(self, arg, log,  *array, **dicts):
        self.arg = arg
        self.log = log
        self.array  = array
        self.dicts  = dicts

        import multiprocessing
        CORES = multiprocessing.cpu_count()*0.8 if multiprocessing.cpu_count() >8 else 8
        os.environ['NUMEXPR_MAX_THREADS'] = '1000' #str(int(CORES))
        os.environ['PATH'] += ':' + self.arg.bedtools

        #bt.helpers.set_bedtools_path(elf.arg.bedtools)
        import importlib
        importlib.reload(bt)

        self.arg.BamDir     = '%s/%s'%(self.arg.indir, self.arg.infetch)
        self.arg.FetchDir   = '%s/%s'%(self.arg.outdir, self.arg.outfetch)
        self.arg.SearchDir  = '%s/%s'%(self.arg.outdir, self.arg.outsearch)
        self.arg.AllTypeDir = '%s/%s'%(self.arg.outdir, self.arg.outregion)

        os.makedirs(self.arg.FetchDir , exist_ok=True)
        os.makedirs(self.arg.SearchDir, exist_ok=True)
        os.makedirs(self.arg.AllTypeDir   , exist_ok=True)

    def eachwork(self, _l ):
        if self.arg.commands in ['Fetch', 'Auto']:
            SoftFetch(  '%s/%s.sorted.bam'%(self.arg.BamDir, _l.sampleid),
                        _l.sampleid,
                        self.arg.FetchDir).getsoft()
    
        if self.arg.commands in ['Search', 'Auto']:
            SearchType('{0}/{1}/{1}.chimeric.bed'.format(self.arg.FetchDir , _l.sampleid),
                        '{0}/{1}/{1}.bed'.format(self.arg.SearchDir, _l.sampleid),
                        Chrom=self.arg.chroms).EachEcDNA()

    def allwork(self, _INdf):
        AllMerge = []
        for _n, _l in _INdf.iterrows():
            if self.arg.chroms:
                EachMerge = '{0}/{1}/{1}.bed.OnlyChrom.Keep'.format(self.arg.SearchDir, _l.sampleid)
            else:
                EachMerge = '{0}/{1}/{1}.bed.Keep'.format(self.arg.SearchDir, _l.sampleid)
            if os.path.exists(  EachMerge ):
                AllMerge.append( pd.read_csv(EachMerge, sep='\t', header=0) )
        if AllMerge:
            AllMerge = pd.concat(AllMerge, axis=0,sort=False)
            SearchType(AllMerge, self.arg.AllTypeDir  + '/All.basedon.Keep', Chrom=self.arg.chroms).AllEcDNA()
        else:
            self.log.CI('cannot find the valid files.')

    def FetchW(self):
        SoftFetch(  '%s/%s.sorted.bam'%(self.arg.BamDir, _l.sampleid),
                    _l.sampleid,
                    self.arg.FetchDir).getsoft()
    def SearchW(self):
        SearchType('{0}/{1}/{1}.chimeric.bed'.format(self.arg.FetchDir , _l.sampleid),
                    '{0}/{1}/{1}.bed'.format(self.arg.SearchDir, _l.sampleid),
                    Chrom=self.arg.chroms).EachEcDNA()
    def Region(self):
        AllMerge = []
        for _n, _l in _INdf.iterrows():
            if self.arg.chroms:
                EachMerge = '{0}/{1}/{1}.bed.OnlyChrom.Keep'.format(self.arg.SearchDir, _l.sampleid)
            else:
                EachMerge = '{0}/{1}/{1}.bed.Keep'.format(self.arg.SearchDir, _l.sampleid)
            if os.path.exists(  EachMerge ):
                AllMerge.append( pd.read_csv(EachMerge, sep='\t', header=0) )
        if AllMerge:
            AllMerge = pd.concat(AllMerge, axis=0,sort=False)
            SearchType(AllMerge, self.arg.AllTypeDir  + '/All.basedon.Keep', Chrom=self.arg.chroms).AllEcDNA()
        else:
            self.log.CI('cannot find the valid files.')

    def Pipeline(self):
        if os.path.exists(self.arg.infile):
            INdf = pd.read_csv(self.arg.infile, sep='\t', comment='#')
        else:
            INdf = pd.DataFrame( {'sampleid' : self.arg.infile.split(',')} )

        if self.arg.commands in ['Fetch', 'Search', 'Region', 'Auto']:
            Parallel( n_jobs=self.arg.njob)( delayed( self.eachwork )(_l) for _n, _l in INdf.iterrows() )
            #for _n, _l in INdf.iterrows():
            #    if _l.sampleid in [ 'simulate', 'HEK293T.CCS.lima.BC1--BC1', 'HEK293T.CCS.lima.BC5--BC5']:
            #        self.eachwork(_l)
        if self.arg.commands in ['Region', 'Auto']:
            self.allwork(INdf)

import argparse
def Args():
    parser = argparse.ArgumentParser(
                formatter_class=argparse.RawDescriptionHelpFormatter,
                prefix_chars='-+',
                conflict_handler='resolve',
                description="",
                epilog="")

    parser.add_argument('-V','--version',action ='version',
                version='EcDNA version 0.1')

    subparsers = parser.add_subparsers(dest="commands",
                    help='models help.')
    P_Common = subparsers.add_parser('Common',conflict_handler='resolve', #add_help=False,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    help='The common parameters used for other models.')
    P_Common.add_argument("-f", "--infile",type=str,
                    help='''the input file or input number split by ",".''')
    P_Common.add_argument("-i", "--indir",type=str,
                    help='''the input directory.''')
    P_Common.add_argument("-o", "--outdir",type=str,default=os.getcwd(),
                    help="output file dir, default=current dir.")
    P_Common.add_argument("-p", "--prefix",type=str,default='',
                    help="output file header, default=None.")
    P_Common.add_argument("-n", "--njob",type=int,default=5,
        help="The maximum number of concurrently running jobs.")
    P_Common.add_argument("-uf", "--outfetch", type=str, default='03.SoftMap',
                    help="out directory for fetch")
    P_Common.add_argument("-if", "--infetch", type=str, default='02.MiniMap',
                    help="input bam directory for fetch ")
    P_Common.add_argument("-us", "--outsearch", type=str, default='03.SoftMap',
                    help="out directory for search")
    P_Common.add_argument("-ur", "--outregion", type=str, default='04.AllRegion',
                    help="out directory for search")
    P_Common.add_argument("-bt", "--bedtools", type=str, default='/share/home/share/software/bedtools2/bin/',
                    help="bedtools path")
    P_Common.add_argument("-ch", "--chroms", action='store_true', default=True,
                    help='''only keep the 23 chromsomes.''')

    P_fetch = subparsers.add_parser('Fetch', conflict_handler='resolve', add_help=False)
    P_Fetch = subparsers.add_parser('Fetch',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common,P_fetch],
                    help='fatch reads information from bam file.')

    P_search = subparsers.add_parser('Search', conflict_handler='resolve', add_help=False)
    P_Search = subparsers.add_parser('Search',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common,P_search],
                    help='search breakpoint  region from bed file.')

    P_region = subparsers.add_parser('Region', conflict_handler='resolve', add_help=False)
    P_Region = subparsers.add_parser('Region',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common,P_region],
                    help='merge all breakpoint region in all samples.')

    P_Autopipe = subparsers.add_parser('Auto', conflict_handler='resolve', prefix_chars='-+',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search, P_region],
                    help='the auto-processing for all.')
    P_Autopipe.add_argument("+P", "++pipeline",nargs='+',
                    help="the auto-processing: standardization, feature selection, Fitting and/or Prediction.")
    P_Autopipe.add_argument('+M','++MODEL' , nargs='+', type=str, default=['Standard'],
                    help='''Chose more the one models from Standard, Fselect,Fitting and Predict used for DIY pipline.''')
    args  = parser.parse_args()
    return args

import logging
class DispatchingFormatter:
    def __init__(self, formatters, default_formatter):
        self._formatters = formatters
        self._default_formatter = default_formatter

    def format(self, record):
        formatter = self._formatters.get(record.name, self._default_formatter)
        return formatter.format(record)

class Logger:
    level_dict = {
        'NOTSET'  : logging.NOTSET,
        'DEBUG'   : logging.DEBUG,
        'INFO'    : logging.INFO,
        'WARNING' : logging.WARNING,
        'ERROR'   : logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }

    ChangeFrom = DispatchingFormatter(
            { 'c' : logging.Formatter( '[%(asctime)s] [%(levelname)-4s]: %(message)s', '%Y-%m-%d %H:%M:%S'),
              'p' : logging.Formatter( '[%(levelname)-4s]: %(message)s'),
              'n' : logging.Formatter( '%(message)s' ),
            }, 
            logging.Formatter('%(message)s')
     )

    def __init__(self, outpath, filemode='w',  clevel = 'INFO', Flevel = 'INFO'):

        logging.basicConfig(
            level    = Logger.level_dict[clevel] ,
            format   = '[%(asctime)s] [%(levelname)-4s]: %(message)s',
            datefmt  = '%Y-%m-%d %H:%M:%S',
            filename = None,
        )

        File = logging.FileHandler(outpath,  mode= filemode)
        File.setLevel(Logger.level_dict[Flevel])
        File.setFormatter(Logger.ChangeFrom)
        logging.getLogger().addHandler(File)

        self.R = logging
        self.C = logging.getLogger('c')
        self.P = logging.getLogger('p')
        self.N = logging.getLogger('n')
        self.CI = logging.getLogger('c').info
        self.NI = logging.getLogger('n').info
        self.CW = logging.getLogger('c').warning
        self.NW = logging.getLogger('n').warning

import os
import time
import traceback
def Commands():
    info ='''
***********************************************************
* Author : Zhou Wei                                       *
* Date   : %s                       *
* E-mail : welljoea@gmail.com                             *
* You are using The scripted by Zhou Wei.                 *
* If you find some bugs, please email to me.              *
* Please let me know and acknowledge in your publication. *
* Sincerely                                               *
* Best wishes!                                            *
***********************************************************
'''%(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

    args = Args()
    os.makedirs( args.outdir, exist_ok=True)
    Log = Logger( '%s/%s_log.log'%(args.outdir, args.commands) )

    Log.NI(info.strip())
    Log.NI("The argument you have set as follows:".center(59, '*'))
    for i,k in enumerate(vars(args),start=1):
        Log.NI('**%s|%-13s: %s'%(str(i).zfill(2), k, str(getattr(args, k))) )
    Log.NI(59 * '*')

    try:
        ecDNA(args, Log).Pipeline()
        Log.CI('Success!!!')
    except Exception:
        Log.CW('Failed!!!')
        traceback.print_exc()
    finally:
        Log.CI('You can check your progress in log file.')
Commands()