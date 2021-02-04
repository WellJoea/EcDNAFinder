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
matplotlib.rcParams['ps.fonttype']  = 42
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

    def clustmap(self, _indf, out, Trans=False):
        linewidths= 0 if min(_indf.shape) > 60  else 0.01
        hm = sns.clustermap(_indf,
                    method='complete',
                    metric='euclidean',
                    z_score=None,
                    #figsize=figsize,
                    linewidths=linewidths,
                    cmap="viridis_r",
                    cbar_pos=(0.02, 0.83, 0.03, 0.11)
                    #center=0,
                    #fmt='.2f',
                    #square=True, 
                    #cbar=True,
                    #yticklabels=Xa,
                    #xticklabels=Xa,
                    #vmin=-1.1,
                    #max=1.1,
                    )
        hm.savefig(out)
        #hm.fig.subplots_adjust(right=.2, top=.3, bottom=.2)
        plt.close()

class Utilities():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.GTFBED = self.arg.gtfbed
        self.hg38gtf  = self.arg.gtf
        self.annotatepeak = self.arg.annopeak
        self.samtools = self.arg.samtools
        self.bedtools = self.arg.bedtools
        self.checkbed = self.arg.checkbed

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

    def annogene(self, _inbed, outpre, minover=30, 
                    biotype=['miRNA','lncRNA', 'protein_coding', 'IG_C_gene', 'IG_D_gene', 
                            'IG_J_gene', 'IG_V_gene', 'TR_C_gene', 'TR_D_gene', 'TR_J_gene']
                            #'TR_V_gene', 'rRNA', 'scaRNA', 'scRNA', 'snoRNA', 'snRNA', 'sRNA'] ):
                ):
        GTFBED = pd.read_csv(self.GTFBED, sep='\t', header=0)
        GTFBED = GTFBED[(GTFBED.gene_biotype.isin(biotype))]
        GTFBED['#chrom'] = GTFBED['#chrom'].astype(str)
        _inbed['#chrom'] = _inbed['#chrom'].astype(str)

        COL1 = ['#chrom', 'start', 'end', 'length']
        COL2 = ['#chrom_g', 'start_g', 'end_g', 'gtype', 'length_g', 'forword_g',  'gene_name', 'gene_id', 'gene_biotype']
        LINE = ['.', -1, -1, '.', '.', '.', '.', '.', '.']

        annobed = bt.BedTool.from_dataframe(GTFBED)
        inbed   = bt.BedTool.from_dataframe(_inbed)
        inbed   = inbed.intersect(annobed, s=False, S=False, loj=True)\
                        .to_dataframe(disable_auto_names=True,  header=None, names=COL1+COL2)

        # reduce running time
        inbed['#chrom'] = inbed['#chrom'].astype(str)
        inbed1 = inbed[(inbed.start_g ==-1)]
        inbed2 = inbed[(inbed.start_g !=-1)].copy()
        inbed2['len_m'] = inbed2[['end', 'end_g']].min(1) - inbed2[['start', 'start_g']].max(1) + 1
        inbed2.loc[(inbed2.len_m < minover), COL2] = LINE
        inbed2 = inbed2[COL1 + COL2]

        inbed = pd.concat([inbed1, inbed2], axis=0, sort=False)
        del(inbed1, inbed2)
        inbed.to_csv(outpre+'.BedAnnotate', sep='\t', index=False)

        inbed = pd.merge(inbed.groupby(by=COL1)['gene_name'].apply(lambda x:x.astype(str).str.cat(sep=';')).reset_index(),
                         inbed.groupby(by=COL1)['gene_biotype'].apply(lambda x:x.astype(str).str.cat(sep=';')).reset_index(),
                         on=COL1, how='outer')

        inbed.to_csv(outpre+'.BedAnnotate.merge', sep='\t', index=False)
        return inbed

    def _mapneigtb(self, _indf, sortC = ['#chrom', 'start', 'end', 'forword'], maxdistance=500, maxreg=True, oriant=False):
        indf   = _indf.sort_values(by=sortC, ascending=[True]*len(sortC))
        cols   = ['#chrom', 'start', 'end',  'forword']
        indf   = indf[cols].to_numpy().tolist()
        mapset = [ indf[0] ]
        for _l in indf:
            _m = -1
            _L = mapset[_m]

            R = str(_l[0]) == str(_L[0])
            F = str(_l[3]) == str(_L[3])
            S = (_L[1] - maxdistance) <= _l[1] <= (_L[1] + maxdistance)
            E = (_L[2] - maxdistance) <= _l[2] <= (_L[2] + maxdistance)

            if (R and S and E) and ( not(oriant and F) ):
                if maxreg:
                    _L[1] = min([_L[1], _l[1]])
                    _L[2] = max([_L[2], _l[2]])
                else:
                    _L[1] = max([_L[1], _l[1]])
                    _L[2] = min([_L[2], _l[2]])
                mapset[_m] = _L
            else:
                mapset.append( _l )
        mapset = pd.DataFrame(mapset, columns=cols)
        return mapset

    def mapneigtb(self, _indf, maxdistance=500, maxreg=True, oriant=False):
        if oriant:
            sortA = ['#chrom', 'forword', 'start', 'end']
            sortD = ['#chrom', 'forword', 'end', 'start']
        else:
            sortA = ['#chrom', 'start', 'end', 'forword']
            sortD = ['#chrom', 'end', 'start', 'forword']

        indf  = _indf.copy()
        mline = 0
        while mline != indf.shape[0]:
            mline = indf.shape[0]
            indf = self._mapneigtb( indf, sortD)
            print(indf.shape)
            indf = self._mapneigtb( indf, sortA)
            print(indf.shape)

        return indf

    def mapanytwo(self, indf, maxdistance=500, maxreg=True, maxline=3000000, oriant=False):
        def _splitmap(_inmap):
            inmap  = _inmap.copy()
            for _n, _l in inmap.iterrows():
                S = inmap.start_n.between(_l.start - maxdistance, _l.start + maxdistance, inclusive=True)
                E = inmap.end_n.between(  _l.end   - maxdistance, _l.end   + maxdistance, inclusive=True)
                inmap.loc[(S & E),'start_n'] = inmap[(S & E)]['start'].min()
                inmap.loc[(S & E),  'end_n'] = inmap[(S & E)]['end'  ].max()
            return inmap

        sortN = ['#chrom', 'start', 'end', 'forword']
        mapsN = ['#chrom', 'start', 'end', 'forword', 'start_n', 'end_n']
        grpby = ['#chrom', 'forword'] if oriant else ['#chrom']

        indf  =  indf.copy().sort_values(by=sortN)
        indf[['start_n', 'end_n']] = indf[['start', 'end']]

        if indf.shape[0] > maxline:
            inmap = indf[mapsN].drop_duplicates(keep='first')
            inmap = Parallel( n_jobs=-1, backend='loky')(delayed(_splitmap )(_g) 
                            for _, _g in inmap.groupby(by=grpby, sort=False))
            inmap = pd.concat(inmap, axis=0)
            indf = indf.merge(inmap, on=sortN, how='left')
        else:
            indf = Parallel( n_jobs=-1, backend='loky')(delayed(_splitmap )(_g) 
                            for _, _g in indf.groupby(by=grpby, sort=False))
            indf = pd.concat(indf, axis=0)

        indf[['start_n', 'end_n']] = indf[['start_n', 'end_n']].astype(int)
        indf[['length_n']] = indf['end_n'] -  indf['start_n'] + 1
        return indf

    @emptyfile
    def bambedflter(self, inbam, output):
        cmd = '''
        {samtools}/samtools view -b -F 256 -F 272 {inbam} | \
        {bedtools}/bedtools intersect -a {checkbed} -b stdin  -wa -wb -loj -bed > {output} 
        '''.format(samtools=self.samtools, 
                    bedtools=self.bedtools, 
                    checkbed=self.checkbed, 
                    inbam=inbam,
                    output=output).strip()
        print(cmd)
        os.system( 'echo " %s" | qsub -V -cwd -pe smp 15 -l h_vmem=60G' %cmd)

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
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.chrs =[str(i) for i in range(1,23)] + ['MT','X','Y'] #remove
        self.minsoflt = self.arg.minsoftdrop
        self.mapq     = self.arg.minmapQ
        self.indel    = self.arg.maskindel
        self.skip     = self.arg.maskskip
        self.hard     = self.arg.maskhard
        self.pad      = self.arg.maskpad
        self.lensoftfq = self.arg.lensoftfq
        self.softfq   = self.arg.getsoftfq

    def _getinfo(self, _info):
        self.info = _info
        self.inid = _info.sampleid
        self.inbam = '%s/%s.sorted.bam'%(self.arg.Bam, self.inid)
        self.outdir= '%s/%s'%(self.arg.Fetch, self.inid )
        self.arg.outpre= '%s/%s'%(self.outdir, self.inid)
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def cigarmerge(self, _ct, indel=100000, skip=1000000000, hard=100000, pad=1000000000, match='Q'):
        indel = self.indel #100000
        skip  = self.skip  #10000000
        hard  = self.hard  #100000
        pad   = self.pad   #10000000

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
        softminlen = self.lensoftfq
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
                        _g.flag, _g.mapping_quality, _g.cigarstring, cigarpos, cigarpos[n], _g.query_length, cigref, _g.query_counts]
                Regs.append(bed)
                start += int(i[1]-1)
        return Regs

    def GetSoft(self, _info, softfq=False, researh=True):
        softfq = self.softfq
        self._getinfo(_info)

        self.log.CI('start finding soft signal: ' + self.inid)
        if researh:
            INsoft = self.bamcigarsoft()
        else:
            INsoft = pd.read_csv('%s/%s.chimeric.txt'%(self.outdir, self.inid), sep='\t', low_memory=False)

        if INsoft.empty:
            self.log.CW(self.inid +' donot find soft signal...')

        BEDS = Parallel( n_jobs=-1, backend='loky')( delayed( self.getsoftregion )(_l) for _n, _l in INsoft.iterrows() )

        if softfq :
            self.getsoftfq(INsoft)
        del(INsoft)

        colm = ['#chrom', 'start', 'end',  'SID', 'length', 'forword', 'query_name', 'flag', 'mapping_quality',
                'cigarstring', 'cigarpos', 'cigarreg' , 'query_length', 'cigarreffilt', 'query_counts']
        BEDS = pd.DataFrame( np.reshape( np.array(BEDS),(-1,len(colm))), columns=colm)
        BEDS.sort_values(by=['#chrom', 'start', 'end', 'query_name', 'cigarreg'],
                            ascending=[True]*5, inplace=True)
        BEDS.to_csv('%s/%s.chimeric.bed'%(self.outdir, self.inid), header=True, index=False, sep='\t')
        Visal().query_length(BEDS , '%s/%s.chimeric.query_length.pdf'%(self.outdir, self.inid))

        self.log.CI('finish finding soft signal: ' + self.inid)

class SearchType():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.Chrom = self.arg.Chrom
        self.overmaperrors = self.arg.overmaperrors
        self.dropcigarover = self.arg.dropcigarover
        self.dropneighbdup = self.arg.dropneighbdup
        self.maxhtdistance = self.arg.maxhtdistance
        self.maxneighbtwoends = self.arg.maxneighbtwoends
        self.maxneighboneend = self.arg.maxneighboneend
        self.neighbmergeways = self.arg.neighbmergeways
        self.maxmasksofttwoends = self.arg.maxmasksofttwoends
        self.maxoverlap = self.arg.maxoverlap
        self.chrs=[str(i) for i in range(1,23)] + ['MT','X','Y'] \
                    + ['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                        '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                        'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                        'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                        'SunTag-CRISPRi', 'V7-MC-HG-FA']

    def _getinfo(self, _info):
        self.info = _info
        self.inid = _info.sampleid
        self.outdir= '%s/%s'%(self.arg.Search, self.inid)
        self.arg.outpre= '%s/%s%s'%(self.outdir, self.inid, self.Chrom)
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getbeddb(self):
        inbed = '{0}/{1}/{1}.chimeric.bed'.format(self.arg.Fetch, self.inid)
        inbed = pd.read_csv( inbed, sep='\t', low_memory=False)

        if inbed.empty:
            inbed=pd.DataFrame( columns = inbed.columns.tolist() + ['raw_order', 'fflag'])
        else:
            #inbed[['Type', 'beds', 'links']] = pd.DataFrame([['',[],[]]]*inbed.shape[0])
            #inbed['links'] = ''
            #inbed['beds'] = inbed[['#chrom', 'start', 'end']].apply(lambda x: '{0}:{1}-{2}'.format(*x), axis=1)
            inbed[['start', 'end']]  = inbed[['start', 'end']].astype(int)
            inbed['#chrom']  = inbed['#chrom'].astype(str)
            inbed['cigarreg'] = inbed.cigarreg.apply(lambda x: tuple([int(i) for i in eval(x)]) )
            inbed['fflag']  = 'DROP'
            inbed['raw_order'] = 1

        COLs  = ['#chrom', 'start', 'end',  'SID', 'length', 'forword', 'query_name', 'query_length',
                 'fflag', 'raw_order','query_counts', 'cigarstring',  'cigarreg']
        inbed = inbed[COLs]
        inbed.sort_values(by=['query_name', 'cigarreg', '#chrom', 'start', 'end' ], ascending=[True]*5, inplace=True)
        inbed['raw_order'] =  inbed.groupby(by=['SID', 'query_name'], sort=False)['raw_order'].apply(np.cumsum)

        if self.Chrom:
            inbed  = inbed[ (inbed['#chrom'].isin(self.chrs) )]
        return inbed

    def dropcigarinclude(self, _G,  errors=30):
        errors = self.overmaperrors  #30
        if _G.shape[0] <2:
            return _G
        else:
            _G = _G.reset_index(drop=True).copy()
            for _n, _l in _G.iterrows():
                for _m, _k in _G.loc[_n+1:, :].iterrows():
                    _s1 = _l.cigarreg[0]
                    _e1 = _l.cigarreg[1]
                    _f1 = _l.forword
                    _s2 = _k.cigarreg[0]
                    _e2 = _k.cigarreg[1]
                    _f2 = _k.forword
                    if   (_s1 <= _s2) and (_e1 >= _e2) and (_f1 == _f2):
                        _G.loc[_m, 'fflag'] ='OVER'
                    elif (_s2 <= _s1) and (_e2 >= _e1) and (_f1 == _f2):
                        _G.loc[_n, 'fflag'] ='OVER'
            return _G

    def maxbeddistance(self, _G):
        if _G.shape[0] <2:
            return _G
        elif _G.shape[0] ==2: #reduce running time
            if _G['#chrom'].unique().size == _G['forword'].unique().size == 1:
                _G['fflag'] = 'HTDIST'
            return _G
        else:
            if _G['#chrom'].unique().size == _G.shape[0]: #reduce running time
                return _G
            else:
                _G = _G.reset_index(drop=True).copy()
                k=[]
                for _n, _l in _G.iterrows() :
                    for _m, _j in _G.loc[_n+1:].iterrows():
                        if (_l['#chrom'] == _j['#chrom']) and (_l['forword'] == _j['forword']):
                            k.append( [_n ,_m, _m - _n, _G.loc[_n:_m,'length'].sum() ])
                if k:
                    k = pd.DataFrame(k, columns=['s','e','d','l'])\
                            .sort_values(by=['l','d'],ascending=[False, False])\
                            .iloc[0,:]
                    _G.loc[k.s : k.e, 'fflag'] = 'HTDIST'
                return _G

    def mergeNeighb(self, _G, maxdistance=200, maxsimilar=20, maxreg=True, oriant=True):
        maxdistance = self.maxneighbtwoends #300
        maxsimilar  = self.maxneighboneend  #50
        maxreg      = self.neighbmergeways  #True

        if _G.shape[0] <= 1: 
            return _G
        elif _G.shape[0] == 2: #bug 2 rows which dup
            if  (_G.iloc[0]['#chrom']  == _G.iloc[-1]['#chrom'])  and \
                (_G.iloc[0]['forword'] == _G.iloc[-1]['forword']) and \
                (abs(_G.iloc[0]['start'] - _G.iloc[-1]['start']) <= maxsimilar) and  \
                (abs(_G.iloc[0]['end']   - _G.iloc[-1]['end'])   <= maxsimilar) :
                _G['fflag'] = 'DUPLIC1'
            return _G
        else:
            overm = [ _G.iloc[0,:].copy() ]
            for _n, _l in _G.iloc[1:,:].iterrows():
                _L = overm[-1].copy()
                if (_l['#chrom']  == _L['#chrom']) and (_l['forword'] == _L['forword']): #reduce running time
                    S1 = (_L['start'] - maxdistance) <= _l['start'] <= (_L['start'] + maxdistance)
                    E1 = (_L['end'] - maxdistance)   <= _l['end']   <= (_L['end'] + maxdistance)
                    S2 = (_L['start'] - maxsimilar)  <= _l['start'] <= (_L['start'] + maxsimilar)
                    E2 = (_L['end'] - maxsimilar)    <= _l['end']   <= (_L['end'] + maxsimilar)

                    if (S1 and E1) or (S2 or E2):
                        if   _L['fflag'] in ['DUPMER1','DUPMER2'] :
                            overm.pop()
                        else:
                            overm[-1]['fflag'] = 'DUPLIC2'
                        _l['fflag'] = 'DUPLIC2'
                        overm.append(_l)

                        _L['start']  = min([_L['start'], _l['start']])
                        _L['end']    = max([_L['end'],   _l['end']]  )
                        _L['length']   = _L['end'] - _L['start'] + 1
                        _L['cigarreg'] = (min(_L['cigarreg']  + _l['cigarreg']), max(_L['cigarreg']  + _l['cigarreg']))
                        _L['cigarstring'] = '%s;%s'%(_L['cigarstring'], _l['cigarstring'] )
                        _L['fflag'] = 'DUPMER2' if (S1 and E1)  else 'DUPMER1'
                        overm.append(_L)
                    else:
                        overm.append(_l)
                else:
                    overm.append(_l)
    
            overm = pd.concat(overm, axis=1).T
            return overm

    def mergeHeadTail(self, _G, maxhtdistance=10000000):
        maxhtdistance = self.maxhtdistance  #10000000 deprecated

        _G = _G.reset_index(drop=True)
        _H = _G.iloc[0,:]
        _T = _G.iloc[-1,:]
        _L = _G.iloc[0,:].copy()

        #if _L > maxhtdistance:
        #    print('warning: the ecDNA breakpoint lenght is large than %s:\n%s'%(_L, _G))

        _G.loc[_H.name, 'fflag'] += ';HEAD'
        _G.loc[_T.name, 'fflag'] += ';TAIL'

        _L['start']  = min([_H['start'], _T['start']])
        _L['end']    = max([_H['end'],   _T['end']])
        _L['length'] = _L['end'] - _L['start'] + 1
        _L['fflag']  += ';HTBREAKP' 
        _L['cigarreg']   = (min(_H['cigarreg']  + _T['cigarreg']), max(_H['cigarreg']  + _T['cigarreg']))
        _L['cigarstring'] = '%s;%s'%(_H['cigarstring'], _T['cigarstring'] )
        _G = pd.concat([_L.to_frame().T, _G], axis=0, sort=False)
        return _G

    def markKeep(self, _G, maxmis=0.1, maxoverlap=800):
        maxmis = self.maxmasksofttwoends #0.1
        maxoverlap = self.maxoverlap #400

        if   _G.shape[0] <2:
            return _G
        elif _G.shape[0]>=2:
            BreakF = _G.iloc[ 0,:]
            BreakL = _G.iloc[-1,:]
            C  = BreakF['#chrom'] == BreakL['#chrom']
            F  = BreakF.forword == BreakL.forword

            QF = BreakF.cigarreg[0] <= BreakF.query_length*maxmis
            QL = BreakL.cigarreg[1] >= BreakF.query_length*(1-maxmis)
            S1 = BreakF.start > BreakL.start
            E1 = BreakF.end   > BreakL.end
            X1 = BreakF.start > BreakL.end - maxoverlap

            S2 = BreakF.start < BreakL.start
            E2 = BreakF.end   < BreakL.end
            X2 = BreakL.start > BreakF.end - maxoverlap

            if (C and F and QF and QL):
                if (S1 and E1 and X1 and BreakF.forword=='+' ) \
                    or (S2 and E2 and X2 and BreakF.forword=='-' ):
                    _G['fflag'] +=';KEEP'
                    _G = self.mergeHeadTail(_G)
                else:
                    _G['fflag'] +=';DISS'
            return _G

    def typeCat(self, indf, dropcigarover=True, dropneighbdup=True):
        dropcigarover = self.dropcigarover #True
        dropneighbdup = self.dropneighbdup #True

        # dropcigarover
        self.log.CI('start droping overlap of mapping region: ' + self.inid)
        if dropcigarover:
            indf =  Parallel( n_jobs= -1, backend='threading')( delayed( self.dropcigarinclude )(_g)
                               for _, _g in indf.groupby(by=['SID', 'query_name'], sort=False))
            indf = pd.concat(indf, axis=0, sort=False)
        OVER = indf[(indf.fflag=='OVER')]
        indf = indf[(indf.fflag!='OVER')]

        # maxbeddistance
        self.log.CI('start computing maximal distance of mapping region: ' + self.inid)
        indf =  Parallel( n_jobs= -1, backend='threading')( delayed( self.maxbeddistance )(_g)
                            for _, _g in indf.groupby(by=['SID', 'query_name'], sort=False))
        indf = pd.concat(indf, axis=0, sort=False)
        DIST  = indf[(indf.fflag!='HTDIST')]
        indf = indf[(indf.fflag=='HTDIST')]

        # mergeNeighb
        self.log.CI('start merging neighbour duplcations of mapping region: ' + self.inid)
        indf = Parallel( n_jobs= -1, backend='threading')( delayed( self.mergeNeighb )(_g)
                    for _, _g in indf.groupby(by=['SID', 'query_name'], sort=False))
        indf = pd.concat(indf, axis=0, sort=False)
        DUPL = indf[ (indf.fflag.str.contains('DUPLIC', regex=False))]
        indf = indf[~(indf.fflag.str.contains('DUPLIC', regex=False))]

        # markKeep
        self.log.CI('start marking and merging head-to-tail mapping region: ' + self.inid)
        indf = Parallel( n_jobs= -1, backend='threading')( delayed( self.markKeep )(_g)
                    for _, _g in indf.groupby(by=['SID', 'query_name'], sort=False))
        indf = pd.concat(indf, axis=0, sort=False)
        DISS = indf[~(indf.fflag.str.contains('KEEP', regex=False))]
        indf = indf[ (indf.fflag.str.contains('KEEP', regex=False))]

        # concat
        MARK = pd.concat( [OVER, DIST, DUPL, DISS, indf], axis=0, sort=False)
        KEEP = indf[ ~(indf.fflag.str.contains('HEAD|TAIL', regex=True))]
        del( OVER, DIST, DUPL, DISS, indf )

        MARK.to_csv(self.arg.outpre+'.Mark', sep='\t', index=False)
        KEEP.to_csv(self.arg.outpre+'.Keep', sep='\t', index=False)

    def TypeBase(self, _info):
        self._getinfo(_info)
        self.log.CI('start searching breakpoin region: ' + self.inid)
        inbed = self._getbeddb()
        if not inbed.empty:
            self.typeCat( inbed )
        else:
            self.log.CW('cannot find any circle region singal: ' + self.inid)
        self.log.CI('finish searching breakpoin region: ' + self.inid)

class MergeReads():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.Chrom = self.arg.Chrom
        self.maxreadstwoends = self.arg.maxreadstwoends
        self.readsmergeways  = self.arg.readsmergeways

    def _getinfo(self, _info):
        self.info = _info
        self.inid = _info.sampleid
        self.outdir= '%s/%s'%(self.arg.Merge, self.inid)
        self.arg.outpre= '%s/%s%s'%(self.outdir, self.inid, self.Chrom)
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getkeep(self):
        self._inbed= '{0}/{1}/{1}{2}.Keep'.format(self.arg.Search, self.inid, self.Chrom)
        if not os.path.exists(self._inbed):
            self.inbed =pd.DataFrame()
            self.log.CW('cannot find the file: ' + self._inbed)
        else:
            self.inbed = pd.read_csv( self._inbed, sep='\t')
            self.inbed[['start', 'end']]  = self.inbed[['start', 'end']].astype(int)
            self.inbed['#chrom']  = self.inbed['#chrom'].astype(str)
        return self

    def orderlinks(self, _G):
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

    def updataLinks(self, _inbed):
        sortN = ['SID', 'query_name', 'raw_order']
        gropN = ['SID', 'query_name']

        inbed = _inbed.copy()
        inbed['raw_order'] = inbed['raw_order'].astype(int)
        inbed = inbed.sort_values(by=sortN) #keep raw order right

        inbed = inbed.merge(inbed.groupby(by=gropN).size().to_frame(name='Type').reset_index(), on=gropN)

        #inbed = Parallel( n_jobs=-1, backend='loky')( delayed( self.orderlinks )(_g) 
        #                        for _l, _g in inbed.groupby(by=gropN, sort=False))
        
        # Reduce compution time
        inbed1 = inbed[(inbed.Type <=1)].copy()
        inbed1['Order'] = 1
        inbed1['Link'] = inbed1[['#chrom', 'start_n', 'end_n', 'forword']]\
                                    .apply(lambda x: '{0}:{1}-{2}'.format(*x[:3]), axis=1)
        inbed1['LINKS'] = inbed1['Link']
        inbed1['forword_n'] = '+'
    
        inbed2 = inbed[(inbed.Type > 1)]
        if inbed2.shape[0] >0:
            inbed2 = Parallel( n_jobs=-1, backend='loky')( delayed( self.orderlinks )(_g) 
                                for _l, _g in inbed2.groupby(by=gropN, sort=False))
            inbed2 = pd.concat(inbed2, axis=0, sort=False)
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

    def mergeLinks(self, _inbed):
        COL1  = ['LINKS', 'SID']
        COL2  = ['#chrom', 'start_n', 'end_n', 'Type', 'length_n', 'forword_n', 'LINKS', 'Order']

        Support = _inbed.loc[(_inbed.Order == 1), COL1]\
                        .groupby(by=['LINKS', 'SID']).size()\
                        .to_frame('support_ID_num').reset_index()

        Supgrpb = Support.groupby(by=['LINKS'])
        Suplist = [ Supgrpb['support_ID_num'].sum().to_frame('support_num'),
                    Supgrpb['support_ID_num'].apply(lambda x: x.astype(str).str.cat(sep=';')).to_frame('support_read_num'),
                    Supgrpb['SID'].size().to_frame('support_ID_num'),
                    Supgrpb['SID'].apply(lambda x: x.str.cat(sep=';')).to_frame('support_IDs'),
                    Support.pivot(index='LINKS', columns='SID', values='support_ID_num').fillna(0) ]
        Support = pd.concat( Suplist, ignore_index=False, join = 'outer', sort = False, axis=1).reset_index()
        del (Suplist, Supgrpb)

        inbed = _inbed[COL2].drop_duplicates(keep='first').copy()
        inbed.rename(columns={'start_n': 'start', 'end_n': 'end', 'length_n': 'length', 'forword_n': 'forword'}, inplace=True)
        inbed = inbed.merge(Support, on='LINKS', how='outer')
        return inbed

    def mergeReads(self, _inbed, Lplot=True, Hplot=False):
        inbed = Utilities(self.arg, self.log)\
                    .mapanytwo(_inbed, maxdistance = self.maxreadstwoends, maxreg = self.readsmergeways)

        inbed = MergeReads(self.arg, self.log).updataLinks(inbed)
        inbed.to_csv(self.arg.outpre+'.Links', sep='\t', index=False)

        inbed = MergeReads(self.arg, self.log).mergeLinks(inbed)
        inbed.to_csv(self.arg.outpre+'.LinksUp', sep='\t', index=False)

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

class RegionCat():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.Chrom = self.arg.Chrom
        self.maxreadstwoends = self.arg.maxreadstwoends
        self.readsmergeways  = self.arg.readsmergeways
    
        self.outdir= self.arg.Region
        self.arg.outpre= '%s/%s'%(self.arg.Region,self.arg.regionpre)
        os.makedirs(self.outdir, exist_ok=True)

    def AllEcDNA(self, _info):
        self.log.CI('start merging all samples region.')
        Allbed = []
        for _n, _l in _info.iterrows():
            EMerge = '{0}/{1}/{1}{2}.Keep'.format(self.arg.Search, _l.sampleid, self.Chrom)
            if os.path.exists( EMerge ):
                Allbed.append( pd.read_csv(EMerge, sep='\t', header=0) )
            else:
                self.log.CW('cannot find the file: '+ EMerge)

        if Allbed:
            Allbed = pd.concat(Allbed, axis=0,sort=False)
            Allbed[['start', 'end', 'length']]  = Allbed[['start', 'end', 'length']].astype(int)
            Allbed['#chrom']  = Allbed['#chrom'].astype(str)
            Allbed.sort_values(by=['#chrom', 'start', 'end'], inplace=True)
            Allbed.to_csv(self.arg.outpre+'.Keep', sep='\t', index=False)
            MergeReads(self.arg, self.log).mergeReads(Allbed, Lplot=True)
        else:
            self.log.CW('cannot find the valid files.')

        self.log.CI('finish merging all samples region.')

class CheckBP():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.mapq = self.arg.minmapQ
        self.overfremin = self.arg.overfremin
        self.overlenmin = self.arg.overlenmin

    def _getinfo(self, _info):
        self.info = _info
        self.inid = _info.sampleid
        self.inbam = '%s/%s.sorted.bam'%(self.arg.Bam, self.inid)
        self.outdir= '%s/%s'%(self.arg.Cheak, self.inid )
        self.arg.outpre= '%s/%s'%(self.outdir, self.inid)
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getkeep(self):
        self.outdir= '%s/%s'%(self.arg.Cheak, 'BPState' )
        self.outpre= '%s/%s'%(self.outdir, 'All.plasmid')
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def Rmerge( self, intervals):
        """
        :param intervals: List[List[int]]
        :return: List[List[int]]
        """
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or merged[-1][-1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][-1] = max(merged[-1][-1], interval[-1])
        merged = sum([i[1]-i[0] + 1 for i in merged ])
        return merged

    def BEDfilter(self, _inbed):
        F = open(_inbed, 'r').readlines()
        F = [ i.strip().split('\t') for i in F if len(i.strip().split('\t')) < 12 ]

        Names = ['#chrom', 'start', 'end', '#chrom_r', 'start_r', 'end_r', 'query_name', 'mapq', 'forword']
        inbed = pd.DataFrame(F, columns=Names)
        inbed[['#chrom', '#chrom_r']] = inbed[['#chrom', '#chrom_r']].astype(str)
        inbed[['start', 'end', 'start_r', 'end_r', 'mapq']] = inbed[['start', 'end', 'start_r', 'end_r', 'mapq']].astype(int)

        checkb = pd.read_csv(self.arg.checkbed, sep='\t')
        checkb['#chrom'] = checkb['#chrom'].astype(str)
        checkb[['start', 'end']] = checkb[['start', 'end']].astype(int)

        inbed = inbed.merge(checkb, on=['#chrom', 'start', 'end'], how='left')

        #####addinfor
        inbed['SID']      = self.inid
        inbed['end_o']    = inbed[['end', 'end_r']].min(1)
        inbed['start_o']  = inbed[['start', 'start_r']].max(1)
        inbed['BPlength'] = inbed['end'] - inbed['start'] + 1
        Names = ['query_name', '#chrom', 'start', 'end', 'SID', 'Links', 'BPlength', 'start_o', 'end_o', 
                 '#chrom_r', 'start_r', 'end_r', 'mapq', 'forword']
        inbed = inbed[Names]
        inbed.sort_values(by=['SID', 'query_name', '#chrom', 'start'], inplace=True)

        #####filter
        inbed1 = inbed[(inbed.mapq >=self.mapq)]
        inbed1 = inbed1.merge(inbed1.groupby(by=['Links', 'query_name', 'start'])\
                                 .apply(lambda x: self.Rmerge(x[['start_o', 'end_o']].values.tolist()))\
                            .to_frame('OVERlen').reset_index(), on=['Links', 'query_name', 'start'], how='left')
        inbed1['OVERfre'] = (inbed1['OVERlen']/inbed1['BPlength']).round(4)
        inbed1 = inbed1[((inbed1.OVERfre >=self.overfremin) | (inbed1.OVERlen >=self.overlenmin))]

        inbed1 = inbed1.merge(inbed1.groupby(by=['Links', 'query_name']).size()\
                                    .to_frame('query_count').reset_index(), on=['Links', 'query_name'], how='left')
        inbed1 = inbed1.merge(inbed1.groupby(by=['Links', 'query_name'])['start'].unique().apply(lambda x:len(x))\
                                    .to_frame('BP_count').reset_index(), on=['Links', 'query_name'], how='left')
        inbed1 = inbed1[((inbed1.query_count >=2) & (inbed1.BP_count >=2))]

        #inbed1 = inbed1.merge(inbed1.groupby(by=['Links', 'query_name'])['forword'].unique().apply(lambda x:len(x))\
        #                            .to_frame('Forwrd_count').reset_index(), on=['Links', 'query_name'], how='left')
        #inbed1 = inbed1[(inbed1.Forwrd_count<=1)]

        inbed2 = pd.concat([inbed, inbed1[Names]], axis=0, sort=False).drop_duplicates(keep=False)
        return inbed1, inbed2

    def BPBed(self, _inline):
        self._getinfo(_inline)
        self.log.CI('start schecking breakpoin region: ' + self.inid)

        bambed = self.arg.outpre + '.breakpoit.reads.txt'
        if not os.path.exists(bambed):
            Utilities(self.arg, self.log).bambedflter(self.inbam, bambed)
        bedKeep, bedDrop = self.BEDfilter(bambed)
        bedKeep.to_csv(self.arg.outpre + '.breakpoint.Keep.txt', sep='\t', index=False)
        bedDrop.to_csv(self.arg.outpre + '.breakpoint.Drop.txt', sep='\t', index=False)
        self.log.CI('finish schecking breakpoin region: ' + self.inid)

    def BPKEEP(self, _indf, Lplot=True):
        indf = _indf.groupby(by=['#chrom', 'Links', 'SID', ])['query_name']\
                    .unique().apply(lambda x:len(x)).to_frame('support_ID_num').reset_index()

        pvot = indf.pivot(index='#chrom', columns='SID', values='support_ID_num').fillna(0).astype(int)
        indf = indf.groupby(by=['#chrom', 'Links'])['support_ID_num'].sum().to_frame('support_num').reset_index()
        indf = indf.merge(pvot.reset_index(), on='#chrom').sort_values(by=['support_num', '#chrom'], ascending=[False, True])
        indf.to_csv(self.outpre+'.Keep.matrix', sep='\t', index=False)

        if Lplot:
            Visal().clustmap(pvot, self.outpre+'.Keep.matrix.pdf')
            Visal().clustmap(np.log2(pvot+1), self.outpre+'.Keep.matrix.log2.pdf')

    def BPStat(self, _info ):
        Parallel( n_jobs=self.arg.njob, verbose=1 )( delayed( self.BPBed )(_l) for _n, _l in _info.iterrows())
        self.log.CI('start stating all samples region.')
        self._getkeep()
        BPKEEP = []
        for _n, _l in _info.iterrows():
            EMerge = '{0}/{1}/{1}.breakpoint.Keep.txt'.format(self.arg.Cheak, _l.sampleid)
            if os.path.exists( EMerge ):
                BPKEEP.append( pd.read_csv(EMerge, sep='\t', header=0) )
            else:
                self.log.CW('cannot find the file: '+ EMerge)
        if BPKEEP:
            BPKEEP = pd.concat(BPKEEP, axis=0,sort=False)
            BPKEEP['#chrom'] = BPKEEP['#chrom'].astype(str)
            BPKEEP.to_csv(self.outpre+'.Keep', sep='\t', index=False)
            self.BPKEEP(BPKEEP)
        else:
            self.log.CW('cannot find the valid files.')
        self.log.CI('finish stating all samples region.')

    def PlotLM(self):
        self._getkeep()
        indf = pd.read_csv(self.outpre+'.Keep.matrix', sep='\t')
        indf.rename(columns={'#chrom' : 'plasmid'}, inplace=True)
        dPCR = '/data/zhouwei/01Projects/03ecDNA/Nanopore/spikei.info.txt'
        indf = indf.merge(dPCR, on='#chrom', how='left')
        print(indf)

class ecDNA():
    'The pipeline used for machine learning models'
    def __init__(self, arg, log,  *array, **dicts):
        self.arg = arg
        self.log = log
        self.array  = array
        self.dicts  = dicts
        self.arg.Chrom  = '.OnlyChrom' if self.arg.Chrom else ''
        self.arg.Bam    = '%s/%s'%(self.arg.indir, self.arg.bamdir)
        self.arg.Fetch  = '%s/%s'%(self.arg.outdir, self.arg.fetchdir)
        self.arg.Search = '%s/%s'%(self.arg.outdir, self.arg.searchdir)
        self.arg.Merge  = '%s/%s'%(self.arg.outdir, self.arg.mergedir)
        self.arg.Region = '%s/%s'%(self.arg.outdir, self.arg.regiondir)
        self.arg.Cheak  = '%s/%s'%(self.arg.outdir, self.arg.checkdir)

        import multiprocessing
        import importlib
        CORES = multiprocessing.cpu_count()*0.8 if multiprocessing.cpu_count() >8 else 8
        os.environ['NUMEXPR_MAX_THREADS'] = '1000' #str(int(CORES))
        os.environ['PATH'] += ':' + self.arg.bedtools
        importlib.reload(bt)
        bt.set_bedtools_path(self.arg.bedtools)
        #bt.helpers.set_bedtools_path(self.arg.bedtools)

    def FetchW(self, _l):
        SoftFetch( self.arg, self.log ).GetSoft(_l)

    def SearchW(self, _l):
        SearchType( self.arg, self.log ).TypeBase(_l)

    def MergeW(self, _l):
        MergeReads( self.arg, self.log ).EachEcDNA(_l)

    def RegionW(self, _L):
        RegionCat( self.arg, self.log ).AllEcDNA(_L)

    def CheckW(self, _L):
        CheckBP( self.arg, self.log ).BPStat(_L)

    def Pipeline(self):
        if os.path.exists(self.arg.infile):
            INdf = pd.read_csv(self.arg.infile, sep='\t', comment='#')
        else:
            INdf = pd.DataFrame( {'sampleid' : self.arg.infile.split(',')} )

        if self.arg.commands in ['Fetch',  'Auto']:
            Parallel( n_jobs=self.arg.njob, verbose=1 )( delayed( self.FetchW )(_l) for _n, _l in INdf.iterrows() )
        if self.arg.commands in ['Search', 'Auto']:
            Parallel( n_jobs=self.arg.njob, verbose=1 )( delayed( self.SearchW )(_l) for _n, _l in INdf.iterrows() )
        if self.arg.commands in ['Merge',  'Auto']:
            Parallel( n_jobs=self.arg.njob, verbose=1 )( delayed( self.MergeW  )(_l) for _n, _l in INdf.iterrows() )
        if self.arg.commands in ['Region', 'Auto']:
            self.RegionW(INdf)
        if self.arg.commands in ['Check']:
            self.CheckW(INdf)
            #self.PlotLM()

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
    P_Common.add_argument("-n", "--njob",type=int,default=5,
                    help="The maximum number of concurrently running jobs.")
    P_Common.add_argument("-bd", "--bamdir", type=str, default='02.MiniMap',
                    help="input bam directory for fetch ")
    P_Common.add_argument("-fd", "--fetchdir", type=str,  default='03.SoftMap',
                    help="out directory for fetch")
    P_Common.add_argument("-sd", "--searchdir", type=str, default='03.SoftMap',
                    help="out directory of search")
    P_Common.add_argument("-md", "--mergedir", type=str,  default='03.SoftMap',
                    help="out directory of merge")
    P_Common.add_argument("-rd", "--regiondir", type=str, default='04.EcRegion',
                    help="out directory for region")
    P_Common.add_argument("-cd", "--checkdir", type=str, default='05.CheakBP',
                    help="out directory for check breakpoint of  plasmid")
    P_Common.add_argument("-ch", "--Chrom", action='store_true', default=True,
                    help="only keep the specified chromosome: 1-22,X,Y,MT.")
    P_Common.add_argument("-bt", "--bedtools", type=str, default='/share/home/share/software/bedtools2/bin/',
                    help="bedtools path")
    P_Common.add_argument("-st", "--samtools", type=str, default='/share/home/share/software/samtools-1.10/bin/',
                help="samtools path")
    P_Common.add_argument("-gb", "--gtfbed", type=str, default='/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf.gene.bed',
                    help="the gene bed file used for annotation of regions")
    P_Common.add_argument("-ap", "--annopeak", type=str, default='/share/home/share/software/Homer/bin/annotatePeaks.pl',
                    help="the annotatePeaks.pl script")
    P_Common.add_argument("-gt", "--gtf", type=str, default='/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf',
                    help="the genome gtf file.")
    P_Common.add_argument("-cb", "--checkbed", type=str, default='/share/home/zhou_wei/Workspace/11Project/02Plasmid/01analysescript/uniqueovr/BEDUniq.region.txt',
                    help="the bed file of plasmid unique region.")

    P_fetch = subparsers.add_parser('Fetch', conflict_handler='resolve', add_help=False)
    P_fetch.add_argument("-ms", "--minsoftdrop",  type=int, default=5,
                    help="the min length of softclip to drop")
    P_fetch.add_argument("-mq", "--minmapQ", type=int, default=0,
                    help="the min mapq of align reads")
    P_fetch.add_argument("-gs", "--getsoftfq", action='store_true', default=False,
                    help="whether to get softclip reads with fastq format.")
    P_fetch.add_argument("-sl", "--lensoftfq",  type=int, default=100,
                    help="the minimun softclip length to save.")
    P_fetch.add_argument("-mi", "--maskindel", type=int, default=100000,
                    help="the number to mask indel in cigar tulpe.")
    P_fetch.add_argument("-ms", "--maskskip", type=int, default=10000000,
                    help="the number to mask skip in cigar tulpe.")
    P_fetch.add_argument("-mh", "--maskhard", type=int, default=100000,
                    help="the number to mask hard softclip in cigar tulpe.")
    P_fetch.add_argument("-mp", "--maskpad", type=int, default=10000000,
                    help="the number to mask pad in cigar tulpe.")
    P_Fetch = subparsers.add_parser('Fetch',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common,P_fetch],
                    help='fatch reads information from bam file.')

    P_search = subparsers.add_parser('Search', conflict_handler='resolve', add_help=False)
    P_search.add_argument("-dc", "--dropcigarover", action='store_true', default=True,
                    help="whether to drop the overlap mapping region.")
    P_search.add_argument("-dc", "--dropneighbdup", action='store_true', default=True,
                    help="whether to drop the duplication of nerghbor mapping region.")


    P_search.add_argument("-oe", "--overmaperrors", type=int, default=30,
                    help="the error margion bases in overlap mapping region.")
    P_search.add_argument("-ht", "--maxhtdistance", type=int, default=10000000,
                    help="if the distance of breakpoint is large than the number, the warning work out.")
    P_search.add_argument("-nt", "--maxneighbtwoends", type=int, default=200,
                    help="the max distance of breakpoint of two ends to merge nerghbour mapping region.")
    P_search.add_argument("-no", "--maxneighboneend", type=int, default=20,
                    help="the max distance of breakpoint of one end to merge nerghbour mapping region.")
    P_search.add_argument("-nw", "--neighbmergeways", action='store_true', default=True,
                    help="whether to use the max distance of breakpoint to merge nerghbour mapping region.")
    P_search.add_argument("-nm", "--maxmasksofttwoends", type=float, default=0.1,
                    help="the max distance of softclip of one end to mask in head-to-tail mapping region.")
    P_search.add_argument("-mo", "--maxoverlap", type=int, default=800,
                    help="the max overlap distance of head-to-tail region.")
    P_Search = subparsers.add_parser('Search',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search],
                    help='search breakpoint region from bed file.')

    P_merge = subparsers.add_parser('Merge', conflict_handler='resolve', add_help=False)
    P_merge.add_argument("-rt", "--maxreadstwoends", type=int, default=500,
                    help="the max distance of breakpoint of two reads to merge.")
    P_merge.add_argument("-rw", "--readsmergeways", action='store_true', default=True,
                    help="whether to use the max distance of breakpoint to merge two reads region.")
    P_Merge = subparsers.add_parser('Merge',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search, P_merge],
                    help='merge breakpoint region from bed file.')

    P_region = subparsers.add_parser('Region', conflict_handler='resolve', add_help=False)
    P_region.add_argument("-rp", "--regionpre", type=str, default='All.circle.region',
                    help="out prefix of regioin out put.")
    P_Region = subparsers.add_parser('Region',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search, P_merge, P_region],
                    help='merge all breakpoint region in all samples.')

    P_check = subparsers.add_parser('Check', conflict_handler='resolve', add_help=False)
    P_check.add_argument("-of", "--overfremin", type=float, default=0.8,
                    help="the minimum overlap ration of breakpiont region.")
    P_check.add_argument("-ol", "--overlenmin", type=int, default=500,
                    help="the minimum overlap lenght of breakpiont region.")
    P_Check = subparsers.add_parser('Check',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search, P_merge, P_check],
                    help='check plasmid unique breakpoint region.')

    P_Autopipe = subparsers.add_parser('Auto', conflict_handler='resolve', prefix_chars='-+',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search, P_merge, P_region],
                    help='the auto-processing for all.')
    P_Autopipe.add_argument("+P", "++pipeline",nargs='+',
                    help="the auto-processing: Fetch, Search, Region.")
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
    os.makedirs(args.outdir, exist_ok=True)
    Log = Logger( '%s/%s_log.log'%(args.outdir, args.commands) )

    Log.NI(info.strip())
    Log.NI("The argument you have set as follows:".center(59, '*'))

    _ml = max([ len(i) for i in vars(args).keys()] )
    for i,(k,v) in enumerate(vars(args).items(), start=1):
        Log.NI( '**{:0>{}}|{:<{}}: {}'.format(i,2, k,_ml, v) )
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
