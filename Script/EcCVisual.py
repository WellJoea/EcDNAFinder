#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : get_links.py
* @Author  : Zhou Wei                                     *
* @Date    : 2021/01/25 19:41:11                          *
* @E-mail  : welljoea@gmail.com                           *
* @Version : --                                           *
* You are using the program scripted by Zhou Wei.         *
* If you find some bugs, please                           *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''

import pandas as pd
import numpy  as np
import re
import os


genebed='/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf.gene.bed'

Gene = pd.read_csv(genebed, sep='\t')
Gene = Gene.loc[(Gene.gene_biotype =='protein_coding'), ['#chrom', 'start', 'end', 'gene_name']]
Gene['#chrom'] = 'hs' + Gene['#chrom'].astype(str).str.lstrip('chr')

class Circos:
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log

    def _getio(self):
        self.circonf = os.path.dirname(os.path.realpath(__file__)) + '/../Circos/links.circos.conf'
        self.infile  = (self.arg.circisin  or '%s/%s.UpFilterTRF'%(self.arg.Update, self.arg.updatepre) )
        self.outdir  = (self.arg.circisout or self.arg.Update)
        self.outdata = self.outdir + '/data/'
        self.outhead = self.arg.cirhead
        self.outpre  = "%s/%s"%(self.outdir, self.outhead)
        os.makedirs(self.outdata, exist_ok=True)

    def _getinfo(self):
        self.Chrs  = [ 'hs' + str(i) for i in range(1,23)] + ['X','Y']
        Links = pd.read_csv(self.infile, sep='\t')
        Links['#chrom'] = 'hs' + Links['#chrom'].astype(str).str.lstrip('chr')
        Drop  = Links.loc[ ~(Links['#chrom'].isin(self.Chrs)), 'LINKS' ]
        self.Links = Links[ ~(Links.LINKS.isin(Drop)) ]

        Gene = pd.read_csv(self.arg.gtfbed, sep='\t')
        Gene['#chrom'] = 'hs' + Gene['#chrom'].astype(str).str.lstrip('chr')
        self.Gene = Gene.loc[(Gene.gene_biotype =='protein_coding'), ['#chrom', 'start', 'end', 'gene_name']]

    def regionstack(self, _G):
        '''
        '#chrom', 'start', 'end', 'support_num'
        '''
        if _G.shape[0] <2:
            return _G
        _G = _G.copy()
        _G[:, 1:3]  = np.sort(_G[:, 1:3]) #_G[:,[1:3]].sort()
        _L = sorted(set(_G[:,1:3].flatten()))
        _L = np.array([ [_G[0,0], _L[i], _L[i+1], 0] for i in range(len(_L) -1 )], dtype=np.object)
        for _l in _G:
            _L[ (_L[:,1]>=_l[1]) & (_L[:,2]<=_l[2]), 3] += _l[3]
        return _L[ (_L[:,3]>0), :]

    def mulitlinks(self, Links):
        Linkmelt = []
        for i in Links.LINKS.unique():
            fillinks = [re.split('[:-]',i) for i in i.split(';')]
            if len(fillinks)>1:
                for j in range(len(fillinks) -1 ):
                    Linkmelt.append( fillinks[j] + fillinks[j+1] + ['type=%s'%len(fillinks)] )

        Linkmelt = pd.DataFrame(Linkmelt, columns=['chra','sa', 'ea', 'chrb', 'sb', 'eb', 'ty'])
        Linkmelt.sort_values(by=['chra','sa', 'ea'],inplace=True)
        Linkmelt.iloc[:,0] = 'hs' + Linkmelt.iloc[:,0].astype(str)
        Linkmelt.iloc[:,3] = 'hs' + Linkmelt.iloc[:,3].astype(str)
        #Linkmelt.iloc[:,1:3].values.sort()
        #Linkmelt.iloc[:,4:6].values.sort()
        Linkmelt.to_csv(self.outdata + '/links.multi.txt', sep='\t',index=False, header=False)

    def linksnum(self, Links):
        Links = Links.copy()
        targetdf = Links[['#chrom', 'start', 'end', 'support_num']]
        K=targetdf.groupby('#chrom').apply(lambda x: self.regionstack(x.values) )
        np.savetxt( self.outdata +'/links.num.txt', np.vstack(K), delimiter='\t',  fmt='%s')

    def linkssite(self, Links):
        targetdf = Links[['#chrom', 'start', 'end', 'Type']].copy()
        targetdf['Type'] = 'type=' + targetdf['Type'].astype(str)
        targetdf.to_csv( self.outdata + '/links.site.txt', sep='\t',index=False, header=False)

    def geneannot(self, Links, Gene):
        targetseri = list(set(';'.join(Links.gene_name).split(';')))
        Gene[ (Gene.gene_name.isin(targetseri))]\
            .to_csv( self.outdata +'/links.gene.txt', sep='\t',index=False, header=False)

    def goplot(self):
        cmd='cd {2} ; {0} -conf {1} -outputdir {2} -outputfile {3}.svg'\
                .format(self.arg.circissw, self.circonf, self.outdir, self.outhead )

        if self.arg.cirplot:
            os.system(cmd)
            try:
                import cairosvg
                cairosvg.svg2pdf(url=self.outpre+'.svg', write_to=self.outpre+'.pdf')
                #from svglib.svglib import svg2rlg
                #from reportlab.graphics import renderPDF
                #drawing = svg2rlg("%s/%s.svg"%(self.outdir, self.outhead) )
                #renderPDF.drawToFile(drawing, "%s/%s.pdf"%(self.outdir, self.outhead))
                #os.system('rsvg-convert {0}/{1}.svg -f pdf -o {0}/{1}.1.pdf'.format(self.outdir, self.outhead))
            except:
                pass
    def ConfData(self):
        self.log.CI('start circos plot.')
        self._getio()
        self._getinfo()
        self.linksnum(self.Links.copy())
        self.mulitlinks(self.Links.copy())
        self.linkssite(self.Links.copy())
        self.geneannot(self.Links.copy(), self.Gene.copy())
        self.goplot()
        self.log.CI('finish circos plot.')
