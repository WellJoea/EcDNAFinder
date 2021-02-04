import pandas as pd
import numpy  as np

IN='/datb/zhouwei/01Projects/03ecDNA/Nanopore/03MINIMAP/04.EcRegion/All.circle.region.UpFilter'
IN='/datb/zhouwei/01Projects/03ecDNA/Nanopore/03MINIMAP/04.EcRegion/All.circle.region.UpMerge_sort'
OU='/datb/zhouwei/01Projects/03ecDNA/Nanopore/03MINIMAP/04.EcRegion/'
IN=pd.read_csv(IN, sep='\t')
IN.sort_values(by=['Type', 'LINKS', 'Order'], inplace=True)
IN['gene_num'] =  IN.gene_name.str.split(';').str.len()

GeneDict={}

for _, _l in IN.iterrows():
    if _l.gene_name=='.':
        continue
    _G = _l.gene_name.split(';')
    _B = _l.gene_biotype.split(';')

    for _i, _g in enumerate(_G):
        GeneDict.setdefault((_g, _B[_i]),[]).append(_l.LINKS)
        #GeneDict.setdefault((_g, _B[_i]),{})[_l.LINKS] = _l[['LINKS', 'Type', 'support_IDs', 'support_read_num']].values

F=open(OU+'./gene.stat.xls', 'w')
F.write('gene_name\tgene_biotype\tLINKS_num\tSIDs_num\treads_num\tMonoreads\tMultireads\t'
        'LINKS\tType\tgene_num\tsupport_IDs\tsupport_read_num\n')
for (gn,gb), v in GeneDict.items():
    Gs = IN[(IN.LINKS.isin(v))]
    LINKS = Gs.LINKS.str.cat(sep=';')
    Type  = Gs.Type.astype(str).str.cat(sep=';')
    gene_num = Gs.gene_num.astype(str).str.cat(sep=';')
    support_IDs = Gs.support_IDs.str.cat(sep=';').split(';')
    support_read_num = Gs.support_read_num.astype(str).str.cat(sep=';').split(';')
    support_dict = {}
    for n, i in enumerate(support_IDs):
        if i in support_dict:
            support_dict[i] += int(support_read_num[n])
        else:
            support_dict[i]  = int(support_read_num[n])
    support_IDs = ';'.join( support_dict.keys() )
    support_read_num = ';'.join( map(str,support_dict.values()) )

    Monoreads  = Gs[(Gs.Type)==1].support_num.sum()
    Multireads = Gs[(Gs.Type) >1].support_num.sum()

    F.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(
             gn, gb, Gs.shape[0], len(support_dict), Monoreads + Multireads, Monoreads, Multireads,
             LINKS, Type, gene_num, support_IDs, support_read_num))
F.close()
