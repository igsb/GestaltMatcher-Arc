import os
import sys
import time
import math
import json
import pickle
import logging
import numpy as np
import pandas as pd

from shutil import copyfile
from statistics import mean
import matplotlib.pyplot as plt

date = '19042023_v1.0.5'
output_dir = os.path.join('gmdb_data/', date)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
input_dir = '../data/GestaltMatcherDB/v1.0.3/gmdb_metadata/'
omim_input_dir = '../data/omim/20042022'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_file = os.path.join(output_dir, 'run.log')
logging.basicConfig(filename=log_file, format='%(asctime)s: %(name)s - %(message)s', datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
console_handle = logging.StreamHandler()
console_handle.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m-%d %H:%M')
console_handle.setFormatter(formatter)
logger.addHandler(console_handle)

input_gmdb_file = os.path.join(input_dir, 'image_metadata_v1.0.3.tsv')
morbid_file = os.path.join(omim_input_dir, 'morbidmap.txt')
ps_file = os.path.join(omim_input_dir, 'phenotypicSeries.txt')
title_file = os.path.join(omim_input_dir, 'mimTitles.txt')
genemap_file = os.path.join(omim_input_dir, 'genemap2.txt')

mapping = GMDB_mapping(morbid_file, ps_file, title_file, genemap_file, output_dir, date, logger)

class GMDB_mapping(object):
    def __init__(self, morbid_file, ps_file, title_file, genemap_file, output_dir, date, logger):

        self.morbid_df = pd.read_csv(morbid_file, skiprows=3, sep='\t')
        self.ps_df = pd.read_csv(ps_file, skiprows=2, sep='\t')
        self.title_df = pd.read_csv(title_file, skiprows=2, sep='\t')
        self.genemap_df = pd.read_csv(genemap_file, skiprows=3, sep='\t')

        self.gene2disorder = {}
        self.disorder2ps = {}
        self.disorder2gene = {}
        self.gene2geneid = {}
        self.ps2disorder = {}
        self.ps_dict = {}
        self.omim2title = {}

        self.init_gene_disorders()
        self.init_ps2disorder()
        self.init_title()
        self.init_gene_map_gene_id()

    def init_gene_map_gene_id(self):
        for index, row in self.genemap_df.iterrows():
            gene = row['Approved Gene Symbol']
            tmp_gene_id = row['Entrez Gene ID']
            if math.isnan(tmp_gene_id):
                gene_id = 'None'
            else:
                gene_id = str(int(tmp_gene_id))
            self.gene2geneid[gene] = gene_id

    def init_gene_disorders(self):
        for index, row in self.morbid_df.iterrows():
            tmp = row['# Phenotype'].split(', ')
            if '(3)' not in row['# Phenotype']:
                continue
            omim_id = tmp[-1].split(' ')[0]
            if not omim_id.isdigit():
                continue
            genes = row['Gene Symbols'].split(', ')

            for g in genes:
                if g not in self.gene2disorder:
                    self.gene2disorder[g] = []
                if int(omim_id) not in self.gene2disorder[g]:
                    self.gene2disorder[g].append(int(omim_id))
            self.disorder2gene[int(omim_id)] = genes[0]

    def get_min_ps(self, synd_id):
        min_ps = ''
        min_len = 10000
        for i in self.disorder2ps[synd_id]:
            if len(self.ps2disorder[i]) < min_len:
                min_len = len(self.ps2disorder[i])
                min_ps = i
        return min_ps

    def init_ps2disorder(self):
        for index, row in self.ps_df.iterrows():
            ps = row['# Phenotypic Series Number'][2:]
            omim_id = row['MIM Number']
            if omim_id.isdigit():
                # child
                omim_id = int(omim_id)
                if omim_id not in self.disorder2ps:
                    self.disorder2ps[omim_id] = []
                if int(ps) not in self.disorder2ps[omim_id]:
                    self.disorder2ps[omim_id].append(int(ps))
                if int(ps) not in self.ps2disorder:
                    self.ps2disorder[int(ps)] = []
                if int(omim_id) not in self.ps2disorder[int(ps)]:
                    self.ps2disorder[int(ps)].append(int(omim_id))
            else:
                # ps
                self.ps_dict[int(ps)] = omim_id
        self.ps_dict[176670] = 'Hutchinson-Gilford progeria'

    def init_title(self):
        for index, row in self.title_df.iterrows():
            omim_id = row['MIM Number']
            title = row['Preferred Title; symbol']
            self.omim2title[omim_id] = title

    def get_synd_name(self, synd_id):
        if synd_id in self.ps_dict:
            return self.ps_dict[synd_id]
        else:
            return self.omim2title[synd_id]

    def get_gene_id_by_genename(self, gene_name):
        if gene_name not in self.gene2geneid:
            return None
        return self.gene2geneid[gene_name]

    def get_disorder_ids_by_ps(self, ps):
        return self.ps2disorder[ps]

    def get_genename_by_disorder_id(self, disorder_id):
        if disorder_id not in self.disorder2gene:
            return None
        return self.disorder2gene[disorder_id]

    def get_genes_by_disorder_ids(self, omim_ids):
        genes = []
        # multi disorder
        # check whether disorder in PS
        tmp_disorder_ids = []
        for omim_id in omim_ids:
            if 'PS' in omim_id:
                found = False
                for omim_id_2 in omim_ids:
                    if omim_id_2 in self.get_disorder_ids_by_ps(int(omim_id[2:])):
                        found = True
                if not found:
                    tmp_disorder_ids.append(omim_id)
            else:
                tmp_disorder_ids.append(omim_id)

        for omim_id in tmp_disorder_ids:
            # single
            if 'PS' in omim_id:
                # PS
                disorder_ids = self.get_disorder_ids_by_ps(int(omim_id[2:]))
                for disorder_id in disorder_ids:
                    gene_name = self.get_genename_by_disorder_id(disorder_id)
                    gene_id = self.get_gene_id_by_genename(gene_name)
                    genes.append({'gene_name': gene_name,
                                  'gene_id': gene_id,
                                  'omim_id': disorder_id,
                                  'disorder_name': self.get_synd_name(disorder_id)})
            else:
                gene_name = self.get_genename_by_disorder_id(int(omim_id))
                gene_id = self.get_gene_id_by_genename(gene_name)
                genes.append({'gene_name': gene_name,
                              'gene_id': gene_id,
                              'omim_id': omim_id,
                              'disorder_name': self.get_synd_name(int(omim_id))})
        return genes
