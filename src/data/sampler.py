import numpy
import numpy as np
import random
import torch
from torch.utils.data import sampler
import time
from random import shuffle
import numpy as np
import os

# Here we define a Sampler that has all the samples of each batch from the same domain,
# same as before but not distributed

# gives equal number of samples per domain, ordered across domains
from tqdm import tqdm

from collections import defaultdict

class BalancedSampler(sampler.Sampler):

    def __init__(self, domain_ids, samples_per_domain, domains_per_batch=5, iters='min'):

        self.n_doms = domains_per_batch
        self.domain_ids = domain_ids
        random.seed(0)

        self.dict_domains = {}
        self.indeces = {}

        for i in range(self.n_doms):
            self.dict_domains[i] = []
            self.indeces[i] = 0

        self.dpb = domains_per_batch
        self.dbs = samples_per_domain
        self.bs = self.dpb * self.dbs

        for idx, d in enumerate(self.domain_ids):
            self.dict_domains[d].append(idx)

        min_dom = 10000000
        max_dom = 0

        for d in self.domain_ids:
            if len(self.dict_domains[d]) < min_dom:
                min_dom = len(self.dict_domains[d])  # 每个dom 照片数量的最小值
            if len(self.dict_domains[d]) > max_dom:  # 每个dom 照片数量的最大值
                max_dom = len(self.dict_domains[d])

        if iters == 'min':
            self.iters = min_dom // self.dbs  # samples per domain
        elif iters == 'max':
            self.iters = max_dom // self.dbs
        else:
            self.iters = int(iters)

        for idx in range(self.n_doms):
            random.shuffle(self.dict_domains[idx])

    def _sampling(self, d_idx, n):
        if self.indeces[d_idx] + n >= len(self.dict_domains[d_idx]):
            self.dict_domains[d_idx] += self.dict_domains[d_idx]
        self.indeces[d_idx] = self.indeces[d_idx] + n
        return self.dict_domains[d_idx][self.indeces[d_idx] - n:self.indeces[d_idx]]

    def _shuffle(self):
        sIdx = []
        for i in range(self.iters):
            for j in range(self.n_doms):
                sIdx += self._sampling(j, self.dbs)
        return np.array(sIdx)

    def __iter__(self):
        return iter(self._shuffle())

    def __len__(self):
        return self.iters * self.bs

class MoreBalancedSampler(sampler.Sampler):

    def __init__(self, domain_ids, cls_ids, clss_per_domain=3, domains_per_batch=5, samples_per_cls=4, iters='min'):

        self.n_doms = domains_per_batch

        self.domain_ids = domain_ids
        random.seed(0)

        nested_dict = lambda: defaultdict(list)
        self.dict_domains_clss_samples = defaultdict(nested_dict)  # [domain][cls]
        self.dict_domains_clss = defaultdict(list)
        self.dict_domains_samples = defaultdict(list)

        self.dpb = domains_per_batch
        self.cpd = clss_per_domain
        self.spc = samples_per_cls
        self.bs = self.dpb * self.cpd * self.spc

        for i in range(len(domain_ids)):
            domain_id = domain_ids[i]
            cls_id = cls_ids[i]
            self.dict_domains_clss_samples[domain_id][cls_id].append(i)
            self.dict_domains_samples[domain_id].append(i)

        for domain_id in range(self.n_doms):
            self.dict_domains_clss[domain_id] = list(self.dict_domains_clss_samples[domain_id].keys())
            random.shuffle(self.dict_domains_clss[domain_id])

        min_dom = 10000000
        max_dom = 0

        for d in self.domain_ids:
            if len(self.dict_domains_samples[d]) < min_dom:
                min_dom = len(self.dict_domains_samples[d])
            if len(self.dict_domains_samples[d]) > max_dom:
                max_dom = len(self.dict_domains_samples[d])
        # print(min_dom, max_dom)
        if iters == 'min':
            self.iters = min_dom // (self.cpd * self.spc)  # 图片数量/ samples_per_domain
        elif iters == 'max':
            self.iters = max_dom // (self.cpd * self.spc)
        else:
            self.iters = int(iters)

        for domain in self.dict_domains_clss_samples.keys():
            for cls in self.dict_domains_clss_samples[domain].keys():
                random.shuffle(self.dict_domains_clss_samples[domain][cls])

        for domain in self.dict_domains_clss.keys():
            random.shuffle(self.dict_domains_clss[domain])

        self.domain_cls_indices = {}

        nested_dict = lambda: defaultdict(int)
        self.domain_cls_sample_indices = defaultdict(nested_dict)

        for dom in range(self.n_doms):
            self.domain_cls_indices[dom] = 0  # 每个dom 到哪个cls了
            for cls in self.dict_domains_clss[dom]:
                self.domain_cls_sample_indices[dom][cls] = 0
                # 每个domain的每个cls到哪个sample了

    def _sampling(self, d_idx):
        return_list = []
        if self.domain_cls_indices[d_idx] + self.cpd > len(self.dict_domains_clss[d_idx]):
            self.dict_domains_clss[d_idx] += self.dict_domains_clss[d_idx]
        self.domain_cls_indices[d_idx] = self.domain_cls_indices[d_idx] + self.cpd

        cls_start = self.domain_cls_indices[d_idx] - self.cpd
        cls_end = self.domain_cls_indices[d_idx]
        for cls_idx in range(cls_start, cls_end):
            cls = self.dict_domains_clss[d_idx][cls_idx]
            if self.domain_cls_sample_indices[d_idx][cls] + self.spc > len(self.dict_domains_clss_samples[d_idx][cls]):
                self.dict_domains_clss_samples[d_idx][cls] += self.dict_domains_clss_samples[d_idx][cls]
            start = self.domain_cls_sample_indices[d_idx][cls]
            self.domain_cls_sample_indices[d_idx][cls] += self.spc
            end = start + self.spc
            return_list.extend(self.dict_domains_clss_samples[d_idx][cls][start:end])

        # if flag_cls == 1:
        #     self.domain_cls_indices[d_idx] = self.domain_cls_indices[d_idx] %

        return return_list

    def _shuffle(self):
        sIdx = []

        for i in tqdm(range(self.iters)):
            for j in range(self.n_doms):
                # 需要每个domain 有的类别
                sIdx += self._sampling(j)
        return numpy.array(sIdx)

    def __iter__(self):
        return iter(self._shuffle())

    def __len__(self):
        return self.iters * self.bs

class MoreMoreBalancedSampler(sampler.Sampler):

    def __init__(self, domain_ids, cls_ids, clss_per_domain=3, domains_per_batch=2, samples_per_cls=4, iters="min"):

        super().__init__(MoreMoreBalancedSampler)

        self.n_doms = domains_per_batch
        self.domain_ids = domain_ids
        random.seed(0)

        nested_dict = lambda: defaultdict(list)
        self.dict_domains_clss_samples = defaultdict(nested_dict)  # [domain][cls]
        self.dict_domains_clss = defaultdict(list)
        self.dict_domains_samples = defaultdict(list)

        self.dpb = domains_per_batch
        self.cpd = clss_per_domain
        self.spc = samples_per_cls
        self.bs = self.dpb * self.cpd * self.spc

        for i in range(len(domain_ids)):
            domain_id = domain_ids[i]
            cls_id = cls_ids[i]
            self.dict_domains_clss_samples[domain_id][cls_id].append(i)
            self.dict_domains_samples[domain_id].append(i)

        for domain_id in range(self.n_doms):
            self.dict_domains_clss[domain_id] = list(self.dict_domains_clss_samples[domain_id].keys())
            self.dict_domains_clss[domain_id].sort()
            # random.shuffle(self.dict_domains_clss[domain_id])

        min_dom = 10000000  # 哪个domain的图片最少
        max_dom = 0  # 哪个domain的图片最多

        for d in self.domain_ids:
            if len(self.dict_domains_samples[d]) < min_dom:
                min_dom = len(self.dict_domains_samples[d])
            if len(self.dict_domains_samples[d]) > max_dom:
                max_dom = len(self.dict_domains_samples[d])

        if iters == 'min':
            self.iters = min_dom // (self.cpd * self.spc)  # 图片数量/ samples_per_domain
        elif iters == 'max':
            self.iters = max_dom // (self.cpd * self.spc)
        else:
            self.iters = int(iters)

        for domain in self.dict_domains_clss_samples.keys():
            for cls in self.dict_domains_clss_samples[domain].keys():
                random.shuffle(self.dict_domains_clss_samples[domain][cls])

        # for domain in self.dict_domains_clss.keys():
        #     random.shuffle(self.dict_domains_clss[domain])

        self.domain_cls_indices = {}

        nested_dict = lambda: defaultdict(int)
        self.domain_cls_sample_indices = defaultdict(nested_dict)

        for dom in range(self.n_doms):
            self.domain_cls_indices[dom] = 0  # 每个dom 到哪个cls了
            for cls in self.dict_domains_clss[dom]:
                self.domain_cls_sample_indices[dom][cls] = 0
                # 每个domain的每个cls到哪个sample了

    def _sampling(self, d_idx):
        return_list = []
        if self.domain_cls_indices[d_idx] + self.cpd > len(self.dict_domains_clss[d_idx]):
            self.dict_domains_clss[d_idx] += self.dict_domains_clss[d_idx]
        self.domain_cls_indices[d_idx] = self.domain_cls_indices[d_idx] + self.cpd

        cls_start = self.domain_cls_indices[d_idx] - self.cpd
        cls_end = self.domain_cls_indices[d_idx]
        for cls_idx in range(cls_start, cls_end):
            cls = self.dict_domains_clss[d_idx][cls_idx]
            if self.domain_cls_sample_indices[d_idx][cls] + self.spc > len(self.dict_domains_clss_samples[d_idx][cls]):
                self.dict_domains_clss_samples[d_idx][cls] += self.dict_domains_clss_samples[d_idx][cls]
            start = self.domain_cls_sample_indices[d_idx][cls]
            self.domain_cls_sample_indices[d_idx][cls] += self.spc
            end = start + self.spc
            return_list.extend(self.dict_domains_clss_samples[d_idx][cls][start:end])

        # if flag_cls == 1:
        #     self.domain_cls_indices[d_idx] = self.domain_cls_indices[d_idx] %

        return return_list

    def _shuffle(self):
        sIdx = []

        for i in tqdm(range(self.iters)):
            for j in range(self.n_doms):
                # 需要每个domain 有的类别
                sIdx += self._sampling(j)
        return numpy.array(sIdx)

    def __iter__(self):
        return iter(self._shuffle())

    def __len__(self):
        return self.iters * self.bs


