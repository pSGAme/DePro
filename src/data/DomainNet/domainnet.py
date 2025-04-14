import os
import numpy as np
import glob


def create_trvalte_splits(args):
    _BASE_PATH = args.dataset_path
    tr_classes = np.load(os.path.join(_BASE_PATH, 'DomainNet', 'train_classes.npy')).tolist() # 245
    va_classes = np.load(os.path.join(_BASE_PATH, 'DomainNet', 'val_classes.npy')).tolist() # 55
    te_classes = np.load(os.path.join(_BASE_PATH, 'DomainNet', 'test_classes.npy')).tolist() # 45
    #print(te_classes) # ["class1", "class2", ... "class_n"]
    semantic_vec = np.load(os.path.join(_BASE_PATH, 'DomainNet', 'w2v_domainnet.npy'), allow_pickle=True, encoding='latin1').item()
    #print(semantic_vec) # a dict, an english word to a vector
    tr_classes += va_classes # 300
    """
    得到所有域在数据集中的路径，包括：
    ./datasets/DomainNet/clipart
    ./datasets/DomainNet/infograph
    ./datasets/DomainNet/painting
    ./datasets/DomainNet/quickdraw
    ./datasets/DomainNet/real
    ./datasets/DomainNet/sketch
    """
    all_domains = []
    for f in os.listdir(os.path.join(args.dataset_path, "DomainNet")):
        if os.path.isdir(os.path.join(args.dataset_path, "DomainNet", f)):
            all_domains.append(f)

    print(all_domains) # ['clipart', 'infograph', 'quickdraw', 'sketch', 'real', 'painting']
    unseen_domain = args.holdout_domain # painting
    query_domains_to_train = [args.seen_domain] # 用于训练的域 infograph
    aux_domains = np.setdiff1d(all_domains, [unseen_domain, args.seen_domain, args.gallery_domain]).tolist() # 筛选出在all_domains中，不在后面这个list中的domain

    if args.include_auxillary_domains:
        query_domains_to_train += aux_domains  # 除了gallery_domain (real) 之外的所有训练集
    print('\nSeen:{}; Unseen:{}; Gallery:{}; Auxillary:{}.'.format(args.seen_domain, unseen_domain, args.gallery_domain, aux_domains))
    # Seen:infograph; Unseen:painting; Gallery:real; Auxillary:['clipart', 'quickdraw', 'sketch'].
    print(f'train_domain:{query_domains_to_train}')
    # train_domain:['infograph', 'clipart', 'quickdraw', 'sketch']

    # split gallery according to classes
    # 1 means 0.92 part of the train data (which will be deleted during the training)is incorporated to test
    splits_gallery = trvalte_per_domain(args, args.gallery_domain, 1, tr_classes, va_classes, te_classes)
    print('{} Seen Test:{}; Unseen Test:{}'.format(args.gallery_domain, len(splits_gallery['te_seen_cls']), len(splits_gallery['te_unseen_cls'])))

    fls_train = splits_gallery['tr']
    # print(len(fls_train)) # 138727
    for dom in query_domains_to_train:
        for cl in tr_classes:
            domain_cl_path = os.path.join(args.dataset_path, "DomainNet", dom, cl)
            fls_train += glob.glob(os.path.join(domain_cl_path, '*.*'))
    print(len(fls_train)) #438670
    splits = {}
    splits['tr'] = np.array(fls_train)

    # the following four lines seems useless
    splits_query_te = trvalte_per_domain(args, args.holdout_domain, 1, tr_classes, va_classes, te_classes)
    splits_gallery_te = trvalte_per_domain(args, args.gallery_domain, 1, tr_classes, va_classes, te_classes)
    splits['query_va'] = np.array(splits_query_te['te'])
    splits['gallery_va'] = np.array(splits_gallery['te'])

    splits_query_te = trvalte_per_domain(args, args.holdout_domain, 0, tr_classes, va_classes, te_classes)
    splits_gallery_te = trvalte_per_domain(args, args.gallery_domain, 0, tr_classes, va_classes, te_classes)

    splits["query_te"] = np.array(splits_query_te['te'])
    splits["gallery_te"] = np.array(splits_gallery_te['te'])

    print('\n# Classes - Tr:{}; Va:{}; Te:{}'.format(len(tr_classes), len(va_classes), len(te_classes)))


    return {'tr_classes':tr_classes, 'va_classes':va_classes, 'te_classes':te_classes, 'semantic_vec':semantic_vec, 'splits':splits}


def trvalte_per_domain(args, domain, gzs, tr_classes, va_classes, te_classes):
    # Split the data in the specified domain according to the defined training, validation, and test classes.
    # Return numpy arrays for the training, validation, and test sets.
    domain_path = os.path.join(args.dataset_path, "DomainNet", domain)  # 這裏本來也有一個domian

    all_fls = np.array(glob.glob(os.path.join(domain_path, '*/*.*'))) # Get all images under the specified domain
    # all paths
    all_clss = np.array([f.split('/')[-2] for f in all_fls]) # Get the classes of all images in this domain
    # all names in english
    fls_tr = []
    fls_va = []
    fls_te_unseen_cls = []
    fls_te_seen_cls = []

    for c in te_classes:
        fls_te_unseen_cls += all_fls[np.where(all_clss==c)[0]].tolist()

    # fls, image_path
    # cls, image_class_name
    for i, c in enumerate(tr_classes):

        sample_c = all_fls[np.where(all_clss==c)[0]]

        if gzs:
            np.random.seed(i)
            tr_samples = np.random.choice(sample_c, int(0.92*len(sample_c)), replace=False)
            te_seen_cls_samples = np.setdiff1d(sample_c, tr_samples)
            fls_te_seen_cls += te_seen_cls_samples.tolist()
        else:
            tr_samples = sample_c
        
        fls_tr += tr_samples.tolist()
    for c in va_classes:
        fls_va += all_fls[np.where(all_clss==c)[0]].tolist()

    splits = {}
    splits['tr'] = fls_tr
    splits['va'] = fls_va
    splits['te_seen_cls'] = fls_te_seen_cls
    splits['te_unseen_cls'] = fls_te_unseen_cls
    splits['te'] = fls_te_seen_cls + fls_te_unseen_cls

    return splits


def seen_cls_te_samples(args, domain, tr_classes, pc_per_cls=0.1):

    domain_path = os.path.join(args.dataset_path, "DomainNet", domain)

    all_fls = np.array(glob.glob(os.path.join(domain_path, '*/*.*')))
    all_clss = np.array([f.split('/')[-2] for f in all_fls])

    fls_te_seen_cls = []
    tr_classes = tr_classes[:45]
    for i, c in enumerate(tr_classes):

        sample_c = all_fls[np.where(all_clss==c)[0]]

        np.random.seed(i)
        te_seen_cls_samples = np.random.choice(sample_c, int(pc_per_cls*len(sample_c)), replace=False)
        fls_te_seen_cls += te_seen_cls_samples.tolist()

    return fls_te_seen_cls

