import sys
from tqdm import tqdm
import os

code_path = '/home/user/Code/DePro'  # e.g. '/home/username/ProS'
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "src"))
sys.path.append(os.path.join(code_path, "clip"))

from DePro import depro
import torch
import math
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data.DomainNet import domainnet
from src.data.Sketchy import sketchy_extended
from src.data.TUBerlin import tuberlin_extended
import numpy as np
import torch.backends.cudnn as cudnn
from src.data.dataloaders import CuMixloader, BaselineDataset
from src.data.sampler import BalancedSampler, MoreBalancedSampler, MoreMoreBalancedSampler
from src.utils import utils, GPUmanager
from src.utils.logger import AverageMeter
from src.utils.metrics import compute_retrieval_metrics
from PIL import Image
from src.losses.sup_con_loss import soft_sup_con_loss, triplet_loss, pairwise_matching_loss, \
    domain_aware_triplet_loss_v1, domain_aware_triplet_loss_v2
from torch import optim, nn
from src.utils.logging import Logger

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
gm = GPUmanager.GPUManager()
gpu_index = gm.auto_choice()

device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.triplet = args.triplet
        self.log_name = args.log_name
        print('\nLoading data...')
        data_input = None
        if args.dataset == 'Sketchy':
            data_input = sketchy_extended.create_trvalte_splits(args)
        if args.dataset == 'DomainNet':
            data_input = domainnet.create_trvalte_splits(args)
        if args.dataset == 'TUBerlin':
            data_input = tuberlin_extended.create_trvalte_splits(args)

        self.tr_classes = data_input['tr_classes']
        self.va_classes = data_input['va_classes']
        self.te_classes = data_input['te_classes']
        self.data_splits = data_input['splits']

        set_seed(args)

        # Imagenet standards
        im_mean = [0.485, 0.456, 0.406]
        im_std = [0.229, 0.224, 0.225]
        # Image transformations
        self.image_transforms = {
            'train':
                transforms.Compose([
                    transforms.RandomResizedCrop((args.image_size, args.image_size), (0.8, 1.0)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(im_mean, im_std)
                ]),
            'eval': transforms.Compose([
                transforms.Resize(args.image_size, interpolation=BICUBIC),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                # lambda image: image.convert("RGB"),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        }

        # class dictionary
        self.dict_clss = utils.create_dict_texts(self.tr_classes)  # 生成 类:index 的一个字典

        # print(self.dict_clss). word to one-hot, not vector
        self.te_dict_class = utils.create_dict_texts(self.tr_classes + self.va_classes + self.te_classes)

        fls_tr = self.data_splits['tr']
        cls_tr = np.array([f.split('/')[-2] for f in fls_tr])
        dom_tr = np.array([f.split('/')[-3] for f in fls_tr])
        tr_domains_unique = np.unique(dom_tr)

        # doamin dictionary
        self.dict_doms = utils.create_dict_texts(tr_domains_unique)

        domain_ids = utils.numeric_classes(dom_tr, self.dict_doms)
        data_train = CuMixloader(fls_tr, cls_tr, dom_tr, self.dict_doms, transforms=self.image_transforms['train'])

        if self.triplet:
            cls_ids = utils.numeric_classes(cls_tr, self.dict_clss)
            train_sampler = MoreMoreBalancedSampler(domain_ids, cls_ids,
                                                    domains_per_batch=len(tr_domains_unique))  # 每个batch的采样都来自同一个domain
        else:
            train_sampler = BalancedSampler(domain_ids, args.batch_size // len(tr_domains_unique),
                                             domains_per_batch=len(tr_domains_unique))  # 每个batch的采样都来自同一个domain
        self.train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, sampler=train_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)
        self.train_loader_for_SP = DataLoader(dataset=data_train, batch_size=400, sampler=train_sampler,
                                              num_workers=args.num_workers,
                                              pin_memory=True)
        data_va_query = BaselineDataset(self.data_splits['query_va'], transforms=self.image_transforms['eval'])
        data_va_gallery = BaselineDataset(self.data_splits['gallery_va'], transforms=self.image_transforms['eval'])

        # PyTorch valid loader for query
        self.va_loader_query = DataLoader(dataset=data_va_query, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers,
                                          pin_memory=True)
        # PyTorch valid loader for gallery
        self.va_loader_gallery = DataLoader(dataset=data_va_gallery, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=True)

        print(
            f'#Tr samples:{len(data_train)}; #Val queries:{len(data_va_query)}; #Val gallery samples:{len(data_va_gallery)}.\n')
        print('Loading Done\n')

        self.model = depro(self.args, self.dict_clss, self.dict_doms, device)
        self.model = self.model.to(device)

        if args.dataset == 'DomainNet':
            self.save_folder_name = 'seen-' + args.seen_domain + '_unseen-' + args.holdout_domain + '_x_' + args.gallery_domain
            if not args.include_auxillary_domains:
                self.save_folder_name += '_noaux'
        elif args.dataset == 'Sketchy':
            if args.is_eccv_split:
                self.save_folder_name = 'eccv_split'
            else:
                self.save_folder_name = 'random_split'
        else:
            self.save_folder_name = ''

        if args.dataset == 'DomainNet' or (args.dataset == 'Sketchy' and args.is_eccv_split):
            self.map_metric = 'mAP@200'
            self.prec_metric = 'prec@200'
        else:
            self.map_metric = 'mAP@all'
            self.prec_metric = 'prec@100'

        self.suffix = '-e-' + str(args.epochs) + '_es-' + str(args.early_stop) + '_opt-' + args.optimizer + \
                      '_bs-' + str(args.batch_size) + '_lr-' + str(args.lr)

        self.path_cp = os.path.join(args.code_path, "src/algos/depro/log",
                                    args.dataset,
                                    self.save_folder_name)
        log_file = os.path.join(self.path_cp, args.log_name + ".txt")

        sys.stdout = Logger(log_file)

        self.start_epoch = 0
        self.best_map = 0
        self.early_stop_counter = 0
        self.last_ckpt_name = 'init'

        print("================Parameters Settings=================")
        print('Parameters:\t' + str(self.args))
        print("----------------Training Settings-------------------")
        print(f"lr = {self.args.lr}")
        print(f"batch_size = {self.args.batch_size}")

        print("---------------- Universal Domain Prompts Settings ----------------")
        print(f"visual UDP number = {self.args.vptNumTokens}")
        print(f"text prompt-setup = {self.args.text}")
        print(f"text UDP number= {self.args.textNumTokens}")
        print(f"UDPs independent? = {self.args.VL_independent}")
        print("---------------- Class Prompts Settings ----------------")
        print(f"Meta-Net used? = {self.args.generator_layer > 0}")
        print(f"Meta-Net layer depth = {self.args.generator_layer}")
        print("---------------- Decouple loss Settings ----------------")
        print(f"Decouple used? = {self.args.decouple}")
        # The above is the all basic Depro settings.
        # To conduct a more fair comparison with ProS, the below, should be set to False.
        # The only difference is that, during the test, we use the normed feature for retrieval, While Depro
        # uses the un-normed feature for retrieval. one can experimentally verify that using normalized features in
        # Depro does not significantly improve the performance.

        print("---------------- Trick&Loss Settings ----------------")
        print(f"LN Trick used? = {self.args.ln_trick}")
        if self.args.ln_trick:
            print("vit out_order = 0")  # NOTE That in ProS, it was set to 1 (I feel weired also).
            # so for a fair comparison with ProS above (without ln trick), we set the out_order=1

        print(f"use triplet loss? = {self.args.triplet}")
        print("==================================================")

    def training_set(self):
        tot = 0
        for name, param in self.model.named_parameters():
            tot += param.numel()
        lr = self.args.lr
        print("======== Training Parameters ========")
        train_parameters = ['text_prompt_learner.ctx', 'text_encoder.text_projection',
                            'visual_encoder.proj', 'visual_encoder.class_embedding',
                            'visual_encoder.prompt_embeddings', 'visual_encoder.meta_net', 'visual_udp_generator']  # 36180992

        train_part = 0
        for name, param in self.model.named_parameters():
            for str in train_parameters:
                flag = 0
                if self.args.ln_trick and "ln" in name:
                    param.requires_grad_(True)
                    print("ln: ", name)
                    train_part += param.numel()
                    flag = 1
                    break
                if name.startswith(str) == True:
                    param.requires_grad_(True)
                    print(name)
                    train_part += param.numel()
                    flag = 1
                    break
            if flag == 0:
                param.requires_grad_(False)
        print(f"tot={tot}, train = {train_part}")
        # NOTE: only give prompt_learner to the optimizer
        if self.args.optimizer == 'sgd':
            custom_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                      weight_decay=self.args.l2_reg, momentum=self.args.momentum, nesterov=False, lr=lr)
        else: # 'adam'
            custom_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                       betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.l2_reg)
        print("===============================================")
        return custom_optimizer

    def post_precess(self, result_unnorm, result_unnorm1, result_unnorm2, result_norm, result_norm1, result_norm2):
        map_unnorm, map_unnorm1, map_unnorm2 = result_unnorm[self.map_metric], \
                                               result_unnorm1[self.map_metric], result_unnorm2[self.map_metric]
        map_norm, map_norm1, map_norm2 = result_norm[self.map_metric], \
                                         result_norm1[self.map_metric], result_norm2[self.map_metric]
        prec_unnorm, prec_unnorm1, prec_unnorm2 = result_unnorm[self.prec_metric], \
                                                  result_unnorm1[self.prec_metric], result_unnorm2[self.prec_metric]
        prec_norm, prec_norm1, prec_norm2 = result_norm[self.prec_metric], \
                                            result_norm1[self.prec_metric], result_norm2[self.prec_metric]
        print("un-norm situation:")
        print(f"learned: map: {map_unnorm}, prec: {prec_unnorm}")
        print(f"11fixed: map: {map_unnorm1}, prec: {prec_unnorm1}")
        print(f"combine: map: {map_unnorm2}, prec: {prec_unnorm2}")

        print("norm situation:")
        print(f"learned: map: {map_norm}, prec: {prec_norm}")
        print(f"11fixed: map: {map_norm1}, prec: {prec_norm1}")
        print(f"combine: map: {map_norm2}, prec: {prec_norm2}")

        return map_norm2, prec_norm2

    def do_training(self):
        optimizer = self.training_set()
        for current_epoch in range(self.start_epoch, self.args.epochs):
            start = time.time()
            self.adjust_learning_rate(optimizer, current_epoch)

            loss = self.do_epoch_with_triplet(2, optimizer, current_epoch)

            print(f"epoch = [{current_epoch + 1}/{self.args.epochs}]loss = {loss}")
            print('\n***Validation***')
            if self.args.dataset == 'DomainNet':
                print("udcdr == 0")
                domain = self.args.holdout_domain
                for includeSeenClassinTestGallery in [0, 1]:
                    test_head_str = 'Query:' + domain + '; Gallery:' + self.args.gallery_domain + '; Generalized:' + str(
                        includeSeenClassinTestGallery)
                    print(test_head_str)

                    splits_query = domainnet.trvalte_per_domain(self.args, domain, 0, self.tr_classes, self.va_classes,
                                                                self.te_classes)
                    splits_gallery = domainnet.trvalte_per_domain(self.args, self.args.gallery_domain,
                                                                  includeSeenClassinTestGallery, self.tr_classes,
                                                                  self.va_classes, self.te_classes)

                    data_te_query = BaselineDataset(np.array(splits_query['te']),
                                                    transforms=self.image_transforms['eval'])
                    data_te_gallery = BaselineDataset(np.array(splits_gallery['te']),
                                                      transforms=self.image_transforms['eval'])

                    # PyTorch test loader for query
                    te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 10,
                                                 shuffle=False,
                                                 num_workers=self.args.num_workers, pin_memory=True)
                    # PyTorch test loader for gallery
                    te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 10,
                                                   shuffle=False,
                                                   num_workers=self.args.num_workers, pin_memory=True)

                    result_unnorm, result_unnorm1, result_unnorm2, result_norm, result_norm1, result_norm2 \
                        = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms,
                                   4, self.args)
                    map_, prec_ = self.post_precess(result_unnorm, result_unnorm1, result_unnorm2, result_norm,
                                                    result_norm1, result_norm2)

                print("udcdr == 1")
                if self.args.holdout_domain == 'quickdraw':
                    p = 0.1
                else:
                    p = 0.25
                splits_query = domainnet.seen_cls_te_samples(self.args, self.args.holdout_domain, self.tr_classes, p)
                splits_gallery = domainnet.seen_cls_te_samples(self.args, self.args.gallery_domain, self.tr_classes, p)

                data_te_query = BaselineDataset(np.array(splits_query), transforms=self.image_transforms['eval'])
                data_te_gallery = BaselineDataset(np.array(splits_gallery), transforms=self.image_transforms['eval'])

                # PyTorch test loader for query
                te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 5, shuffle=False,
                                             num_workers=self.args.num_workers, pin_memory=True)
                # PyTorch test loader for gallery
                te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 5,
                                               shuffle=False,
                                               num_workers=self.args.num_workers, pin_memory=True)
                result_unnorm, result_unnorm1, result_unnorm2, result_norm, result_norm1, result_norm2 \
                    = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms, 4,
                               self.args)
                map_udcdr, prec_udcdr = self.post_precess(result_unnorm, result_unnorm1, result_unnorm2, result_norm,
                                                  result_norm1, result_norm2)

            else:
                data_te_query = BaselineDataset(self.data_splits['query_te'], transforms=self.image_transforms['eval'])
                data_te_gallery = BaselineDataset(self.data_splits['gallery_te'],
                                                  transforms=self.image_transforms['eval'])

                te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 5, shuffle=False,
                                             num_workers=self.args.num_workers, pin_memory=True)
                te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 5,
                                               shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

                print(
                    f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')

                result_unnorm, result_unnorm1, result_unnorm2, result_norm, result_norm1, result_norm2 \
                    = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms, 4,
                               self.args)
                map_, prec_ = self.post_precess(result_unnorm, result_unnorm1, result_unnorm2, result_norm,
                                                result_norm1,
                                                result_norm2)

            end = time.time()
            elapsed = end - start

            print(
                f"Epoch Time:{elapsed // 60:.0f}m{elapsed % 60:.0f}s lr:{utils.get_lr(optimizer):.7f} mAP:{map_:.4f} prec:{prec_:.4f}\n")

            if map_ > self.best_map:
                self.best_map = map_
                self.early_stop_counter = 0
                model_save_name = 'val_map-' + '{0:.4f}'.format(map_) + self.suffix
                utils.save_checkpoint({
                    'epoch': current_epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_map': self.best_map,
                }, directory=self.path_cp, save_name=model_save_name, last_chkpt=self.last_ckpt_name)
                self.last_ckpt_name = model_save_name

            else:
                self.early_stop_counter += 1
                if self.args.early_stop == self.early_stop_counter:
                    print(f"Validation Performance did not improve for {self.args.early_stop} epochs."
                          f"Early stopping by {self.args.epochs - current_epoch - 1} epochs.")
                    break
                print(f"Val mAP hasn't improved from {self.best_map:.4f} for {self.early_stop_counter} epoch(s)!\n")

            print('\n***Training and Validation complete***')

    def adjust_learning_rate(self, optimizer, current_epoch, min_lr=1e-6, ):
        lr = self.args.lr * math.pow(1e-3, float(current_epoch) / 20)
        lr = max(lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def resume_from_checkpoint(self, resume_dict):
        if resume_dict is not None:
            print('==> Resuming from checkpoint: ', resume_dict)
            model_path = os.path.join(self.path_cp, resume_dict + '.pth')
            checkpoint = torch.load(model_path, map_location=device)
            self.start_epoch = checkpoint['epoch'] + 1
            # self.last_chkpt_name = resume_dict
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # self.best_map = checkpoint['best_map']


    def do_epoch_with_triplet(self, stage, custom_optimizer, current_epoch):
        self.model.train()
        batch_time = AverageMeter()

        loss_clss = AverageMeter()
        loss_triplets = AverageMeter()
        loss_mses = AverageMeter()
        losss = AverageMeter()

        # Start counting time
        time_start = time.time()

        train_loader = self.train_loader
        correct = 0
        tot = 0
        for i, (im, cls, dom) in enumerate(train_loader):

            im = im.float().to(device, non_blocking=True)
            cls_numeric = torch.from_numpy(utils.numeric_classes(cls, self.dict_clss)).long().to(device)

            # cls_id = utils.numeric_classes(class_name, self.dict_clss)
            # dom_id = utils.numeric_classes(dom, self.dict_doms)
            dom_numeric = torch.from_numpy(utils.numeric_classes(dom, self.dict_doms)).long().to(device)

            custom_optimizer.zero_grad()

            features, soft_labels = self.model(im, dom, cls, stage)
            feature0, feature1, feature2 = features
            soft_label0, soft_label1 = soft_labels
            hard_labels = cls_numeric

            loss_cls, co = soft_sup_con_loss(feature2, soft_label1, hard_labels, device=device)

            loss_triplet, p1, p2 = domain_aware_triplet_loss_v2(feature2, hard_labels, dom_numeric)
            # loss_triplet, _ = triplet_loss(feature2, hard_labels)

            if self.args.decouple:
                loss_cls2, _ = soft_sup_con_loss(feature1, soft_label0, hard_labels, device=device)
                # loss_triplet2, _ = triplet_loss(feature1, hard_labels)
                loss_triplet2, p1, p2 = domain_aware_triplet_loss_v2(feature1, hard_labels, dom_numeric)
                loss_cls = loss_cls + loss_cls2
                loss_triplet = loss_triplet + loss_triplet2

            with torch.no_grad():
                mse = nn.MSELoss()
                loss_mse = mse(feature2, feature1.detach())

            correct += co
            tot += im.size(0)
            if self.triplet:
                loss = loss_cls + loss_triplet
            else:
                loss = loss_cls
            loss.backward()
            custom_optimizer.step()

            losss.update(loss.item(), im.size(0))
            loss_clss.update(loss_cls.item(), im.size(0))
            loss_triplets.update(loss_triplet.item(), im.size(0))
            loss_mses.update(loss_mse.item(), im.size(0))

            # time
            time_end = time.time()
            batch_time.update(time_end - time_start)
            time_start = time_end
            if (i + 1) % self.args.log_interval == 0:
                print('[Train] Epoch: [{0}/{1}][{2}/{3}]  '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'cls {net1.val:.4f} ({net1.avg:.4f})  '
                      'triplet {net2.val:.4f} ({net2.avg:.4f})  '
                      'mse {net3.val:.4f} ({net3.avg:.4f})  '
                      'tot {net4.val:.4f} ({net4.avg:.4f})  '
                      .format(current_epoch + 1, self.args.epochs, i + 1, len(train_loader), batch_time=batch_time,
                              net1=loss_clss, net2=loss_triplets, net3=loss_mses, net4=losss))
                if self.args.debug_mode == 1:
                    break
        return {'net': losss.avg, 'acc': correct / tot}


@torch.no_grad()
def evaluate(loader_sketch, loader_image, model: depro, dict_clss, dict_doms, stage, args):
    # loader_sketch 是 query
    # loader_image 是 gallery
    # Switch to test mode
    model.eval()

    sketchEmbeddings_fixed = list()
    sketchEmbeddings_learned = list()
    sketchLabels = list()

    for i, (sk, cls_sk, dom) in tqdm(enumerate(loader_sketch), desc='Extrac query feature', total=len(loader_sketch)):

        sk = sk.float().to(device)
        cls_id = utils.numeric_classes(cls_sk, dict_clss)
        dom_id = utils.numeric_classes(dom, dict_doms)

        with torch.no_grad():
            prompts = model.visual_udp_generator(sk)
            sk_em = model.visual_encoder(sk, prompts)  # batch, 512

        sk_em_learned = sk_em[2]
        sk_em_fixed = sk_em[1]
        # sk_em = model.visual_encoder(sk, visual_prompts, text_prompts) # dom_id, cls_id, stage

        sketchEmbeddings_learned.append(sk_em_learned)
        sketchEmbeddings_fixed.append(sk_em_fixed)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        sketchLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    sketchEmbeddings_fixed = torch.cat(sketchEmbeddings_fixed, 0)
    sketchEmbeddings_learned = torch.cat(sketchEmbeddings_learned, 0)
    sketchLabels = torch.cat(sketchLabels, 0)

    realEmbeddings_learned = list()
    realEmbeddings_fixed = list()
    realLabels = list()

    for i, (im, cls_im, dom) in tqdm(enumerate(loader_image), desc='Extrac gallery feature', total=len(loader_image)):

        im = im.float().to(device)
        cls_id = utils.numeric_classes(cls_im, dict_clss)
        dom_id = utils.numeric_classes(dom, dict_doms)
        # Clipart embedding into a semantic space
        with torch.no_grad():
            prompts = model.visual_udp_generator(im)
            im_em = model.visual_encoder(im,prompts)  # batch, 512
            # Accumulate sketch embedding
        im_em_learned = im_em[2]
        im_em_fixed = im_em[1]
        realEmbeddings_learned.append(im_em_learned)
        realEmbeddings_fixed.append(im_em_fixed)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        realLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    realEmbeddings_learned = torch.cat(realEmbeddings_learned, 0)
    realEmbeddings_fixed = torch.cat(realEmbeddings_fixed, 0)
    realLabels = torch.cat(realLabels, 0)

    print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings_learned.shape, realEmbeddings_learned.shape))
    print("computing unormed situation")
    eval_data_unnorm = compute_retrieval_metrics(sketchEmbeddings_learned, sketchLabels, realEmbeddings_learned,
                                                 realLabels)
    eval_data_unnorm1 = compute_retrieval_metrics(sketchEmbeddings_fixed, sketchLabels, realEmbeddings_fixed,
                                                  realLabels)
    sketch = torch.cat((sketchEmbeddings_learned, sketchEmbeddings_fixed), 1)
    real = torch.cat((realEmbeddings_learned, realEmbeddings_fixed), 1)
    eval_data_unnorm2 = compute_retrieval_metrics(sketch, sketchLabels, real, realLabels)
    print(eval_data_unnorm["mAP@200"], eval_data_unnorm["prec@200"])
    print(eval_data_unnorm1["mAP@200"], eval_data_unnorm1["prec@200"])
    print(eval_data_unnorm2["mAP@200"], eval_data_unnorm2["prec@200"])

    print("computing normed situation")

    sketchEmbeddings_learned = sketchEmbeddings_learned / sketchEmbeddings_learned.norm(dim=-1, keepdim=True)
    realEmbeddings_learned = realEmbeddings_learned / realEmbeddings_learned.norm(dim=-1, keepdim=True)
    sketchEmbeddings_fixed = sketchEmbeddings_fixed / sketchEmbeddings_fixed.norm(dim=-1, keepdim=True)
    realEmbeddings_fixed = realEmbeddings_fixed / realEmbeddings_fixed.norm(dim=-1, keepdim=True)
    eval_data_norm = compute_retrieval_metrics(sketchEmbeddings_learned, sketchLabels, realEmbeddings_learned,
                                               realLabels)
    eval_data_norm1 = compute_retrieval_metrics(sketchEmbeddings_fixed, sketchLabels, realEmbeddings_fixed, realLabels)
    sketch = torch.cat((sketchEmbeddings_learned, sketchEmbeddings_fixed), 1)
    real = torch.cat((realEmbeddings_learned, realEmbeddings_fixed), 1)
    eval_data_norm2 = compute_retrieval_metrics(sketch, sketchLabels, real, realLabels)
    print(eval_data_norm["mAP@200"], eval_data_norm["prec@200"])
    print(eval_data_norm1["mAP@200"], eval_data_norm1["prec@200"])
    print(eval_data_norm2["mAP@200"], eval_data_norm2["prec@200"])
    return eval_data_unnorm, eval_data_unnorm1, eval_data_unnorm2, eval_data_norm, eval_data_norm1, eval_data_norm2
