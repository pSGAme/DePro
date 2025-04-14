"""
    Parse input arguments
"""
import argparse
import os


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='UCDR_MAPLE')

        parser.add_argument('-alpha', '--alpha', type=float, default=0.05, metavar='LR',
                            help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-log_name', '--log_name', type=str, default='log',
                            help='log name :)')

        parser.add_argument('--num_shots', default=2, type=int)
        parser.add_argument('--d', default=8, type=int)
        parser.add_argument('--t', default=1, type=int)
        parser.add_argument('--ln', default=1, type=int)
        parser.add_argument('-lr', '--lr', type=float, default=0.0002, metavar='LR',
                            help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-trick', '--trick', type=int, default=1,
                            help='Initial learning rate for optimizer & scheduler')

        # visual prompt
        parser.add_argument('-visual', '--visual', default="VPT", choices=['VPT', 'None'],
                            help='visual prompting methods:)')
        parser.add_argument('-vptNumTokens', '--vptNumTokens', default=4, type=int,
                            help='length of vpt-tokens')

        # text prompt
        parser.add_argument('-ctx_init', '--ctx_init', default='a photo of a',  # 'a photo of a'
                            help='ctx initialization :)')
        parser.add_argument('-text', '--text', default="CoOp", choices=['CoOp', 'DCoOP', 'None'],
                            help='text prompting methods:)')
        parser.add_argument('-textNumTokens', '--textNumTokens', default=4, type=int,
                            help='length of vpt-tokens')
        # maple prompt
        parser.add_argument('-maple', '--maple', choices=[0, 1], default=1, type=int,
                            help='whether to utilize maple')
        parser.add_argument('-maple_depth', '--maple_depth', default=1, type=int,
                            help='maple depth')
        parser.add_argument('-maple_length', '--maple_length', default=4, type=int,
                            help='prompt length')
        parser.add_argument('-start_layer', '--start_layer', default=1, type=int,
                            help='the start of mail layer')
        parser.add_argument('-end_layer', '--end_layer', default=12, type=int,
                            help='the end of mail layer')
        parser.add_argument('-ivlu_start_layer', '--ivlu_start_layer', default=1, type=int,
                            help='the start of ivlu layer')
        parser.add_argument('-ivlu_end_layer', '--ivlu_end_layer', default=12, type=int,
                            help='the end of ivlu layer')
        parser.add_argument('-ivlp', '--ivlp', choices=[0, 1], default=0, type=int,
                            help='whether to conduct independent prompts learning')
        #
        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
        parser.add_argument('-l2', '--l2_reg', default=5e-4, type=float, help='L2 Weight Decay for optimizer')
        parser.add_argument('-e', '--epochs', type=int, default=1, metavar='N',
                            help='Number of epochs to train')

        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
        parser.add_argument('-resume', '--resume_dict', type=str, help='checkpoint file to resume training from')

        # data_root

        code_path = '/home/user/Code/DePro'
        parser.add_argument('-code_path', '--code_path', default=code_path, type=str, help='code path of UCDR')
        data_path = '/data/UCDR/data'
        parser.add_argument('-dataset_path', '--dataset_path', default=data_path, type=str,
                            help='Path of three datasets')

        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['Sketchy', 'DomainNet', 'TUBerlin'])
        parser.add_argument('-eccv', '--is_eccv_split', choices=[0, 1], default=1, type=int, help='whether or not to use eccv18 split\
                            if dataset="Sketchy"')
        # CLIP
        parser.add_argument('-clip_bb', '--clip_backbone', type=str,
                            choices=['RN50x4', 'RN50x16', 'ViT-B/16', 'ViT-B/32'], default='ViT-B/32',
                            help='choose clip backbone')
        parser.add_argument('-CLS_NUM_TOKENS', '--CLS_NUM_TOKENS', default=300, type=int,
                            help='number of Semantic Prompt Units, usually equals to the number of classes')
        parser.add_argument('-DOM_NUM_TOKENS', '--DOM_NUM_TOKENS', default=5, type=int,
                            help='number of Domain Prompt Units, usually equals to the number of domains')

        parser.add_argument('-debug_mode', '--debug_mode', default=0, type=int, help='use debug model')
        parser.add_argument('-dropout', '--dropout', default=0.5, type=float, help='dropout rate')

        # DomainNet specific arguments
        parser.add_argument('-sd', '--seen_domain', default='painting',
                            choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-hd', '--holdout_domain', default='infograph',
                            choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-gd', '--gallery_domain', default='real',
                            choices=['clipart', 'infograph', 'photo', 'painting', 'real'])
        parser.add_argument('-aux', '--include_auxillary_domains', choices=[0, 1], default=1, type=int,
                            help='whether(1) or not(0) to include\domains other than seen domain and gallery')
        parser.add_argument('-udcdr', '--udcdr', choices=[0, 1], default=0, type=int,
                            help='whether or not to evaluate under udcdr protocol')

        # Size parameters
        parser.add_argument('-imsz', '--image_size', default=224, type=int,
                            help='Input size for query/gallery domain sample')

        # Model parameters
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-bs', '--batch_size', default=60, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=6, help='Number of workers in data loader')

        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=2, help='Early stopping epochs.')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=50, metavar='N',
                            help='How many batches to wait before logging training status')

        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
