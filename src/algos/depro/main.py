import sys
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
code_path = '/home/user/Code/DePro_SIGIR'  # insert your code path here
data_path = '/data/UCDR/data'  # insert your data path here
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "src"))
sys.path.append(os.path.join(code_path, "clip"))


from trainer import Trainer
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='DePro')

        # below are the hyper-parameters to adjust
        # baseline setups
        parser.add_argument('-vptNumTokens', '--vptNumTokens', default=4, type=int,
                            help='length of vpt-tokens')
        parser.add_argument('-text', '--text', default="DCoOp", choices=['CoOp', 'DCoOp', 'None'],
                            help='text prompting methods:)')
        parser.add_argument('-textNumTokens', '--textNumTokens', default=1, type=int,
                            help='length of vpt-tokens')
        parser.add_argument('-VL_independent', '--VL_independent', action="store_false",
                            help="whether to use maple")
        # please note that all the above are UDPs

        #
        parser.add_argument('-generator_layer', '--generator_layer', default=0, type=int,
                            help="num of the generator's layer xd")

        parser.add_argument('-domain_specific_trick', '--domain_specific_trick', action="store_true",
                            help="whether to use trick")
        parser.add_argument('-decouple', '--decouple', action="store_true",
                            help="whether to use triplet loss")
        parser.add_argument('-triplet', '--triplet', action="store_true",
                            help="whether to use triplet loss")
        parser.add_argument('-ln_trick', '--ln_trick', action="store_true",
                            help="whether to use trick")

        # something else
        parser.add_argument('-disable_first_pre_ln', '--disable_first_pre_ln', action="store_true",
                            help="whether to disable the pre_ln in generator")
        parser.add_argument('-second_init', '--second_init', default="xavier", choices=['xavier', 'randn'],
                            help="initialization methods in clip-visual")
        parser.add_argument('-first_init', '--first_init', default="xavier", choices=['xavier', 'randn'],
                            help="initialization methods in generator")

        parser.add_argument('-log_name', '--log_name', type=str, default='log',
                            help='log name :)')
        # optimizer
        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
        parser.add_argument('-l2', '--l2_reg', default=0.0, type=float, help='L2 Weight Decay for optimizer')
        parser.add_argument('-e', '--epochs', type=int, default=1, metavar='N',
                            help='Number of epochs to train in stage 2')
        parser.add_argument('-lr', '--lr', type=float, default=0.0001, metavar='LR',
                            help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
        parser.add_argument('-resume', '--resume_dict', type=str, help='checkpoint file to resume training from')

        # data_root
        parser.add_argument('-code_path', '--code_path', default=code_path, type=str, help='code path of ProS')
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
        parser.add_argument('-sd', '--seen_domain', default='quickdraw',
                            choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-hd', '--holdout_domain', default='sketch',
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
        parser.add_argument('-log', '--log_interval', type=int, default=100, metavar='N',
                            help='How many batches to wait before logging training status')

        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()


def main(args):
    trainer = Trainer(args)
    trainer.do_training()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # Parse options
    args = Options().parse()

    print('Parameters:\t' + str(args))
    main(args)
