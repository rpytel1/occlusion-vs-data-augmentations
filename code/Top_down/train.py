from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# import _init_paths
from lib.config import cfg, update_config
from lib.core.loss import JointsMSELoss, ContrastiveJoinMSELoss
from lib.core.function import train, validate, train_contrastive

from lib.dataset.transformations.multi_keypoint_transformation import MultiKeypointTransformation
from lib.dataset.transformations.random_image_transform import RandomImageTransform
from lib.dataset.transformations.targeted_blurring import TargetedBlurring
from lib.dataset.transformations.targeted_cutout import Cutout
from lib.dataset.transformations.wrappers import ToTensorWrapper, NormalizeWrapper
from lib.dataset.transformations.color_jitter import ColorJitterWrapper
from lib.dataset.transformations.targeted_cutmix import TargetedCutmix

from lib.utils.utils import get_optimizer, save_checkpoint, create_logger, get_model_summary

import lib.dataset as dataset
import lib.models as models


def get_annotation_path(cfg):
    if cfg.DATASET.DATASET == "coco":
        prefix = 'person_keypoints' \
            if 'test' not in cfg.DATASET.TRAIN_SET else 'image_info'

        annotation = cfg.DATASET.TRAIN_SET.split("/")[0]

        return os.path.join(cfg.DATASET.ROOT, 'annotations', prefix + '_' + annotation + '.json')
    else:
        return os.path.join(cfg.DATASET.ROOT, 'annot', cfg.DATASET.TRAIN_SET + '.json')


def get_image_dir_path(cfg):
    if cfg.DATASET.DATASET == "coco":
        prefix = 'test2017' if 'test' in cfg.DATASET.TRAIN_SET else cfg.DATASET.TRAIN_SET
        prefix = 'train2017' if 'train' in cfg.DATASET.TRAIN_SET else prefix
        prefix = 'val2017' if 'val' in cfg.DATASET.TRAIN_SET else prefix

        return os.path.join(cfg.DATASET.ROOT, 'images', prefix)
    else:
        return os.path.join(cfg.DATASET.ROOT,'images')

def get_loss(cfg):
    if cfg.CONTRASTIVE.TRAINING:
        return ContrastiveJoinMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT,
                                      remove_anns=cfg.CONTRASTIVE.REMOVE_PARTS).cuda(), JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
    else:
        loss = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
        return loss, loss


def get_transforms(cfg):
    transforms_list = []
    if cfg.DATASET.AUGMENTATIONS.CUTMIX.USE:
        ann_path = get_annotation_path(cfg)
        image_dir = get_image_dir_path(cfg)
        transforms_list.append(
            RandomImageTransform(TargetedCutmix(image_dir, ann_path, remove_anns=cfg.DATASET.AUGMENTATIONS.REMOVE_ANNS,
                                                dataset=cfg.DATASET.DATASET),
                                 cfg.DATASET.AUGMENTATIONS.CUTMIX.PROB, "parts", dataset=cfg.DATASET.DATASET))
    if cfg.DATASET.AUGMENTATIONS.JITTERING.USE:
        transforms_list.append(RandomImageTransform(ColorJitterWrapper(), 0.5, "keypoint", dataset=cfg.DATASET.DATASET))
    if cfg.DATASET.AUGMENTATIONS.BLURRING.USE:
        transforms_list.append(
            RandomImageTransform(
                TargetedBlurring(dataset=cfg.DATASET.DATASET, remove_anns=cfg.DATASET.AUGMENTATIONS.REMOVE_ANNS),
                cfg.DATASET.AUGMENTATIONS.BLURRING.PROB,
                cfg.DATASET.AUGMENTATIONS.BLURRING.MODE, dataset=cfg.DATASET.DATASET))
    if cfg.DATASET.AUGMENTATIONS.CUTOUT.USE:
        transforms_list.append(
            RandomImageTransform(Cutout(dataset=cfg.DATASET.DATASET, remove_anns=cfg.DATASET.AUGMENTATIONS.REMOVE_ANNS),
                                 cfg.DATASET.AUGMENTATIONS.CUTOUT.PROB,
                                 cfg.DATASET.AUGMENTATIONS.CUTOUT.MODE, dataset=cfg.DATASET.DATASET))

    if cfg.DATASET.AUGMENTATIONS.MULTIKEYPOINTS.USE:
        multi_transforms = []
        if cfg.DATASET.AUGMENTATIONS.MULTIKEYPOINTS.CUTOUT:
            multi_transforms.append(Cutout(cfg.DATASET.DATASET))
        if cfg.DATASET.AUGMENTATIONS.MULTIKEYPOINTS.BLURRING:
            multi_transforms.append(TargetedBlurring(cfg.DATASET.DATASET))
        if len(multi_transforms) > 0:
            transforms_list.append(
                MultiKeypointTransformation(multi_transforms, cfg.DATASET.AUGMENTATIONS.MULTIKEYPOINTS.MAX_NUM,
                                            cfg.DATASET.AUGMENTATIONS.MULTIKEYPOINTS.PROB, dataset=cfg.DATASET.DATASET))
    transforms_list.append(ToTensorWrapper())
    transforms_list.append(NormalizeWrapper(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(transforms_list)


def get_val_transforms(cfg):
    transforms_list = []
    if cfg.BLURRING != '':
        transforms_list.append(TargetedBlurring(cfg.DATASET.DATASET, part=cfg.BLURRING, width=cfg.CROP_SIZE))
    if cfg.CUTOUT != '':
        transforms_list.append(Cutout(cfg.DATASET.DATASET, part=cfg.CUTOUT, mean_coloring=cfg.MEAN_COLORING, width=cfg.CROP_SIZE))
    transforms_list.append(ToTensorWrapper())
    transforms_list.append(NormalizeWrapper(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(transforms_list)


def get_train_dataset(cfg):
    tfms = get_transforms(cfg)
    if cfg.CONTRASTIVE.TRAINING:
        train_dataset = eval('dataset.contrastive_' + cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT,
                                                                           cfg.DATASET.TRAIN_SET, True, tfms)
    else:
        train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
                                                               tfms)

    return train_dataset


def get_valid_dataset(cfg):
    tfms = get_val_transforms(cfg)
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, tfms)

    return valid_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--testSet',
                        help='chosen test set',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--cutout',
                        help='If to apply cutout, which part should be cut (either keypoint names or larger parts like: head, left_arm, leg, hip.',
                        type=str,
                        default='')

    parser.add_argument('--meanColoring', action='store_const', const=True, default=False,
                        help='If using cutout, apply mean value instead of blackout')

    parser.add_argument('--blurring',
                        help='If to apply blurring on one of the parts. Again either specific keypoint or larger parts (same as in cutout)',
                        type=str,
                        default='')
    parser.add_argument('--forcedOcclusion',
                        help='For testing purpose to obtain metrics results with reduced number of keypoints',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, 'lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # print(isinstance(model, torch.nn.Module))
    # writer_dict['writer'].add_graph(model, (dump_input,))

    logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion_train, criterion_val = get_loss(cfg)

    # Data loading code
    train_dataset = get_train_dataset(cfg)
    valid_dataset = get_valid_dataset(cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
                                                        last_epoch=last_epoch)

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        if cfg.CONTRASTIVE.TRAINING:
            train_contrastive(cfg, train_loader, model, criterion_train, optimizer, epoch, final_output_dir,
                              writer_dict)
        else:
            train(cfg, train_loader, model, criterion_train, optimizer, epoch, final_output_dir, writer_dict)

        perf_indicator = validate(cfg, valid_loader, valid_dataset, model, criterion_val, final_output_dir, writer_dict)

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
