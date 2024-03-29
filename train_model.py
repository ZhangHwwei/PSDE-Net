import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch import nn
from torch.backends import cudnn
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import logging

from Other_Nets.CoANet.coanet import CoANet
from Other_Nets.Dlinknet import DinkNet34
from Other_Nets.RCFSNet import RCFSNet
from Other_Nets.UNet import UNet
from Other_Nets.deeplabv3plus import DeepLabv3_plus
from Our_method.PSDE_Net import BaseLine, BaseLine_PSC, BaseLine_PSC_RCM
from Train_one_epoch import train_one_epoch
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def worker_init_fn(worker_id):
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=False):
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    train_dataset = ImageToImage2D(config.train_dataset, train_tf, image_size=config.img_size)
    val_dataset = ImageToImage2D(config.val_dataset, val_tf, image_size=config.img_size)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=config.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            worker_init_fn=worker_init_fn,
                            num_workers=config.num_workers,
                            pin_memory=True)

    lr = config.learning_rate
    logger.info(model_type)

    if model_type == 'UNet':
        model = UNet(n_channels=3, n_classes=1)

    elif model_type == 'deeplabv3+':
        model = DeepLabv3_plus(nInputChannels=3, n_classes=1)

    elif model_type == 'Dlinknet34':
        model = DinkNet34(num_classes=1)

    elif model_type == 'RCFSNet':
        model = RCFSNet()

    elif model_type == 'CoANet':
        model = CoANet(backbone='resnet', output_stride=16)

    elif model_type == 'BaseLine':
        model = BaseLine(1)

    elif model_type == 'BaseLine_PSC':
        model = BaseLine_PSC(1)

    elif model_type == 'BaseLine_PSC_RCM':
        model = BaseLine_PSC_RCM(1)

    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
         print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
         model = nn.DataParallel(model)

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize

    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None

    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_iou = 0.0
    best_epoch = 1
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        logger.info('Training with num_workers : {}'.format(config.num_workers))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None,  logger)

        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_acc, val_iou, val_precision, val_recall, val_f1 = train_one_epoch(val_loader,
                                                                                            model, criterion,
                                                                                            optimizer, writer,
                                                                                            epoch, lr_scheduler,
                                                                                            logger)
        # =============================================================
        #       Save best model
        # =============================================================
        if val_iou > max_iou:
            if epoch + 1 > 5:
                logger.info(
                    '\t Saving best model, mean iou increased from: {:.4f} to {:.4f}'.format(max_iou, val_iou))
                max_iou = val_iou
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            logger.info('\t Mean iou:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_iou, max_iou, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=False)
