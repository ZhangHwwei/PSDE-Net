import torch.optim
import os
import time
from utils import *
import Config as config
import warnings
# from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, batch_time,
                  average_loss, average_time, iou, average_iou,
                  oa, average_oa, precision, average_precision,
                  recall, average_recall, f1, f1_recall, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'OA:{:.3f} '.format(oa)
    string += '(Avg {:.4f}) '.format(average_oa)
    string += 'Precision:{:.3f} '.format(precision)
    string += '(Avg {:.4f}) '.format(average_precision)
    string += 'Recall:{:.3f} '.format(recall)
    string += '(Avg {:.4f}) '.format(average_recall)
    string += 'F1:{:.3f} '.format(f1)
    string += '(Avg {:.4f}) '.format(f1_recall)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


##################################################################################
# =================================================================================
#          Train One Epoch
# =================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, logger):
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    time_sum, loss_sum = 0, 0
    f1_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    recall_sum, precision_sum = 0.0, 0.0
    # scaler = GradScaler()
    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images, masks = sampled_batch['image'], sampled_batch['label']
        images, masks = images.cuda(), masks.cuda()

        # ====================================================
        #             Compute loss
        # ====================================================
        # 在训练循环中使用autocast
        '''
        with autocast():
            preds = model(images)
            out_loss = criterion(preds, masks.float())  # Loss
        '''
        preds = model(images)
        out_loss = criterion(preds, masks.float())  # Loss

        if model.training:

            # scaler.scale(out_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        train_iou, train_precision, train_recall = iou_on_batch(masks, preds)
        # train_precision = precision_on_batch(masks, preds)
        # train_recall = recall_on_batch(masks, preds)
        train_oa = acc_on_batch(masks, preds)
        # Calculate F1 score
        if train_precision + train_recall == 0:
            train_f1 = 0  # Handle division by zero
        else:
            train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall)
        batch_time = time.time() - end

        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path + str(epoch) + '/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images, masks, preds, names, vis_path)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss.item()
        iou_sum += len(images) * train_iou
        acc_sum += len(images) * train_oa
        precision_sum += len(images) * train_precision
        recall_sum += len(images) * train_recall
        f1_sum += len(images) * train_f1

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size * (i - 1) + len(images))
            average_time = time_sum / (config.batch_size * (i - 1) + len(images))
            train_iou_average = iou_sum / (config.batch_size * (i - 1) + len(images))
            train_oa_average = acc_sum / (config.batch_size * (i - 1) + len(images))
            train_precision_average = precision_sum / (config.batch_size * (i - 1) + len(images))
            train_recall_average = recall_sum / (config.batch_size * (i - 1) + len(images))
            train_f1_average = f1_sum / (config.batch_size * (i - 1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            train_oa_average = acc_sum / (i * config.batch_size)
            train_precision_average = precision_sum / (i * config.batch_size)
            train_recall_average = recall_sum / (i * config.batch_size)
            train_f1_average = f1_sum / (i * config.batch_size)

        # 清空显存缓冲区tensor
        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_oa, train_oa_average, train_precision, train_precision_average,
                          train_recall, train_recall_average, train_f1, train_f1_average,
                          logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups), logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            writer.add_scalar(logging_mode + '_acc', train_oa, step)
            writer.add_scalar(logging_mode + '_iou', train_precision, step)
            writer.add_scalar(logging_mode + '_acc', train_recall, step)
            writer.add_scalar(logging_mode + '_acc', train_f1, step)
        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_oa_average, train_iou_average, \
        train_precision_average, train_recall_average, train_f1_average
