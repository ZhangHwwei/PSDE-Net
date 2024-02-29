import torch.optim
from sklearn.metrics import accuracy_score

from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

from Other_Nets.CoANet.coanet import CoANet
from Other_Nets.Dlinknet import DinkNet34
from Other_Nets.RCFSNet import RCFSNet
from Other_Nets.UNet import UNet
from Other_Nets.deeplabv3plus import DeepLabv3_plus
from Our_method.PSDE_Net import BaseLine, BaseLine_PSC, BaseLine_PSC_RCM

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from utils import *
import cv2


def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)

    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    recall = recall_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    precision = precision_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    acc = accuracy_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    # Calculate F1 score
    if precision + recall == 0:
        f1 = 0  # Handle division by zero
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    if config.task_name is "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save,(448,448))
        predict_save = cv2.resize(predict_save,(2000,2000))
        cv2.imwrite(save_path,predict_save * 255)
    else:
        cv2.imwrite(save_path,predict_save * 255)
    return precision, recall, f1, iou_pred, acc


def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path):
    model.eval()

    output = model(input_img.cuda())
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    precision_tmp, recall_tmp, f1_tmp, iou_pred_tmp, acc_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
    return precision_tmp, recall_tmp, f1_tmp, iou_pred_tmp, acc_tmp


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    test_session = config.test_session

    if config.task_name is 'DeepGlobe':
        test_num = 1530
        model_type = config.model_name
        print(model_type)
        model_path = "./DeepGlobe/"+model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name is "CHN6-CUG":
        test_num = 448
        model_type = config.model_name
        model_path = "./CHN6-CUG/"+model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    save_path  = config.task_name +'/'+ model_type +'/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

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

    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    precision_pred = 0.0
    recall_pred = 0.0
    f1_pred = 0.0
    acc_pred = 0.0
    iou_pred = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path+str(i)+"_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            precision_t, recall_t, f1_t, \
                iou_pred_t, acc_t = vis_and_save_heatmap(model, input_img, None, lab, vis_path+str(i))

            print(iou_pred_t)
            precision_pred += precision_t
            recall_pred += recall_t
            f1_pred += f1_t
            acc_pred += acc_t
            iou_pred += iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print("precision_pred", precision_pred/test_num)
    print("recall_pred", recall_pred / test_num)
    print("f1_pred", f1_pred / test_num)
    print("acc_pred", acc_pred / test_num)
    print("iou_pred", iou_pred/test_num)




