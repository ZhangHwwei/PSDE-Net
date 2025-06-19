import os
import torch
import time

# PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True  # whether use cosineLR or not
n_channels = 3
n_labels = 1
epochs = 150
img_size = 512
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 50

pretrain = False
# task_name = 'DeepGlobe'
task_name = 'CHN6-CUG'
learning_rate = 0.01
batch_size = 8
num_workers =4

# model_name = 'UNet'
# model_name = 'deeplabv3+'
# model_name = 'Dlinknet34'
# model_name = 'CoANet'
# model_name = 'RCFSNet'
# model_name = 'BaseLine'
# model_name = 'BaseLine_PSC'
model_name = 'BaseLine_PSC_RCM'
if task_name == 'CHN6-CUG':
    train_dataset = '../CHN6-CUG/' + '/train/'
    val_dataset = '../CHN6-CUG/' + '/test/'
    test_dataset = '../CHN6-CUG/' + '/test/'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'

# DistributedDataParallel
local_rank = os.getenv('LOCAL_RANK', -1)

# used in testing phase, copy the session name in training phase
test_session = "Test_session"
