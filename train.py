'''
@saulzhang
The implementation code of the paper "Predicting Future Frames using Retrospective Cycle GAN" with Pytorch
data: Nov,12,2019
'''
import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import metric
import KL_flow as KL
import cv2
import argparse
import lpips.lpips as lpips
from pytorch_msssim import ssim,ms_ssim,SSIM,MS_SSIM
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from models import *
from datasets import *
from utils import *
from laplacian_of_guassian import *
from flownet2.models import FlowNet2C  # the path is depended on where you create this module
from flownet2.utils.frame_utils import read_gen  # the path is depended on where you create this module
import time
from tqdm import tqdm
import scipy
from depthfm import DepthFM

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=9, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=15, help="number of epochs of training")
parser.add_argument("--train_data", type=str, default="/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/dataset/pickle_data/train_data.pkl", help="the path of pickle file about train data")
parser.add_argument("--val_data", type=str, default="/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/dataset/pickle_data/val_data.pkl", help="the path of pickle file about validation data")
parser.add_argument("--test_data", type=str, default="/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/dataset/pickle_data/test_data.pkl", help="the path of pickle file about testing data")
parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=0, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height") #当图片的宽和高设置为256的时候会导致内存溢出
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=2000, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_LoG", type=float, default=0.005, help="cycle loss weight")
parser.add_argument("--lambda_frame_GAN", type=float, default=0.003, help="identity loss weight")
parser.add_argument("--lambda_seq_GAN", type=float, default=0.003, help="identity loss weight")
parser.add_argument("--sequence_len", type=float, default=5, help="the original length of frame sequence(n+1)")
parser.add_argument("--save_model_path", type=str, default="saved_models/cal_dual/", help="the path of saving the models")
parser.add_argument("--save_image_path", type=str, default="saved_images/cal_dual/", help="the path of saving the images")
parser.add_argument("--log_file", type=str, default="log_cal_dual.txt", help="the logging info of training")
parser.add_argument("--ckpt", type=str, default="checkpoints/depthfm-v1.ckpt",
                        help="Path to the model checkpoint")
parser.add_argument("--dtype", type=str, choices=["fp32", "bf16", "fp16"], default="fp16", 
                        help="Run with specific precision. Speeds up inference with subtle loss")
opt = parser.parse_args()##返回一个命名空间,如果想要使用变量,可用args.attr
print(opt)

# Create sample and checkpoint directories
os.makedirs(opt.save_image_path, exist_ok=True)
os.makedirs(opt.save_model_path, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_Limage = torch.nn.L1Loss()
cuda = torch.cuda.is_available()

# Image transformations
transforms_ = [
    transforms.Resize((int(opt.img_height),int(opt.img_width)), Image.BICUBIC),
    transforms.ToTensor(),#Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
                          #Converts a PIL Image or numpy.ndarray (H x W x C) in the range
                          #[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),#[0,1] -> [-1,1]
#     transforms.Normalize([0.485, ], [0.229, ])
]

# Training data loader
dataloader = DataLoader(
    ImageTrainDataset(opt.train_data, transforms_=transforms_,nt=opt.sequence_len),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# val data loader
val_dataloader = DataLoader(
      ImageValDataset(opt.val_data, transforms_=transforms_,nt=opt.sequence_len),
      batch_size=1,
      shuffle=False,
      num_workers=opt.n_cpu,
  )

# test data loader
test_dataloader = DataLoader(
    ImageTestDataset(opt.test_data, transforms_=transforms_,nt=opt.sequence_len),
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,
)


input_shape = (opt.batch_size, opt.sequence_len, opt.channels, opt.img_height, opt.img_width)
G_future = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_past = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = DiscriminatorA(input_shape)
D_B = D_A
print('input_shape',input_shape)
D_F = D_A
#loss_fn = lpips.LPIPS(net='alex')
Laplacian = Laplacian()

if cuda:
    G_future = torch.nn.DataParallel(G_future, device_ids=range(torch.cuda.device_count()))
    G_future = G_future.cuda()
    G_past = torch.nn.DataParallel(G_past, device_ids=range(torch.cuda.device_count()))
    G_past = G_past.cuda()
    D_A = torch.nn.DataParallel(D_A, device_ids=range(torch.cuda.device_count()))
    D_A = D_A.cuda()
    D_B = torch.nn.DataParallel(D_B, device_ids=range(torch.cuda.device_count()))
    D_B = D_B.cuda()
    D_F = torch.nn.DataParallel(D_F, device_ids=range(torch.cuda.device_count()))
    D_F = D_F.cuda()
    criterion_GAN.cuda()
    criterion_Limage.cuda()
    Laplacian = Laplacian.cuda()
    #loss_fn = loss_fn.cuda()
if opt.epoch != 0:
    # Load pretrained models
    G_future.load_state_dict(torch.load(opt.save_model_path+"G_future_%d.pth" %  (opt.epoch)))
    G_past.load_state_dict(torch.load(opt.save_model_path+"G_past_%d.pth" %  (opt.epoch)))
    D_A.load_state_dict(torch.load(opt.save_model_path+"D_A_%d.pth" %  (opt.epoch)))
    D_B.load_state_dict(torch.load(opt.save_model_path+"D_B_%d.pth" %  (opt.epoch)))
    D_F.load_state_dict(torch.load(opt.save_model_path+"D_F_%d.pth" %  (opt.epoch)))
else:
    # Initialize weights
    G_future.apply(weights_init_normal)#apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上
    G_past.apply(weights_init_normal)#apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
    D_F.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_future.parameters(),G_past.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_F = torch.optim.Adam(D_F.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step# lr_lambda为操作学习率的函数
)

lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_F = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_F, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

#保存中间的训练结果
def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    imgs = imgs.type(Tensor)
    input_A = imgs[:,:-1,...]
    input_A = input_A.view((imgs.size(0),-1,)+imgs.size()[3:])
    G_future.eval()
    real_A = Variable(imgs[:,-1:,...])
    A1 = G_future(input_A)
    frames = torch.cat((imgs[0,:-1,],A1[0].unsqueeze(0),imgs[0]), 0)
    image_grid = make_grid(frames,nrow=opt.sequence_len,normalize=False)
    save_image(image_grid, opt.save_image_path+"fake_%s.png" % (batches_done), normalize=False)

def ReverseSeq(Seq):
    length = Seq.size(1)
    return torch.cat([Seq[:,i-2:i+1,...] for i in range(length-1,-1,-3)],1)

count = 0
#################加载Flownet2.0模型
parser0 = argparse.ArgumentParser()
parser0.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser0.add_argument("--rgb_max", type=float, default=255.)
args0 = parser0.parse_args([])
flownet = FlowNet2C(args0).cuda()
    # load the state_dict
dict = torch.load("./flownet2/FlowNet2-C_checkpoint.pth.tar")
flownet.load_state_dict(dict["state_dict"])
loss_fn = lpips.LPIPS(net='alex')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#################加载Flownet2.0模型

#################加载Depthnet模型

depth_model = DepthFM(opt.ckpt)
depth_model.cuda(device).eval()

def get_dtype_from_str(dtype_str):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]
	
dtype = get_dtype_from_str(opt.dtype)
depth_model.depth_model.dtype = dtype
    
#################加载Depthnet模型

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, frame_seq in tqdm(enumerate(dataloader)):

        count = count + 1
        frame_seq = frame_seq.type(Tensor)

        real_A = Variable(frame_seq[:,-1,...]) #[bs,1,c,h,w]
        input_A = Variable(frame_seq[:,:-1,...].view((frame_seq.size(0),-1)+frame_seq.size()[3:]))
        real_B = Variable(frame_seq[:,0,...]) #[bs,1,c,h,w]
        input_B_ = Variable(frame_seq[:,1:,...].view((frame_seq.size(0),-1)+frame_seq.size()[3:]))
        input_B = ReverseSeq(input_B_)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((frame_seq.size(0), *D_A.module.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((frame_seq.size(0), *D_A.module.output_shape))), requires_grad=False)

        #------------------------
        #  Train Generator
        #------------------------
        G_future.train()
        G_past.train()        
        optimizer_G.zero_grad()#梯度清零

        #L_Image loss which minimize the L1 Distance between the image pair
        A1 = G_future(input_A,depth_model) # x^'_{n}  generated future frame
        B1 = G_past(input_B,depth_model) # x^'_{m} generated past frame

        #############################################
        #                                           #
        #                                           #
        #        Optical flow Loss Function         #  
        #                                           # 
        #                                           # 
        #############################################
  
        optical_loss_0_A = []
        optical_loss_1_A = []
		
		depth_loss_0 = []
		depth_loss_1 = []
 
        optical_real_loss_0_A = []
        optical_real_loss_1_A = []
        optical_fake_loss_0_A = []
        optical_fake_loss_1_A = []
        for j in range(9):
                prev = A1[j].cpu().clone()
                curr = A1[j+1].cpu().clone()
                image_grid_prev = prev
                save_image(image_grid_prev, "/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/fake_jpg_A/fake_%s_prev.png" % (j), normalize=False)
                image_grid_curr = curr
                save_image(image_grid_curr, "/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/fake_jpg_A/fake_%s_curr.png" % (j), normalize=False)  
    
                prev_true = real_A[j].cpu().clone()
                curr_true = real_A[j+1].cpu().clone()
                image_grid_prev_true = prev_true
                save_image(image_grid_prev_true, "/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/true_jpg_A/true_%s_prev.png" % (j), normalize=False)
                image_grid_curr_true = curr_true
                save_image(image_grid_curr_true, "/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/true_jpg_A/true_%s_curr.png" % (j), normalize=False)               
                
        for j in range(9):
                pim1_fake = read_gen("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/fake_jpg_A/fake_%s_prev.png" % (j))
                pim2_fake = read_gen("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/fake_jpg_A/fake_%s_curr.png" % (j))
                images_fake = [pim1_fake, pim2_fake]
                images_fake = np.array(images_fake).transpose(3, 0, 1, 2)
                im_fake = torch.from_numpy(images_fake.astype(np.float32)).unsqueeze(0).cuda()
                result_fake = flownet(im_fake).squeeze()
                fake_0 = Image.fromarray(result_fake[0].cpu().detach().numpy())
                scipy.misc.imsave("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_fake_A/fake_%s_prev_0.png" % (j), fake_0)
                fake_1 = Image.fromarray(result_fake[1].cpu().detach().numpy())
                scipy.misc.imsave("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_fake_A/fake_%s_prev_1.png" % (j), fake_1)
                                            
                pim1_real = read_gen("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/true_jpg_A/true_%s_prev.png" % (j))
                pim2_real = read_gen("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/true_jpg_A/true_%s_curr.png" % (j))
                images_real = [pim1_real, pim2_real]
                images_real = np.array(images_real).transpose(3, 0, 1, 2)
                im_real = torch.from_numpy(images_real.astype(np.float32)).unsqueeze(0).cuda()
                result_real= flownet(im_real).squeeze()
                real_0 = Image.fromarray(result_real[0].cpu().detach().numpy())
                scipy.misc.imsave("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_true_A/true_%s_prev_0.png" % (j), real_0)
                real_1 = Image.fromarray(result_real[1].cpu().detach().numpy())
                scipy.misc.imsave("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_true_A/true_%s_prev_1.png" % (j), real_1)
                
        for j in range(9):             
                fake_0 = cv2.imread("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_fake_A/fake_%s_prev_0.png" % (j))
                real_0 = cv2.imread("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_true_A/true_%s_prev_0.png" % (j))
                
                fake_1 = cv2.imread("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_fake_A/fake_%s_prev_1.png" % (j))
                real_1 = cv2.imread("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_true_A/true_%s_prev_1.png" % (j))
                
				#########depth criterion
                fake_0_depth = model.predict_depth(fake_0, num_steps=2, ensemble_size=4)
                fake_0_depth = fake_0_depth.squeeze(0).squeeze(0).cpu().numpy()       # (h, w) in [0, 1]
				
                real_0_depth = model.predict_depth(real_0, num_steps=2, ensemble_size=4)
                real_0_depth = real_0_depth.squeeze(0).squeeze(0).cpu().numpy()       # (h, w) in [0, 1]
				
                fake_1_depth = model.predict_depth(fake_1, num_steps=2, ensemble_size=4)
                fake_1_depth = fake_1_depth.squeeze(0).squeeze(0).cpu().numpy()       # (h, w) in [0, 1]
				
                real_1_depth = model.predict_depth(real_1, num_steps=2, ensemble_size=4)
                real_1_depth = real_1_depth.squeeze(0).squeeze(0).cpu().numpy()       # (h, w) in [0, 1]
				
				loss_depth_0 = criterion_Limage(fake_0_depth,real_0_depth) 
				depth_loss_0.append(loss_depth_0)
				
				loss_depth_1 = criterion_Limage(fake_1_depth,real_1_depth) 
				depth_loss_1.append(loss_depth_1)
				#########depth criterion
                
                fake0 = torch.from_numpy(fake_0.astype(np.float32)).unsqueeze(0).cuda()
                real0 = torch.from_numpy(real_0.astype(np.float32)).unsqueeze(0).cuda()
                loss_opt_0 = criterion_Limage(fake0,real0) 
                optical_loss_0_A.append(loss_opt_0)
                
                fake1 = torch.from_numpy(fake_1.astype(np.float32)).cuda()
                real1 = torch.from_numpy(real_1.astype(np.float32)).cuda()
                loss_opt_1 = criterion_Limage(fake1,real1)
                optical_loss_1_A.append(loss_opt_1)
                
                loss_real_f_0 = criterion_GAN(D_F(real0), valid)
                loss_real_f_1 = criterion_GAN(D_F(real1), valid)
                # Fake loss
                loss_fake_f_0 = criterion_GAN(D_F(fake0), fake)#detach() 将Variable从计算图中抽离出来，进行梯度阶段。注意如果
                loss_fake_f_1 = criterion_GAN(D_F(fake1), fake)
              
                optical_real_loss_0_A.append(loss_real_f_0)      
                optical_real_loss_1_A.append(loss_real_f_1)
                optical_fake_loss_0_A.append(loss_fake_f_0)
                optical_fake_loss_1_A.append(loss_fake_f_1)
				
        optical_loss_total_A = (sum(depth_loss_1) / 9 + sum(depth_loss_0) / 9 + sum(optical_loss_0_A) / 9 + sum(optical_loss_1_A) / 9 + sum(optical_real_loss_0_A) / 9 + sum(optical_real_loss_1_A) / 9 + sum(optical_fake_loss_0_A) / 9 + sum(optical_fake_loss_1_A) / 9 )/8
		#####清空内存
		optical_loss_0_A = []
        optical_loss_1_A = []
		
		depth_loss_0 = []
		depth_loss_1 = []
 
        optical_real_loss_0_A = []
        optical_real_loss_1_A = []
        optical_fake_loss_0_A = []
        optical_fake_loss_1_A = []
		
		######################
        optical_loss_0_B = []
        optical_loss_1_B = []
 
        depth_loss_0 = []
		depth_loss_1 = []
 
        optical_real_loss_0_B = []
        optical_real_loss_1_B = []
        optical_fake_loss_0_B = []
        optical_fake_loss_1_B = []
        for j in range(9):
                prev = B1[j].cpu().clone()
                curr = B1[j+1].cpu().clone()
                image_grid_prev = prev
                save_image(image_grid_prev, "/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/fake_jpg_B/fake_%s_prev.png" % (j), normalize=False)
                image_grid_curr = curr
                save_image(image_grid_curr, "/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/fake_jpg_B/fake_%s_curr.png" % (j), normalize=False)  
    
                prev_true = real_B[j].cpu().clone()
                curr_true = real_B[j+1].cpu().clone()
                image_grid_prev_true = prev_true
                save_image(image_grid_prev_true, "/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/true_jpg_B/true_%s_prev.png" % (j), normalize=False)
                image_grid_curr_true = curr_true
                save_image(image_grid_curr_true, "/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/true_jpg_B/true_%s_curr.png" % (j), normalize=False)  
                
                
        for j in range(9):
                pim1_fake = read_gen("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/fake_jpg_B/fake_%s_prev.png" % (j))
                pim2_fake = read_gen("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/fake_jpg_B/fake_%s_curr.png" % (j))
                images_fake = [pim1_fake, pim2_fake]
                images_fake = np.array(images_fake).transpose(3, 0, 1, 2)
                im_fake = torch.from_numpy(images_fake.astype(np.float32)).unsqueeze(0).cuda()
                result_fake = flownet(im_fake).squeeze()
                fake_0 = Image.fromarray(result_fake[0].cpu().detach().numpy())
                scipy.misc.imsave("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_fake_B/fake_%s_prev_0.png" % (j), fake_0)
                fake_1 = Image.fromarray(result_fake[1].cpu().detach().numpy())
                scipy.misc.imsave("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_fake_B/fake_%s_prev_1.png" % (j), fake_1)
                                            
                pim1_real = read_gen("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/true_jpg_B/true_%s_prev.png" % (j))
                pim2_real = read_gen("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/true_jpg_B/true_%s_curr.png" % (j))
                images_real = [pim1_real, pim2_real]
                images_real = np.array(images_real).transpose(3, 0, 1, 2)
                im_real = torch.from_numpy(images_real.astype(np.float32)).unsqueeze(0).cuda()
                result_real= flownet(im_real).squeeze()
                real_0 = Image.fromarray(result_real[0].cpu().detach().numpy())
                scipy.misc.imsave("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_true_B/true_%s_prev_0.png" % (j), real_0)
                real_1 = Image.fromarray(result_real[1].cpu().detach().numpy())
                scipy.misc.imsave("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_true_B/true_%s_prev_1.png" % (j), real_1)
                
        for j in range(9):             
                fake_0 = cv2.imread("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_fake_B/fake_%s_prev_0.png" % (j))
                real_0 = cv2.imread("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_true_B/true_%s_prev_0.png" % (j))
                
                fake_1 = cv2.imread("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_fake_B/fake_%s_prev_1.png" % (j))
                real_1 = cv2.imread("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/opt_true_B/true_%s_prev_1.png" % (j))
                
				#########depth criterion
                fake_0_depth = model.predict_depth(fake_0, num_steps=2, ensemble_size=4)
                fake_0_depth = fake_0_depth.squeeze(0).squeeze(0).cpu().numpy()       # (h, w) in [0, 1]
				
                real_0_depth = model.predict_depth(real_0, num_steps=2, ensemble_size=4)
                real_0_depth = real_0_depth.squeeze(0).squeeze(0).cpu().numpy()       # (h, w) in [0, 1]
				
                fake_1_depth = model.predict_depth(fake_1, num_steps=2, ensemble_size=4)
                fake_1_depth = fake_1_depth.squeeze(0).squeeze(0).cpu().numpy()       # (h, w) in [0, 1]
				
                real_1_depth = model.predict_depth(real_1, num_steps=2, ensemble_size=4)
                real_1_depth = real_1_depth.squeeze(0).squeeze(0).cpu().numpy()       # (h, w) in [0, 1]
				
				
				loss_depth_0 = criterion_Limage(fake_0_depth,real_0_depth) 
				depth_loss_0.append(loss_depth_0)
				
				loss_depth_1 = criterion_Limage(fake_1_depth,real_1_depth) 
				depth_loss_1.append(loss_depth_1)
				#########depth criterion

                fake0 = torch.from_numpy(fake_0.astype(np.float32)).cuda()
                real0 = torch.from_numpy(real_0.astype(np.float32)).cuda()
                loss_opt_0 = criterion_Limage(fake0,real0) 
                optical_loss_0_B.append(loss_opt_0)
                
                fake1 = torch.from_numpy(fake_1.astype(np.float32)).cuda()
                real1 = torch.from_numpy(real_1.astype(np.float32)).cuda()
                loss_opt_1 = criterion_Limage(fake1,real1)
                optical_loss_1_B.append(loss_opt_1)
				loss_real_f_0 = criterion_GAN(D_F(real0), valid)
                loss_real_f_1 = criterion_GAN(D_F(real1), valid)
#               # Fake loss
                loss_fake_f_0 = criterion_GAN(D_F(fake0), fake)#detach() 将Variable从计算图中抽离出来，进行梯度阶段。注意如果
                loss_fake_f_1 = criterion_GAN(D_F(fake1), fake)
                
                optical_real_loss_0_B.append(loss_real_f_0)      
                optical_real_loss_1_B.append(loss_real_f_1)
                optical_fake_loss_0_B.append(loss_fake_f_0)
                optical_fake_loss_1_B.append(loss_fake_f_1)
        optical_loss_total_B = (sum(depth_loss_1) / 9 + sum(depth_loss_0) / 9 + sum(optical_loss_0_B) / 9 + sum(optical_loss_1_B) / 9+ sum(optical_real_loss_0_B) / 9 + sum(optical_real_loss_1_B) / 9 + sum(optical_fake_loss_0_B) / 9 + sum(optical_fake_loss_1_B) / 9 )/8
        ######################清空内存
        optical_loss_0_B = []
        optical_loss_1_B = []
 
        depth_loss_0 = []
		depth_loss_1 = []
 
        optical_real_loss_0_B = []
        optical_real_loss_1_B = []
        optical_fake_loss_0_B = []
        optical_fake_loss_1_B = []
        
        #############################################
        #                                           #
        #                                           #
        #        Optical flow Loss Function         #  
        #                                           # 
        #                                           # 
        #############################################
        input_A_A1_ = torch.cat((input_A[:,3:,...],A1),1)
        input_A_A1 = ReverseSeq(input_A_A1_)
        input_B_B1 = torch.cat((B1,input_A[:,3:,...]),1)
        A11 = G_future(input_B_B1)# x^''_{n}
        B11 = G_past(input_A_A1)

        loss_A_A1 = criterion_Limage(real_A,A1)
        loss_A_A11 = criterion_Limage(real_A,A11)
        loss_A1_A11 = criterion_Limage(A1,A11)
        loss_B_B1 = criterion_Limage(real_B,B1)
        loss_B_B11 = criterion_Limage(real_B,B11)
        loss_B1_B11 = criterion_Limage(B1,B11)
        
        loss_Image = (loss_A_A1 + loss_A_A11 + loss_A1_A11 + loss_B_B1 + loss_B_B11 + loss_B1_B11 ) / 6
        
        #L_LoG loss 
        L_LoG_A_A1 = criterion_Limage(Laplacian(real_A),Laplacian(A1))
        L_LoG_A_A11 = criterion_Limage(Laplacian(real_A),Laplacian(A11))
        L_LoG_A1_A11 = criterion_Limage(Laplacian(A1),Laplacian(A11))
        L_LoG_B_B1 = criterion_Limage(Laplacian(real_B),Laplacian(B1))
        L_LoG_B_B11 = criterion_Limage(Laplacian(real_B),Laplacian(B11))
        L_LoG_B1_B11 = criterion_Limage(Laplacian(B1),Laplacian(B11))

        loss_LoG = (L_LoG_A_A1 + L_LoG_A_A11 + L_LoG_A1_A11 + L_LoG_B_B1 + L_LoG_B_B11 + L_LoG_B1_B11) / 6

        #GAN frame Loss(Least Square Loss)
        loss_frame_GAN_A1  = criterion_GAN(D_A(A1),valid)# lead the synthetic frame become similiar to the real frame
        loss_frame_GAN_B1  = criterion_GAN(D_A(B1),valid)
        loss_frame_GAN_A11 = criterion_GAN(D_A(A11),valid)
        loss_frame_GAN_B11 = criterion_GAN(D_A(B11),valid)
        #Total frame loss
        loss_frame_GAN = (loss_frame_GAN_A1 + loss_frame_GAN_B1 + loss_frame_GAN_A11 + loss_frame_GAN_B11) / 4

        #GAN seq Loss 
        #four kinds of the synthetic frame sequence
        input_B1_A  = torch.cat((B1,input_A[:,3:,...],real_A),1)
        input_B11_A1 = torch.cat((B11,input_A[:,3:,...],A1),1)
        input_B_A1 = torch.cat((real_B,input_A[:,3:,...],A1),1)
        input_B1_A11 = torch.cat((B1,input_A[:,3:,...],A11),1)
        loss_seq_GAN_B1_A = criterion_GAN(D_B(input_B1_A),valid)
        loss_seq_GAN_B11_A1 = criterion_GAN(D_B(input_B11_A1),valid)
        loss_seq_GAN_B_A1 = criterion_GAN(D_B(input_B_A1),valid)
        loss_seq_GAN_B1_A11 = criterion_GAN(D_B(input_B1_A11),valid)
        # Total seq loss
        loss_seq_GAN = (loss_seq_GAN_B1_A + loss_seq_GAN_B11_A1 + loss_seq_GAN_B_A1 + loss_seq_GAN_B1_A11) / 4

        # Total GAN loss
        total_loss_GAN = loss_Image + optical_loss_total_A + optical_loss_total_B + opt.lambda_LoG*loss_LoG + opt.lambda_frame_GAN *loss_frame_GAN + opt.lambda_seq_GAN*loss_seq_GAN
        total_loss_GAN.backward() #反向传播，对各个变量求导
        optimizer_G.step() # 更新

        #------------------------
        #  Train Discriminator A
        #------------------------
        optimizer_D_A.zero_grad()
        # Real loss
        loss_real_A = criterion_GAN(D_A(real_A), valid)
        loss_real_B = criterion_GAN(D_A(real_B), valid)
        # Fake loss
        loss_fake_A1 = criterion_GAN(D_A(A1.detach()), fake)#detach() 将Variable从计算图中抽离出来，进行梯度阶段。注意如果
                                                            #这里没有加上detach()函数的话会导致梯度传到G，而G的计算图已经被释放会报错
        loss_fake_A11 = criterion_GAN(D_A(A11.detach()), fake)
        loss_fake_B1 = criterion_GAN(D_A(B1.detach()), fake)
        loss_fake_B11 = criterion_GAN(D_A(B11.detach()), fake)
        # Total loss
        loss_D_A = (loss_real_A + loss_real_B + loss_fake_A1 + loss_fake_A11 +loss_fake_B1 + loss_fake_B11 ) / 5
        loss_D_A.backward()#retain_graph=True 
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()
        #real loss
        loss_real = criterion_GAN(D_B(Variable(frame_seq.view((frame_seq.size(0),-1)+frame_seq.size()[3:]))), valid)
        #fake loss
        loss_fake_B1_A = criterion_GAN(D_B(input_B1_A.detach()),fake)
        loss_fake_B11_A1 = criterion_GAN(D_B(input_B11_A1.detach()),fake)
        loss_fake_B_A1 = criterion_GAN(D_B(input_B_A1.detach()),fake)
        loss_fake_B1_A11 = criterion_GAN(D_B(input_B1_A11.detach()),fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake_B1_A + loss_fake_B11_A1 + loss_fake_B_A1 + loss_fake_B1_A11 ) / 5
        loss_D_B.backward()
        optimizer_D_B.step()
        total_loss_D = (loss_D_A + loss_D_B)/2
        # --------------
        #  Log Progress
        # --------------
        batches_done = epoch*len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        if count % 100 == 0:
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, img: %f, LoG: %f, adv_frame: %f, adv_seq: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    total_loss_D.item(),
                    total_loss_GAN.item(),
                    loss_Image.item(),
                    loss_LoG.item(), 
                    loss_frame_GAN.item(),
                    loss_seq_GAN.item(),
                    time_left,
                )
            )
            
        if count > opt.sample_interval/opt.batch_size:
            #sample_images(batches_done)
            count = 0
        time.sleep(0.5)
    # Update learning rates linear decay each 100 epochs
    if epoch % 100 == 0:
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    if epoch % 1 == 0:
    #if epoch == 0:
        num = 0
        tatal_PSNR = 0
        total_SSIM = 0
        total_MSE = 0
        total_MSSSIM = 0
        total_LPIPS = 0        
        ms_ssim_module=MS_SSIM(win_size=7,win_sigma=1.5,data_range=1,size_average=True,channel=3)
        print('Valing')
        for i, frame_seq_test in tqdm(enumerate(val_dataloader)):

            frame_seq_test = frame_seq_test.type(Tensor)
#             print('frame_seq_test',frame_seq_test.shape)
            real_A_test = Variable(frame_seq_test[:,-1,...]) #[bs,1,c,h,w]
#             print('real_A_test',real_A_test.shape)          
            input_A = Variable(frame_seq_test[:,:-1,...].view((frame_seq_test.size(0),-1)+frame_seq_test.size()[3:]))
#             print('input_A',input_A.shape)           
#             print('G_future(input_A)')            
            A1 = G_future(input_A)
#             print('start val')             
            num += 1
            psnr = metric.PSNR(real_A_test.squeeze(0).detach().cpu().clone().numpy(), A1.squeeze(0).detach().cpu().clone().numpy())
            ssim = metric.SSIM(real_A_test.squeeze(0).detach().cpu().clone().numpy(), A1.squeeze(0).detach().cpu().clone().numpy())
            mse = metric.MSE(real_A_test.squeeze(0).detach().cpu().clone().numpy(), A1.squeeze(0).detach().cpu().clone().numpy())*1000     
            ms_ssim_loss=ms_ssim_module(real_A_test.detach(),A1.detach())
                               
            lpips =  loss_fn(real_A_test.detach().cpu().clone(), A1.detach().cpu().clone())    
            
            tatal_PSNR += psnr
            total_SSIM += ssim
            total_MSE += mse
            total_MSSSIM += ms_ssim_loss
            total_LPIPS += lpips
            time.sleep(0.5)    
        testinfo = "Epoch: {} PSNR={}, SSIM={}, MSE={}, MSSSIM={}, LPIPS={}\n".format(epoch,tatal_PSNR/num,total_SSIM/num,total_MSE/num,total_MSSSIM/num,total_LPIPS/num)
        with open(opt.log_file, 'a+') as f:
              f.write(testinfo)

    #save the model
    if opt.checkpoint_interval != -1 and epoch % 1 == 0:
        # Save model checkpoints
        torch.save(G_future.state_dict(),   opt.save_model_path+"G_future_%d.pth" % (epoch+1))
        torch.save(G_past.state_dict(),   opt.save_model_path+"G_past_%d.pth" % (epoch+1))
        torch.save(D_A.state_dict(), opt.save_model_path+"D_A_%d.pth" % (epoch+1))
        torch.save(D_B.state_dict(), opt.save_model_path+"D_B_%d.pth" % (epoch+1))
        torch.save(D_F.state_dict(), opt.save_model_path+"D_F_%d.pth" % (epoch+1))

