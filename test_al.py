from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.CC_al import CrowdCounter_aleatoric
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = "../SHHB_results"
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

if not os.path.exists(exp_name + "/pred"):
    os.mkdir(exp_name + "/pred")

if not os.path.exists(exp_name + "/gt"):
    os.mkdir(exp_name + "/gt")

mean_std = (
    [0.452016860247, 0.447249650955, 0.431981861591],
    [0.23242045939, 0.224925786257, 0.221840232611],
)
img_transform = standard_transforms.Compose(
    [standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)]
)
restore = standard_transforms.Compose(
    [own_transforms.DeNormalize(*mean_std), standard_transforms.ToPILImage()]
)
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = "../ProcessedData/shanghaitech_part_B/test"

model_path = "../08-SANet_all_ep_57_mae_42.4_mse_85.4.pth"

## Modified tester + enabling MC Dropout 

def main(num_samples=100): #num_samples is the number of samples from the model's prediction distribution
    file_list = [filename for root, dirs, filename in os.walk(dataRoot + "/img/")]
    gts_list = []
    preds_list = []
    logvars_list = []
    for i in range(0, num_samples):
      gts, preds, logvars = test(file_list[0], model_path)
      gts_list.append(gts)
      preds_list.append(preds)
      logvars_list.append(logvars)
    return (gts_list, preds_list, logvars_list)


def test(file_list, model_path):

    net = CrowdCounter_aleatoric(cfg.GPU_ID, cfg.NET)

    state = torch.load(model_path)
    net.load_state_dict(state["net"])
    net.cuda()
    #net.eval() #Uncomment if not using Dropout for MC Dropout at test time
    net.train() #Comment if not using Dropout for MC Dropout at test time

    f1 = plt.figure(1)

    gts = []
    preds = []
    logvars = []

    for filename in file_list:
        print(filename)
        imgname = dataRoot + "/img/" + filename
        filename_no_ext = filename.split(".")[0]

        denname = dataRoot + "/den/" + filename_no_ext + ".csv"

        den = pd.read_csv(denname, sep=",", header=None).values
        den = den.astype(np.float32, copy=False)

        img = Image.open(imgname)

        if img.mode == "L":
            img = img.convert("RGB")

        img = img_transform(img)

        gt = np.sum(den)
        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map, logvar = net.test_forward(img)
            gts.append(gt)
            preds.append(pred_map[0].sum().data /100.)
            logvars.append(logvar[0].mean().data)

        sio.savemat(
            exp_name + "/pred/" + filename_no_ext + ".mat",
            {"data": pred_map.squeeze().cpu().numpy() / 100.0},
        )
        sio.savemat(exp_name + "/gt/" + filename_no_ext + ".mat", {"data": den})

        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

        pred = np.sum(pred_map) / 100.0
        pred_map = pred_map / np.max(pred_map + 1e-20)

        den = den / np.max(den + 1e-20)

        den_frame = plt.gca()
        plt.imshow(den, "jet")
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines["top"].set_visible(False)
        den_frame.spines["bottom"].set_visible(False)
        den_frame.spines["left"].set_visible(False)
        den_frame.spines["right"].set_visible(False)
        plt.savefig(
            exp_name + "/" + filename_no_ext + "_gt_" + str(int(gt)) + ".png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=150,
        )

        plt.close()

        pred_frame = plt.gca()
        plt.imshow(pred_map, "jet")
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines["top"].set_visible(False)
        pred_frame.spines["bottom"].set_visible(False)
        pred_frame.spines["left"].set_visible(False)
        pred_frame.spines["right"].set_visible(False)
        plt.savefig(
            exp_name + "/" + filename_no_ext + "_pred_" + str(float(pred)) + ".png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=150,
        )

        plt.close()

        diff = den - pred_map

        diff_frame = plt.gca()
        plt.imshow(diff, "jet")
        plt.colorbar()
        diff_frame.axes.get_yaxis().set_visible(False)
        diff_frame.axes.get_xaxis().set_visible(False)
        diff_frame.spines["top"].set_visible(False)
        diff_frame.spines["bottom"].set_visible(False)
        diff_frame.spines["left"].set_visible(False)
        diff_frame.spines["right"].set_visible(False)
        plt.savefig(
            exp_name + "/" + filename_no_ext + "_diff.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=150,
        )

        plt.close()

    return (gts, preds, logvars)

if __name__ == "__main__":
    gts_list, preds_list, logvars_list = main()
