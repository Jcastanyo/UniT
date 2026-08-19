if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import torchvision
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from PIL import Image as PILImage
import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import wandb
import copy
from UniT.dataset.tactile_representation_dataset import TactileRepresentationDataset
from UniT.taming.models.vqgan import VQModel, WOVQModel
from UniT.dataset.representation_wrapper import RepresentationWrapper
from UniT.taming.logging_util import *

import cv2
import glob
from torchvision import transforms 


def load_images(folder_path, extension="*.png"):

    search_pattern = os.path.join(folder_path, extension)

    image_paths = glob.glob(search_pattern)
    #print(image_paths)

    #print(os.path.basename(image_paths[0]))
    image_paths.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]))
    #print(image_paths)

    images = [cv2.imread(img_path) for img_path in image_paths]

    return images


OmegaConf.register_new_resolver("eval", eval, replace=True)
class TrainVqvaeRepresentationWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        #seed = cfg.seed
        #torch.manual_seed(seed)
        #np.random.seed(seed)
        #random.seed(seed)
        trainer_config = {}
        trainer_config["gpus"] = cfg.gpus
        # configure model
        model = None
        if cfg.with_vq:
            model = VQModel(**cfg.model)
            name = 'vq'
            print('vq model')
        else:
            model = WOVQModel(**cfg.model)
            name = 'wovq'
            print('without vq model')
        self.model = model
        # print model size
        print("Model size: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        sys.path.append(os.getcwd())
        parser = argparse.ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        parser = config_to_parser(cfg.trainer, parser)
        opt, unknown = parser.parse_known_args()

        #nowname = now+name+opt.postfix
        #logdir = os.path.join("representation_models", nowname)
        ckptdir = "/UniT_shared/representation_models/new_experiments/vqgan_mid/2026-03-04T16-17-58vq/checkpoints/checkpoint-epoch=34.ckpt"

        self.model.init_from_ckpt(ckptdir)

        configs = {}
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # trainer and callbacks
        trainer_kwargs = dict()
        # add callbacks
        default_callbacks_cfg = {}
        default_callbacks_cfg["periodic_checkpoint"] = {
            "target": "UniT.taming.logging_util.PeriodicCheckpointCallback",
            "params": {
                "ckptdir": ckptdir,
                "save_interval": 1,  
                "save_last_n": 3,   
                "verbose": True,
            }
        }
        
        callbacks_cfg = lightning_config.get('callbacks') or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        self.trainer = trainer
        self.lightning_config = lightning_config
        self.cpktdir = ckptdir
        self.opt = opt

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        lightning_config = self.lightning_config
        opt = self.opt
        
        #human_demonstrations_img = TactileRepresentationDataset(**cfg.dataset)
        #data = RepresentationWrapper(human_demonstrations_img, cfg, test='test')
#/UniT_shared/data/julio_wth_markers/evaluate/test_rec_without_markers/remain_data_test_rec       
        #images = load_images("./data/julio_wth_markers/train/splitted_images/test/")
        images = load_images("./data/julio_wth_markers/train/img_results/test_color_jitter/vqgan_test_mid_model/orig/")
        print(len(images))

        for idx, image in enumerate(images):

            print(idx)

            b = random.uniform(0.1, 0.3)
            c = random.uniform(0.1, 0.3)
            s = random.uniform(0.1, 0.3)
            h = random.uniform(0.1, 0.3)

            color_jitter = transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)(PILImage.fromarray(image))
            
            cv2.imwrite(f"./data/julio_wth_markers/train/img_results/test_color_jitter/vqgan_test_mid_model/augm/{idx}.png", np.array(color_jitter))

            image = np.array(color_jitter).astype(np.float32) / 255.
            #print(image.shape)
        
            resized_image = cv2.resize(image, (160, 128))

            torch_image = torch.from_numpy(resized_image)
            torch_image = torch.unsqueeze(torch_image, 0)
       
            img_batch = {"image" : torch_image}
            input = self.model.get_input(img_batch, "image")
        

            #quant, emb_loss, info = self.model.encode(input)
            #print(quant.shape)
            h = self.model.encoder(input)
            
            rec_image, _  = self.model(input)
            rec_image = rec_image.to(dtype=torch.float32)
            rec_image = rec_image.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
            rec_image = (rec_image * 255).astype(np.uint8)
        
            #cv2.imwrite(f"/UniT_shared/data/julio_wth_markers/train/img_results/test_color_jitter/vqgan_test_small_model/{idx}.png", rec_image)        
            cv2.imwrite(f"./data/julio_wth_markers/train/img_results/test_color_jitter/vqgan_test_mid_model/rec/{idx}.png", rec_image)        
            

          
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainVqvaeRepresentationWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
