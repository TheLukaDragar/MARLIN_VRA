import argparse
from typing import Optional
from model.regressorVRA import Regressor

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm

from dataset.VRA import VRADataModule
from marlin_pytorch.config import resolve_config
from marlin_pytorch.util import read_yaml
from model.classifier import Classifier
from util.earlystop_lr import EarlyStoppingLR
from util.lr_logger import LrLogger
from util.seed import Seed
from util.system_stats_logger import SystemStatsLogger
from torchmetrics import MeanSquaredError
#plcc and srcc
from scipy.stats import pearsonr, spearmanr

from dataset.VRA import VRABase
import os
from abc import ABC, abstractmethod
from itertools import islice
from typing import Optional

import ffmpeg
import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from marlin_pytorch.util import read_video, padding_video
from util.misc import sample_indexes, read_text, read_json

import albumentations as A
from albumentations.pytorch import ToTensorV2

def gaussian_noise(image, sigma=200):

    #albumentations

    image = A.GaussNoise(p=1, var_limit=(10.0,sigma))(image=image)["image"]
    #apply 2 flips 
    
    #to tensor
   

    return image
    

# for fine-tuning
class CelebvHq(VRABase):

    def __init__(self,
        root_dir: str,
        split: str,
        task: str,
        clip_frames: int,
        temporal_sample_rate: int,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None
    ):
        super().__init__(root_dir, split, task, data_ratio, take_num)
        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate



    def __getitem__(self, index: int):
        y = self.metadata["clips"][self.name_list[index]]["label"]
        y=float(y)
        video_path = os.path.join(self.data_root, self.name_list[index] + ".mp4")

        probe = ffmpeg.probe(video_path)["streams"][0]
        n_frames = int(probe["nb_frames"])

        if n_frames <= self.clip_frames:
            video = read_video(video_path, channel_first=True).video / 255
            # pad frames to 16
            video = padding_video(video, self.clip_frames, "same")  # (T, C, H, W)
            video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
            return video, torch.tensor(y, dtype=torch.long)
        elif n_frames <= self.clip_frames * self.temporal_sample_rate:
            # reset a lower temporal sample rate
            sample_rate = n_frames // self.clip_frames
        else:
            sample_rate = self.temporal_sample_rate
        # sample frames
        video_indexes = sample_indexes(n_frames, self.clip_frames, sample_rate)
        reader = torchvision.io.VideoReader(video_path)
        fps = reader.get_metadata()["video"]["fps"][0]
        reader.seek(video_indexes[0].item() / fps, True)
        frames = []
        for frame in islice(reader, 0, self.clip_frames * sample_rate, sample_rate):
            frames.append(frame["data"])

        pertrubated_frames = []
        for frame in frames:
           
            gaused = gaussian_noise(frame.numpy(), sigma=200)

            #back to tensor
            gaused = torch.from_numpy(gaused)
            pertrubated_frames.append(gaused)



        # print("TETSTSTSTS:")
        video = torch.stack(frames) / 255  # (T, C, H, W)

        #save to debug


        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        assert video.shape[1] == self.clip_frames, video_path

        original_video_to_save = video.permute(1, 2, 3, 0)  # Change to (T, H, W, C)
        # torchvision.io.write_video('./debugvids/original_video.mp4', original_video_to_save * 255, fps)


        pertrubated_video = torch.stack(pertrubated_frames) / 255  # (T, C, H, W)

    

        pertrubated_video = pertrubated_video.permute(1, 0, 2, 3)  # (C, T, H, W)
        assert pertrubated_video.shape[1] == self.clip_frames, video_path

        pertrubated_video_to_save = pertrubated_video.permute(1, 2, 3, 0)  # Change to (T, H, W, C)
        # torchvision.io.write_video('./debugvids/pertrubated_video.mp4', pertrubated_video_to_save * 255, fps)

        return video,pertrubated_video, torch.tensor(y, dtype=torch.float32)


from marlin_pytorch import Marlin



model = Marlin.from_online("marlin_vit_base_ytf")
backbone_config = resolve_config("marlin_vit_base_ytf")

print("backbone_config.n_frames", backbone_config.n_frames)
test_dataset = CelebvHq("/ceph/hpc/data/st2207-pgp-users/ldragar/ds_marlin/", "test", 'vra', backbone_config.n_frames,
            2, 1.0)


test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    drop_last=False,
)

model.eval()
#cuda
model = model.cuda()



#get first batch
with torch.no_grad():
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    ys = []
    cos_sims = []

    for batch in test_loader:
        # print("batch:", batch)
        # print("batch:", batch.shape)
        # print("batch[0].shape:", batch[0].shape)

        x,x_pertrubated, y = batch

        #cuda
        x = x.cuda()
        x_pertrubated = x_pertrubated.cuda()
        y = y.cuda()

        feats = model.extract_features(x,keep_seq=False) #just get the features for the whole video
        feats_pertrubated = model.extract_features(x_pertrubated,keep_seq=False) #just get the features for the whole video
        # print("feats:", feats)
        # print("feats.shape:", feats.shape)
        # print("feats_pertrubated:", feats_pertrubated)
        # print("feats_pertrubated.shape:", feats_pertrubated.shape)

        #compute cosine similarity
        cos_sim = cos(feats,feats_pertrubated)
        print("cos_sims:", cos_sim, "ys:", y)

        ys.append(y.detach().cpu().numpy().flatten())
        cos_sims.append(cos_sim.detach().cpu().numpy().flatten())

    ys = np.concatenate(ys)
    cos_sims = np.concatenate(cos_sims)

    print("ys:", ys)
    print("cos_sims:", cos_sims)


    # Calculate plcc and srcc
    plc = pearsonr(cos_sims, ys)
    print("PLCC:", plc)
    src = spearmanr(cos_sims, ys)
    print("SRCC:", src)

    









        












def train_vra(args, config):
    data_path = args.data_path
    resume_ckpt = args.resume
    n_gpus = args.n_gpus
    max_epochs = args.epochs

    finetune = config["finetune"]
    learning_rate = config["learning_rate"]
    task = config["task"]

    if task == "appearance":
        num_classes = 40
    elif task == "action":
        num_classes = 35

    elif task == "vra":
        num_classes = 1

    else:
        raise ValueError(f"Unknown task {task}")

    if finetune:
        backbone_config = resolve_config(config["backbone"])

        model = Regressor(
            num_classes, config["backbone"], True, args.marlin_ckpt, "regression", config["learning_rate"],
            args.n_gpus > 1,
        )

        dm = VRADataModule(
            data_path, finetune, task,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            clip_frames=backbone_config.n_frames,
            temporal_sample_rate=2
        )

    else:
        model = Regressor(
            num_classes, config["backbone"], False,
            None, "regression", config["learning_rate"], args.n_gpus > 1,
        )

        dm = VRADataModule(
            data_path, finetune, task,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            feature_dir=config["backbone"],
            temporal_reduction=config["temporal_reduction"]
        )

    if args.skip_train:
        dm.setup()
        return resume_ckpt, dm

    strategy = 'auto' if n_gpus <= 1 else "ddp"
    accelerator = "cpu" if n_gpus == 0 else "gpu"

    ckpt_filename = config["model_name"] + "-{epoch}-{val_loss:.3f}"
    ckpt_monitor = "val_loss"

    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    ckpt_callback = ModelCheckpoint(dirpath=f"ckpt/{config['model_name']}", save_last=True,
        filename=ckpt_filename,
        monitor=ckpt_monitor,
        mode="min")

    trainer = Trainer(log_every_n_steps=1, devices=n_gpus, accelerator=accelerator, benchmark=True,
        logger=True, precision=precision, max_epochs=max_epochs,
        strategy=strategy,
        callbacks=[ckpt_callback, LrLogger(), EarlyStoppingLR(1e-6), SystemStatsLogger()])

    trainer.fit(model, dm)

    #last model checkpoint

    last_ckpt = ckpt_callback.last_model_path

    return ckpt_callback.best_model_path, last_ckpt, dm


def evaluate_vra(args, ckpt, dm,type="best"):
    print("Load checkpoint", ckpt)
    model = Regressor.load_from_checkpoint(ckpt)
    accelerator = "cpu" if args.n_gpus == 0 else "gpu"
    trainer = Trainer(log_every_n_steps=1, devices=1 if args.n_gpus > 0 else 0, accelerator=accelerator, benchmark=True,
        logger=False, enable_checkpointing=False)
    Seed.set(42)
    model.eval()

    # # collect predictions
    # preds = trainer.predict(model, dm.test_dataloader())
    # preds = torch.cat(preds)

    # # collect ground truth
    # ys = torch.zeros_like(preds, dtype=torch.bool)
    # for i, (_, y) in enumerate(tqdm(dm.test_dataloader())):
    #     ys[i * args.batch_size: (i + 1) * args.batch_size] = y

    # preds = preds.sigmoid()
    # acc = ((preds > 0.5) == ys).float().mean()
    # auc = model.auc_fn(preds, ys)
    # results = {
    #     "acc": acc,
    #     "auc": auc
    # }
    # print(results)

    preds = trainer.predict(model, dm.test_dataloader())
    preds = torch.cat(preds)

    # collect ground truth
    ys = []
    for _, y in tqdm(dm.test_dataloader()):
        ys.append(y)
    ys = torch.cat(ys)

    #detach
    preds = preds.detach().cpu().numpy()
    ys = ys.detach().cpu().numpy()



    # Calculate regression metrics
    mse = MeanSquaredError()
    mae = torch.nn.L1Loss()
    #remove extra dimension
    preds = preds.squeeze()
    ys = ys.squeeze()

    print("preds:", preds)
    print("ys:", ys)
    
    mse_score = mse(torch.tensor(preds), torch.tensor(ys))
    print("MSE:", mse_score.item())
    mae_score = mae(torch.tensor(preds), torch.tensor(ys))
    print("MAE:", mae_score.item())

    plc = pearsonr(preds, ys)
    print("PLCC:", plc)
    src = spearmanr(preds, ys)
    print("SRCC:", src)

    results = {
        "mse": mse_score.item(),
        "mae": mae_score.item(),
        "plcc": plc,
        "srcc": src

    }
    print(results)

    #save to file
    with open(f"{type}_results.txt", "w") as f:
        print(results, file=f)
    


def evaluate(args):
    config = read_yaml(args.config)
    dataset_name = config["dataset"]

    if dataset_name == "vra":
        ckpt,last_cp, dm = train_vra(args, config)
        evaluate_vra(args, ckpt, dm,"best_cp")
        evaluate_vra(args, last_cp, dm,"last_cp")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser("CelebV-HQ evaluation")
#     parser.add_argument("--config", type=str, help="Path to CelebV-HQ evaluation config file.")
#     parser.add_argument("--data_path", type=str, help="Path to CelebV-HQ dataset.")
#     parser.add_argument("--marlin_ckpt", type=str, default=None,
#         help="Path to MARLIN checkpoint. Default: None, load from online.")
#     parser.add_argument("--n_gpus", type=int, default=1)
#     parser.add_argument("--precision", type=str, default="32")
#     parser.add_argument("--num_workers", type=int, default=8)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--epochs", type=int, default=2000, help="Max epochs to train.")
#     parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
#     parser.add_argument("--skip_train", action="store_true", default=False,
#         help="Skip training and evaluate only.")

#     args = parser.parse_args()
#     if args.skip_train:
#         assert args.resume is not None

#     evaluate(args)
