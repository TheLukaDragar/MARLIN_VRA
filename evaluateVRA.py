import argparse
from model.regressorVRA import Regressor,RegressorFeatureOnly

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


        if args.RegressorFeatureOnly:

            model = RegressorFeatureOnly(num_classes, config["backbone"], True, args.marlin_ckpt, "regression", config["learning_rate"],
            args.n_gpus > 1,
        )
            print("Freeze backbone and train only regressor")
        else:
            model = Regressor(
                num_classes, config["backbone"], True, args.marlin_ckpt, "regression", config["learning_rate"],
                args.n_gpus > 1,
            )
            print("Train backbone and regressor")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CelebV-HQ evaluation")
    parser.add_argument("--config", type=str, help="Path to CelebV-HQ evaluation config file.")
    parser.add_argument("--data_path", type=str, help="Path to CelebV-HQ dataset.")
    parser.add_argument("--marlin_ckpt", type=str, default=None,
        help="Path to MARLIN checkpoint. Default: None, load from online.")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2000, help="Max epochs to train.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--skip_train", action="store_true", default=False,
        help="Skip training and evaluate only.")

    parser.add_argument("--RegressorFeatureOnly", action="store_true", default=False,
        help="Freeze backbone and train only regressor")
    


    args = parser.parse_args()
    if args.skip_train:
        assert args.resume is not None

    evaluate(args)
