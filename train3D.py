from dataset import create_dataset
from model.UNet3DNewNew import UNet
from utils.engine3D import GaussianDiffusionTrainer2
from utils.tools3D import train_one_epoch, load_yaml
import torch
from utils.callbacks import ModelCheckpoint
from Sophia import SophiaG
from Adam import DAdaptAdam
from lookahead import Lookahead
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

def train(config):
    consume = config["consume"]
    if consume:
        cp = torch.load(config["consume_path"])
        config = cp["config"]
    print(config)

    device = torch.device(config["device"])
    loader = create_dataset(**config["Dataset"])
    start_epoch = 1

    model = UNet(**config["Model"]).to(device)
    #optimizer = SophiaG(model.parameters(),
    #lr=(1.25e-4),
    #betas=(0.965, 0.99),
    #rho=0.04,
    #weight_decay=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-1)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)  # 每个epoch衰减为原来的0.9倍
    # scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])  # 使用余弦退火调整学习率
    #0.011.25e-4lookahead_opti
    # mizer = Lookahead(optimizer, k=5, alpha=0.5)
    trainer = GaussianDiffusionTrainer2(model, **config["Trainer"]).to(device)

    model_checkpoint = ModelCheckpoint(**config["Callback"])

    if consume:
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        model_checkpoint.load_state_dict(cp["model_checkpoint"])
        start_epoch = cp["start_epoch"] + 1

    # 在训练循环开始之前输出模型架构信息
    # print("Model Architecture:")
    # print(model)

    # print("Down blocks:")
    # for i, module in enumerate(model.down_blocks):
    #     print(f"Down block {i}: {module}")

    # print("Middle block:")
    # print(model.middle_block)

    # print("Up blocks:")
    # for i, module in enumerate(model.up_blocks):
    #    print(f"Up block {i}: {module}")

    # print("Output layer:")
    # print(model.out)

    for epoch in range(start_epoch, config["epochs"] + 1):
        loss = train_one_epoch(trainer, loader, optimizer, device, epoch, save_input=True, samples_dir="samples")
        model_checkpoint.step(loss, model=model.state_dict(), config=config,
                              optimizer=optimizer.state_dict(), start_epoch=epoch,
                              model_checkpoint=model_checkpoint.state_dict())

        scheduler.step()  # 在每个epoch结束后更新学习率

    #for epoch in range(start_epoch, config["epochs"] + 1):
        #loss = train_one_epoch( trainer, loader, lookahead_optimizer, device, epoch)
        #model_checkpoint.step(loss, model=model.state_dict(), config=config,
        #                      optimizer=lookahead_optimizer.optimizer.state_dict(), start_epoch=epoch,
        #                      model_checkpoint=model_checkpoint.state_dict())


if __name__ == "__main__":
    config = load_yaml("config3D.yml", encoding="utf-8")
    train(config)
