from utils.engine3D import DDPMSampler, DDIMSampler
from model.UNet3DNewNew import UNet
import torch
from utils.tools3D import save_slice_image1, save_slice_image, save_image
from argparse import ArgumentParser
import numpy as np
import tifffile as tiff  # 需要安装 tifffile 包

def parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])

    # generator param
    parser.add_argument("-bs", "--batch_size", type=int, default=16)

    # sampler param
    parser.add_argument("--result_only", default=False, action="store_true")
    parser.add_argument("--interval", type=int, default=50)

    # DDIM sampler param
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "quadratic"])

    # save image param
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None)
    parser.add_argument("--to_grayscale", default=False, action="store_true")

    args = parser.parse_args()
    return args


@torch.no_grad()
def generate(args):
    device = torch.device(args.device)

    cp = torch.load(args.checkpoint_path)
    model = UNet(**cp["config"]["Model"])
    model.load_state_dict(cp["model"])
    model.to(device)
    model = model.eval()

    if args.sampler == "ddim":
        sampler = DDIMSampler(model, **cp["config"]["Trainer"]).to(device)
    elif args.sampler == "ddpm":
        sampler = DDPMSampler(model, **cp["config"]["Trainer"]).to(device)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    image_size = (128,128,128)
    z_t = torch.randn((args.batch_size, cp["config"]["Model"]["in_channels"], *image_size), device=device)

    extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)
    x = sampler(z_t, only_return_x_0=args.result_only, interval=args.interval, **extra_param)

    # 保存3D图像的切片
    if args.result_only:
        for i in range(x.shape[2]):  # 遍历深度维度
            save_slice_image(x, slice_index=i, axis=2, show=False, path=f"{args.image_save_path}_slice_{i}.png", to_grayscale=args.to_grayscale)


    if args.result_only:
        for i in range(x.shape[0]):  # 遍历批次维度
            # 提取整个3D图像
            img_3d = x[i].cpu().numpy()  # 将整个3D图像转换为numpy数组
            # 归一化到 [0, 1] 并转换到 [0, 255]
            img_3d = (img_3d - img_3d.min()) / (img_3d.max() - img_3d.min()) * 255.0
            img_3d = img_3d.astype(np.uint8)  # 转换为整数

            # 构建保存路径
            tiff_path = f"{args.image_save_path}_batch_{i}.tif"
            # 保存整个3D图像为一个TIF文件
            tiff.imwrite(tiff_path, img_3d, photometric='minisblack')


if __name__ == "__main__":
    args = parse_option()
    generate(args)

