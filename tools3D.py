from typing import Optional, Union
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from PIL import Image
from pathlib2 import Path
import numpy as np
import yaml
import os
import tifffile as tiff

def load_yaml(yml_path: Union[Path, str], encoding="utf-8"):
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    with yml_path.open('r', encoding=encoding) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg


def train_one_epoch(trainer, loader, optimizer, device, epoch, save_input=True, samples_dir="samples"):
    trainer.train()
    total_loss, total_num = 0., 0

    # 创建一个文件夹来保存样本数据
    os.makedirs(samples_dir, exist_ok=True)

    with tqdm(loader, dynamic_ncols=True, colour="#ff924a") as data:
        for batch_idx, (images, _) in enumerate(data):
            # 保存输入数据为TIF格式
            if save_input:
                for i, img in enumerate(images):
                    tiff_path = os.path.join(samples_dir, f"input_epoch{epoch}_batch{batch_idx}_img{i}.tif")
                    # 直接保存3D图像数据，不进行范围转换
                    tiff.imwrite(tiff_path, img.cpu().numpy())

            # 训练过程
            optimizer.zero_grad()
            x_0 = images.to(device)
            loss = trainer(x_0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_num += x_0.shape[0]

            data.set_description(f"Epoch: {epoch}")
            data.set_postfix(ordered_dict={
                "train_loss": total_loss / total_num,
            })

    return total_loss / total_num


def save_image(images: torch.Tensor, nrow: int = 8, show: bool = True, path: Optional[str] = None,
               format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    concat all image into a picture.

    Parameters:
        images: a tensor with shape (batch_size, channels, height, width).
        nrow: decide how many images per row. Default `8`.
        show: whether to display the image after stitching. Default `True`.
        path: the path to save the image. if None (default), will not save image.
        format: image format. You can print the set of available formats by running `python3 -m PIL`.
        to_grayscale: convert PIL image to grayscale version of image. Default `False`.
        **kwargs: other arguments for `torchvision.utils.make_grid`.

    Returns:
        concat image, a tensor with shape (height, width, channels).
    """
    images = images * 0.5 + 0.5
    grid = make_grid(images, nrow=nrow, **kwargs)  # (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid


def save_slice_image(image_3d: torch.Tensor, slice_index: int, axis: int = 2, show: bool = True, path: Optional[str] = None, format: Optional[str] = None, to_grayscale: bool = False):
    """
    保存 3D 图像的一个切片。
    ... [省略参数说明] ...
    """
    # 选择切片
    if axis == 0:
        slice_image = image_3d[:, :, slice_index, :, :]
    elif axis == 1:
        slice_image = image_3d[:, :, :, slice_index, :]
    elif axis == 2:
        slice_image = image_3d[:, :, :, :, slice_index]
    else:
        raise ValueError("Invalid axis for slicing.")

    # 调整图像尺寸
    slice_image = slice_image.squeeze(axis)  # 移除切片轴
    slice_image = slice_image * 0.5 + 0.5  # 归一化

    # 转换为 PIL 图像
    slice_image = slice_image.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    slice_image = np.squeeze(slice_image)  # 如果 batch_size 为 1，则移除批次维度

    # 确保图像是2D的
    if slice_image.ndim == 3:
        slice_image = slice_image[0]  # 选择第一个通道

    im = Image.fromarray(slice_image)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return slice_image
# save_sample_slice_image 函数类似，但处理包含多个样本的张量

def save_slice_image1(image_3d, slice_index, axis=2, show=True, path=None, to_grayscale=False):
    # 确保输入是4维张量
    if image_3d.ndim != 4:
        raise ValueError("Input tensor must be 4-dimensional.")

    # 选择切片
    if axis == 0:
        slice_image = image_3d[slice_index, :, :, :]
    elif axis == 1:
        slice_image = image_3d[:, slice_index, :, :]
    elif axis == 2:
        slice_image = image_3d[:, :, slice_index, :]
    else:
        raise ValueError("Invalid axis for slicing.")

    # 转换为 PIL 图像
    slice_image = slice_image.squeeze()  # 移除单维度条目
    slice_image = slice_image * 0.5 + 0.5  # 归一化
    slice_image = slice_image.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    slice_image = np.squeeze(slice_image)  # 如果 batch_size 为 1，则移除批次维度

    # 确保图像是2D的
    if slice_image.ndim == 3:
        slice_image = slice_image[0]  # 选择第一个通道

    im = Image.fromarray(slice_image)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path)
    if show:
        im.show()

    return slice_image