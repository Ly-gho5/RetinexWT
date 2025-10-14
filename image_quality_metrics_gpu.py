import argparse
import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from torchmetrics.functional import structural_similarity_index_measure as ssim_loss
from tqdm import tqdm

# ------------------ argparse ------------------
parser = argparse.ArgumentParser(description='PSNR SSIM LPIPS script', add_help=False)
parser.add_argument('--input_images_path', default='')
parser.add_argument('--image2smiles2image_save_path', default='')
parser.add_argument('--gpu_id', default='0', type=str, help='GPU id to use')
parser.add_argument('-v', '--version', type=str, default='0.1')
args = parser.parse_args()

# ------------------ 设置 GPU ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ 工具函数 ------------------
def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg"])

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    return img

class DataLoaderVal(Dataset):
    def __init__(self, target_transform=None):
        super().__init__()
        gt_dir = args.input_images_path
        input_dir = args.image2smiles2image_save_path
        clean_files = sorted(os.listdir(gt_dir))
        noisy_files = sorted(os.listdir(input_dir))
        self.clean_filenames = [os.path.join(gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(input_dir, x) for x in noisy_files if is_png_file(x)]
        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        clean = torch.from_numpy(load_img(self.clean_filenames[index])).permute(2, 0, 1)
        noisy = torch.from_numpy(load_img(self.noisy_filenames[index])).permute(2, 0, 1)
        return clean, noisy, os.path.basename(self.clean_filenames[index]), os.path.basename(self.noisy_filenames[index])

# ------------------ 主函数 ------------------
def calculate_metrics():
    print("开始计算图像质量评价指标...")
    print(f"原始图像路径: {args.input_images_path}")
    print(f"处理后图像路径: {args.image2smiles2image_save_path}")

    if not os.path.exists(args.input_images_path) or not os.path.exists(args.image2smiles2image_save_path):
        print("错误：图像文件夹路径不存在！")
        return

    dataset = DataLoaderVal()
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    # 初始化 LPIPS 模型（放入GPU）
    loss_fn = lpips.LPIPS(net='alex', version=args.version).to(device)

    psnr_list, ssim_list = [], []

    for clean, noisy, _, _ in tqdm(loader, desc="计算PSNR和SSIM"):
        clean = clean.to(device)
        noisy = noisy.to(device)

        # PSNR
        gt_np = clean.squeeze().permute(1, 2, 0).cpu().numpy()
        pred_np = torch.clamp(noisy, 0, 1).squeeze().permute(1, 2, 0).cpu().numpy()
        psnr_list.append(psnr_loss(pred_np, gt_np))

        # SSIM
        ssim_score = ssim_loss(noisy, clean, data_range=1.0)
        ssim_list.append(ssim_score.item())

    psnr_avg = sum(psnr_list) / len(psnr_list)
    ssim_avg = sum(ssim_list) / len(ssim_list)

    # ------------------ LPIPS ------------------
    print("\n正在计算 LPIPS 指标...")
    files = os.listdir(args.input_images_path)
    total_lpips = 0
    count = 0

    for file in tqdm(files, desc="计算LPIPS"):
        img0_path = os.path.join(args.input_images_path, file)
        img1_path = os.path.join(args.image2smiles2image_save_path, file)
        if os.path.exists(img0_path) and os.path.exists(img1_path):
            try:
                img0 = lpips.im2tensor(lpips.load_image(img0_path)).to(device)
                img1 = lpips.im2tensor(lpips.load_image(img1_path)).to(device)
                dist = loss_fn(img0, img1)
                total_lpips += dist.item()
                count += 1
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
    
    avg_lpips = total_lpips / count if count > 0 else 0

    # ------------------ 输出 ------------------
    print("\n" + "="*50)
    print("图像质量评价结果")
    print("="*50)
    print(f"处理的图像数量: {count}")
    print(f"PSNR: {psnr_avg:.4f} dB")
    print(f"SSIM: {ssim_avg:.4f}")
    print(f"LPIPS: {avg_lpips:.4f}")
    print("="*50)

if __name__ == '__main__':
    calculate_metrics()
