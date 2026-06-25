import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('Agg')

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.ultra_video_snow_net_v3_optimized import build_ultra_v3_optimized

from kitti_test_dataset import get_kitti_test_dataloader

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

def calculate_psnr(img1, img2, max_value=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_value / torch.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11):
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def calculate_lpips(img1, img2, lpips_model):
    if lpips_model is None:
        return torch.tensor(0.0)
    with torch.no_grad():
        img1_norm = img1 * 2 - 1
        img2_norm = img2 * 2 - 1
        return lpips_model(img1_norm, img2_norm).mean()

def pad_to_multiple(tensor, multiple=8):
    _, _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    return tensor, (pad_h, pad_w)

def unpad(tensor, pad_h, pad_w):
    if pad_h > 0:
        tensor = tensor[:, :, :-pad_h, :]
    if pad_w > 0:
        tensor = tensor[:, :, :, :-pad_w]
    return tensor

def tensor_to_image(tensor):
    img = tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def save_comparison(input_img, output_img, gt_img, save_path, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(input_img)
    axes[0].set_title('输入图像 (Input Snow)', fontsize=14, fontproperties='SimHei')
    axes[0].axis('off')
    
    axes[1].imshow(output_img)
    axes[1].set_title('模型输出 (Model Output Desnowed)', fontsize=14, fontproperties='SimHei')
    axes[1].axis('off')
    
    axes[2].imshow(gt_img)
    axes[2].set_title('真实清晰 (Ground Truth Clean)', fontsize=14, fontproperties='SimHei')
    axes[2].axis('off')
    
    psnr, ssim, lpips_val = metrics
    metrics_text = f"PSNR: {psnr:.4f} dB  |  SSIM: {ssim:.6f}  |  LPIPS: {lpips_val:.6f}"
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def test():
    data_root = './KITTI_snow'
    
    checkpoint_paths = [
        './'
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        dataset2_dir = './checkpoint/kitti/'
        if os.path.exists(dataset2_dir):
            pth_files = [f for f in os.listdir(dataset2_dir) if f.endswith('.pth')]
            if pth_files:
                best_files = [f for f in pth_files if 'BEST_MODEL' in f]
                if best_files:
                    best_files.sort(key=lambda x: float(x.split('PSNR')[1].split('.pth')[0]) if 'PSNR' in x else 0, reverse=True)
                    checkpoint_path = os.path.join(dataset2_dir, best_files[0])
                else:
                    checkpoint_path = os.path.join(dataset2_dir, pth_files[0])
        
        if checkpoint_path is None:
            checkpoint_path = checkpoint_paths[0]
    
    output_dir = './test_results'
    
    num_frames = 5
    base_channels = 48
    num_blocks = [2, 2, 4]
    num_heads = [1, 2, 4]
    
    test_videos = ['video25']

    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("UltraVideoSnowNet V3-Optimized - KITTI Video029测试")
    print("="*80)
    print(f"\n配置:")
    print(f"   数据目录: {data_root}")
    print(f"   测试视频: {test_videos}")
    print(f"   模型权重: {checkpoint_path}")
    print(f"   输出目录: {output_dir}")
    print(f"   处理模式: 完整尺寸（不裁剪）")
    print(f"   保存设置: 所有处理后的原图（与输入尺寸一致）")
    print("="*80 + "\n")
    
    print(f"加载KITTI测试数据（完整尺寸） - 只加载 {test_videos}...")
    try:
        test_loader = get_kitti_test_dataloader(
            data_root=data_root,
            split='test',
            window_size=num_frames,
            num_workers=0,
            video_filter=test_videos
        )
        print(f"测试数据加载成功，共 {len(test_loader)} 个样本\n")
        
        if len(test_loader) == 0:
            print("❌ 警告：没有找到任何测试样本！")
            print("可能的原因：")
            print("1. 指定的视频不存在")
            print("2. 视频数据不完整")
            print("3. 文件格式不匹配")
            print(f"请检查数据目录：{data_root}/Test/ 和 {data_root}/Test_GT/")
            return
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("   这通常是因为数据集路径不正确或数据集为空")
        print("   请检查KITTI_snow数据集是否已正确放置")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    print("创建模型...")
    model = build_ultra_v3_optimized(
        num_frames=num_frames,
        base_channels=base_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        use_checkpoint=False
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        print(f"Checkpoint包含的键: {list(checkpoint.keys())[:10]}...")
        
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            print("检测到训练checkpoint格式")
            if 'epoch' in checkpoint:
                print(f"   训练轮次: {checkpoint['epoch']}")
            if 'psnr' in checkpoint:
                print(f"   最佳PSNR: {checkpoint.get('psnr', 'N/A')}")
        elif isinstance(checkpoint, dict) and any(key.startswith(('coarse_net.', 'fine_net.')) for key in checkpoint.keys()):
            model_state_dict = checkpoint
            print("检测到模型权重格式")
        else:
            model_state_dict = checkpoint
            print("未知checkpoint格式，尝试直接加载")
        
        try:
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            
            if len(missing_keys) > 0:
                print(f"缺少的权重键数量: {len(missing_keys)}")
                if len(missing_keys) <= 10:
                    print(f"   缺少的键: {missing_keys}")
                else:
                    print(f"   前10个缺少的键: {missing_keys[:10]}")
            
            if len(unexpected_keys) > 0:
                print(f"多余的权重键数量: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 10:
                    print(f"   多余的键: {unexpected_keys}")
                else:
                    print(f"   前10个多余的键: {unexpected_keys[:10]}")
            
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                print("模型权重完美匹配加载成功")
            else:
                print("模型权重部分加载成功（存在不匹配的键）")
            print()
            
        except Exception as e:
            print(f"模型权重加载失败: {e}")
            print("将使用随机初始化的模型\n")
    else:
        print(f"警告: 未找到模型权重 {checkpoint_path}")
        print("将使用随机初始化的模型\n")
    
    model.eval()
    
    lpips_model = None
    if LPIPS_AVAILABLE:
        print("初始化 LPIPS 模型...")
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
        for param in lpips_model.parameters():
            param.requires_grad = False
            print("LPIPS 模型加载成功\n")
    else:
        print("LPIPS 不可用\n")
        
    print("="*80)
    print("开始测试...")
    print("="*80 + "\n")
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    results = []
    
    start_time = time.time()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing KITTI Video029")
        for idx, (input_frames, gt_frame, sample_info) in enumerate(pbar):
            video_name = sample_info['snow_video'][0]
            frame_idx = sample_info['frame_idx'].item()
            
            input_frames = input_frames.to(device)
            gt_frame = gt_frame.to(device)
            
            _, _, _, H_orig, W_orig = input_frames.shape
            
            input_frames_padded, (pad_h, pad_w) = pad_to_multiple(input_frames, multiple=8)
            
            output_padded = model(input_frames_padded)
            
            output = unpad(output_padded, pad_h, pad_w)
            
            psnr = calculate_psnr(output, gt_frame)
            ssim = calculate_ssim(output, gt_frame)
            lpips_val = calculate_lpips(output, gt_frame, lpips_model)
            
            total_psnr += psnr.item()
            total_ssim += ssim.item()
            total_lpips += lpips_val.item()
            
            results.append({
                'idx': idx + 1,
                'video': video_name,
                'frame': frame_idx,
                'size': f'{H_orig}×{W_orig}',
                'psnr': psnr.item(),
                'ssim': ssim.item(),
                'lpips': lpips_val.item()
            })
            
            pbar.set_postfix({
                'Size': f'{H_orig}×{W_orig}',
                'PSNR': f'{psnr.item():.2f}',
                'SSIM': f'{ssim.item():.4f}',
                'LPIPS': f'{lpips_val.item():.4f}'
            })
            
            output_img = tensor_to_image(output[0])
            video_output_dir = os.path.join(output_dir, 'output_images', video_name)
            os.makedirs(video_output_dir, exist_ok=True)
            output_save_path = os.path.join(video_output_dir, f'{frame_idx:05d}.png')
            output_img.save(output_save_path)
            
            mid_idx = num_frames // 2
            input_mid = input_frames[0, :, mid_idx, :, :]
            
            input_img = tensor_to_image(input_mid)
            gt_img = tensor_to_image(gt_frame[0])
            
            save_path = os.path.join(output_dir, 'comparison_images', f'comparison_{idx+1:04d}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_comparison(
                input_img, output_img, gt_img, save_path,
                (psnr.item(), ssim.item(), lpips_val.item())
            )
            
    elapsed_time = time.time() - start_time
    
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    avg_lpips = total_lpips / len(test_loader)
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    print(f"\n平均指标:")
    print(f"   PSNR:  {avg_psnr:.4f} dB")
    print(f"   SSIM:  {avg_ssim:.6f}")
    if LPIPS_AVAILABLE:
        print(f"   LPIPS: {avg_lpips:.6f}")
    print(f"\n总耗时: {elapsed_time:.2f}秒")
    print(f"平均速度: {elapsed_time/len(test_loader):.3f}秒/样本")
    print(f"\n结果保存在: {output_dir}/")
    print(f"   - 处理后图像: {output_dir}/output_images/{test_videos[0]}/ ({len(test_loader)} 张)")
    print(f"   - 对比图: {output_dir}/comparison_images/ ({len(test_loader)} 张)")
    print(f"图像尺寸: 完整原始尺寸（与输入一致，自动padding处理）")
    print("="*80 + "\n")

if __name__ == '__main__':
    test()