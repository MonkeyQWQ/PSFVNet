import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.ultra_video_snow_net_v3_optimized import build_ultra_v3_optimized

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

class Dataset2FullSizeDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_root, split='val', window_size=5):
        super().__init__()
        
        self.data_root = data_root
        self.split = split
        self.window_size = window_size
        
        self.snow_dir = os.path.join(data_root, split, 'snow')
        self.clean_dir = os.path.join(data_root, split, 'clean')
        
        self.samples = []
        self._scan_videos()
        
        print(f"✅ Dataset2FullSizeDataset ({split}):")
        print(f"   数据目录: {os.path.join(data_root, split)}")
        print(f"   样本数量: {len(self.samples)}")
        print(f"   测试模式: 完整尺寸（不裁剪）")
        
    def _scan_videos(self):
        if not os.path.exists(self.snow_dir):
            print(f"❌ 错误: 目录不存在 {self.snow_dir}")
            return
            
        video_dirs = sorted([d for d in os.listdir(self.snow_dir) 
                            if os.path.isdir(os.path.join(self.snow_dir, d))])
        
        for video_id in video_dirs:
            snow_video_dir = os.path.join(self.snow_dir, video_id)
            clean_video_dir = os.path.join(self.clean_dir, video_id)
            
            frames = sorted([f for f in os.listdir(snow_video_dir) 
                           if f.endswith(('.jpg', '.png'))])
            
            for i in range(len(frames) - self.window_size + 1):
                window_frames = frames[i:i + self.window_size]
                mid_idx = self.window_size // 2
                
                sample = {
                    'video_id': video_id,
                    'frame_idx': i,
                    'snow_frames': [os.path.join(snow_video_dir, f) for f in window_frames],
                    'clean_frame': os.path.join(clean_video_dir, window_frames[mid_idx])
                }
                self.samples.append(sample)
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        snow_frames = [Image.open(path).convert('RGB') for path in sample['snow_frames']]
        clean_frame = Image.open(sample['clean_frame']).convert('RGB')
        
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        
        snow_tensors = [to_tensor(f) for f in snow_frames]
        clean_tensor = to_tensor(clean_frame)
        
        input_frames = torch.stack(snow_tensors, dim=1)
        
        info = {
            'video_id': sample['video_id'],
            'frame_idx': sample['frame_idx'],
            'size': (clean_frame.height, clean_frame.width)
        }
        
        return input_frames, clean_tensor, info

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
    is_5d = tensor.dim() == 5
    
    if is_5d:
        B, C, T, H, W = tensor.shape
    else:
        B, C, H, W = tensor.shape
        
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    
    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0)
        
    padding = (0, pad_w, 0, pad_h)
    
    if is_5d:
        tensor_4d = tensor.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        padded_4d = F.pad(tensor_4d, padding, mode='reflect')
        _, _, H_pad, W_pad = padded_4d.shape
        padded = padded_4d.reshape(B, T, C, H_pad, W_pad).permute(0, 2, 1, 3, 4)
    else:
        padded = F.pad(tensor, padding, mode='reflect')
        
    return padded, (pad_h, pad_w)

def unpad(tensor, pad_h, pad_w):
    if pad_h == 0 and pad_w == 0:
        return tensor
        
    _, _, H, W = tensor.shape
    return tensor[:, :, :H-pad_h, :W-pad_w]

def tensor_to_image(tensor):
    img = tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def save_comparison(input_img, output_img, gt_img, save_path, metrics, info):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(input_img)
    axes[0].set_title('Input (Snow)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(output_img)
    axes[1].set_title('Output (Desnowed)', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(gt_img)
    axes[2].set_title('Ground Truth (Clean)', fontsize=14)
    axes[2].axis('off')
    
    psnr, ssim, lpips_val = metrics
    h, w = info['size']
    video_id = info['video_id']
    frame_idx = info['frame_idx']
    
    metrics_text = f"PSNR: {psnr:.4f} dB  |  SSIM: {ssim:.6f}  |  LPIPS: {lpips_val:.6f}\n"
    metrics_text += f"Video: {video_id}  |  Frame: {frame_idx}  |  Size: {h}×{w}"
    
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
             
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def test():
    data_root = './rvsd'
    checkpoint_path = './'
    output_dir = './test_results'
    
    num_frames = 5
    base_channels = 48
    num_blocks = [2, 2, 4]
    num_heads = [1, 2, 4]
    
    save_all_results = True
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("🚀 Dataset2 完整尺寸测试")
    print("="*80)
    print(f"\n📁 配置:")
    print(f"   数据目录: {data_root}")
    print(f"   模型权重: {checkpoint_path}")
    print(f"   输出目录: {output_dir}")
    print(f"   处理模式: 完整尺寸（不裁剪）")
    print(f"   保存策略: 保存所有结果，按视频分文件夹")
    print("="*80 + "\n")
    
    print("📥 加载测试数据（完整尺寸）...")
    
    dataset = Dataset2FullSizeDataset(
        data_root=data_root,
        split='val',
        window_size=num_frames
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✅ 测试数据加载成功，共 {len(test_loader)} 个样本\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}\n")
    
    print("🏗️  创建模型...")
    model = build_ultra_v3_optimized(
        num_frames=num_frames,
        base_channels=base_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        use_checkpoint=False
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"📥 加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"   Checkpoint信息: Epoch {checkpoint['epoch']}, "
                      f"PSNR {checkpoint.get('psnr', 0):.4f} dB")
        else:
            model.load_state_dict(checkpoint)
            
        print("✅ 模型权重加载成功\n")
    else:
        print(f"⚠️  警告: 未找到模型权重 {checkpoint_path}")
        print("将使用随机初始化的模型\n")
        
    model.eval()
    
    lpips_model = None
    if LPIPS_AVAILABLE:
        print("📐 初始化 LPIPS 模型...")
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
        for param in lpips_model.parameters():
            param.requires_grad = False
        print("✅ LPIPS 模型加载成功\n")
    else:
        print("⚠️  LPIPS 不可用（安装: pip install lpips）\n")
        
    print("="*80)
    print("🔍 开始测试...")
    print("="*80 + "\n")
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    
    results = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for idx, (input_frames, gt_frame, info) in enumerate(pbar):
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
            
            result = {
                'idx': idx + 1,
                'video_id': info['video_id'][0],
                'frame_idx': info['frame_idx'][0].item(),
                'size': (H_orig, W_orig),
                'psnr': psnr.item(),
                'ssim': ssim.item(),
                'lpips': lpips_val.item()
            }
            results.append(result)
            
            pbar.set_postfix({
                'Video': info['video_id'][0],
                'Size': f'{H_orig}×{W_orig}',
                'PSNR': f'{psnr.item():.2f}',
                'SSIM': f'{ssim.item():.4f}',
                'LPIPS': f'{lpips_val.item():.4f}'
            })
            
            if save_all_results:
                video_id = info['video_id'][0]
                video_output_dir = os.path.join(output_dir, video_id)
                os.makedirs(video_output_dir, exist_ok=True)
                
                output_img = tensor_to_image(output[0])
                frame_idx = info['frame_idx'][0].item()
                
                output_path = os.path.join(video_output_dir, f'{frame_idx:05d}.jpg')
                output_img.save(output_path, quality=95)
                
                if idx < 10:
                    mid_idx = num_frames // 2
                    input_mid = input_frames[0, :, mid_idx, :, :]
                    input_img = tensor_to_image(input_mid)
                    gt_img = tensor_to_image(gt_frame[0])
                    
                    comparison_path = os.path.join(video_output_dir, 
                                                  f'{frame_idx:05d}_comparison.png')
                    save_comparison(
                        input_img, output_img, gt_img, comparison_path,
                        (psnr.item(), ssim.item(), lpips_val.item()),
                        {'video_id': video_id, 
                         'frame_idx': frame_idx,
                         'size': (H_orig, W_orig)}
                    )
                    
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    avg_lpips = total_lpips / len(test_loader)
    
    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("="*80)
    print(f"\n📊 平均指标:")
    print(f"   PSNR:  {avg_psnr:.4f} dB")
    print(f"   SSIM:  {avg_ssim:.6f}")
    print(f"   LPIPS: {avg_lpips:.6f}")
    
    best_sample = max(results, key=lambda x: x['psnr'])
    worst_sample = min(results, key=lambda x: x['psnr'])
    
    print(f"\n🏆 最佳样本:")
    print(f"   Video {best_sample['video_id']} Frame {best_sample['frame_idx']}: "
          f"PSNR={best_sample['psnr']:.4f} dB, SSIM={best_sample['ssim']:.6f}")
          
    print(f"\n⚠️  最差样本:")
    print(f"   Video {worst_sample['video_id']} Frame {worst_sample['frame_idx']}: "
          f"PSNR={worst_sample['psnr']:.4f} dB, SSIM={worst_sample['ssim']:.6f}")
          
    report_path = os.path.join(output_dir, 'test_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Dataset2 完整尺寸测试报告\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"数据目录: {data_root}\n")
        f.write(f"模型权重: {checkpoint_path}\n")
        f.write(f"测试样本数: {len(test_loader)}\n")
        f.write(f"测试模式: 完整尺寸（不裁剪）\n\n")
        
        f.write(f"平均指标:\n")
        f.write(f"  PSNR:  {avg_psnr:.4f} dB\n")
        f.write(f"  SSIM:  {avg_ssim:.6f}\n")
        f.write(f"  LPIPS: {avg_lpips:.6f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("详细结果:\n")
        f.write("="*80 + "\n")
        f.write(f"{'序号':<6} {'视频ID':<10} {'帧号':<6} {'尺寸':<12} {'PSNR (dB)':<12} {'SSIM':<12} {'LPIPS':<12}\n")
        f.write("-"*80 + "\n")
        
        for res in results:
            f.write(f"{res['idx']:<6} {res['video_id']:<10} {res['frame_idx']:<6} "
                   f"{res['size'][0]}×{res['size'][1]:<6} "
                   f"{res['psnr']:<12.4f} {res['ssim']:<12.6f} {res['lpips']:<12.6f}\n")
                   
    video_ids = set([res['video_id'] for res in results])
    num_videos = len(video_ids)
    
    print(f"\n📄 详细报告保存在: {report_path}")
    print(f"📁 测试结果保存在: {output_dir}/")
    print(f"   - 共 {num_videos} 个视频文件夹")
    print(f"   - 每个视频的去雪结果独立保存")
    print(f"   - 前10个样本额外保存对比图")
    print(f"💾 图像尺寸: 完整原始尺寸（未裁剪）")
    print("\n📂 输出结构:")
    print(f"   {output_dir}/")
    for vid in sorted(list(video_ids))[:3]:
        print(f"   ├── {vid}/")
        print(f"   │   ├── 00000.jpg  (去雪结果)")
        print(f"   │   ├── 00001.jpg")
        print(f"   │   └── ...")
    if num_videos > 3:
        print(f"   └── ... (共{num_videos}个视频文件夹)")
    print("="*80 + "\n")

if __name__ == '__main__':
    test()