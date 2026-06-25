import os
import sys
import time
import logging
import warnings
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from PIL import Image
import PIL.PngImagePlugin

warnings.filterwarnings('ignore')
PIL.PngImagePlugin.logger.setLevel(logging.ERROR)

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.ultra_video_snow_net_v3_optimized import build_ultra_v3_optimized

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset'))
from kitti_snow_dataset import get_kitti_dataloader

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  警告: lpips 未安装，将跳过LPIPS计算")

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_v3_optimized_kitti_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def calculate_psnr(img1, img2, max_value=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_value / torch.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, 1, padding=window_size//2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, 1, padding=window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.avg_pool2d(img1 * img1, window_size, 1, padding=window_size//2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 * img2, window_size, 1, padding=window_size//2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, window_size, 1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        return 1 - calculate_ssim(img1, img2, self.window_size)

def calculate_lpips(img1, img2, lpips_model):
    if not LPIPS_AVAILABLE or lpips_model is None:
        return torch.tensor(0.0)

    img1_normalized = img1 * 2 - 1
    img2_normalized = img2 * 2 - 1

    with torch.no_grad():
        lpips_value = lpips_model(img1_normalized, img2_normalized)

    return lpips_value.mean()

class UltraLossV3Optimized(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()

    def forward(self, outputs, gt, input_frame):
        fine_out, coarse_out, atm, trans, density = outputs

        loss_final = self.l1_loss(fine_out, gt) + 0.15 * self.ssim_loss(fine_out, gt)

        loss_coarse = self.l1_loss(coarse_out, gt) + 0.1 * self.ssim_loss(coarse_out, gt)

        B, C, H, W = gt.shape
        atm_expanded = atm
        trans_expanded = trans.expand(B, 1, H, W)

        physical_out = (input_frame - atm_expanded * (1 - trans_expanded)) / (trans_expanded + 1e-6)
        physical_out = torch.clamp(physical_out, 0, 1)

        loss_physical = 0.1 * self.l1_loss(coarse_out, physical_out)

        total_loss = (
            1.0 * loss_final +
            0.5 * loss_coarse +
            0.1 * loss_physical
        )

        loss_dict = {
            'total': total_loss,
            'final': loss_final,
            'coarse': loss_coarse,
            'physical': loss_physical
        }

        return total_loss, loss_dict

def validate(model, test_loader, device, lpips_model=None, max_batches=None):
    model.eval()

    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    count = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="🔍 验证中", leave=False)
        for batch_idx, (input_frames, gt_frame) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_frames = input_frames.to(device)
            gt_frame = gt_frame.to(device)

            output = model(input_frames)

            psnr = calculate_psnr(output, gt_frame)
            ssim = calculate_ssim(output, gt_frame)
            lpips_val = calculate_lpips(output, gt_frame, lpips_model)

            total_psnr += psnr.item()
            total_ssim += ssim.item()
            total_lpips += lpips_val.item()
            count += 1

            postfix = {'PSNR': f'{psnr.item():.2f}', 'SSIM': f'{ssim.item():.4f}'}
            if LPIPS_AVAILABLE and lpips_model is not None:
                postfix['LPIPS'] = f'{lpips_val.item():.4f}'
            pbar.set_postfix(postfix)

    avg_psnr = total_psnr / count if count > 0 else 0
    avg_ssim = total_ssim / count if count > 0 else 0
    avg_lpips = total_lpips / count if count > 0 else 0

    model.train()
    return avg_psnr, avg_ssim, avg_lpips

def train():
    data_root = './KITTI_snow'
    num_frames = 5
    epochs = 150
    batch_size = 4
    learning_rate = 2e-4
    base_channels = 48
    num_blocks = [2, 2, 4]
    num_heads = [1, 2, 4]
    use_checkpoint = True
    use_amp = True
    
    eval_interval = 1
    save_interval = 10
    val_max_batches = None
    
    checkpoint_dir = './checkpoint/1'
    log_dir = './train_log'
    
    resume_training = False

    print("\n" + "=" * 80)
    print("📊 训练配置")
    print("=" * 80)
    print(f"📁 数据目录: {data_root}")
    print(f"📦 数据集: KITTI_snow (视频序列)")
    print(f"📊 Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🎯 学习率: {learning_rate}")
    print(f"🔧 基础通道数: {base_channels}")
    print(f"🧱 网络深度: {num_blocks}")
    print(f"👁️  注意力头数: {num_heads}")
    print(f"💾 梯度检查点: {use_checkpoint}")
    print(f"⚡ 混合精度: {use_amp}")
    print(f"✅ 每轮验证: eval_interval={eval_interval}")
    print(f"💾 保存间隔: save_interval={save_interval}")
    print(f"🔄 继续训练: {resume_training}")
    print(f"💾 最优模型: 每次新最优都单独保存（不删除旧模型）✅✅✅")
    print("=" * 80 + "\n")
    
    log_file = setup_logging(log_dir)
    logging.info("UltraVideoSnowNet V3-Optimized KITTI训练开始")

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("📥 加载训练数据...")
    train_loader = get_kitti_dataloader(
        data_root=data_root,
        split='train',
        batch_size=batch_size,
        window_size=num_frames,
        crop_size=224,
        is_testing=False,
        num_workers=4
    )
    print(f"✅ 训练数据加载成功，共 {len(train_loader)} 个batch\n")
    
    print("📥 加载验证数据...")
    test_loader = get_kitti_dataloader(
        data_root=data_root,
        split='test',
        batch_size=1,
        window_size=num_frames,
        crop_size=224,
        is_testing=True,
        num_workers=2
    )
    print(f"✅ 验证数据加载成功，共 {len(test_loader)} 个batch\n")

    print("🏗️  创建模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_ultra_v3_optimized(
        num_frames=num_frames,
        base_channels=base_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        use_checkpoint=use_checkpoint
    ).to(device)
    
    print("✅ 模型创建成功")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型参数量: {total_params / 1e6:.2f}M\n")
    
    lpips_model = None
    if LPIPS_AVAILABLE:
        print("📐 初始化 LPIPS 模型...")
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
        for param in lpips_model.parameters():
            param.requires_grad = False
        print("✅ LPIPS 模型加载成功\n")
    else:
        print("⚠️  LPIPS 不可用，将跳过LPIPS计算\n")

    criterion = UltraLossV3Optimized()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = GradScaler() if use_amp else None

    start_epoch = 0
    best_psnr = 0
    
    if resume_training:
        latest_path = os.path.join(checkpoint_dir, 'latest.pth')
        if os.path.exists(latest_path):
            print(f"📥 加载检查点: {latest_path}")
            try:
                checkpoint_data = torch.load(latest_path)
                model.load_state_dict(checkpoint_data['model_state_dict'])
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                start_epoch = checkpoint_data['epoch']
                best_psnr = checkpoint_data.get('best_psnr', 0)
                print(f"✅ 成功加载检查点，从 epoch {start_epoch} 继续训练")
                print(f"📊 当前最佳PSNR: {best_psnr:.2f} dB\n")
            except Exception as e:
                logging.warning(f"⚠️  加载检查点失败: {e}")
                logging.warning("将从头开始训练...\n")
                start_epoch = 0
        else:
            logging.info("🆕 未找到latest.pth，从头开始训练\n")
    else:
        logging.info("🆕 从头开始训练（不加载checkpoint）\n")

    logging.info(f"开始训练，共 {epochs} 轮，从 epoch {start_epoch} 开始")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        epoch_lpips = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        for batch_idx, (input_frames, gt_frame) in enumerate(pbar):
            input_frames = input_frames.to(device)
            gt_frame = gt_frame.to(device)
            
            mid_idx = num_frames // 2
            input_mid_frame = input_frames[:, :, mid_idx, :, :]
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    outputs = model(input_frames, return_intermediate=True)
                    loss, loss_dict = criterion(outputs, gt_frame, input_mid_frame)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_frames, return_intermediate=True)
                loss, loss_dict = criterion(outputs, gt_frame, input_mid_frame)
                loss.backward()
                optimizer.step()
            
            final_out = outputs[0]
            psnr = calculate_psnr(final_out, gt_frame)
            ssim = calculate_ssim(final_out, gt_frame)
            lpips_val = calculate_lpips(final_out, gt_frame, lpips_model)
            
            epoch_loss += loss.item()
            epoch_psnr += psnr.item()
            epoch_ssim += ssim.item()
            epoch_lpips += lpips_val.item()
            
            current_lr = optimizer.param_groups[0]['lr']
            postfix = {
                'Loss': f"{loss.item():.4f}",
                'LR': f"{current_lr:.6f}",
                'PSNR': f"{psnr.item():.2f}",
                'SSIM': f"{ssim.item():.4f}"
            }
            if LPIPS_AVAILABLE and lpips_model is not None:
                postfix['LPIPS'] = f"{lpips_val.item():.4f}"
            pbar.set_postfix(postfix)
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = epoch_psnr / len(train_loader)
        avg_ssim = epoch_ssim / len(train_loader)
        avg_lpips = epoch_lpips / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n{'='*80}")
        print(f"✅ Epoch [{epoch+1}/{epochs}] 完成")
        if LPIPS_AVAILABLE wildlife and lpips_model is not None:
            print(f"📊 Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
        else:
            print(f"📊 Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")
        print(f"{'='*80}\n")
        
        if LPIPS_AVAILABLE and lpips_model is not None:
            logging.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
        else:
            logging.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
        
        latest_path = os.path.join(checkpoint_dir, 'latest.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'best_psnr': best_psnr
        }, latest_path)
        
        if (epoch + 1) % save_interval == 0:
            for old_file in os.listdir(checkpoint_dir):
                if old_file.startswith('checkpoint_') and old_file.endswith('.pth'):
                    os.remove(os.path.join(checkpoint_dir, old_file))
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch+1:03d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'best_psnr': best_psnr
            }, checkpoint_path)
            print(f"💾 检查点已保存: checkpoint_{epoch+1:03d}.pth\n")
            logging.info(f"Checkpoint saved: checkpoint_{epoch+1:03d}.pth")
        
        if (epoch + 1) % eval_interval == 0:
            print(f"🔍 开始验证 (Epoch {epoch+1})...")
            val_psnr, val_ssim, val_lpips = validate(model, test_loader, device, lpips_model, max_batches=val_max_batches)
            
            if LPIPS_AVAILABLE and lpips_model is not None:
                print(f"✅ 验证完成 - 平均PSNR: {val_psnr:.3f} dB | 平均SSIM: {val_ssim:.4f} | 平均LPIPS: {val_lpips:.4f}")
                print(f"📊 当前 PSNR: {val_psnr:.3f} | SSIM: {val_ssim:.4f} | LPIPS: {val_lpips:.4f} (最佳PSNR: {best_psnr:.3f})\n")
                logging.info(f"Validation - PSNR: {val_psnr:.3f} dB, SSIM: {val_ssim:.4f}, LPIPS: {val_lpips:.4f}")
            else:
                print(f"✅ 验证完成 - 平均PSNR: {val_psnr:.3f} dB | 平均SSIM: {val_ssim:.4f}")
                print(f"📊 当前 PSNR: {val_psnr:.3f} | SSIM: {val_ssim:.4f} (最佳PSNR: {best_psnr:.3f})\n")
                logging.info(f"Validation - PSNR: {val_psnr:.3f} dB, SSIM: {val_ssim:.4f}")
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                
                best_model_path = os.path.join(
                    checkpoint_dir,
                    f'BEST_MODEL_epoch{epoch+1:03d}_PSNR{val_psnr:.3f}.pth'
                )
                torch.save(model.state_dict(), best_model_path)
                
                print(f"🎉 新的最优模型! PSNR: {val_psnr:.3f} dB")
                print(f"💾 已保存: {os.path.basename(best_model_path)}")
                print(f"📝 注意: 旧的最优模型已保留，不删除\n")
                logging.info(f"New best model saved with PSNR: {val_psnr:.3f} dB at {best_model_path}")
    
    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print("=" * 80)
    print(f"📊 最佳验证PSNR: {best_psnr:.3f} dB")
    print(f"📁 模型保存目录: {checkpoint_dir}")
    print(f"📝 日志文件: {log_file}")
    print(f"💾 所有最优模型已保存（未删除旧模型）")
    print("=" * 80)
    
    logging.info("训练完成")
    logging.info(f"最佳验证PSNR: {best_psnr:.3f} dB")

def main():
    print("\n" + "=" * 80)
    print("🚀 UltraVideoSnowNet V3-Optimized - KITTI_snow 数据集训练")
    print("=" * 80)
    print("\n" + "=" * 80 + "\n")
    
    try:
        train()
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        logging.info("训练被用户中断")
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        logging.error(f"训练出错: {e}", exc_info=True)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()