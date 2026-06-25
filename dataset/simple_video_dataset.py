import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SimpleVideoDataset(Dataset):
    def __init__(self, data_root, split='train', window_size=5, 
                 crop_size=224, is_testing=False, full_size=False):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.window_size = window_size
        self.crop_size = crop_size
        self.is_testing = is_testing
        self.full_size = full_size
        
        self.snow_dir = os.path.join(data_root, split, 'snow')
        self.clean_dir = os.path.join(data_root, split, 'clean')
        
        self.samples = []
        self._scan_videos()
        
    def _scan_videos(self):
        video_dirs = sorted([d for d in os.listdir(self.snow_dir) 
                            if os.path.isdir(os.path.join(self.snow_dir, d))])
        
        for video_id in video_dirs:
            snow_video_dir = os.path.join(self.snow_dir, video_id)
            clean_video_dir = os.path.join(self.clean_dir, video_id)
            
            frames = sorted([f for f in os.listdir(snow_video_dir) if f.endswith('.jpg')])
            
            for i in range(len(frames) - self.window_size + 1):
                window_frames = frames[i:i + self.window_size]
                mid_idx = self.window_size // 2
                
                sample = {
                    'video_id': video_id,
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
        
        if not self.is_testing and random.random() > 0.5:
            snow_frames = [f.transpose(Image.FLIP_LEFT_RIGHT) for f in snow_frames]
            clean_frame = clean_frame.transpose(Image.FLIP_LEFT_RIGHT)
        
        if not self.full_size:
            w, h = snow_frames[0].size
            if w > self.crop_size and h > self.crop_size:
                if not self.is_testing:
                    left = random.randint(0, w - self.crop_size)
                    top = random.randint(0, h - self.crop_size)
                else:
                    left = (w - self.crop_size) // 2
                    top = (h - self.crop_size) // 2
                    
                snow_frames = [f.crop((left, top, left + self.crop_size, top + self.crop_size)) 
                              for f in snow_frames]
                clean_frame = clean_frame.crop((left, top, left + self.crop_size, top + self.crop_size))
        
        to_tensor = transforms.ToTensor()
        snow_tensors = [to_tensor(f) for f in snow_frames]
        clean_tensor = to_tensor(clean_frame)
        
        input_frames = torch.stack(snow_tensors, dim=1)
        
        return input_frames, clean_tensor


def get_simple_video_dataloader(data_root, split='train', batch_size=4, 
                                 window_size=5, crop_size=224, is_testing=False,
                                 num_workers=4, full_size=False):
    dataset = SimpleVideoDataset(
        data_root=data_root,
        split=split,
        window_size=window_size,
        crop_size=crop_size,
        is_testing=False,
        full_size=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train' and not is_testing),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader