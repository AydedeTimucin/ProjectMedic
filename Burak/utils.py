from torch.utils.data import Dataset
from PIL import Image
class train_patch_dataset(Dataset):
    def __init__(self, wsi_paths, mask_paths, transform=None):
        self.wsi_paths = wsi_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.wsi_paths)

    def __getitem__(self, idx):
        wsi_path = self.wsi_paths[idx]
        mask_path = self.mask_paths[idx]

        wsi = Image.open(wsi_path)
        mask = Image.open(mask_path)

        if self.transform:
            wsi = self.transform(wsi)
            mask = self.transform(mask)

        return wsi, mask, idx
    
    