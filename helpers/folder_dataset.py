import os

from PIL import Image
from torch.utils.data import Dataset


class FolderImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.samples = []

        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(self.extensions):
                    full_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(full_path, root_dir)
                    self.samples.append((full_path, rel_path))

        self.samples.sort(key=lambda item: item[1].replace("\\", "/").lower())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, rel_path = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, {"image_path": path, "relative_path": rel_path}
