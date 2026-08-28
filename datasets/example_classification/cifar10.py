from pathlib import Path

from omegaconf import DictConfig

import torchvision
import torchvision.transforms as transforms


"""
The dataset you want to define is a `torch.utils.data.Dataset` object. 
In our example we inherit from a built in dataset from pytorch instead.
"""

class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, cfg: DictConfig, split="training"):
        """
        A example dataset class for CIFAR10 image classification. All datasets should have the same arguments.
        Args:
            cfg: a DictConfig object defined by `configurations/dataset/example_cifar10.yaml`.
            split: a string indicating which split of the dataset to use. typically "training", "validation", or "test".
        """

        self.cfg = cfg
        self.mean = cfg.mean
        self.std = cfg.std
        self.debug = cfg.debug
        if split == "training":
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
            train = True
        elif split == "test" or split == "validation":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
            train = False
        else:
            raise ValueError(f"split {split} not available for cifar 10s")

        data_path = Path(cfg.data_dir)
        if not data_path.is_absolute():
            try:
                from hydra.utils import get_original_cwd

                base_dir = Path(get_original_cwd())
            except ValueError:
                # Hydra not initialized (e.g. unit tests); fall back to repo root.
                base_dir = Path(__file__).resolve().parents[2]
            data_path = base_dir / data_path

        # torchvision expects `root` to contain `cifar-10-batches-py/`.
        root = data_path.parent if data_path.name == self.base_folder else data_path
        dataset_dir = root / self.base_folder
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"CIFAR-10 dataset not found at {dataset_dir}. "
                "Update dataset.data_dir in configurations/dataset/example_cifar10.yaml."
            )

        super().__init__(root=str(root), train=train, download=False, transform=transform)
