from typing import Any, Dict, Optional

from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset, random_split
from dust3r.datasets import get_data_loader
from dust3r.datasets import *

class MultiViewDUSt3RDataModule(LightningDataModule):
    """LightningDataModule for the custom dataset.

     A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    """
    def __init__(
        self,
        train_datasets: list[str],
        validation_datasets: list[str],
        batch_size_per_device: int = 64,
        num_workers: int = 12,
        pin_memory: bool = True,
    ) -> None:
        """Initialize a CustomDataModule.

        :param train_dataset: Path to the training dataset.
        :param test_dataset: Path to the testing dataset.
        :param batch_size: Batch size for training and evaluation.
        :param num_workers: Number of workers for data loading.
        :param pin_memory: Whether to pin memory.
        """
        super().__init__()

        self.train_datasets = train_datasets
        self.validation_datasets = validation_datasets
        self.batch_size_per_device = batch_size_per_device
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

    def prepare_data(self) -> None:
        """Download or prepare the dataset if needed."""
        # Implement any dataset preparation steps if needed.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`.
        """
        pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        # Assert every dataset is a string
        assert all(isinstance(dataset, str) for dataset in self.hparams.train_datasets), "All datasets must be strings"

        # Concatenate all train dataset strings into a single string with "+" separator
        train_datasets_concat = " + ".join(self.hparams.train_datasets)
        print("Building train Data loader for dataset: ", train_datasets_concat)
        self.train_loader = get_data_loader(
            train_datasets_concat,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_mem=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

        # Set epoch for train and validation loaders (if applicable)
        if hasattr(self.train_loader, "dataset") and hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(0)
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(0)

        return self.train_loader

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        assert all(isinstance(dataset, str) for dataset in self.hparams.validation_datasets), "All datasets must be strings"

        # Evaluate each string in the validation datasets list to get actual datasets
        val_datasets = [eval(dataset) for dataset in self.hparams.validation_datasets]

        # Create individual validation data loaders for each dataset
        val_loaders = [
            get_data_loader(
                dataset,
                batch_size=self.batch_size_per_device,
                num_workers=self.num_workers,
                pin_mem=self.pin_memory,
                shuffle=False,
                drop_last=False,  # set to False if you want to keep the last batch, e.g., for precise evaluation
            )
            for dataset in val_datasets
        ]

        # Set epoch for each validation loader (if applicable)
        for loader in val_loaders:
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(0)
            if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(0)

        # Combine the validation data loaders using CombinedLoader with 'sequential' mode
        # this will return a single dataloader that will iterate over all validation datasets sequentially
        # this way, each batch will only contain samples from a single dataset
        # this is important because the resolutions might be different between datasets
        print("Building validation CombinedLoader for datasets: ", self.hparams.validation_datasets)
        self.val_loader = CombinedLoader(val_loaders, mode='sequential')

        return self.val_loader

    # def test_dataloader(self) -> DataLoader[Any]:
    #     """Create and return the test dataloader.

    #     :return: The test dataloader.
    #     """
    #     return self.val_loader

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Generate and save the datamodule state.

        :return: A dictionary containing the datamodule state.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Reload datamodule state given datamodule `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = MultiViewDUSt3RDataModule(train_dataset="path/to/train", test_dataset="path/to/test")
