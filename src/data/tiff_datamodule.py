from typing import Any, Optional, List

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.tiff_dataset import TiffDataset, spatial_cross_val_split, filter_by_bounds
from src import utils

log = utils.get_pylogger(__name__)


class TIFFDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

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

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        tif_dir: str = "/workspace/data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        patch_size: int = 33,
        predict_bounds: Optional[List[str]] = None,
        uscan_only: bool = False,
        test_set: int = 2,
        val_set: int = 3
    ) -> None:
        """Initialize a `TIFFDataModule`.

        :param tif_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param uscan_only: Whether to use US/Canada or US/Canada/Australia data. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations - we might use this later with custom tronsforms,
        # default 3 band RGB image transform WILL NOT NECESSARILY WORK
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if stage in ["fit","validate","test"]:
            # loads and splits datasets for train / val / test
            if not self.data_train or not self.data_val or not self.data_test:
                log.debug(f"Instantiating base dataset.")
                self.data_train = TiffDataset(
                    tif_dir=self.hparams.tif_dir, 
                    patch_size=self.hparams.patch_size,
                    stage=stage,
                    uscan_only=self.hparams.uscan_only,
                )
                log.debug(f"Splitting base dataset into train / val / test.")
                self.data_train, self.data_val, self.data_test = spatial_cross_val_split(self.data_train, k=6, test_set=self.hparams.test_set, val_set=self.hparams.val_set)
                log.info(f"Spatial cross val ouput: train pos - {self.data_train.valid_patches[:,2].sum()}, train neg - {len(self.data_train)-self.data_train.valid_patches[:,2].sum()}.")
                log.info(f"Spatial cross val ouput: val pos - {self.data_val.valid_patches[:,2].sum()}, val neg - {len(self.data_val)-self.data_val.valid_patches[:,2].sum()}.")
                log.info(f"Spatial cross val ouput: test pos - {self.data_test.valid_patches[:,2].sum()}, test neg - {len(self.data_test)-self.data_test.valid_patches[:,2].sum()}.")
        elif stage == "predict":
            # loads datasets to produce a prediction map
            if not self.data_predict:
                self.data_predict = TiffDataset(
                    tif_dir=self.hparams.tif_dir, 
                    patch_size=self.hparams.patch_size,
                    stage=stage,
                    uscan_only=self.hparams.uscan_only,
                )
                self.data_predict = filter_by_bounds(self.data_predict, self.hparams.predict_bounds)
                log.info(f"Used bounds to filter patches - number of patches {len(self.data_predict)}.")
        else:
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = TIFFDataModule()