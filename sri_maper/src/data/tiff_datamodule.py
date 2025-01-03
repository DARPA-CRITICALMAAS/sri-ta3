from typing import Any, Optional, List
from functools import partial

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import sri_maper.src.data.tiff_dataset as dataset_utils
from sri_maper.src import utils

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
        window_size: int = 5,
        multiplier: int = 20,
        downsample: bool = True,
        oversample: bool = True,
        log_path: str = "/workspace/logs/",
        likely_neg_range: List[float] = [0.25,0.75],
        frac_train_split: float = 0.5,
        specified_split: Optional[List[List[float]]] = None,
        num_pca_components: Optional[int] = None,
        in_pca_space: Optional[bool] = False,
        store_all_unlabeled_csv: Optional[bool] = False,
        seed: int = 0,
    ) -> None:
        """Initialize a `TIFFDataModule`.

        :param tif_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param window_size: The window size. Defaults to `5`.
        :param multiplier: The multiplier for upsampling positives in the train_data split. Defaults to `20`.
        :param downsample: Whether to downsample the negative (i.e., unlabeled) class. Defaults to `True`.
        :param oversample: Whether to oversample the positive (i.e., known deposits) in train_data split. Defaults to `True`.
        :param log_path: The path to the log directory. Defaults to `"logs/"`.
        :param likely_neg_range: The range of values to consider as likely negatives. Defaults to `[0.25,0.75]`.
        :param frac_train_split: The fraction of the data to use for training. Defaults to `0.5`.
        :param specified_split: The coordinates to use for splitting the data. Defaults to `None`.
        :param num_pca_components: The number of PCA components to use. Defaults to `None`.
        :param in_pca_space: Whether to use PCA space to do downsampling. Defaults to `False`.
        :param seed: The random seed. Defaults to `0`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

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
                self.data_train = dataset_utils.TiffDataset(
                    tif_dir=self.hparams.tif_dir,
                    window_size=self.hparams.window_size,
                    stage=stage,
                    num_pca_components=self.hparams.num_pca_components,
                )
                # downsample to likely negatives
                if self.hparams.downsample:
                    def simple_feat_extractor(X, window_size):
                        if type(X) is torch.Tensor:
                            X = X.detach().cpu().numpy()
                        return X[:, window_size//2, window_size//2]
                    init_feat_extractor = partial(simple_feat_extractor, window_size=self.data_train.window_size)
                    self.data_train = dataset_utils.pu_downsample(
                        self.data_train,
                        init_feat_extractor,
                        multiplier=self.hparams.multiplier,
                        likely_neg_range=self.hparams.likely_neg_range,
                        seed=self.hparams.seed,
                        log_path=self.hparams.log_path,
                        in_pca_space=self.hparams.in_pca_space,
                        store_all_unlabeled_csv=self.hparams.store_all_unlabeled_csv,
                    )
                log.debug(f"Splitting base dataset into train / val / test.")
                if self.hparams.specified_split:
                    log.debug(f"Splitting using specified coordinates.")
                    self.data_train, self.data_val, self.data_test = dataset_utils.specified_split(self.data_train, pos_train_coordinates=self.hparams.specified_split, train_split=self.hparams.frac_train_split, seed=self.hparams.seed)
                else:
                    # random split
                    log.debug(f"Randomly splitting.")
                    if self.hparams.frac_train_split < 1.0:
                        self.data_train, self.data_val, self.data_test = dataset_utils.random_proportionate_split(self.data_train, train_split=self.hparams.frac_train_split, seed=self.hparams.seed)
                    else:
                        _, self.data_val, self.data_test = dataset_utils.random_proportionate_split(self.data_train, train_split=0.5, seed=self.hparams.seed)
                # oversample to balance
                self.data_train = dataset_utils.balance_data(self.data_train, multiplier=self.hparams.multiplier, oversample=self.hparams.oversample, seed=self.hparams.seed)
                dataset_utils.store_samples(self.data_train, self.hparams.log_path, "train")
                dataset_utils.store_samples(self.data_val, self.hparams.log_path, "valid")
                dataset_utils.store_samples(self.data_test, self.hparams.log_path, "test")
        elif stage == "predict":
            # loads datasets to produce a prediction map
            if not self.data_predict:
                self.data_predict = dataset_utils.TiffDataset(
                    tif_dir=self.hparams.tif_dir,
                    window_size=self.hparams.window_size,
                    stage=stage,
                    num_pca_components=self.hparams.num_pca_components
                )
                self.data_predict = dataset_utils.filter_by_bounds(self.data_predict)
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
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=self.hparams.num_workers > 0,
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
            # drop_last=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return DataLoader(
            dataset=self.data_predict,
            batch_size=128,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = TIFFDataModule()
