from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from sri_maper.src import utils
log = utils.get_pylogger(__name__)


class SSCMALitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
        ) -> None:
        """Initialize a `SSCMALitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model.
        :param gain: The weight on the positive class, helps with dataset inbalance.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # metric objects for calculating reconstruction ability of the model
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.val_psnr = PeakSignalNoiseRatio()

        self.test_ssim = StructuralSimilarityIndexMeasure()
        self.test_psnr = PeakSignalNoiseRatio()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation ssim/psnr
        self.val_ssim_best = MaxMetric()
        self.val_psnr_best = MaxMetric()

    def forward(
            self,
            x: torch.Tensor,
            pca_matrix: Union[torch.Tensor, None]=None,
        ) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: The input tensor for forward pass (i.e. window from the datacube).
        :param pca_matrix: The PCA matrix to apply to the input tensor. (optional)

        :return: A tensor of logits.
        """
        return self.net(x, pca_matrix)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        self.val_ssim.reset()
        self.val_psnr.reset()

        self.val_ssim_best.reset()
        self.val_psnr_best.reset()

    def compute_loss(
            self,
            img: torch.Tensor,
            pred: torch.Tensor,
            mask: torch.Tensor
        ) -> torch.Tensor:
        # calculates L2 loss
        loss = torch.pow(pred - img, 2).mean()
        return loss

    def model_step(
            self, batch: Tuple[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        img = batch[0]
        pca_matrix = batch[-1]
        pca_matrix = pca_matrix.detach().half() if len(pca_matrix.shape) != 1 else None

        img_input, pred_img, mask = self.forward(img, pca_matrix)
        loss = self.compute_loss(img_input, pred_img, mask)
        return loss, img_input.detach(), pred_img.detach(), mask.detach()

    def training_step(
            self, batch: Tuple[torch.Tensor], batch_idx: int
        ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, _, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss.item())
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_start(self) -> None:
        "Lightning hook that is called when a training epoch begins."
        pass

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, img, pred_img, _ = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss.item())
        if self.hparams.net.image_size < 11:
            self.val_ssim(F.interpolate(img, size=(11, 11)), \
                            F.interpolate(pred_img, size=(11, 11)))
        else:
            self.val_ssim(img, pred_img)
        self.val_psnr(img, pred_img)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", self.val_ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        ssim = self.val_ssim.compute()  # get current val ssim
        self.val_ssim_best(ssim)  # update best so far val ssim
        psnr = self.val_psnr.compute()  # get current val psnr
        self.val_psnr_best(psnr)  # update best so far val psnr

        self.log("val/ssim_best", self.val_ssim_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/psnr_best", self.val_psnr_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, img, pred_img, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss.item())
        if self.hparams.net.image_size < 11:
            self.test_ssim(F.interpolate(img, size=(11, 11)), \
                            F.interpolate(pred_img, size=(11, 11)))
        else:
            self.test_ssim(img, pred_img)
        self.test_psnr(img, pred_img)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ssim", self.test_ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/psnr", self.test_psnr, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single predict step on a batch of data from the predict set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        imgs, _, lons, lats, _, _, pca_matrix = batch
        img_input = torch.einsum('ijkl,ijm->imkl', imgs, pca_matrix.detach().half()) if len(pca_matrix.shape) != 1 else imgs
        
        feats, _, _ = self.net.encoder(img_input)
        feats = feats[:,0,:].detach().cpu().squeeze()

        return torch.concat((torch.stack((lons.detach().cpu(), lats.detach().cpu()), dim=1), feats), dim=1)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SSCMALitModule(None, None, None, None)
