from typing import Any, Dict, Tuple

import torch
import math
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from captum.attr import IntegratedGradients

from src import utils
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
        mc_samples: int,
        warmup_epoch: int,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param gain: The weight on the positive class, helps with dataset inbalance.
        """
        super().__init__()
        # self.example_input_array = torch.Tensor(16, 23, 33, 33)
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net

        # loss function
        # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.hparams.gain))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

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
        self, img: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        
        # calculates L1 loss
        # loss = torch.abs(pred[mask==1] - img[mask==1]).mean()
        # calculates L2 loss
        loss = torch.pow(pred[mask==1] - img[mask==1], 2).mean()
        # calculates SSIM loss
        # loss += 0.75 * (1.0 - ssim(torch.where(mask == 1, img, 0.0), torch.where(mask == 1, pred, 0.0), reduction="sum") / mask.sum())
        
        return loss

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        img, _ = batch
        pred_img, mask = self.forward(img)
        loss = self.compute_loss(img, pred_img, mask)
        
        return loss, img, pred_img, mask

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
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

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, img, pred_img, mask = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss.item())
        self.val_ssim(img.detach() * mask.detach(), pred_img.detach() * mask.detach())
        self.val_psnr(img.detach() * mask.detach(), pred_img.detach() * mask.detach())

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

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, img, pred_img, mask = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss.item())
        self.test_ssim(img.detach() * mask.detach(), pred_img.detach() * mask.detach())
        self.test_psnr(img.detach() * mask.detach(), pred_img.detach() * mask.detach())

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ssim", self.test_ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/psnr", self.test_psnr, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single predict step on a batch of data from the predict set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # extracts feature attributions
        ig = IntegratedGradients(self.net)
        attribution = ig.attribute(batch[0].requires_grad_(), n_steps=50).mean(dim=(-1,-2))

        # enables Monte Carlo Dropout
        self.net.activate_dropout()

        # generates MC samples
        preds = torch.sigmoid(
            self.forward(
                batch[0].tile((self.hparams.mc_samples,1,1,1))
            ).reshape(self.hparams.mc_samples,-1)
        ).detach()

        # computes mean and std of MC samples
        means = preds.mean(dim=0).squeeze()
        stds = preds.std(dim=0).squeeze()
        
        return torch.concat((torch.stack((batch[2], batch[3], means, stds), dim=1), attribution), dim=1)
        
    def on_predict_epoch_end(self, results):
        self.trainer.results = torch.vstack(results[0]).cpu().numpy()

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
            lr_func = lambda epoch: min((epoch + 1) / (self.hparams.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / self.trainer.max_epochs * math.pi) + 1))
            scheduler = self.hparams.scheduler(optimizer=optimizer, lr_lambda=lr_func)
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