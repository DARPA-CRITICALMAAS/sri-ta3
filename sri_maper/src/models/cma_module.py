from typing import Any, Dict, Tuple

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, MulticlassAccuracy, BinaryAccuracy, BinaryMatthewsCorrCoef, BinaryF1Score
from captum.attr import IntegratedGradients
import pandas as pd

from sri_maper.src import utils
log = utils.get_pylogger(__name__)


class CMALitModule(LightningModule):
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
        gain: float,
        mc_samples: int,
        smoothing: float,
        threshold: float = 0.5,
        temperature: float = 1.0,
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
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.hparams.gain))

        # metric objects for calculating and averaging AUC across batches
        self.val_auc = BinaryAUROC(thresholds=None)
        self.test_auc = BinaryAUROC(thresholds=None)

        # metric objects for calculating and averaging area under
        # the precision-recall curve (AUPRC) across batches
        self.val_auprc = BinaryAveragePrecision(thresholds=None)
        self.test_auprc = BinaryAveragePrecision(thresholds=None)

        # additional metrics
        self.test_bal_acc = MulticlassAccuracy(num_classes=2, average="macro")
        self.test_acc = BinaryAccuracy(threshold=self.hparams.threshold)
        self.test_mcc = BinaryMatthewsCorrCoef(threshold=self.hparams.threshold)
        self.test_f1 = BinaryF1Score(threshold=self.hparams.threshold)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation AUC
        self.val_auc_best = MaxMetric()
        self.val_auprc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: The input tensor for forward pass (i.e. window from the datacube).

        :return: A tensor of logits.
        """
        return self.net(x)
    
    def calibrated_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a calibrated forward pass through the model `self.net`.

        :param x: The input tensor for forward pass (i.e. window from the datacube).

        :return: A tensor of calibrated logits.
        """
        return self.net(x) / torch.tensor(self.hparams.temperature).to(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_auc.reset()
        self.val_auc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], calibrated: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing (in order) the input tensor, target
            labels.
        :param calibrated: A flag to indicate whether to use a calibrated forward pass.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        if calibrated:
            logits = self.calibrated_forward(x)
        else:
            logits = self.forward(x)
        loss = self.criterion(logits, y.unsqueeze(1) * (1.0 - self.hparams.smoothing) + 0.5 * self.hparams.smoothing)
        preds = torch.sigmoid(logits)
        return loss, preds.detach(), y.detach()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing (in order) the input tensor, target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss.item())
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing (in order) the input tensor, target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss.item())
        self.val_auc(preds, targets)
        self.val_auprc(preds.squeeze(), targets.squeeze().to(torch.int))
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        auc = self.val_auc.compute()  # get current val auc
        auprc = self.val_auprc.compute()
        self.val_auc_best(auc)  # update best so far val auc
        self.val_auprc_best(auprc)
        # log `val_auc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/auprc_best", self.val_auprc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch, calibrated=True)

        # update and log metrics
        self.test_loss(loss.item())
        self.test_auc(preds, targets)
        self.test_auprc(preds.squeeze(), targets.squeeze().to(torch.int))
        self.test_bal_acc((preds.squeeze() > self.hparams.threshold).to(torch.int), targets)
        self.test_acc(preds.squeeze(), targets)
        self.test_mcc(preds.squeeze(), targets)
        self.test_f1(preds.squeeze(), targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/bal_acc", self.test_bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mcc", self.test_mcc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single predict step on a batch of data from the predict set.

        :param batch: A batch of data (a tuple) containing (in order) the input tensor, target
            labels, prediction longitudes, and prediction latitudes.
        :param batch_idx: The index of the current batch.

        :return: A tensor containing (in order):
            - Prediction Longitude
            - Prediction Latitude
            - Prediction Likelihood
            - Prediction Uncertainty
            - Prediction Feature Attributions
        """
        # extracts feature attributions
        ig = IntegratedGradients(self.net)
        attribution = ig.attribute(batch[0].requires_grad_(), n_steps=12).mean(dim=(-1,-2)).detach()

        # enables Monte Carlo Dropout
        if self.hparams.mc_samples > 1:
            self.net.activate_dropout()

        # generates MC samples
        preds = torch.sigmoid(
            self.calibrated_forward(
                batch[0].tile((self.hparams.mc_samples,1,1,1))
            ).reshape(self.hparams.mc_samples,-1)
        ).detach()

        # computes mean and std of MC samples
        means = preds.mean(dim=0).squeeze()
        stds = preds.std(dim=0).squeeze()
        
        return torch.concat((torch.stack((batch[2], batch[3], means, stds), dim=-1), attribution), dim=-1)
        
    def on_predict_epoch_end(self, results):
        results = torch.concat(results[0]).cpu().numpy()
        cols = ["lon","lat","mean","std"] + [f"attr{n}" for n in range(results.shape[-1]-4)]
        res_df = pd.DataFrame(data=results, columns=cols)
        res_df.to_csv(f"gpu_{self.trainer.strategy.global_rank}_result.csv",index=False)
        self.trainer.strategy.barrier()

        # TODO DEBUG following
        # if self.trainer.strategy.world_size > 1:
            # num_dims = results.shape[-1]
            # results = self.all_gather(results).reshape((-1,num_dims))
        # if self.trainer.strategy.global_rank == 0:
            # self.trainer.results = results.cpu().numpy()

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
    
    def set_temperature(self, temperature: float) -> None:
        self.hparams.temperature = temperature

    def set_threshold(self, threshold: float) -> None:
        self.hparams.threshold = threshold
        self.test_acc = BinaryAccuracy(threshold=self.hparams.threshold)
        self.test_mcc = BinaryMatthewsCorrCoef(threshold=self.hparams.threshold)
        self.test_f1 = BinaryF1Score(threshold=self.hparams.threshold)


if __name__ == "__main__":
    _ = CMALitModule(None, None, None, None)