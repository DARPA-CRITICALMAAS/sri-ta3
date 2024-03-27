from typing import List, Optional, Tuple
import hydra
from omegaconf import DictConfig
from torch import set_float32_matmul_precision
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from sklearn.metrics import f1_score

from sri_maper.src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def test(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Preprocessing rasters...")
    hydra.utils.call(cfg.preprocess)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # preparation
    datamodule.setup("validate")
    model = model.__class__.load_from_checkpoint(cfg.ckpt_path)

    # temperature scaling
    if "temperature" not in cfg.model:
        model_calibrator = utils.BinaryTemperatureScaling(model)
        opt_temp = model_calibrator.calibrate(datamodule, cfg.trainer.limit_val_batches)
        del model_calibrator
        log.info(f"Optimal temperature: {opt_temp:.3f}")
        model.set_temperature(opt_temp)
    else:
        model.set_temperature(cfg.model.temperature)

    # theshold selection
    if "threshold" not in cfg.model:
        threshold_selector = utils.ThresholdMoving(model)
        opt_thr = threshold_selector.search_threshold(f1_score, datamodule, cfg.trainer.limit_val_batches)
        del threshold_selector
        log.info(f"Optimal threshold: {opt_thr:.3f}")
        model.set_threshold(opt_thr)
    else:
        model.set_threshold(cfg.model.threshold)

    log.info("Testing!")
    trainer.test(model=model, datamodule=datamodule)

    test_metrics = trainer.callback_metrics

    return test_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs/", config_name="test.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)
    # test the model
    test(cfg)


if __name__ == "__main__":
    set_float32_matmul_precision('medium')
    main()
