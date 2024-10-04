import optuna

from typing import List, Dict
from sri_maper.src import utils
from sri_maper.src.train import train

def optuna_objective(
    trial: optuna.Trial,
    fixed_overrides: List[str],
    optuna_params_dict: Dict,
    metric: str = 'val/auprc_best'
) -> float:
    optuna_overrides = []
    # Suggest hyperparameters
    for key, value in optuna_params_dict.items():
        if key == "smoothing":
            optuna_overrides.append(value(trial.suggest_discrete_uniform(key, low=0.0, high=0.3, q=0.1)))
        elif key == "dropout":
            optuna_overrides.append(value(trial.suggest_discrete_uniform(key, low=0.1, high=0.4, q=0.1)))
        elif key == "negative_sampling_fraction":
            optuna_overrides.append(value(
                trial.suggest_discrete_uniform(key+"_low", low=0.0, high=0.1, q=0.02),
                trial.suggest_discrete_uniform(key+"_high", low=0.9, high=1.0, q=0.02)
            ))
        # if key == "fraction_train_split":
        #     optuna_overrides.append(value(trial.suggest_discrete_uniform(key, low=0.5, high=0.9, q=0.1)))
        # elif key == "upsample_multiplier":
        #     optuna_overrides.append(value(trial.suggest_discrete_uniform(key, low=10.0, high=30.0, q=10.0)))
        # elif key == "learning_rate":
        #     optuna_overrides.append(value(trial.suggest_loguniform(key, low=1e-4, high=1e-2)))
        # elif key == "weight_decay":
        #     optuna_overrides.append(value(trial.suggest_loguniform(key, low=1e-4, high=1e-2)))
        # elif key == "smoothing":
        #     optuna_overrides.append(value(trial.suggest_discrete_uniform(key, low=0.0, high=0.3, q=0.1)))
        # elif key == "likely_negative_range":
        #     optuna_overrides.append(value(
        #         trial.suggest_discrete_uniform(key+"_low", low=0.0, high=0.1, q=0.02),
        #         trial.suggest_discrete_uniform(key+"_high", low=0.9, high=1.0, q=0.02)
        #     ))
        # elif key == "dropout":
        #     optuna_overrides.append(value(
        #         trial.suggest_discrete_uniform(key+"_0", low=0.0, high=0.5, q=0.1),
        #         trial.suggest_discrete_uniform(key+"_1", low=0.0, high=0.5, q=0.1),
        #         trial.suggest_discrete_uniform(key+"_2", low=0.0, high=0.5, q=0.1)
        #     )) # - after schemas get updated

    # Build the Hydra config
    optuna_cfg = utils.build_hydra_config_notebook(overrides=fixed_overrides + optuna_overrides)

    utils.print_config_tree(optuna_cfg)
    optuna_metrics, _ = train(optuna_cfg)

    # Return the score for Optuna to minimize or maximize
    return optuna_metrics[metric]


def run_optuna_study(
    fixed_overrides: List[str],
    optuna_params_dict: Dict,
    n_trials: int = 10
):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, fixed_overrides, optuna_params_dict),
        n_trials=n_trials
    )
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    optuna_output_overrides = []
    for key, value in optuna_params_dict.items():
        if key == "negative_sampling_fraction":
            optuna_output_overrides.append(
                value(
                    trial.params['negative_sampling_fraction_low'],
                    trial.params['negative_sampling_fraction_high']
                )
            )
        else:
            optuna_output_overrides.append(value(trial.params[key]))

        # if key == "likely_negative_range":
        #     optuna_output_overrides.append(
        #         value(
        #             trial.params['likely_negative_range_low'],
        #             trial.params['likely_negative_range_high']
        #         )
        #     )
        # elif key == "dropout":
        #     optuna_output_overrides.append(
        #         value(
        #             trial.params['dropout_0'],
        #             trial.params['dropout_1'],
        #             trial.params['dropout_2']
        #         )
        #     )
        # else:
        #     optuna_output_overrides.append(value(trial.params[key]))
        # # - after schemas get updated

    return optuna_output_overrides, trial
