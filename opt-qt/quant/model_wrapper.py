from .opt_wrapper import wrap_opt_model
from .qwen_wrapper import wrap_qwen_model


def wrap_model_by_family(
    model,
    quant_config,
    mode="scale_inspection",
    stat_manager=None,
):
    family = quant_config.get("model_family", "opt").lower()

    if family == "opt":
        return wrap_opt_model(
            model,
            quant_config,
            mode=mode,
            stat_manager=stat_manager,
        )

    if family in {"qwen", "qwen2", "qwen2.5"}:
        return wrap_qwen_model(
            model,
            quant_config,
            mode=mode,
            stat_manager=stat_manager,
        )

    raise ValueError(f"Unsupported model_family: {family}")