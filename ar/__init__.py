from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "ActivationReasoning",
    "LogicConfig",
    "LogicConfigNoSteering",
    "LogicConfigDefault",
    "LogicalParser",
]

if TYPE_CHECKING:
    from .config import LogicConfig, LogicConfigNoSteering, LogicConfigDefault
    from .model.activation_reasoning import ActivationReasoning
    from .model.logic import LogicalParser


def __getattr__(name: str):
    if name in {"LogicConfig", "LogicConfigNoSteering", "LogicConfigDefault"}:
        module = import_module("ar.config")
        return getattr(module, name)
    if name == "ActivationReasoning":
        module = import_module("ar.model.activation_reasoning")
        return getattr(module, name)
    if name == "LogicalParser":
        module = import_module("ar.model.logic")
        return getattr(module, name)
    raise AttributeError(f"module 'ar' has no attribute '{name}'")
