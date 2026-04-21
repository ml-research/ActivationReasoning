from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "ActivationReasoning",
    "LogicConfig",
    "LogicConfigDefault",
    "LogicConfigNoSteering",
    "LogicalParser",
]

if TYPE_CHECKING:
    from ar import (
        ActivationReasoning,
        LogicConfig,
        LogicConfigDefault,
        LogicConfigNoSteering,
        LogicalParser,
    )


def __getattr__(name: str):
    if name in __all__:
        module = import_module("ar")
        return getattr(module, name)
    raise AttributeError(f"module 'activation_reasoning' has no attribute '{name}'")
