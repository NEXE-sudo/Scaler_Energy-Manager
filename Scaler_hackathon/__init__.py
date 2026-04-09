"""Scaler Hackathon Environment."""

from .client import EnergyGridEnv
from .models import EnergyGridAction, EnergyGridObservation

# Aliases for backward compatibility
ScalerHackathonAction = EnergyGridAction
ScalerHackathonObservation = EnergyGridObservation

__all__ = [
    "EnergyGridAction",
    "EnergyGridObservation",
    "ScalerHackathonAction",
    "ScalerHackathonObservation",
    "EnergyGridEnv",
]
