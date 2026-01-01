"""
Base class for pipeline stages
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch


class BaseStage(ABC):
    """Abstract base class for all pipeline stages"""
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda:0"):
        self.config = config
        self.device = device
        self.model = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model for this stage"""
        pass
    
    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run this stage of the pipeline"""
        pass
    
    def to(self, device: str) -> "BaseStage":
        """Move model to device"""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self
    
    def unload_model(self) -> None:
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
