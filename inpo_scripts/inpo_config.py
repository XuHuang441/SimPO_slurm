from dataclasses import dataclass, field
from scripts.simpo_config import SimPOConfig

@dataclass
class INPOConfig(SimPOConfig):
    """
    Configuration class for INPOTrainer.
    """
    # The mixing ratio between the reference model and the historical models.
    ratio: float = 1/3
    # The eta parameter for the INPO loss.
    eta: float = 0.0075
    beta: float = 0.001
    # The maximum number of historical models to consider.
    max_history_t: int = 2
    # The loss type for INPO, we'll use a specific name.
    loss_type: str = field(default="inpo_squared_error", metadata={"help": "The loss type to use."})
