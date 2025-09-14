import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from scripts.simpo_trainer import SimPOTrainer
from inpo_scripts.inpo_config import INPOConfig
from transformers import AutoModelForCausalLM, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer


class INPOTrainer(SimPOTrainer):
    def __init__(
            self,
            model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            args: Optional[INPOConfig] = None,
            **kwargs,
    ):

        super().__init__(model=model, args=args, **kwargs)

        # INPO parameters
        self.ratio = args.ratio
        self.eta = args.eta
        self.max_history_t = args.max_history_t
        self.beta = args.eta

        print(f"INPOTrainer initialized with ratio: {self.ratio}, eta: {self.eta}, max_history_t: {self.max_history_t}")

    def inpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            chosen_probs: Union[torch.FloatTensor, None] = None,
            chosen_probs_win: Union[torch.FloatTensor, None] = None,
            chosen_probs_lose: Union[torch.FloatTensor, None] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Computes the INPO loss.
        This is a direct adaptation of your provided loss function.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        logits_w = policy_chosen_logps - reference_chosen_logps
        logits_l = policy_rejected_logps - reference_rejected_logps

        loss_w = (logits_w - (1 / self.beta) * (chosen_probs_win - 0.5)) ** 2
        loss_l = (logits_l - (1 / self.beta) * (chosen_probs_lose - 0.5)) ** 2
        losses = (loss_w + loss_l) / 2

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: str = "train",
    ):
        """
        Compute the INPO loss and other metrics for the given batch.
        This method is overridden from SimPOTrainer.
        """
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        # 1. Get policy logps using the efficient concatenated_forward from SimPOTrainer
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,  # SimPOTrainer's forward returns this, useful for SFT loss
        ) = self.concatenated_forward(model, batch)

        chosen_probs = torch.tensor(batch["chosen_probs"], dtype=float, device=policy_chosen_logps.device)
        chosen_probs_win = torch.tensor(batch["chosen_probs_win"], dtype=float, device=policy_chosen_logps.device)
        chosen_probs_lose = torch.tensor(batch["chosen_probs_lose"], dtype=float, device=policy_chosen_logps.device)

        # 2. Get reference and history logps from the pre-computed batch data
        # Ensure they are on the correct device
        reference_chosen_logps = batch['reference_chosen_logps'].to(self.accelerator.device)
        reference_rejected_logps = batch['reference_rejected_logps'].to(self.accelerator.device)

        # 3. Compute INPO loss
        losses, chosen_rewards, rejected_rewards = self.inpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_probs,
            chosen_probs_win,
            chosen_probs_lose,
        )

        loss = losses.mean()

        # 4. Calculate metrics (same as your original implementation)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return loss, metrics