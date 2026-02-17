import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel


class PlausibilityConfig(PretrainedConfig):
    model_type = "plausibility-regression"

    def __init__(
        self,
        base_model_name=None,
        pooling="cls",  # "cls", "mean", or "attention"
        loss_type="huber",
        huber_delta=1.0,
        use_ranking_loss=False,
        ranking_weight=0.1,
        use_uncertainty_loss=False,  
        uncertainty_weight=0.3,  
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.pooling = pooling
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.use_ranking_loss = use_ranking_loss
        self.ranking_weight = ranking_weight
        self.use_uncertainty_loss = use_uncertainty_loss
        self.uncertainty_weight = uncertainty_weight


class PlausibilityRegressionModel(PreTrainedModel):
    config_class = PlausibilityConfig

    def __init__(self, config: PlausibilityConfig, freeze_transformer=False):
        super().__init__(config)

        self.transformer = AutoModel.from_pretrained(config.base_model_name)
        hidden_size = self.transformer.config.hidden_size

        if freeze_transformer:
            for p in self.transformer.parameters():
                p.requires_grad = False

        # Attention pooling layer (if using attention pooling)
        if config.pooling == "attention":
            self.attention_pooling = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )

        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Tanh(),  # bounded but less saturating than sigmoid
        )

        if config.loss_type == "mse":
            self.base_loss = nn.MSELoss()
        elif config.loss_type == "mae":
            self.base_loss = nn.L1Loss()
        elif config.loss_type == "huber":
            self.base_loss = nn.SmoothL1Loss(beta=config.huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")

    def _pool(self, outputs, attention_mask):
        if self.config.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-6)
            return summed / counts
        elif self.config.pooling == "attention":
            # Learnable attention-weighted pooling
            attention_scores = self.attention_pooling(outputs.last_hidden_state)  # [batch, seq_len, 1]
            
            # Mask padding tokens
            attention_scores = attention_scores.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), 
                float('-inf')
            )
            
            # Softmax to get attention weights
            attention_weights = torch.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]
            
            # Weighted sum
            pooled = torch.sum(outputs.last_hidden_state * attention_weights, dim=1)  # [batch, hidden_size]
            return pooled
        else:  # cls
            return outputs.last_hidden_state[:, 0, :]

    def _scale_output(self, x):
        """
        Map tanh output from [-1, 1] to [1, 5]
        """
        return 3.0 + 2.0 * x

    def _ranking_loss(self, predictions, labels):
        """
        RankNet-style pairwise logistic loss
        Encourages correct ordering for Spearman
        """
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
        label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)

        target = torch.sign(label_diff)
        mask = target != 0

        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)

        loss = torch.log1p(torch.exp(-target[mask] * pred_diff[mask]))
        return loss.mean()

    def _uncertainty_loss(self, predictions, labels, std_devs):
        """
        Uncertainty-aware loss: penalize predictions outside human disagreement range
        
        Args:
            predictions: Model predictions [batch_size]
            labels: Ground truth labels [batch_size]
            std_devs: Standard deviations from human annotations [batch_size]
        """
        error = torch.abs(predictions - labels)
        
        # Convert std_devs to tensor if needed
        if not isinstance(std_devs, torch.Tensor):
            std_devs = torch.tensor(std_devs, device=predictions.device, dtype=torch.float)
        
        # Ensure minimum uncertainty of 1.0
        std_devs = torch.clamp(std_devs, min=1.0)
        
        # Penalize errors that exceed the standard deviation
        # If error is within std_dev, penalty is 0
        # If error exceeds std_dev, penalty grows linearly
        uncertainty_penalty = torch.relu(error - std_devs)
        
        return uncertainty_penalty.mean()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        std_devs=None,  # NEW: pass standard deviations
        **kwargs,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled = self._pool(outputs, attention_mask)
        raw_output = self.regressor(pooled).squeeze(-1)
        logits = self._scale_output(raw_output)

        loss = None
        if labels is not None:
            base_loss = self.base_loss(logits, labels)
            total_loss = base_loss

            # Add ranking loss if enabled
            if self.config.use_ranking_loss and labels.size(0) > 1:
                rank_loss = self._ranking_loss(logits, labels)
                total_loss = total_loss + self.config.ranking_weight * rank_loss

            # Add uncertainty loss if enabled and std_devs provided
            if self.config.use_uncertainty_loss and std_devs is not None:
                uncertainty_loss = self._uncertainty_loss(logits, labels, std_devs)
                total_loss = total_loss + self.config.uncertainty_weight * uncertainty_loss

            loss = total_loss

        return {"loss": loss, "logits": logits}


def create_model(
    model_name,
    freeze_transformer=False,
    pooling="mean",
    loss_type="huber",
    huber_delta=1.0,
    use_ranking_loss=False,
    ranking_weight=0.25,
    use_uncertainty_loss=False,  
    uncertainty_weight=0.3,  
):
    config = PlausibilityConfig(
        base_model_name=model_name,
        pooling=pooling,
        loss_type=loss_type,
        huber_delta=huber_delta,
        use_ranking_loss=use_ranking_loss,
        ranking_weight=ranking_weight,
        use_uncertainty_loss=use_uncertainty_loss,
        uncertainty_weight=uncertainty_weight,
    )

    return PlausibilityRegressionModel(
        config=config,
        freeze_transformer=freeze_transformer,
    )