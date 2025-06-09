import sys
import logging
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

from src.models import model_paths
from src.models.MMoE import create_mmoe_predictor
from src.all_loss import loss_fn_1, symmetric_contrastive_loss
from peft import get_peft_model, LoraConfig

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class M3Jepa(nn.Module):
    """Multi-modal Joint Embedding Predictive Architecture (JEPA) model.

    This model integrates vision and text encoders with a predictor (MLP or MMoE)
    for text-to-image and image-to-text tasks, using L2 and contrastive losses.

    Args:
        image_encoder_name (str): Name of the image encoder model.
        text_encoder_name (str): Name of the text encoder model.
        num_experts (int): Number of experts for MMoE predictor.
        hidden_size (int): Size of hidden layers in predictor.
        k_expert (int): Number of top experts to select in MMoE.
        dropout_rate (float): Dropout probability.
        predictor_type (str): Type of predictor ('mlp' or 'mmoe_mlp').
    """
    def __init__(
        self,
        image_encoder_name: str,
        text_encoder_name: str,
        num_experts: int,
        hidden_size: int,
        k_expert: int,
        dropout_rate: float,
        predictor_type: str
    ):
        super().__init__()
        self.task_map = {'text2image': self._text2image_forward, 'image2text': self._image2text_forward}

        # Initialize vision and text models
        self.image_model = self._load_image_model(image_encoder_name)
        self.text_model = self._load_text_model(text_encoder_name)

        image_dim = self.image_model.config.hidden_size
        text_dim = self.text_model.config.hidden_size
        logger.info(f"Image model dim: {image_dim}, Text model dim: {text_dim}")

        # Modality embeddings
        self.image_to_text_embedding = nn.Parameter(torch.empty(1, 1, image_dim))
        self.text_to_image_embedding = nn.Parameter(torch.empty(1, 1, text_dim))

        # Predictor
        if predictor_type == 'mlp':
            self.predictor = Creat_MLP_predictor(
                hidden_sizes=[hidden_size, hidden_size],
                image_dim=image_dim,
                text_dim=text_dim,
                drop=dropout_rate
            )
        elif predictor_type == 'mmoe_mlp':
            self.predictor = create_mmoe_predictor(
                num_experts=num_experts,
                image_dim=image_dim,
                text_dim=text_dim,
                hidden_size=hidden_size,
                k=k_expert,
                drop=dropout_rate
            )
        else:
            raise ValueError(f"Invalid predictor_type: {predictor_type}. Expected 'mlp' or 'mmoe_mlp'.")

        # Contrastive learning embeddings
        self.cl_image_emb = nn.Linear(image_dim, hidden_size, bias=True)
        self.cl_text_emb = nn.Linear(text_dim, hidden_size, bias=True)

        # Loss weights
        self.l2_weight = 1.0
        self.cl_weight = 1.0
        self.task_weights = {'text2image': 0.5, 'image2text': 0.5}

        self._initialize_weights()
        logger.info(f"Initialized M3Jepa with predictor: {predictor_type}")

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        def init_linear(layer: nn.Module) -> None:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.kaiming_uniform_(self.image_to_text_embedding, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.text_to_image_embedding, nonlinearity='relu')
        init_linear(self.cl_image_emb)
        init_linear(self.cl_text_emb)

    def _load_image_model(self, image_encoder_name: str) -> AutoModel:
        """Load and configure the image model with LoRA.

        Args:
            image_encoder_name (str): Name of the image encoder.

        Returns:
            AutoModel: Configured image model.
        """
        model = AutoModel.from_pretrained(
            model_paths.vm_model_path[image_encoder_name],
            output_hidden_states=True
        )
        model.config.gradient_checkpointing = True
        model = self._apply_lora_to_vm_layers(model)
        for name, param in model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
        return model

    def _load_text_model(self, text_encoder_name: str) -> AutoModelForCausalLM:
        """Load and configure the text model with LoRA.

        Args:
            text_encoder_name (str): Name of the text encoder.

        Returns:
            AutoModelForCausalLM: Configured text model.
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_paths.lm_model_path[text_encoder_name],
            output_hidden_states=True
        )
        model.config.use_flash_attention_2 = True
        model.config.gradient_checkpointing = True
        model = self._apply_lora_to_lm_layers(model)
        for name, param in model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
        return model

    def _apply_lora_to_lm_layers(self, model: AutoModelForCausalLM, num_layers: int = 3) -> AutoModelForCausalLM:
        """Apply LoRA to the last num_layers of the language model.

        Args:
            model (AutoModelForCausalLM): Text model.
            num_layers (int): Number of layers to apply LoRA to.

        Returns:
            AutoModelForCausalLM: Model with LoRA applied.
        """
        total_layers = len(model.model.layers)
        if num_layers > total_layers:
            logger.warning(f"Requested {num_layers} layers for LoRA, but model has {total_layers}. Using {total_layers}.")
            num_layers = total_layers

        selected_layers = model.model.layers[-num_layers:]
        logger.info(f"Applying LoRA to {num_layers} of {total_layers} LM layers")
        lora_config = LoraConfig(
            peft_type='feature_extraction',
            r=4,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='lora_only',
            init_lora_weights='pissa',
            target_modules=["q_proj", "v_proj"]
        )

        for layer in selected_layers:
            layer.self_attn = get_peft_model(layer.self_attn, lora_config)
        return model

    def _apply_lora_to_vm_layers(self, model: AutoModel, num_layers: int = 3) -> AutoModel:
        """Apply LoRA to the last num_layers of the vision model.

        Args:
            model (AutoModel): Image model.
            num_layers (int): Number of layers to apply LoRA to.

        Returns:
            AutoModel: Model with LoRA applied.
        """
        total_layers = len(model.encoder.layer)
        if num_layers > total_layers:
            logger.warning(f"Requested {num_layers} layers for LoRA, but model has {total_layers}. Using {total_layers}.")
            num_layers = total_layers

        selected_layers = model.encoder.layer[-num_layers:]
        logger.info(f"Applying LoRA to {num_layers} of {total_layers} VM layers")
        lora_config = LoraConfig(
            peft_type='feature_extraction',
            r=4,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='lora_only',
            init_lora_weights='pissa',
            target_modules=["query", "value"]
        )

        for layer in selected_layers:
            layer.attention.attention = get_peft_model(layer.attention.attention, lora_config)
        return model

    def _compute_loss(
        self,
        pred_emb: torch.Tensor,
        target_emb: torch.Tensor,
        cl_source_emb: torch.Tensor,
        cl_target_emb: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined L2 and contrastive loss.

        Args:
            pred_emb (torch.Tensor): Predicted embedding.
            target_emb (torch.Tensor): Target embedding for L2 loss.
            cl_source_emb (torch.Tensor): Source embedding for contrastive loss.
            cl_target_emb (torch.Tensor): Target embedding for contrastive loss.

        Returns:
            torch.Tensor: Weighted combined loss.
        """
        l2_loss = loss_fn_1(pred_emb, target_emb)
        cl_loss = symmetric_contrastive_loss(cl_source_emb, cl_target_emb)
        return self.cl_weight * cl_loss + self.l2_weight * l2_loss

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        image_pixel_values: torch.Tensor,
        task: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the M3Jepa model.

        Args:
            text_input_ids (torch.Tensor): Text input IDs.
            text_attention_mask (torch.Tensor): Text attention mask.
            image_pixel_values (torch.Tensor): Image pixel values.
            task (str): Task type ('text2image' or 'image2text').

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Weighted loss, predicted embedding, target L2 embedding,
                contrastive source embedding, contrastive target embedding.

        Raises:
            ValueError: If task is not 'text2image' or 'image2text'.
        """
        if task not in self.task_map:
            raise ValueError(f"Invalid task: {task}. Expected 'text2image' or 'image2text'.")

        # Encode text and image
        text_embedding = self.text_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_embedding = torch.mean(torch.stack(text_embedding.hidden_states[-2:]), dim=0)

        image_embedding = self.image_model(pixel_values=image_pixel_values)
        image_embedding = torch.mean(torch.stack(image_embedding.hidden_states[-2:]), dim=0)

        return self.task_map[task](text_embedding, image_embedding)

    def _text2image_forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for text-to-image task.

        Args:
            text_embedding (torch.Tensor): Text embedding.
            image_embedding (torch.Tensor): Image embedding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Weighted loss and embeddings.
        """
        t2i_embedding = self.text_to_image_embedding.expand(text_embedding.shape[0], text_embedding.shape[1], -1)
        text_embedding = text_embedding + t2i_embedding

        pred_image_emb, cl_text_emb = self.predictor(text_embedding, 'text2image')
        pred_image_emb = pred_image_emb.mean(dim=1)
        l2_image_emb = image_embedding.mean(dim=1)

        cl_text_emb = cl_text_emb.mean(dim=1)
        cl_image_emb = self.cl_image_emb(image_embedding).mean(dim=1)

        loss = self._compute_loss(pred_image_emb, l2_image_emb, cl_text_emb, cl_image_emb)
        return self.task_weights['text2image'] * loss, pred_image_emb, l2_image_emb, cl_text_emb, cl_image_emb

    def _image2text_forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for image-to-text task.

        Args:
            text_embedding (torch.Tensor): Text embedding.
            image_embedding (torch.Tensor): Image embedding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Weighted loss and embeddings.
        """
        i2t_embedding = self.image_to_text_embedding.expand(image_embedding.shape[0], image_embedding.shape[1], -1)
        image_embedding = image_embedding + i2t_embedding

        pred_text_emb, cl_image_emb = self.predictor(image_embedding, 'image2text')
        pred_text_emb = pred_text_emb.mean(dim=1)
        l2_text_emb = text_embedding.mean(dim=1)

        cl_image_emb = cl_image_emb.mean(dim=1)
        cl_text_emb = self.cl_text_emb(text_embedding).mean(dim=1)

        loss = self._compute_loss(pred_text_emb, l2_text_emb, cl_image_emb, cl_text_emb)
        return self.task_weights['image2text'] * loss, pred_text_emb, l2_text_emb, cl_image_emb, cl_text_emb

def init_m3jepa(
    image_encoder_name: str,
    text_encoder_name: str,
    num_experts: int,
    hidden_size: int,
    k_expert: int,
    dropout_rate: float,
    predictor_type: str
) -> M3Jepa:
    """Initialize the M3Jepa model.

    Args:
        image_encoder_name (str): Name of the image encoder model.
        text_encoder_name (str): Name of the text encoder model.
        num_experts (int): Number of experts for MMoE predictor.
        hidden_size (int): Size of hidden layers in predictor.
        k_expert (int): Number of top experts to select in MMoE.
        dropout_rate (float): Dropout probability.
        predictor_type (str): Type of predictor ('mlp' or 'mmoe_mlp').

    Returns:
        M3Jepa: Initialized model.
    """
    return M3Jepa(
        image_encoder_name=image_encoder_name,
        text_encoder_name=text_encoder_name,
        num_experts=num_experts,
        hidden_size=hidden_size,
        k_expert=k_expert,
        dropout_rate=dropout_rate,
        predictor_type=predictor_type
    )
