import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

class MoE_predictor(nn.Module):
    """Mixture of Experts (MoE) predictor for text-to-image and image-to-text tasks.

    This model uses a gating mechanism to select top-k experts and combines their outputs
    for task-specific predictions, supporting both L2 and contrastive learning objectives.

    Args:
        num_experts (int): Number of expert networks.
        image_dim (int): Dimension of image features.
        text_dim (int): Dimension of text features.
        hidden_size (int, optional): Size of hidden layers. Defaults to 256.
        k (int, optional): Number of top experts to select. Defaults to 2.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(self, num_experts: int, image_dim: int, text_dim: int, 
                 hidden_size: int = 256, k: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.hidden_size = hidden_size

        # Layer normalization
        self.l2_norm = nn.LayerNorm(hidden_size)
        self.cl_norm = nn.LayerNorm(hidden_size)

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_size, num_experts)

        # Task-specific embeddings
        self.image_embed = nn.Linear(image_dim, hidden_size, bias=True)
        self.text_embed = nn.Linear(text_dim, hidden_size, bias=True)

        # L2 prediction heads
        self.text_to_image = nn.Linear(hidden_size, image_dim, bias=True)
        self.image_to_text = nn.Linear(hidden_size, text_dim, bias=True)

        # Contrastive learning heads
        self.cl_text_to_image = nn.Linear(hidden_size, hidden_size, bias=True)
        self.cl_image_to_text = nn.Linear(hidden_size, hidden_size, bias=True)

        # Task embeddings
        self.cl_task_embedding = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.l2_task_embedding = nn.Parameter(torch.empty(1, 1, hidden_size))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        def init_linear(layer: nn.Module) -> None:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Initialize experts
        for expert in self.experts:
            expert.apply(init_linear)

        # Initialize other linear layers
        for module in [self.gate, self.image_embed, self.text_embed, 
                      self.text_to_image, self.image_to_text,
                      self.cl_text_to_image, self.cl_image_to_text]:
            init_linear(module)

        # Initialize task embeddings
        nn.init.kaiming_uniform_(self.cl_task_embedding, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.l2_task_embedding, nonlinearity='relu')

    def forward(self, x: torch.Tensor, task: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the MoE predictor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            task (str): Task type, either 'text2image' or 'image2text'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: L2 and contrastive learning outputs.

        Raises:
            ValueError: If task is not 'text2image' or 'image2text'.
        """
        # Apply task-specific embedding
        x = F.gelu(self._apply_task_embedding(x, task))

        # Expand task embeddings
        l2_task_emb = self.l2_task_embedding.expand(x.shape[0], x.shape[1], -1)
        cl_task_emb = self.cl_task_embedding.expand(x.shape[0], x.shape[1], -1)
        l2_input = x + l2_task_emb
        cl_input = x + cl_task_emb

        # Process through MoE and normalization
        l2_output = self._forward_moe(l2_input)
        l2_output = self.l2_norm(l2_output)
        l2_output = F.gelu(l2_output) + l2_input

        cl_output = self._forward_moe(cl_input)
        cl_output = self.cl_norm(cl_output)
        cl_output = F.gelu(cl_output) + cl_input

        # Apply task-specific output heads
        l2_result, cl_result = self._apply_task_output(l2_output, cl_output, task)
        return l2矫正l2_result, cl_result

    def _forward_moe(self, x: torch.Tensor) -> torch.Tensor:
        """Mixture of Experts forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Combined output from selected experts.
        """
        gate_output = self.gate(x)
        topk_values, topk_indices = torch.topk(F.softmax(gate_output, dim=-1), self.k, dim=-1)

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        # Select top-k experts
        batch_size, seq_len, _ = x.size()
        topk_indices = topk_indices.unsqueeze(-1).expand(batch_size, seq_len, self.k, expert_outputs.size(-1))
        selected_experts = torch.gather(expert_outputs, 2, topk_indices)

        # Combine weighted expert outputs
        weighted_output = selected_experts * topk_values.unsqueeze(-1).expand_as(selected_experts)
        result = weighted_output.sum(dim=2)

        return result

    def _apply_task_embedding(self, x: torch.Tensor, task: str) -> torch.Tensor:
        """Apply task-specific embedding to input.

        Args:
            x (torch.Tensor): Input tensor.
            task (str): Task type ('text2image' or 'image2text').

        Returns:
            torch.Tensor: Embedded input.

        Raises:
            ValueError: If task is invalid.
        """
        if task == 'text2image':
            return self.text_embed(x)
        elif task == 'image2text':
            return self.image_embed(x)
        else:
            raise ValueError(f"Invalid task: {task}. Expected 'text2image' or 'image2text'.")

    def _apply_task_output(self, l2_output: torch.Tensor, cl_output: torch.Tensor, 
                         task: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply task-specific output heads.

        Args:
            l2_output (torch.Tensor): L2 output tensor.
            cl_output (torch.Tensor): Contrastive learning output tensor.
            task (str): Task type ('text2image' or 'image2text').

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: L2 and contrastive outputs.

        Raises:
            ValueError: If task is invalid.
        """
        if task == 'text2image':
            return self.text_to_image(l2_output), self.cl_text_to_image(cl_output)
        elif task == 'image2text':
            return self.image_to_text(l2_output), self.cl_image_to_text(cl_output)
        else:
            raise ValueError(f"Invalid task: {task}. Expected 'text2image' or 'image2text'.")
