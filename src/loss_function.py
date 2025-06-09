# Copyright 2025 Your Name or Organization
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import all_reduce, ReduceOp
from typing import Optional

from src.utils.distributed import AllReduce

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def contrastive_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.03) -> torch.Tensor:
    """Compute contrastive loss between predicted and target embeddings.

    Args:
        pred (torch.Tensor): Predicted embeddings, shape (batch_size, embedding_dim).
        target (torch.Tensor): Target embeddings, shape (batch_size, embedding_dim).
        temperature (float, optional): Temperature scaling factor. Defaults to 0.03.

    Returns:
        torch.Tensor: Contrastive loss value.

    Raises:
        ValueError: If pred and target have incompatible shapes or are on different devices.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    if pred.device != target.device:
        raise ValueError("Pred and target must be on the same device")

    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    logits = torch.matmul(pred, target.t()) / temperature
    labels = torch.arange(pred.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)

    logger.debug(f"Contrastive loss: {loss.item():.4f}")
    return loss

def symmetric_contrastive_loss(image_embed: torch.Tensor, text_embed: torch.Tensor, 
                             temperature: float = 0.03) -> torch.Tensor:
    """Compute symmetric contrastive loss for image and text embeddings.

    Args:
        image_embed (torch.Tensor): Image embeddings, shape (batch_size, embedding_dim).
        text_embed (torch.Tensor): Text embeddings, shape (batch_size, embedding_dim).
        temperature (float, optional): Temperature scaling factor. Defaults to 0.03.

    Returns:
        torch.Tensor: Symmetric contrastive loss value, averaged over image-to-text and text-to-image directions.

    Raises:
        ValueError: If embeddings have incompatible shapes or are on different devices.
    """
    image_to_text_loss = contrastive_loss(image_embed, text_embed, temperature)
    text_to_image_loss = contrastive_loss(text_embed, image_embed, temperature)
    loss = (image_to_text_loss + text_to_image_loss) / 2

    logger.debug(f"Symmetric contrastive loss: {loss.item():.4f} "
                 f"(image-to-text: {image_to_text_loss.item():.4f}, "
                 f"text-to-image: {text_to_image_loss.item():.4f})")
    return loss

def loss_fn_1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute smooth L1 loss between predicted and target embeddings with distributed reduction.

    Args:
        pred (torch.Tensor): Predicted embeddings, shape (batch_size, embedding_dim).
        target (torch.Tensor): Target embeddings, shape (batch_size, embedding_dim).

    Returns:
        torch.Tensor: Smooth L1 loss value, reduced across distributed processes if applicable.

    Raises:
        ValueError: If pred and target have incompatible shapes or are on different devices.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    if pred.device != target.device:
        raise ValueError("Pred and target must be on the same device")

    loss = F.smooth_l1_loss(pred, target)
    loss = AllReduce.apply(loss)
    
    logger.debug(f"Smooth L1 loss: {loss.item():.4f}")
    return loss
