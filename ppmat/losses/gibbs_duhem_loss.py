# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Loss functions for GDI-NN with Gibbs-Duhem constraints.

This module provides loss functions that combine prediction loss with
Gibbs-Duhem thermodynamic constraint enforcement.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Dict, Optional, Callable


class GibbsDuhemLoss(nn.Layer):
    """Gibbs-Duhem constraint loss.

    This loss enforces the Gibbs-Duhem thermodynamic constraint for binary mixtures:
    x1 * d(ln(gamma1))/dx1 + x2 * d(ln(gamma2))/dx1 = 0

    This ensures thermodynamic consistency of the predicted activity coefficients.

    Args:
        lambda_gd: Weight for Gibbs-Duhem loss (default: 1.0)
        loss_type: Type of loss to use ('mse', 'mae', 'huber') (default: 'mse')
        create_graph: If True, enables higher-order gradients (default: True)
                     Set to False when using dropout to avoid gradient issues
    """

    def __init__(
        self,
        lambda_gd: float = 1.0,
        loss_type: str = "mse",
        create_graph: bool = True
    ):
        super().__init__()
        self.lambda_gd = lambda_gd
        self.loss_type = loss_type
        self.create_graph = create_graph
    
    def forward(
        self,
        ln_gamma1: paddle.Tensor,
        ln_gamma2: paddle.Tensor,
        x1: paddle.Tensor,
        model_output_fn: Optional[Callable] = None
    ) -> paddle.Tensor:
        """Compute Gibbs-Duhem constraint loss.
        
        Args:
            ln_gamma1: Predicted ln(gamma1) [batch_size, 1]
            ln_gamma2: Predicted ln(gamma2) [batch_size, 1]
            x1: Composition of solvent 1 [batch_size, 1]
            model_output_fn: Optional function to compute model outputs given x1
                This is useful if the model is a function of composition
                
        Returns:
            Gibbs-Duhem constraint loss (scalar)
        """
        x1.stop_gradient = False

        if model_output_fn is not None:
            # Use provided function to compute model outputs
            # This is useful when the model is explicitly a function of x1
            outputs = model_output_fn(x1)
            ln_gamma1 = outputs['ln_gamma1']
            ln_gamma2 = outputs['ln_gamma2']
        
        # Compute d(ln(gamma1))/dx1
        dln_gamma1_dx1 = paddle.grad(
            outputs=ln_gamma1,
            inputs=x1,
            create_graph=self.create_graph,
            retain_graph=True,
            allow_unused=True
        )[0]

        # Compute d(ln(gamma2))/dx1
        dln_gamma2_dx1 = paddle.grad(
            outputs=ln_gamma2,
            inputs=x1,
            create_graph=self.create_graph,
            retain_graph=True,
            allow_unused=True
        )[0]

        # Handle None gradients (when prediction mode or no gradient flow)
        if dln_gamma1_dx1 is None:
            dln_gamma1_dx1 = paddle.zeros_like(x1)
        if dln_gamma2_dx1 is None:
            dln_gamma2_dx1 = paddle.zeros_like(x1)
        
        # Gibbs-Duhem constraint: x1*dln_gamma1/dx1 + (1-x1)*dln_gamma2/dx1 = 0
        gd_constraint = x1 * dln_gamma1_dx1 + (1 - x1) * dln_gamma2_dx1
        
        # Compute loss based on loss_type
        if self.loss_type == "mse":
            loss = paddle.mean(gd_constraint ** 2)
        elif self.loss_type == "mae":
            loss = paddle.mean(paddle.abs(gd_constraint))
        elif self.loss_type == "huber":
            # Paddle's huber_loss is in nn.functional or use smooth_l1_loss
            loss = paddle.nn.functional.smooth_l1_loss(gd_constraint, paddle.zeros_like(gd_constraint))
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        
        # Apply weight
        loss = self.lambda_gd * loss
        
        return loss


class GDICombinedLoss(nn.Layer):
    """Combined prediction and Gibbs-Duhem loss for GDI-NN.
    
    This loss combines:
    1. Prediction loss (MSE on ln(gamma) values)
    2. Gibbs-Duhem constraint loss
    
    The Gibbs-Duhem loss can be optionally delayed (started after a certain epoch)
    to allow the model to learn basic predictions first.
    
    Args:
        lambda_gd: Weight for Gibbs-Duhem loss (default: 1.0)
        gd_start_epoch: Epoch to start applying Gibbs-Duhem loss (default: 0)
        current_epoch_fn: Function to get current epoch (optional)
        loss_type: Type of prediction loss ('mse', 'mae', 'huber') (default: 'mse")
        gd_loss_type: Type of Gibbs-Duhem loss ('mse', 'mae', 'huber') (default: 'mse")
        use_ln_gamma: Whether to compute loss on ln(gamma) instead of gamma (default: True)
    """
    
    def __init__(
        self,
        lambda_gd: float = 1.0,
        gd_start_epoch: int = 0,
        current_epoch_fn: Optional[Callable] = None,
        loss_type: str = "mse",
        gd_loss_type: str = "mse",
        use_ln_gamma: bool = True
    ):
        super().__init__()
        self.lambda_gd = lambda_gd
        self.gd_start_epoch = gd_start_epoch
        self.current_epoch_fn = current_epoch_fn
        self.loss_type = loss_type
        self.gd_loss_type = gd_loss_type
        self.use_ln_gamma = use_ln_gamma
        
        # Initialize Gibbs-Duhem loss
        self.gd_loss_fn = GibbsDuhemLoss(
            lambda_gd=lambda_gd,
            loss_type=gd_loss_type
        )
    
    def forward(
        self,
        pred_dict: Dict[str, paddle.Tensor],
        label_dict: Dict[str, paddle.Tensor],
        x1: Optional[paddle.Tensor] = None,
        model_output_fn: Optional[Callable] = None
    ) -> Dict[str, paddle.Tensor]:
        """Compute combined loss.
        
        Args:
            pred_dict: Dictionary of predictions containing:
                - gamma1: Predicted gamma1
                - gamma2: Predicted gamma2
                - ln_gamma1: Predicted ln(gamma1)
                - ln_gamma2: Predicted ln(gamma2)
            label_dict: Dictionary of labels containing:
                - gamma1: Target gamma1
                - gamma2: Target gamma2
            x1: Composition of solvent 1 [batch_size, 1] (required for GD loss)
            model_output_fn: Optional function to compute model outputs given x1
                
        Returns:
            Dictionary of losses:
                - pred_loss: Prediction loss
                - gd_loss: Gibbs-Duhem loss (0 if before gd_start_epoch)
                - total_loss: Combined loss
        """
        # Extract predictions and labels
        gamma1_pred = pred_dict['gamma1']
        gamma2_pred = pred_dict['gamma2']
        ln_gamma1_pred = pred_dict['ln_gamma1']
        ln_gamma2_pred = pred_dict['ln_gamma2']
        
        gamma1_label = label_dict['gamma1']
        gamma2_label = label_dict['gamma2']
        
        # Compute prediction loss
        if self.use_ln_gamma:
            # Use ln(gamma) for better scaling
            # Avoid log(0) by clamping
            ln_gamma1_label = paddle.log(paddle.maximum(gamma1_label, paddle.ones_like(gamma1_label) * 1e-6))
            ln_gamma2_label = paddle.log(paddle.maximum(gamma2_label, paddle.ones_like(gamma2_label) * 1e-6))
            
            pred_loss = self._compute_loss(
                ln_gamma1_pred, ln_gamma1_label
            ) + self._compute_loss(
                ln_gamma2_pred, ln_gamma2_label
            )
        else:
            # Use gamma directly
            pred_loss = self._compute_loss(
                gamma1_pred, gamma1_label
            ) + self._compute_loss(
                gamma2_pred, gamma2_label
            )
        
        # Compute Gibbs-Duhem loss (check if we should apply it)
        gd_loss = paddle.zeros([1], dtype=pred_loss.dtype)
        
        if x1 is not None:
            # Check if we should apply GD loss
            apply_gd_loss = True
            if self.current_epoch_fn is not None:
                current_epoch = self.current_epoch_fn()
                if current_epoch < self.gd_start_epoch:
                    apply_gd_loss = False
            
            if apply_gd_loss:
                gd_loss = self.gd_loss_fn(
                    ln_gamma1_pred, ln_gamma2_pred, x1, model_output_fn
                )
        
        # Total loss
        total_loss = pred_loss + gd_loss
        
        return {
            'pred_loss': pred_loss,
            'gd_loss': gd_loss,
            'total_loss': total_loss
        }
    
    def _compute_loss(
        self,
        pred: paddle.Tensor,
        label: paddle.Tensor
    ) -> paddle.Tensor:
        """Compute loss based on loss_type.
        
        Args:
            pred: Predictions
            label: Labels
            
        Returns:
            Loss value
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(pred, label)
        elif self.loss_type == "mae":
            loss = paddle.mean(paddle.abs(pred - label))
        elif self.loss_type == "huber":
            loss = paddle.nn.functional.smooth_l1_loss(pred, label)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        
        return loss
    
    def set_current_epoch_fn(self, fn: Callable):
        """Set function to get current epoch.
        
        Args:
            fn: Function that returns current epoch as integer
        """
        self.current_epoch_fn = fn


class WeightedGDICombinedLoss(GDICombinedLoss):
    """Weighted combined loss with different weights for gamma1 and gamma2.
    
    This allows asymmetric weighting of the two activity coefficients,
    useful when one solvent is more important or has different uncertainty.
    
    Args:
        weight_gamma1: Weight for gamma1 loss (default: 1.0)
        weight_gamma2: Weight for gamma2 loss (default: 1.0)
        lambda_gd: Weight for Gibbs-Duhem loss (default: 1.0)
        gd_start_epoch: Epoch to start applying Gibbs-Duhem loss (default: 0)
        current_epoch_fn: Function to get current epoch (optional)
        loss_type: Type of prediction loss (default: 'mse')
        gd_loss_type: Type of Gibbs-Duhem loss (default: 'mse')
        use_ln_gamma: Whether to compute loss on ln(gamma) (default: True)
    """
    
    def __init__(
        self,
        weight_gamma1: float = 1.0,
        weight_gamma2: float = 1.0,
        lambda_gd: float = 1.0,
        gd_start_epoch: int = 0,
        current_epoch_fn: Optional[Callable] = None,
        loss_type: str = "mse",
        gd_loss_type: str = "mse",
        use_ln_gamma: bool = True
    ):
        super().__init__(
            lambda_gd=lambda_gd,
            gd_start_epoch=gd_start_epoch,
            current_epoch_fn=current_epoch_fn,
            loss_type=loss_type,
            gd_loss_type=gd_loss_type,
            use_ln_gamma=use_ln_gamma
        )
        self.weight_gamma1 = weight_gamma1
        self.weight_gamma2 = weight_gamma2
    
    def forward(
        self,
        pred_dict: Dict[str, paddle.Tensor],
        label_dict: Dict[str, paddle.Tensor],
        x1: Optional[paddle.Tensor] = None,
        model_output_fn: Optional[Callable] = None
    ) -> Dict[str, paddle.Tensor]:
        """Compute weighted combined loss."""
        # Extract predictions and labels
        gamma1_pred = pred_dict['gamma1']
        gamma2_pred = pred_dict['gamma2']
        ln_gamma1_pred = pred_dict['ln_gamma1']
        ln_gamma2_pred = pred_dict['ln_gamma2']
        
        gamma1_label = label_dict['gamma1']
        gamma2_label = label_dict['gamma2']
        
        # Compute weighted prediction loss
        if self.use_ln_gamma:
            ln_gamma1_label = paddle.log(paddle.maximum(gamma1_label, paddle.ones_like(gamma1_label) * 1e-6))
            ln_gamma2_label = paddle.log(paddle.maximum(gamma2_label, paddle.ones_like(gamma2_label) * 1e-6))
            
            pred_loss = (
                self.weight_gamma1 * self._compute_loss(ln_gamma1_pred, ln_gamma1_label) +
                self.weight_gamma2 * self._compute_loss(ln_gamma2_pred, ln_gamma2_label)
            )
        else:
            pred_loss = (
                self.weight_gamma1 * self._compute_loss(gamma1_pred, gamma1_label) +
                self.weight_gamma2 * self._compute_loss(gamma2_pred, gamma2_label)
            )
        
        # Compute Gibbs-Duhem loss
        gd_loss = paddle.zeros([1], dtype=pred_loss.dtype)
        
        if x1 is not None:
            apply_gd_loss = True
            if self.current_epoch_fn is not None:
                current_epoch = self.current_epoch_fn()
                if current_epoch < self.gd_start_epoch:
                    apply_gd_loss = False
            
            if apply_gd_loss:
                gd_loss = self.gd_loss_fn(
                    ln_gamma1_pred, ln_gamma2_pred, x1, model_output_fn
                )
        
        # Total loss
        total_loss = pred_loss + gd_loss
        
        return {
            'pred_loss': pred_loss,
            'gd_loss': gd_loss,
            'total_loss': total_loss
        }
