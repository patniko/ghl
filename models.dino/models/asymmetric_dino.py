import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DINOPredictionHead(nn.Module):
    # Same as previous implementation
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(hidden_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x

class DINOV2(nn.Module):
    def __init__(self, student, teacher, student_dim=768, teacher_dim=768, 
                 out_dim=65536, warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_teacher_temp_epochs=30, nepochs=100, 
                 momentum_teacher=0.996, center_momentum=0.9,
                 freeze_teacher_backbone=False, freeze_student_backbone=False):
        super().__init__()
        self.student = student
        self.teacher = teacher
        
        # Freeze backbones if requested
        if freeze_teacher_backbone:
            for param in self.teacher.parameters():
                param.requires_grad = False
                
        if freeze_student_backbone:
            for param in self.student.parameters():
                param.requires_grad = False
        
        # Projection heads
        self.student_proj = DINOPredictionHead(
            student_dim, student_dim // 4, out_dim)
        self.teacher_proj = DINOPredictionHead(
            teacher_dim, teacher_dim // 4, out_dim)
        
        # Initialize teacher with student weights (only if not frozen)
        if not freeze_teacher_backbone:
            # Skip weight copying for asymmetric models
            if student_dim != teacher_dim:
                print("Skipping weight copying for asymmetric models")
                for t_param in self.teacher.parameters():
                    t_param.requires_grad = False
            else:
                for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                    t_param.data.copy_(s_param.data)
                    t_param.requires_grad = False
            
        # DINO specific parameters
        self.epoch = 0
        self.nepochs = nepochs
        self.momentum_teacher = momentum_teacher
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Temperature parameters
        self.teacher_temp_schedule = torch.cat((
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        
    @torch.no_grad()
    def update_teacher(self):
        # EMA update for teacher
        # Skip EMA update for asymmetric models
        if hasattr(self, 'student_dim') and hasattr(self, 'teacher_dim') and self.student_dim != self.teacher_dim:
            return
            
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            if t_param.requires_grad:  # Only update if not frozen
                try:
                    t_param.data = self.momentum_teacher * t_param.data + (1 - self.momentum_teacher) * s_param.data
                except RuntimeError:
                    # Skip parameters with mismatched shapes
                    pass
            
    @torch.no_grad()
    def update_center(self, teacher_output):
        # Update center used for teacher output centering
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        
    def forward(self, student_input, teacher_input, teacher_metadata=None):
        # Student forward pass (only lead I)
        student_output = self.student(student_input)
        student_proj = self.student_proj(student_output)
        
        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_output = self.teacher(teacher_input, teacher_metadata)
            teacher_proj = self.teacher_proj(teacher_output)
            
            # Update center
            self.update_center(teacher_proj)
            
            # Apply centering and sharpening
            teacher_proj = teacher_proj - self.center
            teacher_temp = self.teacher_temp_schedule[self.epoch]
            teacher_proj = teacher_proj / teacher_temp
            
        return student_proj, teacher_proj
    
    def set_epoch(self, epoch):
        self.epoch = epoch
