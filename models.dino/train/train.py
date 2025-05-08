import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from models.asymmetric_dino import DINOV2
from models.ecg_transformer import ECGTransformer
from data.ptbxl import PTBXLDataset
from data.custom_dataset import CustomECGDataset
from utils.config import load_config
from utils.logger import setup_logger

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Asymmetric DINO V2 for ECG Analysis')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.training.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config.__dict__, f)
    
    # Setup logger
    logger = setup_logger(output_dir)
    writer = SummaryWriter(log_dir=output_dir)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create datasets
    if config.phase == "pretrain":
        # PTB-XL dataset for pretraining
        student_dataset = PTBXLDataset(
            config.data.ptbxl_path,
            sample_rate=config.data.sample_rate,
            duration=config.data.duration,
            leads=['I'],  # Student only sees lead I
            augment=True,
            metadata=False  # Student doesn't get metadata
        )
        
        teacher_dataset = PTBXLDataset(
            config.data.ptbxl_path,
            sample_rate=config.data.sample_rate,
            duration=config.data.duration,
            leads=config.data.lead_names,  # Teacher sees all 12 leads
            augment=True,
            metadata=True  # Teacher gets metadata
        )
        
        # Combine into a single dataset that returns (student_input, teacher_input, teacher_metadata)
        class CombinedDataset(Dataset):
            def __init__(self, student_ds, teacher_ds):
                self.student_ds = student_ds
                self.teacher_ds = teacher_ds
                
            def __len__(self):
                return len(self.student_ds)
                
            def __getitem__(self, idx):
                student_ecg, _ = self.student_ds[idx]
                teacher_ecg, teacher_metadata = self.teacher_ds[idx]
                return student_ecg, teacher_ecg, teacher_metadata
                
        train_dataset = CombinedDataset(student_dataset, teacher_dataset)
        
    else:  # finetune
        # Custom high-quality dataset for finetuning
        train_dataset = CustomECGDataset(
            config.data.custom_dataset_path,
            sample_rate=config.data.sample_rate,
            duration=config.data.duration,
            student_leads=['I'],
            teacher_leads=config.data.lead_names,
            augment=True,
            metadata=True
        )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    # Create models
    student = ECGTransformer(
        in_channels=1,  # Single lead
        embed_dim=config.model.student_dim,
        depth=config.model.student_depth,
        num_heads=config.model.student_num_heads,  # Use student-specific number of heads
        mlp_ratio=config.model.mlp_ratio,
        patch_size=config.model.patch_size,
        overlap=config.model.overlap,
        num_patches=config.model.num_patches,
        dropout=config.model.dropout,
        use_metadata=False  # Student doesn't use metadata
    )
    
    teacher = ECGTransformer(
        in_channels=12,  # 12 leads
        embed_dim=config.model.teacher_dim,
        depth=config.model.teacher_depth,
        num_heads=config.model.teacher_num_heads,  # Use teacher-specific number of heads
        mlp_ratio=config.model.mlp_ratio,
        patch_size=config.model.patch_size,
        overlap=config.model.overlap,
        num_patches=config.model.num_patches,
        dropout=config.model.dropout,
        use_metadata=True,  # Teacher uses metadata
        metadata_dim=config.model.metadata_dim
    )
    
    # Create DINO model
    model = DINOV2(
        student,
        teacher,
        student_dim=config.model.student_dim,
        teacher_dim=config.model.teacher_dim,
        out_dim=config.model.out_dim,
        warmup_teacher_temp=config.training.warmup_teacher_temp,
        teacher_temp=config.training.teacher_temp,
        warmup_teacher_temp_epochs=config.training.warmup_epochs,
        nepochs=config.training.epochs,
        momentum_teacher=config.training.momentum_teacher,
        center_momentum=config.training.center_momentum,
        freeze_teacher_backbone=config.training.freeze_teacher_backbone,
        freeze_student_backbone=config.training.freeze_student_backbone
    ).to(device)
    
    # Load checkpoint if resuming
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        model.set_epoch(epoch)
        
        total_loss = 0.0
        for batch_idx, (student_input, teacher_input, teacher_metadata) in enumerate(train_loader):
            student_input = student_input.to(device)
            teacher_input = teacher_input.to(device)
            
            # Move metadata to device
            teacher_metadata = {k: v.to(device) for k, v in teacher_metadata.items()}
            
            # Forward pass
            student_proj, teacher_proj = model(student_input, teacher_input, teacher_metadata)
            
            # Compute loss
            loss = dino_loss(student_proj, teacher_proj)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update teacher with EMA
            model.update_teacher()
            
            total_loss += loss.item()
            
            # Log batch progress
            if batch_idx % config.training.log_every == 0:
                logger.info(f"Epoch {epoch+1}/{config.training.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                writer.add_scalar('Loss/batch', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{config.training.epochs}, Avg Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        
        # Save checkpoint
        if (epoch + 1) % config.training.save_every == 0 or (epoch + 1) == config.training.epochs:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    logger.info("Training complete!")
    writer.close()

def dino_loss(student_output, teacher_output, eps=1e-5):
    """DINO loss function."""
    student_out = student_output / student_output.sum(dim=-1, keepdim=True)
    teacher_out = teacher_output / teacher_output.sum(dim=-1, keepdim=True)
    
    # Cross-entropy loss
    loss = - (teacher_out * torch.log(student_out + eps)).sum(dim=-1).mean()
    return loss

if __name__ == "__main__":
    main()
