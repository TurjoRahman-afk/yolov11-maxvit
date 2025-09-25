#!/usr/bin/env python3
"""
YOLOv11 + MaxViT Training Script for AutoDL
Author: TurjoRahman-afk
"""

import os
import torch
from ultralytics import YOLO
import wandb
from pathlib import Path

def setup_training_environment():
    """Setup training environment on AutoDL"""
    print("🔧 Setting up training environment...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name()}")
        print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ No GPU detected, using CPU")
    
    # Create necessary directories
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("runs/train", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    return torch.cuda.is_available()

def train_yolov11_maxvit(
    data_path="datasets/coco8/coco8.yaml",
    epochs=100,
    batch_size=16,
    image_size=640,
    workers=8,
    project_name="yolov11-maxvit-training"
):
    """Train YOLOv11 + MaxViT model"""
    
    # Setup environment
    gpu_available = setup_training_environment()
    
    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project=project_name,
        name=f"yolov11-maxvit-{epochs}ep",
        config={
            "model": "yolov11C3TR",
            "epochs": epochs,
            "batch_size": batch_size,
            "image_size": image_size,
            "architecture": "YOLOv11 + MaxViT"
        }
    )
    
    print("🚀 Starting YOLOv11 + MaxViT Training...")
    print(f"📊 Dataset: {data_path}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📦 Batch Size: {batch_size}")
    print(f"🖼️  Image Size: {image_size}")
    
    try:
        # Load your custom YOLOv11C3TR model
        model = YOLO('ultralytics/cfg/models/11/yolov11C3TR.yaml')
        
        print("✅ Model loaded successfully!")
        print(f"📋 Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        
        # Start training
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            workers=workers,
            device=0 if gpu_available else 'cpu',
            project='runs/train',
            name='yolov11_maxvit',
            exist_ok=True,
            save_period=10,  # Save checkpoint every 10 epochs
            val=True,
            plots=True,
            verbose=True,
            # Optimization settings for MaxViT
            optimizer='AdamW',
            lr0=0.001,
            weight_decay=0.0005,
            momentum=0.9,
            warmup_epochs=3,
            cos_lr=True,
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
        )
        
        print("🎉 Training completed successfully!")
        
        # Save the final model
        model.save('weights/yolov11_maxvit_final.pt')
        print("💾 Model saved to weights/yolov11_maxvit_final.pt")
        
        # Run validation
        print("🔍 Running validation...")
        val_results = model.val()
        
        print("📊 Training Results Summary:")
        print(f"📈 mAP50: {val_results.box.map50:.4f}")
        print(f"📈 mAP50-95: {val_results.box.map:.4f}")
        
        return model, results, val_results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise e
    finally:
        wandb.finish()

def quick_test_training():
    """Quick test with minimal epochs for validation"""
    print("🧪 Running quick test training...")
    return train_yolov11_maxvit(
        epochs=5,
        batch_size=8,
        project_name="yolov11-maxvit-test"
    )

def full_training():
    """Full training configuration"""
    print("🏋️ Running full training...")
    return train_yolov11_maxvit(
        epochs=300,
        batch_size=32,  # Adjust based on your GPU memory
        project_name="yolov11-maxvit-full"
    )

if __name__ == "__main__":
    print("🚀 YOLOv11 + MaxViT Training Script")
    print("="*50)
    
    # Check if dataset exists
    if not os.path.exists("datasets/coco8/coco8.yaml"):
        print("⚠️ Dataset not found. Please run setup_autodl.sh first")
        exit(1)
    
    # Choose training mode
    mode = input("Choose training mode:\n1. Quick test (5 epochs)\n2. Full training (300 epochs)\nEnter choice (1 or 2): ")
    
    if mode == "1":
        model, results, val_results = quick_test_training()
    elif mode == "2":
        model, results, val_results = full_training()
    else:
        print("Invalid choice. Running quick test...")
        model, results, val_results = quick_test_training()
    
    print("✅ Training script completed!")
