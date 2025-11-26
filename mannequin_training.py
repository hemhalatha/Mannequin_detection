from roboflow import Roboflow
import supervision as sv
import cv2
from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import torch
import shutil
from google.colab import drive

# Mount Google Drive for backups
drive.mount('/content/drive')

# Your existing Roboflow setup
rf = Roboflow(api_key="1VLQJvQUHxOft3M83dmb")
project = rf.workspace("hemu13").project("alu-j6shp")
model = project.version(7).model

# Download dataset
dataset = project.version(2).download("yolov8")
dataset_location = dataset.location
print(f"Dataset downloaded to: {dataset_location}")

class MemoryEfficientYOLOv8Trainer:
    def __init__(self):
        self.setup_environment()
        self.drive_backup_dir = "/content/drive/MyDrive/yolov8_backups"
        self.setup_drive_backup()
        self.current_training_dir = None

    def setup_environment(self):
        """Setup training environment for Colab T4"""
        print("Setting up Colab T4 environment...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        torch.cuda.empty_cache()

    def setup_drive_backup(self):
        """Setup Google Drive backup directory"""
        os.makedirs(self.drive_backup_dir, exist_ok=True)
        print(f"Google Drive backup directory: {self.drive_backup_dir}")

        self.model_backup_dir = os.path.join(self.drive_backup_dir, "models")
        self.results_backup_dir = os.path.join(self.drive_backup_dir, "results")
        os.makedirs(self.model_backup_dir, exist_ok=True)
        os.makedirs(self.results_backup_dir, exist_ok=True)
        print("Google Drive backup structure created!")

    def backup_to_drive(self, source_path, backup_type="model", epoch=None):
        """Backup files to Google Drive"""
        try:
            if not os.path.exists(source_path):
                print(f"Warning: Source path does not exist: {source_path}")
                return False

            if backup_type == "model":
                dest_dir = self.model_backup_dir
                if epoch is not None:
                    filename = os.path.basename(source_path)
                    name, ext = os.path.splitext(filename)
                    if isinstance(epoch, str):
                        new_filename = f"{name}_{epoch}{ext}"
                    else:
                        new_filename = f"{name}_epoch_{epoch:03d}{ext}"
                    dest_path = os.path.join(dest_dir, new_filename)
                else:
                    dest_path = os.path.join(dest_dir, os.path.basename(source_path))
            else:
                dest_dir = self.results_backup_dir
                dest_path = os.path.join(dest_dir, os.path.basename(source_path))

            if os.path.isdir(source_path):
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path)
            else:
                shutil.copy2(source_path, dest_path)

            print(f"[BACKUP SUCCESS] Saved to Google Drive: {dest_path}")
            return True

        except Exception as e:
            print(f"[BACKUP FAILED] Error: {e}")
            return False

    def on_train_epoch_end(self, trainer):
        """Callback function called at the end of each epoch"""
        epoch = trainer.epoch

        # Backup every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n{'='*50}")
            print(f"EPOCH {epoch + 1} BACKUP TO GOOGLE DRIVE")
            print(f"{'='*50}")

            # Backup last.pt (most recent weights)
            last_model_path = Path(trainer.save_dir) / 'weights' / 'last.pt'
            if last_model_path.exists():
                self.backup_to_drive(str(last_model_path), "model", epoch=epoch+1)

            # Backup best.pt if it exists
            best_model_path = Path(trainer.save_dir) / 'weights' / 'best.pt'
            if best_model_path.exists():
                self.backup_to_drive(str(best_model_path), "model", epoch="best")

            print(f"{'='*50}\n")

    def verify_dataset(self, data_yaml_path):
        """Verify dataset structure and contents"""
        print(f"Looking for data.yaml at: {data_yaml_path}")

        if not os.path.exists(data_yaml_path):
            print(f"data.yaml not found at {data_yaml_path}")
            print("Looking for data.yaml in directory...")

            dir_path = os.path.dirname(data_yaml_path)
            if os.path.exists(dir_path):
                all_files = os.listdir(dir_path)
                print(f"Files in directory: {all_files}")

                yaml_files = [f for f in all_files if f.endswith('.yaml')]
                print(f"YAML files found: {yaml_files}")

                if yaml_files:
                    data_yaml_path = os.path.join(dir_path, yaml_files[0])
                    print(f"Using YAML file: {data_yaml_path}")

        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        print("Dataset configuration:")
        print(f"Number of classes: {data_config['nc']}")
        print(f"Class names: {data_config['names']}")

        return data_config

    def train_memory_efficient(self, data_yaml_path, model_size='s', epochs=150):
        """Train YOLOv8 with improved accuracy settings for Colab T4"""

        data_config = self.verify_dataset(data_yaml_path)

        print(f"Loading YOLOv8{model_size} model...")
        model = YOLO(f'yolov8{model_size}.pt')

        # Add callback for periodic backups
        model.add_callback("on_train_epoch_end", self.on_train_epoch_end)

        # Improved training configuration for higher accuracy
        training_config = {
            'data': data_yaml_path,
            'epochs': epochs,
            'imgsz': 640,
            'batch': 16,  # Optimal batch size for T4
            'patience': 30,  # Increased patience for better convergence
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'cache': False,
            'device': 0,
            'workers': 8,
            'project': '/content/yolov8_training',
            'name': f'alu_detection_high_accuracy',
            'exist_ok': True,

            # Improved optimizer settings for accuracy
            'optimizer': 'AdamW',
            'lr0': 0.001,  # Initial learning rate
            'lrf': 0.001,  # Final learning rate (10x lower for fine-tuning)
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,  # Increased warmup for stability
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,

            # Enhanced loss weights for better detection
            'box': 8.0,  # Bounding box loss weight
            'cls': 0.3,  # Classification loss weight
            'dfl': 1.8,  # Distribution focal loss weight

            # Training enhancements
            'close_mosaic': 10,  # Disable mosaic in last N epochs
            'amp': True,  # Automatic Mixed Precision
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,

            # Multi-scale training for better generalization
            'rect': False,  # Rectangular training
            'cos_lr': True,  # Cosine learning rate scheduler
            'label_smoothing': 0.0,  # Label smoothing
            'nbs': 64,  # Nominal batch size
            'plots': True,  # Save training plots
        }

        print("Starting high-accuracy training for Colab T4...")
        print(f"Training on {data_config['nc']} classes: {data_config['names']}")
        print(f"Google Drive backups enabled every 10 epochs: {self.drive_backup_dir}")
        print(f"Augmentation strategy: Enhanced for better generalization")

        torch.cuda.empty_cache()

        # Start training
        results = model.train(**training_config)

        # Store training directory for later use
        self.current_training_dir = results.save_dir

        # Backup final results to Google Drive
        self.backup_final_results(model, results, epochs)

        return results, model

    def backup_final_results(self, model, results, total_epochs):
        """Backup final training results to Google Drive"""
        print("\n" + "="*50)
        print("FINAL GOOGLE DRIVE BACKUP")
        print("="*50)

        # Backup the best model
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        if os.path.exists(best_model_path):
            self.backup_to_drive(str(best_model_path), "model", epoch="best_final")

        # Backup the last model
        last_model_path = results.save_dir / 'weights' / 'last.pt'
        if os.path.exists(last_model_path):
            self.backup_to_drive(str(last_model_path), "model", epoch="last_final")

        # Backup results directory
        results_dir = results.save_dir
        if os.path.exists(results_dir):
            self.backup_to_drive(str(results_dir), "results")

        # Backup training arguments
        args_file = results.save_dir / 'args.yaml'
        if os.path.exists(args_file):
            self.backup_to_drive(str(args_file), "results")

        print("[COMPLETE] Final backup completed!")

    def manual_backup_current_state(self, training_dir=None):
        """Manually backup current training state to Google Drive"""
        print("\n[MANUAL BACKUP] Starting manual backup...")

        if training_dir is None:
            if self.current_training_dir:
                training_dir = str(self.current_training_dir)
            else:
                base_dir = "/content/yolov8_training"
                if os.path.exists(base_dir):
                    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
                    if dirs:
                        latest_dir = max(dirs, key=lambda x: os.path.getctime(os.path.join(base_dir, x)))
                        training_dir = os.path.join(base_dir, latest_dir)

        if training_dir and os.path.exists(training_dir):
            print(f"Backing up: {training_dir}")
            self.backup_to_drive(training_dir, "results")

            weights_dir = os.path.join(training_dir, "weights")
            if os.path.exists(weights_dir):
                for model_file in ['best.pt', 'last.pt']:
                    model_path = os.path.join(weights_dir, model_file)
                    if os.path.exists(model_path):
                        self.backup_to_drive(model_path, "model")
        else:
            print("[ERROR] No training directory found for backup")

# Initialize trainer
trainer = MemoryEfficientYOLOv8Trainer()

# Find the correct data.yaml file
def find_data_yaml(dataset_location):
    """Find the correct data.yaml file"""
    if os.path.exists(f"{dataset_location}/data.yaml"):
        return f"{dataset_location}/data.yaml"
    else:
        for file in os.listdir(dataset_location):
            if file.endswith('.yaml'):
                return f"{dataset_location}/{file}"
        for root, dirs, files in os.walk(dataset_location):
            for file in files:
                if file.endswith('.yaml'):
                    return os.path.join(root, file)
    return None

# Find the correct data.yaml path
data_yaml_path = find_data_yaml(dataset_location)
if data_yaml_path:
    print(f"Found data.yaml at: {data_yaml_path}")
else:
    print("Could not find data.yaml file!")
    print("Directory contents:")
    for item in os.listdir(dataset_location):
        print(f"  {item}")

# Start training with memory-efficient settings
if data_yaml_path and os.path.exists(data_yaml_path):
    print("\n" + "="*60)
    print("STARTING TRAINING WITH GOOGLE DRIVE BACKUP")
    print("="*60)

    results, model = trainer.train_memory_efficient(
        data_yaml_path=data_yaml_path,
        model_size='s',
        epochs=150
    )

    # Final manual backup to ensure everything is saved
    trainer.manual_backup_current_state()

    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"All models and results backed up to: {trainer.drive_backup_dir}")
    print("You can access these files from any Colab account by mounting Google Drive")

else:
    print("ERROR: Could not find data.yaml file. Please check the dataset download.")

# Utility function to restore from backup
def restore_from_drive(backup_dir=None):
    """Restore models from Google Drive backup"""
    if backup_dir is None:
        backup_dir = "/content/drive/MyDrive/yolov8_backups"

    models_dir = os.path.join(backup_dir, "models")
    if os.path.exists(models_dir):
        print("Available backup models:")
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        for model_file in sorted(model_files):
            print(f"  - {model_file}")

        if model_files:
            best_model = [f for f in model_files if 'best' in f]
            if best_model:
                best_model_path = os.path.join(models_dir, best_model[-1])
                shutil.copy(best_model_path, '/content/restored_best.pt')
                print(f"[RESTORED] Model saved to: /content/restored_best.pt")
                return '/content/restored_best.pt'

    print("[ERROR] No backup models found")
    return None

print(f"\nTo restore models later, run: restore_from_drive()")
print(f"Backup location: {trainer.drive_backup_dir}")
