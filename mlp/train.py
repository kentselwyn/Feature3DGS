import os
import time
import torch
import logging
import torch.nn as nn
import cProfile
import pstats
import io
from .utils.tools import set_seed
from .utils.modules import get_mlp
from .utils.tensor import batch_to_device
from .utils.image import ImagePreprocessor, load_image
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from argparse import ArgumentParser
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from encoders.superpoint.superpoint import SuperPoint
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = datetime.now().strftime('%H_%M')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def lr_schedule(epoch):
    start = lr_schedule_epoch
    if epoch < start:
        return 1.0
    else:
        factor = 0.9 ** ((epoch - start) // 500)
        return factor


def save_profiling_results(profiler, output_dir, filename):
    """Save profiling results to file and print top functions"""
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    # Save detailed results to file
    profile_output_path = os.path.join(output_dir, f"{filename}.txt")
    with open(profile_output_path, 'w') as f:
        f.write(s.getvalue())
    
    # Print top 20 most time-consuming functions
    print(f"\n{'='*60}")
    print(f"TOP 20 TIME-CONSUMING FUNCTIONS - {filename.upper()}")
    print(f"{'='*60}")
    ps.print_stats(20)
    print(f"Full profiling results saved to: {profile_output_path}")
    print(f"{'='*60}\n")


def profile_feature_extraction(args, encoder, output_dir):
    """Profile the feature extraction process"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Profile feature extraction with GPU preloading option
    dataset = feature_set(args, encoder, preload_to_gpu=args.preload_gpu)
    
    profiler.disable()
    save_profiling_results(profiler, output_dir, "feature_extraction_profile")
    return dataset


def profile_training_epoch_lightweight(model, loader, optimizer, scheduler, l2_loss, writer, epoch, profile_dir):
    """Profile a single training epoch with lightweight profiling - only profile first 10 batches"""
    model.train(True)
    total_loss = 0
    
    # Profile only first 10 batches for performance analysis
    profiler = cProfile.Profile()
    
    for index, data in enumerate(loader):
        # Enable profiling for first 10 batches
        if index < 10:
            profiler.enable()
        
        # Data might be on GPU (if preloaded) or will be moved in __getitem__
        optimizer.zero_grad()
        pred = model(data)
        loss = l2_loss(pred, data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Disable profiling after operations
        if index < 10:
            profiler.disable()
            
        total_loss += loss.item()
        if index % 99 == 0:
            total_iteration = len(loader) * epoch + index
            writer.add_scalar("train/mse", loss.item(), total_iteration)
    
    # Save profiling results for the sampled batches
    save_profiling_results(profiler, profile_dir, f"training_epoch_{epoch+1}_lightweight_profile")
    return total_loss / len(loader)


def profile_training_epoch(model, loader, optimizer, scheduler, l2_loss, writer, epoch):
    """Profile a single training epoch - processes ALL batches like the original"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    model.train(True)
    total_loss = 0
    
    for index, data in enumerate(loader):
        # Data might be on GPU (if preloaded) or will be moved in __getitem__
        optimizer.zero_grad()
        pred = model(data)
        loss = l2_loss(pred, data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if index % 99 == 0:
            total_iteration = len(loader) * epoch + index
            writer.add_scalar("train/mse", loss.item(), total_iteration)
    
    profiler.disable()
    return profiler, total_loss / len(loader)


def manage_checkpoints(save_dir, max_checkpoints=10):
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoint_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    if len(checkpoint_files) > max_checkpoints:
        for old_ckpt in checkpoint_files[:-max_checkpoints]:
            old_ckpt_path = os.path.join(save_dir, old_ckpt)
            os.remove(old_ckpt_path)


class feature_set(Dataset):
    def __init__(self, args, encoder, preload_to_gpu=True) -> None:
        train_path = Path(args.path)/"train/rgb"
        test_path = Path(args.path)/"test/rgb"
        train_images = list(train_path.glob("**/" + "*.png"))
        test_images = list(test_path.glob("**/" + "*.png"))
        all_images = train_images + test_images
        preprocessor = ImagePreprocessor({})
        self.features = []
        self.preload_to_gpu = preload_to_gpu
        
        if preload_to_gpu:
            logger.info(f"Extracting and preloading {len(all_images)} features to GPU...")
        else:
            logger.info(f"Extracting {len(all_images)} features (storing on CPU)...")
            
        with torch.no_grad():
            for idx, img_p in enumerate(all_images):
                img = load_image(img_p)
                data = {}
                data["image"] = img.unsqueeze(0).to(device)
                pred = encoder(data)
                
                if preload_to_gpu:
                    # Keep features on GPU
                    feat = pred["descriptors"].squeeze(0)
                else:
                    # Store on CPU (original behavior)
                    feat = pred["descriptors"].squeeze(0).cpu()
                    
                self.features.append(feat)
                
                # Log progress for large datasets
                if (idx + 1) % 500 == 0 or idx == len(all_images) - 1:
                    logger.info(f"Processed {idx + 1}/{len(all_images)} images...")
        
        if preload_to_gpu:
            logger.info(f"Successfully preloaded {len(self.features)} features to GPU!")
            
            # Check GPU memory usage and warn if usage is high
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated_memory_gb = torch.cuda.memory_allocated() / (1024**3)
                usage_percentage = (allocated_memory_gb / gpu_memory_gb) * 100
                
                logger.info(f"GPU memory: {allocated_memory_gb:.2f}GB / {gpu_memory_gb:.2f}GB allocated ({usage_percentage:.1f}%)")
                
                if usage_percentage > 80:
                    logger.warning(f"High GPU memory usage ({usage_percentage:.1f}%)! Consider reducing batch_size if you encounter OOM errors.")
                elif usage_percentage > 90:
                    logger.error(f"Very high GPU memory usage ({usage_percentage:.1f}%)! OOM errors are likely. Consider reducing dataset size or batch_size.")
        else:
            logger.info(f"Successfully loaded {len(self.features)} features to CPU!")
            
    def __getitem__(self, idx):
        desc = self.features[idx]
        # If features are on CPU, move to GPU on-demand
        if not self.preload_to_gpu:
            desc = desc.to(device)
        return desc
        
    def __len__(self):
        return len(self.features)


def main_original(args, conf):
    """Original main function without profiling for better performance"""
    encoder = SuperPoint(conf).eval().to(device)
    # Initialize dataset and dataloader with GPU preloading option
    dataset = feature_set(args, encoder, preload_to_gpu=args.preload_gpu)
    loader = DataLoader(dataset, batch_size=args.batch_size, 
                       shuffle=True, num_workers=args.num_workers)
    # Model setup
    model = get_mlp(args.dim)
    model.to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    best_vloss = float("inf")
    l2_loss = nn.MSELoss()
    # Logging
    writer = SummaryWriter(f"{args.out_path}/runs/{timestamp}")
    logger.info("Training started.")
    for epoch in tqdm(range(args.epochs)):
        logger.info(f"EPOCH {epoch + 1}:")
        start_time = time.time()
        model.train(True)
        total_loss = 0
        
        for index, data in enumerate(loader):
            # Data might be on GPU (if preloaded) or will be moved in __getitem__
            optimizer.zero_grad()
            pred = model(data)
            loss = l2_loss(pred, data)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if index % 99 == 0:
                total_iteration = len(loader) * epoch + index
                writer.add_scalar("train/mse", loss.item(), total_iteration)
        total_loss /= len(loader)
        logger.info(f"[E {epoch} | iter {index}] loss {total_loss:.6f}")

        epoch_time = time.time() - start_time
        logger.info(f'Time taken for EPOCH {epoch + 1}: {epoch_time:.2f} seconds')
        if total_loss < best_vloss:
            logger.info(f'New checkpoint! total_loss: {total_loss:.6f}')
            best_vloss = total_loss
            checkpoint_path = f'{args.out_path}/epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            manage_checkpoints(args.out_path, max_checkpoints=10)


def main(args, conf):
    # Create profiling output directory
    profile_dir = os.path.join(args.out_path, "profiling")
    os.makedirs(profile_dir, exist_ok=True)
    
    logger.info("Starting profiled training...")
    
    # Profile encoder initialization
    logger.info("Profiling encoder initialization...")
    profiler_init = cProfile.Profile()
    profiler_init.enable()
    encoder = SuperPoint(conf).eval().to(device)
    profiler_init.disable()
    save_profiling_results(profiler_init, profile_dir, "encoder_initialization_profile")
    
    # Profile feature extraction
    logger.info("Profiling feature extraction...")
    dataset = profile_feature_extraction(args, encoder, profile_dir)
    
    # Profile dataloader creation
    logger.info("Profiling dataloader creation...")
    profiler_loader = cProfile.Profile()
    profiler_loader.enable()
    loader = DataLoader(dataset, batch_size=args.batch_size, 
                       shuffle=True, num_workers=args.num_workers)
    profiler_loader.disable()
    save_profiling_results(profiler_loader, profile_dir, "dataloader_creation_profile")
    
    # Profile model setup
    logger.info("Profiling model setup...")
    profiler_model = cProfile.Profile()
    profiler_model.enable()
    model = get_mlp(args.dim)
    model.to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    profiler_model.disable()
    save_profiling_results(profiler_model, profile_dir, "model_setup_profile")
    
    best_vloss = float("inf")
    l2_loss = nn.MSELoss()
    
    # Logging
    writer = SummaryWriter(f"{args.out_path}/runs/{timestamp}")
    logger.info("Training started with profiling enabled.")
    
    # Profile first few epochs
    epochs_to_profile = min(3, args.epochs)
    logger.info(f"Will profile first {epochs_to_profile} epochs in {args.profile_mode} mode...")
    
    for epoch in tqdm(range(args.epochs)):
        logger.info(f"EPOCH {epoch + 1}:")
        start_time = time.time()
        
        if epoch < epochs_to_profile:
            if args.profile_mode == "full":
                # Full profiling - profiles entire epoch (slower but complete)
                logger.info(f"Full profiling epoch {epoch + 1}...")
                epoch_profiler, total_loss = profile_training_epoch(
                    model, loader, optimizer, scheduler, l2_loss, writer, epoch
                )
                save_profiling_results(epoch_profiler, profile_dir, f"training_epoch_{epoch+1}_full_profile")
            else:
                # Lightweight profiling - profiles sample batches (faster)
                logger.info(f"Lightweight profiling epoch {epoch + 1}...")
                total_loss = profile_training_epoch_lightweight(
                    model, loader, optimizer, scheduler, l2_loss, writer, epoch, profile_dir
                )
        else:
            # Regular training without profiling for remaining epochs
            model.train(True)
            total_loss = 0
            
            for index, data in enumerate(loader):
                # Data might be on GPU (if preloaded) or will be moved in __getitem__
                optimizer.zero_grad()
                pred = model(data)
                loss = l2_loss(pred, data)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                if index % 99 == 0:
                    total_iteration = len(loader) * epoch + index
                    writer.add_scalar("train/mse", loss.item(), total_iteration)
            
            total_loss /= len(loader)
        
        logger.info(f"[E {epoch} | iter {index}] loss {total_loss:.6f}")

        epoch_time = time.time() - start_time
        logger.info(f'Time taken for EPOCH {epoch + 1}: {epoch_time:.2f} seconds')
        
        if total_loss < best_vloss:
            logger.info(f'New checkpoint! total_loss: {total_loss:.6f}')
            best_vloss = total_loss
            checkpoint_path = f'{args.out_path}/epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            manage_checkpoints(args.out_path, max_checkpoints=10)
    
    logger.info(f"Training completed. Profiling results saved in: {profile_dir}")
    
    # Create a summary of all profiling results
    summary_path = os.path.join(profile_dir, "profiling_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("PROFILING SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Profiling mode used: {args.profile_mode}\n")
        f.write(f"GPU preloading: {'Enabled' if args.preload_gpu else 'Disabled'}\n\n")
        if args.preload_gpu:
            f.write("OPTIMIZATION: GPU Preloading Enabled\n")
            f.write("- All features are preloaded to GPU during dataset initialization\n")
            f.write("- No data transfers needed during training (except for first batch creation)\n")
            f.write("- This should significantly reduce per-batch overhead\n")
            f.write("- Higher GPU memory usage but faster training\n\n")
        else:
            f.write("STANDARD MODE: CPU Storage with On-Demand Transfer\n")
            f.write("- Features stored on CPU, transferred to GPU per-batch in __getitem__\n")
            f.write("- Lower GPU memory usage but slower training due to transfers\n")
            f.write("- Use this mode if you encounter GPU memory issues\n\n")
        f.write("This directory contains the following profiling results:\n\n")
        f.write("1. encoder_initialization_profile.txt - Time spent initializing SuperPoint encoder\n")
        f.write("2. feature_extraction_profile.txt - Time spent extracting features from all images\n")
        if args.preload_gpu:
            f.write("   (Note: This includes GPU preloading time)\n")
        else:
            f.write("   (Note: Features stored on CPU)\n")
        f.write("3. dataloader_creation_profile.txt - Time spent creating the DataLoader\n")
        f.write("4. model_setup_profile.txt - Time spent setting up MLP model and optimizer\n")
        if args.profile_mode == "full":
            f.write("5. training_epoch_X_full_profile.txt - Complete profiling of entire training epochs\n")
            f.write("   (Note: Full profiling adds significant overhead but gives complete picture)\n\n")
        else:
            f.write("5. training_epoch_X_lightweight_profile.txt - Profiling of sample batches from training epochs\n")
            f.write("   (Note: Lightweight profiling has minimal overhead, profiles first 10 batches per epoch)\n\n")
        f.write("Each file contains detailed function-level timing information.\n")
        f.write("Look for functions with high 'cumtime' (cumulative time) values to identify bottlenecks.\n\n")
        f.write("PERFORMANCE IMPACT:\n")
        f.write(f"- Profile mode: {args.profile_mode}\n")
        f.write(f"- GPU preloading: {'Enabled' if args.preload_gpu else 'Disabled'}\n")
        if args.profile_mode == "full":
            f.write("- Expected profiling overhead: 30-50% slower due to comprehensive profiling\n")
        else:
            f.write("- Expected profiling overhead: 5-10% slower due to lightweight profiling\n")
        if args.preload_gpu:
            f.write("- Training should be faster due to GPU preloading optimization\n")
        else:
            f.write("- Training will have per-batch GPU transfer overhead\n")
    
    logger.info(f"Profiling summary created: {summary_path}")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--lr_schedule_epoch", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_kpts", type=int, default=64)
    parser.add_argument("--sp_th", type=float, default=0.0)
    parser.add_argument("--profile", action="store_true", help="Enable detailed profiling with cProfile")
    parser.add_argument("--profile_mode", type=str, choices=["full", "lightweight"], default="lightweight", 
                       help="Profiling mode: 'full' profiles entire epochs (slower), 'lightweight' profiles sample batches (faster)")
    parser.add_argument("--preload_gpu", action="store_true", default=True, help="Preload all features to GPU for faster training (default: True)")
    parser.add_argument("--no_preload_gpu", dest="preload_gpu", action="store_false", help="Store features on CPU and transfer on-demand (slower but uses less GPU memory)")
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.out_path+"/log.txt", mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    lr_schedule_epoch = args.lr_schedule_epoch
    model_conf = {
        "max_num_keypoints": args.num_kpts,
        "detection_threshold": args.sp_th,
        "force_num_keypoints": True,
    }
    model_conf = OmegaConf.create(model_conf)
    
    if args.profile:
        logger.info("Running with detailed profiling enabled...")
        main(args, model_conf)
    else:
        logger.info("Running without profiling (use --profile flag to enable profiling)...")
        # Run original main function without profiling
        main_original(args, model_conf)
