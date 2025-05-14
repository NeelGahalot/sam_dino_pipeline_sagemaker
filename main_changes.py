import logging
import os
from datetime import datetime

def main():
    # Set up random seed for reproducibility
    args = parse_args() 
    args.epochs = int(os.environ.get('EPOCHS', args.epochs))
    args.batch_size = int(os.environ.get('BATCH_SIZE', args.batch_size))
    
    # Create logging directory and setup logger (Change 1 & 3)
    os.makedirs(os.path.join(args.model_dir, 'logs'), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, 'logs', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info("\nCurrent GPU details:")
        current_device = torch.cuda.current_device()
        logger.info(f"Current GPU ID: {current_device}")
        logger.info(f"Current GPU name: {torch.cuda.get_device_name(current_device)}")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated(current_device)/1024**3:.2f}GB")
        logger.info(f"Memory cached: {torch.cuda.memory_reserved(current_device)/1024**3:.2f}GB")
        logger.info(f"CUDA version: {torch.version.cuda}")

    # Hyperparameters
    batch_size = args.batch_size
    num_workers = 2
    learning_rate = 0.0005
    epochs = args.epochs
    print_freq = 10
    num_classes = 2  # Binary segmentation
    HOME = '/app'
    
    base_dir = args.train_dir
    image_dir = os.path.join(base_dir, 'samples')
    mask_dir = os.path.join(base_dir, 'binary_masks')
    splits_dir = os.path.join(base_dir, 'splits')
    
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Mask directory: {mask_dir}")
    logger.info(f"Splits directory: {splits_dir}")
    
    # Log transforms (Change 3)
    logger.info("Image transforms:")
    for transform in train_img_transform.transforms:
        logger.info(f"  - {transform.__class__.__name__}: {transform}")
    
    logger.info("Mask transforms:")
    for transform in train_mask_transform.transforms:
        logger.info(f"  - {transform.__class__.__name__}: {transform}")

    # Create dataset and dataloader
    dataset = LeafDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split_file=os.path.join(splits_dir, 'train.txt'),
        img_transform=train_img_transform,
        mask_transform=train_mask_transform
    )

    # Create validation dataset
    val_dataset = LeafDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split_file=os.path.join(splits_dir, 'val.txt'),
        img_transform=train_img_transform,
        mask_transform=train_mask_transform
    )

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    logger.info(f"Training dataset size: {len(dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    build_sam_seg = build_sam_vit_h_seg_cnn
    # Model loading 
    sam2_checkpoint = "/app/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    #build the model
    if not os.path.exists(sam2_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {sam2_checkpoint}")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_encoder = sam2_model.image_encoder.to(device)
    model = _build_sam_seg_model(sam2_encoder, 2).to(device)
    model = model.to(device)

    # Freeze image encoder weights
    for name, param in model.named_parameters():
        if param.requires_grad and "image_encoder" in name or "iou" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Loss functions
    dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)
    ce_loss = torch.nn.CrossEntropyLoss()

    # Create save directory
    save_dir = args.model_dir
    tensorboard_dir = os.path.join(save_dir, 'tensorboard')
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))

    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train(train_loader, model, optimizer, dice_loss, ce_loss, epoch, device, print_freq, writer)
        # Validate
        val_loss = validate(val_loader, model, dice_loss, ce_loss, epoch, device, writer)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save model if validation loss improves
        is_best = val_loss < best_loss
        if is_best:
            logger.info('New best checkpoint')
            best_loss = val_loss
            best_model_path = os.path.join(args.model_dir, 'model_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            }, best_model_path)
            logger.info('New best checkpoint saved')
        
        # Save regular checkpoint
        if True:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_path)
            logger.info(f'Checkpoint for epoch - {epoch}, periodically saved')
            
    final_model_path = os.path.join(args.model_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # Add hyperparameter logging with environment variables (Change 2)
    writer.add_hparams(
        {
            'batch_size': args.batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'num_workers': num_workers,
            'is_human_annotation': os.environ.get('IS_HUMAN_ANNOTATION', 'unknown'),
            'annotation_method': os.environ.get('ANNOTATION_METHOD', 'unknown')
        },
        {
            'hparam/best_val_loss': best_loss
        }
    )
    
    writer.close()
    logger.info('Training completed')