import os
import torch

def save_checkpoint_accelerate(args, epoch, model, optimizer, scheduler, accelerator):
    if accelerator.is_main_process:
        ckpt = {
            "epoch": epoch,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        ckpt_file = os.path.join(
            args.ckpt_dir, 
            args.exp_name, 
            "checkpoint-{:04d}.pth.gz".format(epoch)
        )
        accelerator.print("==> Saving checkpoint '{}'".format(ckpt_file))
        torch.save(ckpt, ckpt_file)

def load_checkpoint_accelerate(args, model, optimizer, scheduler, accelerator):
    if args.resume_dir is not None:
        ckpt_file = os.path.join(
            args.resume_dir,
            "checkpoint-{:04d}.pth.gz".format(args.resume_epoch)
        )
    else:
        ckpt_file = os.path.join(
            args.ckpt_dir,
            args.exp_name, 
            "checkpoint-{:04d}.pth.gz".format(args.resume_epoch)
        )
    accelerator.print("\n==> Loading checkpoint '{}'...".format(ckpt_file))
    accelerator.print("\n==> Resuming from epoch {}...".format(args.resume_epoch))
    ckpt = torch.load(ckpt_file, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    try: 
        model.load_state_dict(ckpt["model"])
    except ValueError:
        accelerator.print("==> Model state dict not Matched")
        model.load_state_dict(ckpt["model"], strict=False)
    except RuntimeError as e:
        accelerator.print("==> Model state dict not Matched")
        print("RuntimeError: %s"%e)
        model.load_state_dict(ckpt["model"], strict=False)
    
    if not args.use_new_optimizer:
        if optimizer is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                accelerator.print("==> Optimizer state dict not loaded")
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"]