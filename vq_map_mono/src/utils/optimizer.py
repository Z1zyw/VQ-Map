import torch
import math

def build_optimizer(args, model, group_params: bool = False):
    
    if len(list(model.parameters())) == 0:
        return None, None
    
    if not group_params:
        param  = filter(lambda par: par.requires_grad, model.parameters())
    else:
        param = [
            {'params':[p for n, p in model.named_parameters() if p.requires_grad and 'backbone' not in n and 'encoder' not in n], 'lr':args.lr}, 
            {'params':[p for n, p in model.named_parameters() if p.requires_grad and ('backbone' in n or 'encoder' in n)], 'lr':args.lr_backbone},
        ]
    
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            param,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.lr_step_size, 
            gamma=args.lr_gamma
        )
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            param,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.lr_step_size,
            gamma= args.lr_gamma
        )
    elif args.optimizer.lower() == 'adamw':
        base_lr = args.lr
        optimizer = torch.optim.AdamW(
            param,
            lr=base_lr,
            weight_decay=args.weight_decay
        )
        def warm_up_coslr(current_epoch, warm_up_epoch, max_epoch):
            if current_epoch < warm_up_epoch:
                return (current_epoch + 1) * 1.0 / warm_up_epoch
            else:
                return 0.5 * (1 + math.cos(math.pi * (current_epoch - warm_up_epoch) / (max_epoch - warm_up_epoch))) + 0.1
                # return 0.5 * (1 + math.cos(math.pi * (current_epoch - warm_up_epoch) / (max_epoch - warm_up_epoch))) + 0.01
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: warm_up_coslr(step, 5, args.num_epochs))
    
    else:
        raise ValueError('Optimizer {} is not supported'.format(args.optimizer))
    
    return optimizer, scheduler