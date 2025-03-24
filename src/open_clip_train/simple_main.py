import copy
import glob
import logging
import os
import re
import subprocess
import sys
import random
import math
import time
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim

import wandb

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from open_clip_train.data import get_data
from open_clip_train.distributed import  init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from open_clip_train.train import train_one_epoch, evaluate
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync
from open_clip_train.train import AverageMeter, get_autocast, get_input_dtype, backward, unwrap_model
from .dist_utils import (
    destroy_process_group,
    dist_barrier,
    dist_init,
    gather_object,
    get_dist_local_rank,
    get_dist_rank,
    get_dist_size,
    is_dist_initialized,
    is_master
)
from .data_provider.coyo import CoyoDataProvider, CoyoDataProviderConfig
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

LATEST_CHECKPOINT_NAME = "latest.pt"

def get_dist_size() -> int:
    return int(os.environ["WORLD_SIZE"])

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)
    dist_init()
    args.distributed = is_dist_initialized()
    args.rank = get_dist_rank()
    args.local_rank = get_dist_local_rank()
    args.world_size = get_dist_size()
    device = torch.device(f"cuda:{get_dist_local_rank()}")
    cfg: CoyoDataProviderConfig = CoyoDataProviderConfig()
    cfg.data_dir = args.data_dir
    cfg.wds_meta_path = args.wds_meta_path
    cfg.batch_size = args.batch_size
    cfg.mean = (0.48145466,0.4578275,0.40821073)
    cfg.std = (0.26862954,0.26130258,0.27577711)
    cfg.resolution = 224
    data_provider = CoyoDataProvider(cfg)
    args.distill = None
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    # device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master():
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    # args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        # if args.remote_sync is not None:
        #     checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
        #     if args.save_most_recent:
        #         print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
        #         return -1
        #     if args.remote_sync_protocol != 's3':
        #         print('Error. Sync protocol not supported when using resume latest.')
        #         return -1
        if is_master():
            resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
            if not os.path.exists(resume_from):
                # If no latest checkpoint has been saved yet, don't try to resume
                resume_from = None
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        args.resume = resume_from


    # if args.precision == 'fp16':
    #     logging.warning(
    #         'It is recommended to use AMP mixed-precision instead of FP16. '
    #         'FP16 support needs further verification and tuning, especially for train.')

    # if args.horovod:
    #     logging.info(
    #         f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
    #         f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    # elif args.distributed:
    #     logging.info(
    #         f'Running in distributed mode with multiple processes. Device: {args.device}.'
    #         f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    # else:
    #     logging.info(f'Running with a single process. Device {args.device}.')

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )

    random_seed(args.seed, get_dist_rank())

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    # if args.lock_image:
    #     # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
    #     model.lock_image_tower(
    #         unlocked_groups=args.lock_image_unlocked_groups,
    #         freeze_bn_stats=args.lock_image_freeze_bn_stats)
    # if args.lock_text:
    #     model.lock_text_tower(
    #         unlocked_layers=args.lock_text_unlocked_layers,
    #         freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master():
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    opt = getattr(args, 'opt', 'adamw').lower()
    if opt.startswith('timm/'):
        from timm.optim import create_optimizer_v2
        timm_opt = opt.split('timm/')[-1]
        opt_kwargs = {}
        assert (args.beta1 is None) == (args.beta2 is None), \
            'When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified).'
        if args.beta1 is not None:
            opt_kwargs['betas'] = (args.beta1, args.beta2)
        if args.momentum is not None:
            opt_kwargs['momentum'] = args.momentum
        optimizer = create_optimizer_v2(
            model,
            timm_opt,
            lr=args.lr,
            weight_decay=args.wd,
            eps=args.eps,
            **opt_kwargs,
        )
    else:
        # If some params are not passed, we use the default values based on model name.
        # import ipdb; ipdb.set_trace()
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
        if opt == 'adamw':
            optimizer = optim.AdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        else:
            assert False, f'Unknown optimizer {opt}'

        if is_master():
            if is_master():
                defaults = copy.deepcopy(optimizer.defaults)
                defaults['weight_decay'] = args.wd
                defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
                logging.info(
                    f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
                )

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    start_epoch = 0
    batch_index = 0
    # import ipdb; ipdb.set_trace()
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            batch_index = checkpoint["batch_index"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch}, batch_index {batch_index})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    # assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, args.total_steps)
    # if 'train' in data and optimizer is not None:
    #     total_steps = (data["train"].dataloader.num_batches) * args.epochs
    #     if args.lr_scheduler == "cosine":
    #         scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    #     elif args.lr_scheduler == "const":
    #         scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
    #     elif args.lr_scheduler == "const-cooldown":
    #         assert args.epochs_cooldown is not None,\
    #             "Please specify the number of cooldown epochs for this lr schedule."
    #         cooldown_steps = (data["train"].dataloader.num_batches) * args.epochs_cooldown
    #         scheduler = const_lr_cooldown(
    #             optimizer, args.lr, args.warmup, total_steps,
    #             cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
    #     else:
    #         logging.error(
    #             f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
    #         exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master()
    writer = None

    if args.wandb and is_master():
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        # you will have to configure this for your project!
        import hashlib
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project_name,
            name=args.name,
            id=hashlib.sha1((args.wandb_project_name + args.name).encode("utf-8")).hexdigest(),
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')

        if args.grad_checkpointing and args.distributed:
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        model = torch.compile(original_model)

    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master():
            logging.info(f'Start epoch {epoch}')

        # train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        device = torch.device(args.device)
        autocast = get_autocast(args.precision, device_type=device.type)
        input_dtype = get_input_dtype(args.precision)

        model.train()
        if epoch == start_epoch:
            data_provider.sampler.set_epoch(epoch)
            data_provider.set_batch_index(batch_index)
        else:
            data_provider.sampler.set_epoch(epoch)
            batch_index = 0
        dataloader = data_provider.data_loader
        num_batches_per_epoch = len(dataloader)
        sample_digits = math.ceil(math.log(len(dataloader.dataset) + 1, 10))

        losses_m = {}
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        end = time.time()
        for i, batch in enumerate(dataloader):
            step = num_batches_per_epoch * epoch + batch_index

            if not args.skip_scheduler:
                scheduler(step)

            images, texts = batch
            texts = texts["caption"]
            texts = tokenizer(texts)
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            data_time_m.update(time.time() - end)
            optimizer.zero_grad()

            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            batch_index += 1
            backward(total_loss, scaler)

            if scaler is not None:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                optimizer.step()

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

            batch_time_m.update(time.time() - end)
            end = time.time()
            batch_count = batch_index + 1
            if is_master() and (batch_index % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
                batch_size = len(images)
                num_samples = batch_count * batch_size * args.world_size
                samples_per_epoch = len(dataloader.dataset)
                percent_complete = 100.0 * batch_count / num_batches_per_epoch

                # NOTE loss is coarsely sampled, just master node and per log update
                for key, val in losses.items():
                    if key not in losses_m:
                        losses_m[key] = AverageMeter()
                    losses_m[key].update(val.item(), batch_size)

                logit_scale_scalar = logit_scale.item()
                loss_log = " ".join(
                    [
                        f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                        for loss_name, loss_m in losses_m.items()
                    ]
                )
                samples_per_second = args.batch_size * args.world_size / batch_time_m.val
                samples_per_second_per_gpu = args.batch_size / batch_time_m.val
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "load_data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "logit_scale": logit_scale_scalar,
                    "lr": optimizer.param_groups[0]["lr"]
                }            
                log_data.update({name:val.val for name,val in losses_m.items()})

                log_data = {"train/" + name: val for name, val in log_data.items()}
                
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    log_data['step'] = step  # for backwards compatibility
                    wandb.log(log_data, step=step)
                
                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()
            if is_master() and (batch_index % args.save_checkpoint_steps == 0):
                checkpoint_dict = {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "state_dict": original_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if scaler is not None:
                    checkpoint_dict["scaler"] = scaler.state_dict()
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{epoch}_batch_{batch_index}.pt"),
                )
                # save latest checkpoint
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)
                logging.info(f"Saved checkpoint at epoch {epoch}, batch {batch_index}.")
            if is_master() and (batch_index % args.evaluation_steps == 0):
                evaluate(model, data, epoch, args, tb_writer=writer, tokenizer=tokenizer, batch_index=batch_index)
            if batch_index + epoch * num_batches_per_epoch >= args.total_steps:
                break
        # end for
        # completed_epoch = epoch + 1

        # if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
        #     evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

        # # Saving checkpoints.
        # if args.save_logs:
        #     checkpoint_dict = {
        #         "epoch": completed_epoch,
        #         "name": args.name,
        #         "state_dict": original_model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #     }
        #     if scaler is not None:
        #         checkpoint_dict["scaler"] = scaler.state_dict()

        #     if completed_epoch == args.epochs or (
        #         args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
        #     ):
        #         torch.save(
        #             checkpoint_dict,
        #             os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
        #         )

        #     if args.save_most_recent:
        #         # try not to corrupt the latest checkpoint if save fails
        #         tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
        #         latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
        #         torch.save(checkpoint_dict, tmp_save_path)
        #         os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master():
        wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
