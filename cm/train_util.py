import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

from .fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

# auxiliary functions for stochastic patch based training 
import random
from torchvision import transforms 
import torchvision.transforms.functional as TF

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

rotation_transform = MyRotationTransform(angles=[0, 90, 180, 270])
hflip_transform = transforms.RandomHorizontalFlip()

def sample_patch(data,batch_sz,patch_sz,n_patches,device,augment_pipe):
  n, n_ch, w, h = data.size()
  n_mat = n_ch#//2 #//= 2
  h_start = np.random.choice(np.array(range(0,h-patch_sz)), batch_sz*n_patches)
  w_start = np.random.choice(np.array(range(0,w-patch_sz)), batch_sz*n_patches)
  out = th.zeros((n*n_patches,n_mat,patch_sz,patch_sz)).to(device)
  k = 0
  for j in range(n):
    for i in range(n_patches):
      idx_h = th.tensor(range(h_start[k], h_start[k]+patch_sz)).to(device)
      idx_w = th.tensor(range(w_start[k], w_start[k]+patch_sz)).to(device)
      if augment_pipe is None:
        data[j,:,:,:] = hflip_transform(data[j,:,:,:])
        data[j,:,:,:] = rotation_transform(data[j,:,:,:])
      out[k,:,:,:] = data[j,0:n_mat,:,:].index_select(1, idx_h).index_select(2, idx_w)
      k += 1
  return out

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        aug_dim = None
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        
        self.aug_dim = aug_dim

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
                
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step
    
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                ),
            )

        dist_util.sync_params(self.model.parameters())
        dist_util.sync_params(self.model.buffers())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params
    
    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
    
    def run_loop(self):
        while not self.lr_anneal_steps or self.step < self.lr_anneal_steps:
            batch, cond = next(self.data)
            batch =  sample_patch(batch,batch.size(0),256,1,device='cpu',augment_pipe=None)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self.step += 1
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        # zero out gradients 
        self.mp_trainer.zero_grad()
        # microbatch will give the increments 
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            # t/sigma, lambda(t)/lambda(sigma)
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                aug_dim=self.aug_dim,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        #frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        #logger.logkv("step", self.step + self.resume_step)
        #logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    #filename = f"model{(self.step+self.resume_step):06d}.pt"
                    filename = f"model{(self.step):06d}.pt"
                else:
                    #filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                    filename = f"ema_{rate}_{(self.step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                #bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                bf.join(get_blob_logdir(), f"opt{(self.step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()

class CMTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        target_model,
        teacher_model,
        teacher_diffusion,
        training_mode,
        ema_scale_fn,
        total_training_steps,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.training_mode = training_mode
        self.ema_scale_fn = ema_scale_fn
        self.target_model = target_model
        self.teacher_model = teacher_model
        self.teacher_diffusion = teacher_diffusion
        self.total_training_steps = total_training_steps
        
        if target_model:
            self._load_and_sync_target_parameters()
            self.target_model.requires_grad_(False)
            self.target_model.train()

            self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
                self.target_model.named_parameters()
            )
            self.target_model_master_params = make_master_params(
                self.target_model_param_groups_and_shapes
            )

        if teacher_model:
            self._load_and_sync_teacher_parameters()
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

        self.global_step = self.step
        if training_mode == "progdist":
            self.target_model.eval()
            _, scale = ema_scale_fn(self.global_step)
            if scale == 1 or scale == 2:
                _, start_scale = ema_scale_fn(0)
                n_normal_steps = int(np.log2(start_scale // 2)) * self.lr_anneal_steps
                step = self.global_step - n_normal_steps
                if step != 0:
                    self.lr_anneal_steps *= 2
                    self.step = step % self.lr_anneal_steps
                else:
                    self.step = 0
            else:
                self.step = self.global_step % self.lr_anneal_steps

    def _load_and_sync_target_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            target_name = name.replace("model", "target_model")
            resume_target_checkpoint = os.path.join(path, target_name)
            #if bf.exists(resume_target_checkpoint) and (dist.get_rank() < 4): # note: or dist.get_rank() == 1 added by me. Seems necessary to resume with mpiexec and n>1.
            logger.log(
                f"loading model from checkpoint: {resume_target_checkpoint}..."
            )
            self.target_model.load_state_dict(
                dist_util.load_state_dict(
                    resume_target_checkpoint, map_location=dist_util.dev()
                ),
            )

        dist_util.sync_params(self.target_model.parameters())
        dist_util.sync_params(self.target_model.buffers())
    
    def _load_and_sync_teacher_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            teacher_name = name.replace("model", "teacher_model")
            resume_teacher_checkpoint = os.path.join(path, teacher_name)

            if bf.exists(resume_teacher_checkpoint): # and (dist.get_rank() < 4): # note: or dist.get_rank() == 1 added by me. Seems necessary to resume with mpiexec and n>1.
                logger.log(
                    f"loading model from checkpoint: {resume_teacher_checkpoint}..."
                )
                self.teacher_model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_teacher_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.teacher_model.parameters())
        dist_util.sync_params(self.teacher_model.buffers())
    
    def run_loop(self):
        saved = False
        while (
            not self.lr_anneal_steps
            or self.step < self.lr_anneal_steps
            or self.global_step < self.total_training_steps
        ):
            batch, cond = next(self.data)
            batch =  sample_patch(batch,batch.size(0),256,1,device='cpu',augment_pipe=None)
            self.run_step(batch, cond)
            saved = False
            if (
                self.global_step
                and self.save_interval != -1
                and self.global_step % self.save_interval == 0
            ):
                self.save()
                saved = True
                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            if self.global_step % self.log_interval == 0:
                logger.dumpkvs()

        if not saved:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
            if self.target_model:
                self._update_target_ema()
            if self.training_mode == "progdist":
                self.reset_training_for_progdist()
            self.step += 1
            self.global_step += 1

        self._anneal_lr()
        self.log_step()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad(): 
            update_ema(
                self.target_model_master_params,
                self.mp_trainer.master_params,
                rate=target_ema,
            )

            master_params_to_model_params(
                self.target_model_param_groups_and_shapes,
                self.target_model_master_params,
            )

    def reset_training_for_progdist(self):
        assert self.training_mode == "progdist", "Training mode must be progdist"
        if self.global_step > 0:
            scales = self.ema_scale_fn(self.global_step)[1]
            scales2 = self.ema_scale_fn(self.global_step - 1)[1]
            if scales != scales2:
                with th.no_grad():
                    update_ema(
                        self.teacher_model.parameters(),
                        self.model.parameters(),
                        0.0,
                    )
                # reset optimizer
                self.opt = RAdam(
                    self.mp_trainer.master_params,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )

                self.ema_params = [
                    copy.deepcopy(self.mp_trainer.master_params)
                    for _ in range(len(self.ema_rate))
                ]
                if scales == 2:
                    self.lr_anneal_steps *= 2
                self.teacher_model.eval()
                self.step = 0

    def forward_backward(self, batch, cond):
        # zero out gradients 
        self.mp_trainer.zero_grad()
        # microbatch will give the increments 
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            ema, num_scales = self.ema_scale_fn(self.global_step)
            if self.training_mode == "progdist":
                if num_scales == self.ema_scale_fn(0)[1]:
                    compute_losses = functools.partial(
                        self.diffusion.progdist_losses,
                        self.ddp_model,
                        micro,
                        num_scales,
                        target_model=self.teacher_model,
                        target_diffusion=self.teacher_diffusion,
                        model_kwargs=micro_cond,
                    )
                else:
                    compute_losses = functools.partial(
                        self.diffusion.progdist_losses,
                        self.ddp_model,
                        micro,
                        num_scales,
                        target_model=self.target_model,
                        target_diffusion=self.diffusion,
                        model_kwargs=micro_cond,
                    )

            elif self.training_mode == "consistency_distillation":
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.ddp_model,
                    micro,
                    num_scales,
                    target_model=self.target_model,
                    teacher_model=self.teacher_model,
                    teacher_diffusion=self.teacher_diffusion,
                    aug_dim=self.aug_dim,
                    model_kwargs=micro_cond,
                )

            elif self.training_mode == "consistency_training":
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.ddp_model,
                    micro,
                    num_scales,
                    target_model=self.target_model,
                    aug_dim=self.aug_dim,
                    model_kwargs=micro_cond,
                )
            else:
                raise ValueError(f"Unknown training mode {self.training_mode}")

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            self.mp_trainer.backward(loss)

    def save(self):
        import blobfile as bf

        step = self.global_step

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{step:06d}.pt"
                else:
                    filename = f"ema_{rate}_{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        logger.log("saving optimizer state...")
        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{step:06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        if dist.get_rank() == 0:
            if self.target_model:
                logger.log("saving target model state")
                filename = f"target_model{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.target_model.state_dict(), f)
            if self.teacher_model and self.training_mode == "progdist":
                logger.log("saving teacher model state")
                filename = f"teacher_model{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.teacher_model.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()

    def log_step(self):
        step = self.global_step
        logger.logkv("step", step)
        logger.logkv("samples", (step + 1) * self.global_batch)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
