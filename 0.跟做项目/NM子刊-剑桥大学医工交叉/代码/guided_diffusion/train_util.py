import copy
import functools
import os
import time
import sys
import blobfile as bf
import numpy as np
import torch
import torch as th
import torch.distributed as dist
from scipy.stats import truncnorm
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from collections import Counter
import time
from torch.utils.data import DataLoader
from guided_diffusion import dist_util, logger
#from . import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
#from .fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
#from .nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
#from .resample import LossAwareSampler, UniformSampler
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def np2tensor(arr):
    return th.from_numpy(np.array(arr)).float()


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
            SR_times=10,
            epoch = 1000
    ):
        # 初始化计时器字典，用于记录各部分耗时
        self.time_stats = {
            "data_loading": 0.0,
            "model_forward": 0.0,
            "loss_calculation": 0.0,
            "backward_pass": 0.0,
            "optimizer_step": 0.0,
            "total_step": 0.0,
            "counts": 0
        }
        
        self.epoch = epoch
        self.model = model
        self.SR_times=SR_times
        self.diffusion = diffusion
        
        # 记录数据加载时间
        data_load_start = time.time()
        self.data = data
        self.data_loader = self.get_dataloader()
        self.time_stats["data_loading"] = time.time() - data_load_start
        logger.log(f"Initial data loader creation took: {self.time_stats['data_loading']:.4f}s")
        
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
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # Removed world_size multiplication
        if th.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self.sync_cuda = th.cuda.is_available()

        # 记录模型加载时间
        model_load_start = time.time()
        self._load_parameters()  # Modified to remove sync
        model_load_time = time.time() - model_load_start
        logger.log(f"Model loading took: {model_load_time:.4f}s")
        
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        print(f'Number of Trainable Parameters: {count_parameters(model)}')

        # 在文件顶部添加导入
        try:
            # 检查是否有torch.optim.fused模块
            from torch.optim.fused import FusedAdam
            has_fused_adam = True
        except ImportError:
            has_fused_adam = False
            print("PyTorch中没有FusedAdam，将使用标准AdamW优化器")

        # ... 现有代码 ...
        
        print(f'Number of Trainable Parameters: {count_parameters(model)}')
        
        # 使用PyTorch的FusedAdam替代AdamW
        if has_fused_adam and th.cuda.is_available():
            self.opt = FusedAdam(
                self.mp_trainer.master_params, 
                lr=self.lr, 
                weight_decay=self.weight_decay,
                adam_w_mode=True  # 启用AdamW模式
            )
            print("使用PyTorch FusedAdam优化器")
        else:
            self.opt = AdamW(
                self.mp_trainer.master_params, 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
            print("使用标准AdamW优化器")
        

        # Use model directly without DDP wrapping
        self.ddp_model = self.model
        
    def load_dataloader(self, deterministic=False):
        if deterministic:
            loader = DataLoader(
                self.data, batch_size=self.batch_size, shuffle=False, 
                num_workers=min(64, os.cpu_count()/2), drop_last=True, 
                pin_memory=True, prefetch_factor=16, 
                persistent_workers=True,
            )
        else:
            # 增加num_workers并使用pin_memory=True加速数据传输
            loader = DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, 
                num_workers=min(64, os.cpu_count()/2), drop_last=True, 
                pin_memory=True, prefetch_factor=16, 
                persistent_workers=True,
            )

        while True:
            yield from loader
    def get_dataloader(self):
        loader = self.load_dataloader()
        # 预取数据以减少等待时间
        prefetch_queue = []
        max_prefetch = 4  # 预取3个批次
        
        # 初始预填充队列
        for _ in range(max_prefetch):
            try:
                SR_ST, spot_ST, WSI_5120, Gene_index_map = next(loader)
                model_kwargs = {"low_res": spot_ST, "WSI_5120": WSI_5120, "Gene_index_map": Gene_index_map}
                prefetch_queue.append((SR_ST, model_kwargs))
            except StopIteration:
                break
        
        while prefetch_queue:
            # 返回当前批次
            yield prefetch_queue.pop(0)
            
            # 添加新批次到队列
            try:
                SR_ST, spot_ST, WSI_5120, Gene_index_map = next(loader)
                model_kwargs = {"low_res": spot_ST, "WSI_5120": WSI_5120, "Gene_index_map": Gene_index_map}
                prefetch_queue.append((SR_ST, model_kwargs))
            except StopIteration:
                continue
    def _load_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

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
        Epoch=self.epoch
        # Initialize timing variables
        loop_start_time = time.time()
        last_log_time = loop_start_time
        iter_num=int(99*Epoch/self.batch_size)+1
        while (self.step <= iter_num):
            step_start_time = time.time()
            # if self.step <iter_num*0.01:
            #     ratio=0.1
            # elif  self.step <iter_num*0.03:
            #     ratio = 0.3
            # elif  self.step <iter_num*0.04:
            #     ratio = 0.4
            # elif  self.step <iter_num*0.05:
            #     ratio = 0.5
            # elif  self.step <iter_num*0.06:
            #     ratio = 0.6
            # elif  self.step <iter_num*0.07:
            #     ratio = 0.7
            # elif  self.step <iter_num*0.08:
            #     ratio = 0.8
            # elif  self.step <iter_num*0.09:
            #     ratio = 0.9
            # else:
            ratio=1.0

            # 记录数据加载时间
            data_load_start = time.time()
            batch, cond = next(self.data_loader)
            data_load_time = time.time() - data_load_start
            self.time_stats["data_loading"] += data_load_time
            
            # 记录整个步骤的时间
            self.run_step(batch, cond, ratio)
            
            # Time calculation logic
            current_time = time.time()
            step_duration = current_time - step_start_time
            self.time_stats["total_step"] += step_duration
            self.time_stats["counts"] += 1
            
            total_duration = current_time - loop_start_time
            
            # Calculate remaining time
            remaining_steps = iter_num - self.step
            avg_time_per_step = total_duration / (self.step + 1)  # +1 to avoid division by zero
            estimated_remaining = avg_time_per_step * remaining_steps
            
            # Format time display
            m, s = divmod(total_duration, 60)
            h, m = divmod(m, 60)
            current_time_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
            
            m, s = divmod(estimated_remaining, 60)
            h, m = divmod(m, 60)
            remain_time_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
            
            # Log time at log_interval
            if self.step % self.log_interval == 0:
                # 计算平均时间
                if self.time_stats["counts"] > 0:
                    avg_data_loading = self.time_stats["data_loading"] / self.time_stats["counts"]
                    avg_model_forward = self.time_stats["model_forward"] / self.time_stats["counts"]
                    avg_loss_calculation = self.time_stats["loss_calculation"] / self.time_stats["counts"]
                    avg_backward_pass = self.time_stats["backward_pass"] / self.time_stats["counts"]
                    avg_optimizer_step = self.time_stats["optimizer_step"] / self.time_stats["counts"]
                    avg_total_step = self.time_stats["total_step"] / self.time_stats["counts"]
                    
                    logger.log(f"Step {self.step}/{iter_num} | "
                            f"Elapsed: {current_time_str} | "
                            f"Remain: {remain_time_str} | "
                            f"Speed: {step_duration:.2f}s/step")
                    
                    logger.log(f"Time breakdown - "
                            f"Data loading: {avg_data_loading:.4f}s ({100*avg_data_loading/avg_total_step:.1f}%) | "
                            f"Forward: {avg_model_forward:.4f}s ({100*avg_model_forward/avg_total_step:.1f}%) | "
                            f"Loss calc: {avg_loss_calculation:.4f}s ({100*avg_loss_calculation/avg_total_step:.1f}%) | "
                            f"Backward: {avg_backward_pass:.4f}s ({100*avg_backward_pass/avg_total_step:.1f}%) | "
                            f"Optimizer: {avg_optimizer_step:.4f}s ({100*avg_optimizer_step/avg_total_step:.1f}%)")
                    
                    # 重置计时器
                    if self.step > 0 and self.step % (self.log_interval * 10) == 0:
                        self.time_stats = {
                            "data_loading": 0.0,
                            "model_forward": 0.0,
                            "loss_calculation": 0.0,
                            "backward_pass": 0.0,
                            "optimizer_step": 0.0,
                            "total_step": 0.0,
                            "counts": 0
                        }

            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step!=0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
    
    def run_step(self, batch, cond, ratio):
        # 记录前向传播和损失计算时间
        forward_backward_start = time.time()
        self.forward_backward(batch, cond, ratio)
        forward_backward_time = time.time() - forward_backward_start
        
        # 记录优化器步骤时间
        optimizer_start = time.time()
        took_step = self.mp_trainer.optimize(self.opt)
        optimizer_time = time.time() - optimizer_start
        self.time_stats["optimizer_step"] += optimizer_time
        
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond, ratio):
        self.mp_trainer.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(th.device("cuda" if th.cuda.is_available() else "cpu"))
            if self.SR_times==5:
                micro=F.interpolate(micro, size=(256, 256))
            micro_cond = {
                k: v[i: i + self.microbatch].to(th.device("cuda" if th.cuda.is_available() else "cpu"))
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], th.device("cuda" if th.cuda.is_available() else "cpu"))

            # 记录模型前向传播时间
            forward_start = time.time()
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                ratio,
                model_kwargs=micro_cond,
            )

            # No need for DDP sync checking
            losses = compute_losses()
            forward_time = time.time() - forward_start
            self.time_stats["model_forward"] += forward_time
            
            # 记录损失计算时间
            loss_calc_start = time.time()
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            loss_calc_time = time.time() - loss_calc_start
            self.time_stats["loss_calculation"] += loss_calc_time
            
            # 记录反向传播时间
            backward_start = time.time()
            self.mp_trainer.backward(loss)
            backward_time = time.time() - backward_start
            self.time_stats["backward_pass"] += backward_time

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                    with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                        th.save(state_dict, f)
                # else:
                #     filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"


        save_checkpoint(0, self.mp_trainer.master_params)
        # for rate, params in zip(self.ema_rate, self.ema_params):
        #     save_checkpoint(rate, params)

        # if dist.get_rank() == 0:
        #     # with bf.BlobFile(
        #         # bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
        #         # "wb",
        #     # ) as f:
        #         # th.save(self.opt.state_dict(), f)

        dist.barrier()

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
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
