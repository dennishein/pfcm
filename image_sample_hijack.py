"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import time 
import tqdm 

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample

# auxiliary for quantitative image evaluation
import lpips
loss_fn_alex = lpips.LPIPS(net='alex').to(th.float32)
#from piq import LPIPS
#loss_lpips = LPIPS(replace_pooling=True, reduction="none")
from skimage.metrics import structural_similarity as ssim

def PSNR(gt, img):
    """ PSNR values expected in [0,1]"""
    mse = np.mean(np.square((gt - img)))
    return 20 * np.log10(1) - 10 * np.log10(mse)

def SSIM(gt, img):
    """ SSIM values expected in [0,1]
    See https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity
    """
    return ssim(gt,img,data_range=1)

def LPIPS(gt,img):
    """ LPIPS values expected in [0,1]. Will map to [-1,1] within function"""
    return loss_fn_alex(gt*2-1,img*2-1).item()
    
#def LPIPS(gt,img):
#    """ LPIPS values expected in [0,1]."""
#    img = img.unsqueeze(0).unsqueeze(1)
#    gt = gt.unsqueeze(0).unsqueeze(1)
#    return loss_lpips(gt,img).item()


def standardize(X):
  min_val = X.min()
  max_val = X.max()
  X = (X-min_val)/(max_val-min_val)
  X *= 2.0
  X -= 1.0
  assert X.min() == -1.0 
  assert X.max() == 1.0
  return X

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=os.path.join("./results",args.model_path.split("/")[2],args.data))
    os.makedirs(os.path.join("./results",args.model_path.split("/")[2],args.data),exist_ok=True)

    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    model.to(dist_util.dev())
    #print(model.dtype)
    if args.use_fp16:
        model.convert_to_fp16()
    #print(model.dtype)
    #model.convert_to_fp16()
    model.eval()
    #print(model.dtype)
    #assert 1 == 2

    logger.log("sampling...")
    if args.sampler == "multistep" or args.sampler == "multistep_h":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)
    
    # for conditional generation
    x_cond = None
    img_sz = args.image_size 
    if args.sampler.split('_')[-1] == 'h' or args.sampler.split('_')[-1] == 'hijack':
        print('loading condition data...')
        x_cond = th.load('./datasets/'+args.data+'.pt').to(th.float32)
        if x_cond.size(1)==1:
            x_cond = th.cat((x_cond,x_cond),1)
            print(x_cond.size())
        if args.minmax is not None:
            minmax = th.load('./datasets/'+args.minmax+'.pt')
            x_cond = (x_cond-minmax[0])/(minmax[1]-minmax[0]) # map to [0,1]
        # we assume that that is in [0,1] if no minmax
        print(x_cond.min())
        print(x_cond.max())
        x_cond = x_cond*2.0-1.0
        print(x_cond.min())
        print(x_cond.max())
        n_ch = x_cond.size(1) // 2
        x_cond, x_truth = x_cond[:,0:n_ch,:,:], x_cond[:,n_ch:,:,:]
        img_sz = x_cond.size(-1)
     
    num_samples = args.num_samples
    if x_cond is not None:
      num_samples = x_cond.size(0)  
    
    count = 0
    total_time = 0
    pbar = tqdm.tqdm(total=num_samples)
    if args.aug_dim is not None:
      logger.log(f'Sampling with D={args.aug_dim}')
      #print(f'Sampling with D={args.aug_dim}')
    while len(all_images) * args.batch_size < num_samples:
        x_cond_batch = th.index_select(x_cond,0,th.tensor(range(count*args.batch_size,(count+1)*args.batch_size)))
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
            
        start = time.time()
        sample = karras_sample(
            diffusion,
            model,
            (args.batch_size, n_ch, img_sz, img_sz),
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            x_cond=x_cond_batch,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
            D = args.aug_dim,
            hijack = args.hijack,
            weight = args.weight
        )
        end = time.time()
        total_time += (end-start)
        pbar.update(args.batch_size*dist.get_world_size())
        #sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = ((sample + 1) * 127.5).clamp(0, 255)#.to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        #logger.log(f"created {len(all_images) * args.batch_size} samples")
        count +=1
          
    #print(f'Total time required: {total_time}')
    pbar.close()
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    # For simplicity just append notebook (temporary)
    # Compute quantitative metrics
    x_truth = (x_truth+1.0)/2.0
    arr = th.tensor(arr) / 255.0 # map to [0,1]
    
    lpips_list = []
    ssim_list = []
    psnr_list = []
    for k in range(x_truth.size(0)):  
        img = arr[k,:,:,0].clone().detach()  
        lpips_temp = LPIPS(x_truth[k,0,:,:], img)
        ssim_temp = SSIM(x_truth[k,0,:,:].numpy(), img.numpy())
        psnr_temp = PSNR(x_truth[k,0,:,:].numpy(), img.numpy())

        logger.log(f"Img: {k+1} LPIPS: {round(lpips_temp,4)} SSIM: {round(ssim_temp,4)} PSNR: mean {round(psnr_temp,3)}")

        lpips_list.append(lpips_temp)
        ssim_list.append(ssim_temp)
        psnr_list.append(psnr_temp)

    lpips_mean = np.mean(lpips_list)
    lpips_std = np.std(lpips_list)
    ssim_mean = np.mean(ssim_list)
    ssim_std = np.std(ssim_list)
    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list)

    logger.log(f"*********************************************************************************************")
    logger.log(f"LPIPS: mean {round(lpips_mean,6)} std {round(lpips_std,6)} SSIM: mean {round(ssim_mean,4)} std {round(ssim_std,4)} PSNR: mean {round(psnr_mean,3)} std {round(psnr_std,3)}")
    logger.log(f"*********************************************************************************************")
    
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
        data="val_mayo_1_alt_selected",
        minmax=None,
        aug_dim = None,
        hijack = 1,
        weight = 1.0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
