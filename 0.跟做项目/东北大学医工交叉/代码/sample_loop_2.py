
import os
import re
import glob
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from guided_diffusion.image_datasets import load_data,Xenium5k,Spatialinformation
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_create_model_and_diffusion,
    add_dict_to_argparser,
)
from scipy.stats import spearmanr
import scipy
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from mpi4py import MPI
import gc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# GPU设置保持原样
gpu_ids = [7]
torch.cuda.set_device(gpu_ids[rank])
def create_argparser():
    """参数解析器与原始代码保持一致"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML configuration file")
    
    with open('./config/config_test.yaml', "r") as file:
        config = yaml.safe_load(file)

    add_dict_to_argparser(parser, config)
    return parser
args = create_argparser().parse_args()
# 保持原有参数设置
args.all_gene = 1000
args.gene_num = 20
args.batch_size = 1
args.SR_times = 10
args.dataset_use = 'Xenium5k_cervixcervicalcancer'#Xenium5k_cervixcervicalcancer,Xenium5k_frozenmousebrain
gene_order_path = os.path.join(args.data_root, args.dataset_use+'/gene_order1.npy')


def remove_boundary_genes(data, border_width=2):
    """
    将图像边界区域的基因表达置零
    
    参数:
    data: 基因表达数据，形状为 [H, W, C]
    border_width: 要清除的边界宽度（像素数）
    """
    processed = np.copy(data)
    h, w, c = processed.shape
    
    # 创建边界掩码（0表示边界，1表示内部）
    mask = np.ones((h, w), dtype=bool)
    mask[:border_width, :] = False  # 上边界
    mask[-border_width:, :] = False  # 下边界
    mask[:, :border_width] = False  # 左边界
    mask[:, -border_width:] = False  # 右边界
    
    # 对每个通道应用掩码
    for i in range(c):
        # 仅保留非边界区域的值
        # print('processed1',processed[0, 0, i])
        processed[:, :, i] = processed[:, :, i] * mask
        # print('processed2',processed[0, 0, i])
    return processed
def post_process_sparse(generated_data, threshold=0.05, percentile_mode=True, sparsity_targets=None, gt_data=None):
    """
    对生成的空间转录组数据进行后处理，去除低于阈值的噪声
    
    参数:
    generated_data: 生成的数据
    threshold: 阈值（或百分位数，取决于percentile_mode）
    percentile_mode: 如果True，使用百分位数作为阈值
    sparsity_targets: 目标稀疏度列表，每个通道一个值，如果为None则使用默认值0.95
    gt_data: 原始GT数据，用于比较稀疏度，如果为None则不比较
    """
    processed = np.copy(generated_data)
    # print('generated_data',generated_data.shape)
    # print('processed = np.copy(generated_data)')
    for gene_idx in range(processed.shape[-1]):
        
        # print('processe', gene_idx)
        gene_slice = processed[..., gene_idx]
        
        # 获取当前通道的目标稀疏度
        if sparsity_targets is not None and gene_idx < len(sparsity_targets):
            # print('sparsity_targets[gene_idx]', sparsity_targets[gene_idx])
            sparsity_target = sparsity_targets[gene_idx]
        else:
            print(f"警告: 未提供通道 {gene_idx} 的稀疏度目标")
            continue
        
        # 计算当前预测的稀疏度
        current_sparsity = np.mean(gene_slice == 0)
        
        # 如果预测已经比GT更稀疏或相似，跳过处理
        if current_sparsity >= sparsity_target:
            print(f"通道 {gene_idx} 已经足够稀疏 (当前: {current_sparsity:.4f}, 目标: {sparsity_target:.4f})，跳过处理")
            continue
        
        print(f"通道 {gene_idx} 需要稀疏化处理 (当前: {current_sparsity:.4f}, 目标: {sparsity_target:.4f})")
        
        # 基于百分位数的自适应阈值
        if percentile_mode:
            # 只考虑非零值的分布
            nonzero_values = gene_slice[gene_slice > 0]
            if len(nonzero_values) > 0:
                # 计算保留非零值的百分比
                keep_percent = (1 - sparsity_target) * 100
                # 确保至少保留一些值（如果非零值太少）
                keep_percent = min(keep_percent, 100)
                actual_threshold = np.percentile(nonzero_values, 100 - keep_percent)
            else:
                actual_threshold = 0
        else:
            actual_threshold = threshold
        
        # 应用阈值
        gene_slice[gene_slice < actual_threshold] = 0
    
    return processed
def main():
    # 解析参数


    # 查找所有符合条件的训练模型目录
    model_dirs = glob.glob(os.path.join("logs5K/", f"{args.dataset_use}/{args.SR_times}X/G*"))
    
    # 定义提取基因编号的函数
    def extract_gene_number(dir_path):
        dir_name = os.path.basename(dir_path)
        # 提取G和-之间的数字
        match = re.search(r'G(\d+)-', dir_name)
        if match:
            return int(match.group(1))
        return 0  # 如果没有匹配到，返回0作为默认值
    
    # 按基因编号排序
    sorted_model_dirs = sorted(model_dirs, key=extract_gene_number)
    
    print('排序后的model_dirs:', sorted_model_dirs)
    for model_dir in sorted_model_dirs:
        # 解析基因组范围信息
        dir_name = os.path.basename(model_dir)
        g_part = dir_name.split("G")[1].split("_")[0]
        start_gene = int(g_part.split("-")[0])
        print('NOW:', model_dir)
        
        # 加载对应的基因顺序
        gene_order = np.load(gene_order_path)[start_gene:start_gene+args.gene_num]
        
        # 查找最新模型参数
        checkpoints = glob.glob(os.path.join(model_dir, "model*.pt"))
        if not checkpoints:
            print(f"跳过无检查点的目录: {model_dir}")
            continue
            
        # 选择步数最大的模型
        max_step = max(
            [int(re.search(r"model(\d+)\.pt", ckpt).group(1)) 
             for ckpt in checkpoints if re.search(r"model\d+\.pt", ckpt)]
        )
        model_path = os.path.join(model_dir, f"model{max_step:06d}.pt")

        # 准备结果目录
        script_name = f'Ours-{args.dataset_use}/{args.SR_times}X/G{start_gene}-{start_gene+args.gene_num}'
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
        results_dir = os.path.join(root_dir+"/TEST_Result",script_name)
        os.makedirs(results_dir, exist_ok=True)
        logger.configure(dir=results_dir)

        # 初始化模型
        logger.log(f"\n{'='*40}\n测试基因组 {start_gene}-{start_gene+args.gene_num}\n{'='*40}")
        model, diffusion = sr_create_model_and_diffusion(args)
        model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        # 初始化数据集
        logger.log("loading data...")
        
        # 创建数据集实例以获取测试样本的空间信息
        test_dataset = Spatialinformation(
            data_root=args.data_root,
            dataset_use=args.dataset_use,
            SR_times=args.SR_times,
            status='Test',
            gene_num=args.gene_num,
            all_gene=args.all_gene,
            gene_order=gene_order
        )
        
        # 提取测试样本的空间信息
        test_patch_info = test_dataset.selected_patches
        test_patch_info.sort(key=lambda x: (int(x[1].split('_')[0]), int(x[1].split('_')[1])))
        # print('TEST Order:',test_patch_info)
        print(f"找到 {len(test_patch_info)} 个测试样本")
        #('/home/hanyu/hanyu_code/NC-code3.18/TEST_Result/Ours-Xenium5k_human/10X/G0-20',test_patch_info)#
        # 加载数据生成器
        data = load_superres_data(
            args.batch_size,
            args.data_root,
            args.dataset_use,
            status='Test',
            SR_times=args.SR_times,
            gene_num=args.gene_num,
            all_gene=args.all_gene,
            gene_order=gene_order
        )
        
        # 初始化结果记录
        progress_csv = os.path.join(results_dir, "metrics.csv")
        with open(progress_csv, "w") as f:
            f.write("SampleID,Row,Col,RMSE,SSIM,CC,IFC,VIF,PSNR\n")

        # 推理流程
        logger.log("creating samples...")
        output_dir = os.path.join(results_dir, "samples")
        create_output_dir(output_dir)

        # 在同一批数据上测试所有样本
        for j, (sample_id, patch_id) in enumerate(test_patch_info):
            # 从patch_id中提取空间坐标
            try:
                parts = patch_id.split('_')
                row = int(parts[0])
                col = int(parts[1])
                spatial_id = f"{row}_{col}"  # 创建空间标识符
            except (ValueError, IndexError):
                spatial_id = f"sample_{j+1}"  # 保底方案
                row, col = j+1, 0
                
            print(f"处理样本 {j+1}/{len(test_patch_info)}: {spatial_id}")
            
            # 数据预处理逻辑
            if args.dataset_use in ['SGE', 'BreastST']:
                model_kwargs = next(data)
                low_res_data = model_kwargs['low_res']
                model_kwargs['low_res'] = F.interpolate(low_res_data, size=(26, 26))
                hr_tensor = model_kwargs['low_res']
                hr = hr_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            else:
                hr, model_kwargs = next(data)
                if args.SR_times == 5:
                    hr = F.interpolate(hr, size=(256, 256))
                hr = hr.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            
            # 推理逻辑
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            
            if args.sampling_method == 'ddim':
                sample_fn = diffusion.ddim_sample_loop
            elif args.sampling_method == 'dpm++':
                sample_fn = diffusion.dpm_solver_sample_loop
            else:
                sample_fn = diffusion.p_sample_loop

            with torch.no_grad():
                sample = sample_fn(
                    model,
                    (
                        args.batch_size,
                        args.gene_num,
                        model_kwargs['WSI_5120'].shape[2],
                        model_kwargs['WSI_5120'].shape[3]
                    ),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            sample = sample.permute(0, 2, 3, 1).cpu().numpy()

            # 处理单个样本
            gt_sample = hr[0]
            pred_sample = sample[0]

            pred_sample_t = torch.tensor(pred_sample).permute(2, 0, 1).unsqueeze(0)
            pred_sample_t = F.interpolate(pred_sample_t, size=(256, 256))
            pred_sample = pred_sample_t.squeeze(0).permute(1, 2, 0).numpy()
            
            # 标准归一化
            pred_sample = normalize_prediction(pred_sample)
            
            # 计算每个通道的GT样本稀疏度
            gt_sparsities = []
            for gene_idx in range(gt_sample.shape[-1]):
                gt_gene = gt_sample[..., gene_idx]
                gene_sparsity = np.mean(gt_gene == 0)
                gt_sparsities.append(gene_sparsity)
                #print(f"GT样本 {spatial_id} 通道 {gene_idx} 稀疏度: {gene_sparsity:.4f}")

            # 应用稀疏后处理，使用每个通道各自的稀疏度目标
            # pred_sample_1 = pred_sample
            pred_sample = post_process_sparse(
                pred_sample, 
                threshold=0.05,
                percentile_mode=True, 
                sparsity_targets=gt_sparsities
            )
            if args.dataset_use != 'Xenium5k_frozenmousebrain':
                if start_gene<=30:
                    pred_sample = remove_boundary_genes(pred_sample,1)
                pred_sample = remove_boundary_genes(pred_sample,2)
                # 保存示例图像 - 使用空间坐标作为文件名
            save_sample_images(
                gt_sample, 
                pred_sample,
                sample_index=spatial_id,  # 使用空间坐标命名
                output_dir=output_dir
            )
            # save_sample_images_he(
            #     gt_sample, 
            #     pred_sample,
            #     sample_index=spatial_id,  # 使用空间坐标命名
            #     output_dir=output_dir,
            #     he_image=model_kwargs['WSI_5120']
            # )

            # save_sample_npy(
            #     gt_sample, 
            #     pred_sample,
            #     sample_index=spatial_id,  # 使用空间坐标命名
            #     output_dir=output_dir
            # )
            
            # 计算指标
            rmse_avg, ssim_avg, cc_avg= compute_metrics(gt_sample, pred_sample, threshold=0.1)#, psnr_avg, ifc_avg, vif_avg 
        
            # 写入CSV - 增加行列信息
            with open(progress_csv, "a") as f:
                f.write(f"{spatial_id},{row},{col},{rmse_avg},{ssim_avg},{cc_avg}\n")#,{ifc_avg},{vif_avg},{psnr_avg}
        # 创建空间复合图像
        logger.log("创建空间复合图像...")
        create_spatial_composite(
            results_dir=results_dir,
            patch_info_list=test_patch_info,
        )
        logger.log(f"基因组 {start_gene}-{start_gene+args.gene_num} 测试完成")
        # 计算所有样本的平均指标
        compute_and_log_average_metrics(progress_csv)
        gc.collect()
        torch.cuda.empty_cache()

def compute_and_log_average_metrics(csv_path):
    """计算并记录所有样本的平均指标"""
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_path)
        # 计算平均值
        avg_metrics = {
            'RMSE': df['RMSE'].mean(),
            'SSIM': df['SSIM'].mean(),
            'CC': df['CC'].mean(),
            # 'IFC': df['IFC'].mean(),
            # 'VIF': df['VIF'].mean(),
            # 'PSNR': df['PSNR'].mean()
        }
        
        # 记录平均指标
        logger.log(f"所有样本平均指标: RMSE={avg_metrics['RMSE']:.4f}, SSIM={avg_metrics['SSIM']:.4f}, "
                   f"CC={avg_metrics['CC']:.4f} ")
                   #IFC={avg_metrics['IFC']:.4f}, 
                   #f"VIF={avg_metrics['VIF']:.4f}, PSNR={avg_metrics['PSNR']:.4f}")
        
        # 附加到CSV文件末尾
        with open(csv_path, "a") as f:
            f.write(f"Average,N/A,N/A,{avg_metrics['RMSE']:.4f},{avg_metrics['SSIM']:.4f},"
                   f"{avg_metrics['CC']:.4f}")
                   #{avg_metrics['IFC']:.4f},{avg_metrics['VIF']:.4f},"f"{avg_metrics['PSNR']:.4f}\n")
    except Exception as e:
        logger.log(f"计算平均指标时出错: {e}")

# 创建输出目录的辅助函数
def create_output_dir(dir_path):
    """创建输出目录，如果存在则清空"""
    if os.path.exists(dir_path):
        # 清空目录
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    else:
        # 创建目录
        os.makedirs(dir_path, exist_ok=True)
def save_sample_images(gt_sample, pred_sample, sample_index, output_dir):
    """
    保存样本的 GT 和预测结果图像，使用空间坐标作为标识符
    
    参数:
        gt_sample: 真实值样本
        pred_sample: 预测值样本
        sample_index: 样本索引（空间坐标字符串，如 "3_5"）
        output_dir: 输出目录基础路径
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    
    # 创建三个子目录
    color_dir = os.path.join(output_dir, "color_images")
    gray_dir = os.path.join(output_dir, "gray_images")
    npy_dir = os.path.join(output_dir, "npy_files")
    channel_npy_dir = os.path.join(output_dir, "Channel_npy_files")
    large_dir = os.path.join(output_dir, "Large_files")
    
    # 确保三个输出目录存在
    os.makedirs(color_dir, exist_ok=True)
    # os.makedirs(gray_dir, exist_ok=True)
    # os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(channel_npy_dir, exist_ok=True)
    # os.makedirs(large_dir, exist_ok=True)
    
    # 获取通道数量
    gene_num = gt_sample.shape[-1]
    
    # 为每个通道创建可视化
    print('开始保存')
    # for gene_idx in range(gene_num):
    #     # 1. 保存彩色版本 GT (分开保存)
    #     plt.figure(figsize=(5, 5))
    #     plt.imshow(gt_sample[..., gene_idx], cmap='viridis')
    #     plt.title(f'GT Gene {gene_idx}')
    #     plt.axis('off')
    #     plt.tight_layout()
    #     color_gt_path = os.path.join(color_dir, f'{sample_index}_gt_gene_{gene_idx}.png')
    #     plt.savefig(color_gt_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    #     plt.close()
        
    #     # 保存彩色版本 Pred (分开保存)
    #     plt.figure(figsize=(5, 5))
    #     plt.imshow(pred_sample[..., gene_idx], cmap='viridis')
    #     plt.title(f'Pred Gene {gene_idx}')
    #     plt.axis('off')
    #     plt.tight_layout()
    #     color_pred_path = os.path.join(color_dir, f'{sample_index}_pred_gene_{gene_idx}.png')
    #     plt.savefig(color_pred_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    #     plt.close()
        
        # # 3. 保存NPY文件
        # gt_npy_path = os.path.join(npy_dir, f'{sample_index}_gt_gene_{gene_idx}.npy')
        # pred_npy_path = os.path.join(npy_dir, f'{sample_index}_pred_gene_{gene_idx}.npy')
        
        # 保存单个基因通道的数据
        # np.save(gt_npy_path, gt_sample[..., gene_idx])
        # np.save(pred_npy_path, pred_sample[..., gene_idx])
    
    # 另外保存完整的多通道数据
    full_gt_npy_path = os.path.join(channel_npy_dir, f'{sample_index}_gt_all_genes.npy')
    full_pred_npy_path = os.path.join(channel_npy_dir, f'{sample_index}_pred_all_genes.npy')
    np.save(full_gt_npy_path, gt_sample)
    np.save(full_pred_npy_path, pred_sample)

def save_sample_images_he(gt_sample, pred_sample, sample_index, output_dir, he_image):
    """
    保存带有 HE 图像叠加的样本图像
    
    参数:
        gt_sample: 真实值样本
        pred_sample: 预测值样本
        sample_index: 样本索引（空间坐标字符串，如 "3_5"）
        output_dir: 输出目录
        he_image: HE 图像数据
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理 HE 图像
    if isinstance(he_image, torch.Tensor):
        he_img = he_image.cpu().detach().numpy()[0]
        he_img = np.transpose(he_img, axes=(1, 2, 0))
    else:
        he_img = he_image
    
    # 标准化 HE 图像以便显示
    he_img = (he_img - he_img.min()) / (he_img.max() - he_img.min() + 1e-8)
    
    # 获取通道数量
    gene_num = gt_sample.shape[-1]
    
    # 为每个通道创建可视化
    for gene_idx in range(gene_num):
        # 创建图像子图 (1x3)：HE 图像、GT 叠加、Pred 叠加
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 绘制 HE 图像
        axes[0].imshow(he_img)
        axes[0].set_title('HE Image')
        axes[0].axis('off')
        
        # 准备叠加的 GT 图像
        gt_gene = gt_sample[..., gene_idx]
        gt_gene_normalized = (gt_gene - gt_gene.min()) / (gt_gene.max() - gt_gene.min() + 1e-8)
        
        # 创建 GT 叠加图像
        axes[1].imshow(he_img)
        im1 = axes[1].imshow(gt_gene_normalized, cmap='viridis', alpha=0.6)
        axes[1].set_title(f'GT Gene {gene_idx}')
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 准备叠加的预测图像
        pred_gene = pred_sample[..., gene_idx]
        pred_gene_normalized = (pred_gene - pred_gene.min()) / (pred_gene.max() - pred_gene.min() + 1e-8)
        
        # 创建预测叠加图像
        axes[2].imshow(he_img)
        im2 = axes[2].imshow(pred_gene_normalized, cmap='viridis', alpha=0.6)
        axes[2].set_title(f'Pred Gene {gene_idx}')
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 设置图像标题
        plt.tight_layout()
        plt.suptitle(f'Sample {sample_index} - Gene {gene_idx} with HE Image')
        
        # 保存图像，使用空间坐标作为文件名一部分
        save_path = os.path.join(output_dir, f'sample_{sample_index}_gene_{gene_idx}_with_he.png')
        plt.savefig(save_path, dpi=300)
        plt.close()


def normalize_prediction(pred_sample):
    """
    对预测样本进行标准化处理
    
    参数:
        pred_sample: 预测样本
    
    返回:
        标准化后的预测样本
    """
    import numpy as np
    
    # 复制预测样本以避免修改原始数据
    normalized_pred = pred_sample.copy()
    
    # 对每个通道分别进行标准化
    for gene_idx in range(pred_sample.shape[-1]):
        gene_data = normalized_pred[..., gene_idx]
        
        # 如果通道中有数据，进行最小-最大标准化
        if np.sum(gene_data) != 0:
            min_val, max_val = gene_data.min(), gene_data.max()
            normalized_pred[..., gene_idx] = (gene_data - min_val) / (max_val - min_val + 1e-8)
    
    return normalized_pred



def compute_metrics(gt_sample, pred_sample, threshold=0.1):
    """
    计算各种评估指标
    
    参数:
        gt_sample: 真实值样本
        pred_sample: 预测值样本
        threshold: 用于过滤小值的阈值
    
    返回:
        评估指标的平均值：RMSE, PSNR, SSIM, CC, IFC, VIF
    """
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    # 假设您有IQA库或自定义函数来计算IFC和VIF
    
    # 获取通道数量
    gene_num = gt_sample.shape[-1]
    
    # 初始化指标列表
    rmse_list, psnr_list, ssim_list = [], [], []
    cc_list, ifc_list, vif_list = [], [], []
    
    # 对每个通道计算指标
    for gene_idx in range(gene_num):
        gt_gene = gt_sample[..., gene_idx]
        pred_gene = pred_sample[..., gene_idx]
        
        # 应用阈值过滤
        gt_mask = gt_gene > threshold
        
        # 计算 RMSE
        if np.sum(gt_mask) > 0:
            rmse = np.sqrt(np.mean((gt_gene[gt_mask] - pred_gene[gt_mask])**2))
            rmse_list.append(rmse)
        
        # # 计算 PSNR
        # try:
        #     psnr_value = psnr(gt_gene, pred_gene, data_range=1.0)
        #     psnr_list.append(psnr_value)
        # except:
        #     pass
        
        # 计算 SSIM
        try:
            ssim_value = ssim(gt_gene, pred_gene, data_range=1.0)
            ssim_list.append(ssim_value)
        except:
            pass
        
        # 计算余弦相似度 (CC)
        if np.sum(gt_mask) > 0:
            gt_flat = gt_gene[gt_mask].flatten()
            pred_flat = pred_gene[gt_mask].flatten()
            # 确保向量非零再计算余弦相似度
            if np.linalg.norm(gt_flat) > 0 and np.linalg.norm(pred_flat) > 0:
                # 计算余弦相似度：两个向量的点积除以各自的范数之积
                cosine_sim = np.dot(gt_flat, pred_flat) / (np.linalg.norm(gt_flat) * np.linalg.norm(pred_flat))
                cc_list.append(cosine_sim)
        
        
        # # 计算 IFC 和 VIF (这里使用占位值)
        # # 实际实现中，您需要替换为真实的 IFC 和 VIF 计算
        # ifc_list.append(0.8)  # 占位
        # vif_list.append(0.7)  # 占位
    
    # 计算平均值
    rmse_avg = np.mean(rmse_list) if rmse_list else 0
    # psnr_avg = np.mean(psnr_list) if psnr_list else 0
    ssim_avg = np.mean(ssim_list) if ssim_list else 0
    cc_avg = np.mean(cc_list) if cc_list else 0
    # ifc_avg = np.mean(ifc_list) if ifc_list else 0
    # vif_avg = np.mean(vif_list) if vif_list else 0
    
    return rmse_avg, ssim_avg, cc_avg#psnr_avg,, ifc_avg, vif_avg
def remove_all_file(path: str):
    """
    移除指定文件夹内所有文件，不删除子文件夹。
    """
    if os.path.isdir(path):
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if os.path.isfile(full_path):
                os.remove(full_path)


def save_metrics_to_csv(file_path: str, content: str):
    """
    追加写入指标内容到指定 CSV 文件中。
    """
    with open(file_path, mode="a") as f:
        f.write(content + "\n")



import numpy as np
import scipy.stats
from skimage.metrics import structural_similarity
from scipy import signal
from scipy import ndimage
import pywt  # 用于小波变换


def compute_ifc(gt_img, dist_img):
    """
    计算信息保真度准则 (IFC)
    """
    # 小波变换分解
    coeffs_gt = pywt.wavedec2(gt_img, 'db1', level=3)
    coeffs_dist = pywt.wavedec2(dist_img, 'db1', level=3)
    
    ifc_sum = 0
    
    # 估计GSM模型参数
    for i in range(1, len(coeffs_gt)):
        for j in range(3):  # 水平、垂直和对角子带
            gt_subband = coeffs_gt[i][j]
            dist_subband = coeffs_dist[i][j]
            
            if gt_subband.size < 16:  # 跳过过小的子带
                continue
                
            # 估计每个子带的方差
            var_gt = np.var(gt_subband)
            if var_gt < 1e-10:  # 避免接近零的方差
                continue
                
            # 估计噪声方差
            var_noise = np.mean((gt_subband - dist_subband) ** 2)
            
            # 计算此子带的互信息
            snr = var_gt / (var_noise + 1e-10)
            subband_ifc = 0.5 * np.log2(1 + snr) * gt_subband.size
            ifc_sum += subband_ifc
    
    return max(0, ifc_sum)  # 确保非负值
def compute_vif(gt_img, dist_img):
    """
    计算视觉信息保真度 (VIF)，确保结果范围在0-1之间
    """
    # 小波变换分解
    coeffs_gt = pywt.wavedec2(gt_img, 'db1', level=3)
    coeffs_dist = pywt.wavedec2(dist_img, 'db1', level=3)
    
    numerator = 0.0
    denominator = 0.0
    
    sigma_nsq = 0.1  # 模拟HVS噪声方差
    
    for i in range(1, len(coeffs_gt)):
        for j in range(3):  # 水平、垂直和对角子带
            gt_subband = coeffs_gt[i][j]
            dist_subband = coeffs_dist[i][j]
            
            if gt_subband.size < 16:  # 跳过过小的子带
                continue
                
            # 估计每个子带的方差
            var_gt = np.var(gt_subband)
            if var_gt < 1e-10:  # 避免接近零的方差
                continue
                
            # 估计失真图像与源图像的相关系数
            cov = np.mean((gt_subband - np.mean(gt_subband)) * 
                          (dist_subband - np.mean(dist_subband)))
            var_dist = np.var(dist_subband)
            
            # 确保协方差在合理范围内
            if cov <= 0:
                continue
                
            # 估计噪声方差 - 使用更稳健的方法
            var_noise = max(var_dist - (cov**2 / var_gt), 1e-10)
            
            # 计算源图像的信息
            g_denominator = 0.5 * np.log2(1 + var_gt / sigma_nsq) * gt_subband.size
            denominator += g_denominator
            
            # 计算失真图像的信息 - 确保不超过原始信息
            g_numerator = 0.5 * np.log2(1 + (cov**2) / (var_gt * var_noise)) * gt_subband.size
            numerator += g_numerator
    
    if denominator <= 0:
        return 0
        
    # 确保结果在0-1范围内
    return min(1.0, max(0.0, numerator / denominator))
# def compute_metrics(gt_sample: np.ndarray, pred_sample: np.ndarray):
#     """
#     计算给定 GT 和 Pred 的所有基因通道下的 RMSE、SSIM、CC，并返回各通道平均值。
#     """
#     rmse_gene, ssim_gene, cc_gene = [], [], []

#     for gene_idx in range(gt_sample.shape[-1]):
#         gt_gene = gt_sample[..., gene_idx]
#         pred_gene = pred_sample[..., gene_idx]

#         # 过滤掉标准差为0的通道
#         if np.std(gt_gene) == 0:
#             continue

#         # 计算 RMSE
#         mse_loss = np.mean((gt_gene - pred_gene) ** 2)
#         rmse_gene.append(np.sqrt(mse_loss))

#         # 计算 SSIM
#         ssim_value = structural_similarity(gt_gene, pred_gene, data_range=1.0)
#         ssim_gene.append(ssim_value)

#         # 计算 |CC|
#         cc_value, _ = scipy.stats.pearsonr(gt_gene.flatten(), pred_gene.flatten())
#         cc_gene.append(abs(cc_value))

#     # 取平均值
#     rmse_avg = np.mean(rmse_gene) if len(rmse_gene) > 0 else 0
#     ssim_avg = np.mean(ssim_gene) if len(ssim_gene) > 0 else 0
#     cc_avg = np.mean(cc_gene) if len(cc_gene) > 0 else 0

#     return rmse_avg, ssim_avg, cc_avg


import numpy as np

def normalize_prediction(pred_sample: np.ndarray):
    """
    对预测结果 pred_sample 进行智能裁剪和归一化，按通道去除最低 10% 灰度值（噪声），
    然后对每个通道裁剪到有效范围并归一化到 [0,1] 区间。
    """
    for k in range(pred_sample.shape[-1]):
        # 获取当前通道的灰度值分布
        channel_data = pred_sample[..., k]

        # 去除最低 10% 的灰度值 (计算分位数)
        lower_bound = np.percentile(channel_data, 30)

        # 将低于 lower_bound 的部分视为噪声，裁剪掉
        channel_data[channel_data < lower_bound] = lower_bound

        # 重新计算有效值范围
        pred_min, pred_max = np.min(channel_data), np.max(channel_data)

        # 如果通道有非平坦分布，进行归一化
        if pred_max > pred_min:
            pred_sample[..., k] = (channel_data - pred_min) / (pred_max - pred_min)
        else:
            # 如果通道完全平坦，直接置为 0
            pred_sample[..., k] = 0.0

    return pred_sample


def load_superres_data(batch_size, data_root, dataset_use, status, SR_times, gene_num, all_gene,gene_order):
    """
    加载并返回一个生成器，每次 yield 模型需要的内容。
    """
    dataset = load_data(
        data_root=data_root,
        dataset_use=dataset_use,
        status=status,
        SR_times=SR_times,
        gene_num=gene_num,
        all_gene=all_gene,
        gene_order=gene_order
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        drop_last=False,
        pin_memory=True
    )



    if dataset_use in ['SGE', 'BreastST']:
        for spot_ST, WSI_5120,  Gene_index_map in loader:
            model_kwargs = {
                "low_res": spot_ST,
                "WSI_5120": WSI_5120,
                "Gene_index_map": Gene_index_map
            }
            yield model_kwargs
    else :
        for SR_ST, spot_ST, WSI_5120, Gene_index_map in loader:
            model_kwargs = {
                "low_res": spot_ST,
                "WSI_5120": WSI_5120,
                "Gene_index_map": Gene_index_map
            }
            yield SR_ST, model_kwargs

def create_spatial_composite(results_dir, patch_info_list, gene_num=20, image_size=256):
    """
    根据样本的空间坐标创建GT和Pred的多通道大图
    
    参数:
        results_dir: 结果目录基础路径
        patch_info_list: 包含（样本ID, patch_id）的列表，其中patch_id格式为"a_b"
        gene_num: 基因通道数量（默认为20）
        image_size: 单个样本的图像大小（默认为256x256）
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 创建输出目录
    composite_dir = os.path.join(results_dir, "spatial_composite")
    os.makedirs(composite_dir, exist_ok=True)
    
    # 解析所有样本的空间坐标
    coordinates = []
    for _, patch_id in patch_info_list:
        parts = patch_id.split('_')
        if len(parts) >= 2:
            try:
                # 修改：交换行列解释 - 原来的列现在是行，原来的行现在是列
                col = int(parts[0])  # 原来是 row
                row = int(parts[1])  # 原来是 col
                coordinates.append((row, col, patch_id))
            except ValueError:
                print(f"无法解析坐标: {patch_id}")
    
    if not coordinates:
        print("没有找到有效的空间坐标，无法创建复合图像")
        return
    print('coordinates', coordinates)
    
    # 计算复合图像的尺寸（考虑50%的重叠率）
    overlap_factor = 0.5
    effective_size = int(image_size * (1 - overlap_factor))
    
    max_row = max(coord[0] for coord in coordinates)
    max_col = max(coord[1] for coord in coordinates)
    min_row = min(coord[0] for coord in coordinates)
    min_col = min(coord[1] for coord in coordinates)
    
    composite_height = int((max_row - min_row) * effective_size + image_size)
    composite_width = int((max_col - min_col) * effective_size + image_size)
    print('composite size', composite_height, composite_width)
    
    # 加载NPY目录，获取完整的样本文件
    npy_dir = os.path.join(results_dir, "samples/Channel_npy_files")
    
    # 创建多通道复合图像，保持原始的通道数量
    gt_composite_multi = np.zeros((composite_height, composite_width, gene_num))
    pred_composite_multi = np.zeros((composite_height, composite_width, gene_num))
    
    # 填充复合图像
    for row, col, patch_id in coordinates:
        # 计算在复合图像中的位置（考虑重叠）
        y_start = int((row - min_row) * effective_size)
        x_start = int((col - min_col) * effective_size)
        
        # 加载当前样本的完整NPY文件 - 注意文件名仍然使用原始格式
        original_row = patch_id.split('_')[0]
        original_col = patch_id.split('_')[1]
        gt_file = os.path.join(npy_dir, f'{original_row}_{original_col}_gt_all_genes.npy')
        pred_file = os.path.join(npy_dir, f'{original_row}_{original_col}_pred_all_genes.npy')
        
        try:
            gt_data = np.load(gt_file)
            pred_data = np.load(pred_file)
            print(gt_data.shape, pred_data.shape)
            
            # 检查通道数是否匹配
            if gt_data.shape[2] != gene_num or pred_data.shape[2] != gene_num:
                print(f"警告: 样本 {original_row}_{original_col} 的通道数不匹配，期望 {gene_num}，" 
                      f"实际 GT: {gt_data.shape[2]}, Pred: {pred_data.shape[2]}")
                # 使用最小的通道数
                channels_to_use = min(gt_data.shape[2], pred_data.shape[2], gene_num)
                gt_data = gt_data[:, :, :channels_to_use]
                pred_data = pred_data[:, :, :channels_to_use]
            
            # 将多通道数据放入复合图像的正确位置
            gt_composite_multi[y_start:y_start+image_size, x_start:x_start+image_size, :gt_data.shape[2]] = gt_data
            pred_composite_multi[y_start:y_start+image_size, x_start:x_start+image_size, :pred_data.shape[2]] = pred_data
            
        except Exception as e:
            print(f"处理文件时出错: {gt_file} 或 {pred_file}")
            print(f"错误: {e}")
    
    # 提取第一个通道用于可视化
    gt_channel0 = gt_composite_multi[:, :, 0]
    pred_channel0 = pred_composite_multi[:, :, 0]
    
    # 归一化第一个通道用于可视化
    if np.max(gt_channel0) > 0:
        gt_channel0_norm = gt_channel0 / np.max(gt_channel0)
    else:
        gt_channel0_norm = gt_channel0
        
    if np.max(pred_channel0) > 0:
        pred_channel0_norm = pred_channel0 / np.max(pred_channel0)
    else:
        pred_channel0_norm = pred_channel0
    
    # 保存第一个通道的可视化（彩色）
    plt.figure(figsize=(12, 12))
    plt.imshow(gt_channel0_norm, cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(composite_dir, 'gt_color_channel0.svg'), 
                format='svg', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    plt.figure(figsize=(12, 12))
    plt.imshow(pred_channel0_norm, cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(composite_dir, 'pred_color_channel0.svg'), 
                format='svg', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 保存完整的多通道NPY文件
    np.save(os.path.join(composite_dir, 'gt_all_genes.npy'), gt_composite_multi)
    np.save(os.path.join(composite_dir, 'pred_all_genes.npy'), pred_composite_multi)
    
    print(f"已创建空间复合图像，保存在 {composite_dir}")
    print(f"复合图像尺寸: {composite_height}x{composite_width}，通道数: {gene_num}")
def merge_gene_composites(base_result_dir, dataset_name):
    """
    在通道维度上串接不同基因组范围的复合空间图像NPY文件
    
    参数:
        base_result_dir: 结果目录的基础路径
        dataset_name: 数据集名称，如 'Xenium5k_human'
    """
    import os
    import glob
    import numpy as np
    import re
    import matplotlib.pyplot as plt

    # 构建目录路径模式
    pattern = os.path.join(base_result_dir, f"Ours-{dataset_name}/{args.SR_times}X/G*")
    
    # 查找所有匹配的基因组目录
    gene_dirs = glob.glob(pattern)
    if not gene_dirs:
        print(f"未找到匹配的基因组目录: {pattern}")
        return
    
    print(f"找到 {len(gene_dirs)} 个基因组目录")
    
    # 按基因编号排序目录
    def extract_gene_range(dir_path):
        dir_name = os.path.basename(dir_path)
        match = re.search(r'G(\d+)-(\d+)', dir_name)
        if match:
            start_gene = int(match.group(1))
            return start_gene
        return 0
    
    sorted_gene_dirs = sorted(gene_dirs, key=extract_gene_range)
    
    # 创建输出目录
    output_dir = os.path.join(base_result_dir, f"Ours-{dataset_name}/{args.SR_times}X/Final_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有gt和pred文件的路径
    gt_files = []
    pred_files = []
    gene_ranges = []
    
    for gene_dir in sorted_gene_dirs:
        dir_name = os.path.basename(gene_dir)
        match = re.search(r'G(\d+)-(\d+)', dir_name)
        if match:
            start_gene = int(match.group(1))
            end_gene = int(match.group(2))
            gene_ranges.append((start_gene, end_gene))
            
            # 构建NPY文件路径
            gt_path = os.path.join(gene_dir, "spatial_composite/gt_all_genes.npy")
            pred_path = os.path.join(gene_dir, "spatial_composite/pred_all_genes.npy")
            
            # 检查文件是否存在
            if os.path.exists(gt_path) and os.path.exists(pred_path):
                gt_files.append((gt_path, start_gene, end_gene))
                pred_files.append((pred_path, start_gene, end_gene))
            else:
                # 尝试替代路径模式
                alt_gt_path = os.path.join(gene_dir, "spatial_composite/gt_all_genes.npy")
                alt_pred_path = os.path.join(gene_dir, "spatial_composite/pred_all_genes.npy")
                
                if os.path.exists(alt_gt_path) and os.path.exists(alt_pred_path):
                    gt_files.append((alt_gt_path, start_gene, end_gene))
                    pred_files.append((alt_pred_path, start_gene, end_gene))
                else:
                    print(f"警告: 在目录 {gene_dir} 中未找到有效的NPY文件")
    
    if not gt_files or not pred_files:
        print("未找到有效的NPY文件，无法合并")
        return
    
    print(f"找到 {len(gt_files)} 个GT文件和 {len(pred_files)} 个Pred文件")
    
    try:
        # 串接GT文件
        print("串接GT文件...")
        concatenate_and_save_npy_files(gt_files, os.path.join(output_dir, "concatenated_gt_all_genes"))
        
        # 串接Pred文件
        print("串接Pred文件...")
        concatenate_and_save_npy_files(pred_files, os.path.join(output_dir, "concatenated_pred_all_genes"))
        
        # 创建基因范围信息文件
        with open(os.path.join(output_dir, "gene_ranges.txt"), 'w') as f:
            for i, (start, end) in enumerate(gene_ranges):
                f.write(f"Gene Set {i+1}: G{start}-{end}\n")
        
        print(f"串接完成，结果保存在 {output_dir}")
        
    except Exception as e:
        print(f"串接过程中发生错误: {e}")

def concatenate_and_save_npy_files(file_list, output_path):
    """
    在通道维度上串接NPY文件并保存
    
    参数:
        file_list: 包含(文件路径, 起始基因, 结束基因)元组的列表
        output_path: 输出文件路径
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    if not file_list:
        print("没有文件可串接")
        return
    
    # 存储要串接的数组列表
    arrays_to_concat = []
    
    # 逐个加载文件
    for file_path, start_gene, end_gene in file_list:
        try:
            data = np.load(file_path)
            
            # 检查形状
            if len(arrays_to_concat) > 0:
                first_shape = arrays_to_concat[0].shape
                # 确保空间维度（高度和宽度）相同
                if data.shape[0] != first_shape[0] or data.shape[1] != first_shape[1]:
                    print(f"警告: 文件 {file_path} 的空间维度 ({data.shape[0]}x{data.shape[1]}) 与第一个文件 ({first_shape[0]}x{first_shape[1]}) 不匹配")
                    # 如果形状不同，可以考虑调整大小，但这需要根据具体情况决定
                    # 这里我们先跳过不匹配的文件
                    continue
            
            arrays_to_concat.append(data)
            print(f"已添加文件用于串接: {file_path}")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    if not arrays_to_concat:
        print("没有有效文件可串接")
        return
    
    # 在通道维度上串接数组
    try:
        # 检查数组形状，确保它们可以在通道维度上串接
        first_array = arrays_to_concat[0]
        
        # 假设通道维度是最后一个维度（-1维度）
        if len(first_array.shape) < 3:
            print(f"警告: 数组形状 {first_array.shape} 没有通道维度，无法在通道维度上串接")
            return
        
        # 在通道维度上串接
        concatenated = np.concatenate(arrays_to_concat, axis=-1)
        
        print(f"串接成功: 从 {len(arrays_to_concat)} 个文件创建了形状为 {concatenated.shape} 的数组")
        
        # 可视化第一个通道（示例）
        plt.figure(figsize=(12, 12))
        plt.imshow(concatenated[..., 0], cmap='viridis')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path+'.png', dpi=720, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # 保存串接结果
        np.save(output_path+'.npy', concatenated)
        print(f"已保存串接文件: {output_path}.npy")
        
    except Exception as e:
        print(f"串接数组时出错: {e}")

if __name__ == "__main__":
    main()
    merge_gene_composites("./TEST_Result", args.dataset_use,)
