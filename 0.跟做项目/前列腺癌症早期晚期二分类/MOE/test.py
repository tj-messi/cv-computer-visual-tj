import shutil
import os
import subprocess
import multiprocessing
import shutil
from multiprocessing import Pool
from PIL import Image
import os
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes

from convert import *

def delete_non_empty_folder(folder_path: str):
    try:
        shutil.rmtree(folder_path)  # 删除非空文件夹及其中的所有内容
        print(f"文件夹 {folder_path} 和其中的所有内容已删除")
    except OSError as e:
        print(f"无法删除文件夹 {folder_path}: {e}")

def load_and_convert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
                          min_component_size: int = 50):
    seg = io.imread(input_seg)
    seg[:]=1
    image = io.imread(input_image)
    # print(image.shape)
    if image.shape[-1] == 4:
        image = image[..., :3] 

    image = image.sum(2)
    mask = image == (3 * 255)
    # the dataset has large white areas in which road segmentations can exist but no image information is available.
    # Remove the road label in these areas
    mask = generic_filter_components(mask, filter_fn=lambda ids, sizes: [i for j, i in enumerate(ids) if
                                                                         sizes[j] > min_component_size])
    mask = binary_fill_holes(mask)
    seg[mask] = 0
    seg = seg[..., 0]
    # print(seg.shape)
    io.imsave(output_seg, seg, check_contrast=False)
    image_pil = Image.fromarray(image.astype(np.uint8))  # 转换为 PIL 图像对象
    image_pil = image_pil.convert('RGB')  # 确保是 RGB 图像
    image_pil.save(output_image)  # 保存图像

def sort_rawdata_main(man,kind):
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    source = '/media/tongji/Medical-SAM2-zjz/data/USVideo_final'

    dataset_name = 'Dataset1234_Prostate'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(source, 'train')
    test_source = join(source, 'train')

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(join(train_source, kind, man), join=False, suffix='png')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(train_source,kind, man, v),
                         join(train_source,kind, man, v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v),
                         50
                     ),)
                )
            )

        # test set
        valid_ids = subfiles(join(test_source, kind ,man), join=False, suffix='png')
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(test_source,kind, man, v),
                         join(test_source, kind, man,v),
                         join(imagests, v[:-4] + '_0000.png'),
                         join(labelsts, v),
                         50
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'prostate': 1},
                          num_train, '.png', dataset_name=dataset_name)
    
def main():
    dataset_base_folder = '/media/tongji/Medical-SAM2-zjz/data/USVideo_final/train'
    for kind in os.listdir(dataset_base_folder):
        #print(kind)
        kind_dataset_base_folder = os.path.join(dataset_base_folder,kind)
        for man in os.listdir(kind_dataset_base_folder):
            man_kind_dataset_base_folder = os.path.join(kind_dataset_base_folder,man)
            print(man_kind_dataset_base_folder)

            # 排版数据集
            sort_rawdata_main(man,kind)

            # 预处理数据
            command = [
                    "nnUNetv2_plan_and_preprocess",
                    "-d", "1234",
                    "--verify_dataset_integrity"
                ]
            subprocess.run(command)

            # 获取0,1分割
            os.makedirs(os.path.dirname(f"/media/tongji/nnUNet-master/out/{kind}/{man}"), exist_ok=True)
            command = [
            "nnUNetv2_predict", 
            "-i", "/media/tongji/nnUNet-master/zjz-nnUNetFrame/nnUNet_raw/Dataset1234_Prostate/imagesTr",
            "-o", f"/media/tongji/nnUNet-master/out/{kind}/{man}",
            "-d", "1234",
            "-c", "2d",
            "--save_probabilities"
            ]

            subprocess.run(command)

            # 转化为0,255分割

            directory_path = f"/media/tongji/nnUNet-master/out/{kind}/{man}"
            png_files = get_png_files(directory_path)

            base_out_fold = f"/media/tongji/nnUNet-master/out_convert/{kind}/{man}/"
            os.makedirs(os.path.dirname(base_out_fold), exist_ok=True)

            for i in range(len(png_files)):
                convert_mask(os.path.join(directory_path,png_files[i]),os.path.join(base_out_fold,png_files[i]))

            delete_non_empty_folder("/media/tongji/nnUNet-master/zjz-nnUNetFrame/nnUNet_raw/Dataset1234_Prostate")
            delete_non_empty_folder("/media/tongji/nnUNet-master/zjz-nnUNetFrame/nnUNet_preprocessed/Dataset1234_Prostate")

if __name__ == "__main__" :
    main()