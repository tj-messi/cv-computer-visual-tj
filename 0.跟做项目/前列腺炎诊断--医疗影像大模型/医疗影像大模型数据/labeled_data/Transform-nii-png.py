import nibabel as nib
import matplotlib.pyplot as plt
import imageio
import os


nii_file = 'case000000.nii'
img = nib.load(nii_file)
img_fdata = img.get_fdata() # 获取图像数据
print(img_fdata.shape)  # 打印形状 ((81, 118, 88, 1))

save_dir = 'train_png/images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    # 假设 img_fdata 形状为 (x, y, z, t)
if len(img_fdata.shape) == 4:
    (x, y, z, t) = img_fdata.shape
    # 选择其中一个时间点或通道
    img_slice = img_fdata[:, :, :, 0]  # 选取第一个时间点或通道
elif len(img_fdata.shape) == 3:
    (x, y, z) = img_fdata.shape
    img_slice = img_fdata[:, :, z // 2]  # 选取中间切片
else:
    raise ValueError("Unexpected data shape: {}".format(img_fdata.shape))

    for slice_index in range(z):
        slice_data = img_fdata[:, :, slice_index] # 获取切片数据
        imageio.imwrite(os.path.join(save_dir, f'slice_{slice_index}.png'), slice_data)