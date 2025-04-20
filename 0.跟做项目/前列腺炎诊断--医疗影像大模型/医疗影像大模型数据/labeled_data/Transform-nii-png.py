import nibabel as nib
import matplotlib.pyplot as plt
import imageio
import os


nii_file = 'case000000.nii'
save_dir = ' D:cv/cv-computer-visual/0.跟做项目/前列腺炎诊断--医疗影像大模型/医疗影像大模型数据/labeled_data/train_png/label'
img = nib.load(nii_file)

img_fdata = img.get_fdata() # 获取图像数据

# 选择一个切片进行可视化
slice_index = 0
slice_data = img_fdata[45, :, :]
slice_data = slice_data[:,0,:]
plt.imshow(slice_data, cmap='gray')
plt.axis('off')
#plt.show()

plt.savefig(os.path.join(save_dir, f'slice_{slice_index}.png'), bbox_inches='tight', pad_inches=0)

