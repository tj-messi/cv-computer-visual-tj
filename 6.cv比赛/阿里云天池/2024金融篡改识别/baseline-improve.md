#baseline-improve

把数据集用完全

	def normalize_polygon(polygon, img_width, img_height):
	    return [(x / img_width, y / img_height) for x, y in polygon]
	
	for row in training_anno.iloc[10000：].iterrows():
	    shutil.copy(row[1].Path, 'yolo_seg_dataset/train')
	
	    img = cv2.imread(row[1].Path)
	    img_height, img_width = img.shape[:2]
	    txt_filename = os.path.join('yolo_seg_dataset/train/' + row[1].Path.split('/')[-1][:-4] + '.txt')
	    with open(txt_filename, 'w') as up:
	        for polygon in row[1].Polygons:
	            normalized_polygon = normalize_polygon(polygon, img_width, img_height)
	            normalized_coords = ' '.join([f'{coord[0]:.3f} {coord[1]:.3f}' for coord in normalized_polygon])
	            up.write(f'0 {normalized_coords}\n')

10000：

代表把本数据集中后1w的数据全部使用为数据集

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729310510229.png)

然后采用剩余的作为验证集

