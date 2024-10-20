#baseline-improve

###把数据集用完全

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

	for row in training_anno.iloc[:10000].iterrows():
	    shutil.copy(row[1].Path, 'yolo_seg_dataset/valid')
	
	    img = cv2.imread(row[1].Path)
	    img_height, img_width = img.shape[:2]
	    txt_filename = os.path.join('yolo_seg_dataset/valid/' + row[1].Path.split('/')[-1][:-4] + '.txt')
	    with open(txt_filename, 'w') as up:
	        for polygon in row[1].Polygons:
	            normalized_polygon = normalize_polygon(polygon, img_width, img_height)
	            normalized_coords = ' '.join([f'{coord[0]:.3f} {coord[1]:.3f}' for coord in normalized_polygon])
	            up.write(f'0 {normalized_coords}\n')

：10000

代表把前10000个作为验证集

###把验证test改一下

	import numpy as np
	Polygon = []
	for path in tqdm(test_imgs[:]):
	    results = model(path, verbose=False)
	    result = results[0]
	    if result.masks is None:
	        Polygon.append([])
	    else:
	        image_polygons=[]
	        for mask in result.masks.xy:
	            mask_coords=np.array(mask)
	            
	            x_min=np.min(mask_coords[:,0])
	            x_max=np.max(mask_coords[:,0])
	            y_min=np.min(mask_coords[:,1])
	            y_max=np.max(mask_coords[:,1])
	       
	            bounding_bos=[
	                [x_min,y_min],
	                [x_min,y_max],
	                [x_max,y_max],
	                [x_max,y_min]
	            ]
	            
	            image_polygons.append(bounding_bos)
	        
	        Polygon.append(image_polygons)

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1729325490585.png)