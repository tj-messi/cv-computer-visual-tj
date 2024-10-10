#2024金融篡改识别

##注册比赛
在阿里云天池大赛里面找到对应比赛报名即可

##启动魔搭Notebook
启用一个GPU环境

从git上拉下原始环境

	git lfs install
	git clone https://www.modelscope.cn/datasets/Datawhale/dw_AI_defense_track2.git

从.ipynb文件中打开执行文件

##读取数据集

	!apt update > /dev/null; apt install aria2 git-lfs axel -y > /dev/null
	!pip install ultralytics==8.2.0 numpy pandas opencv-python Pillow matplotlib > /dev/null
	!axel -n 12 -a http://mirror.coggle.club/seg_risky_testing_data.zip; unzip -q seg_risky_testing_data.zip
	!axel -n 12 -a  http://mirror.coggle.club/seg_risky_training_data_00.zip; unzip -q seg_risky_training_data_00.zip

##执行yolo

		import os, shutil
		import cv2
		import glob
		import json
		import pandas as pd
		import numpy as np
		import matplotlib.pyplot as plt
		
		training_anno = pd.read_csv('http://mirror.coggle.club/seg_risky_training_anno.csv')
		
		train_jpgs = [x.replace('./', '') for x in glob.glob('./0/*.jpg')]
		training_anno = training_anno[training_anno['Path'].isin(train_jpgs)]
		training_anno['Polygons'] = training_anno['Polygons'].apply(json.loads)
		
		training_anno.head()
		
		training_anno.shape
		
		np.array(training_anno['Polygons'].iloc[4], dtype=np.int32)
		
		idx = 23
		img = cv2.imread(training_anno['Path'].iloc[idx])
		
		plt.figure(figsize=(12, 6))
		plt.subplot(121)
		plt.imshow(img)
		plt.title("Original Image")
		plt.axis('off')
		
		plt.subplot(122)
		img = cv2.imread(training_anno['Path'].iloc[idx])
		polygon_coords = np.array(training_anno['Polygons'].iloc[idx], dtype=np.int32)
		
		for polygon_coord in polygon_coords:
		    cv2.polylines(img, np.expand_dims(polygon_coord, 0), isClosed=True, color=(0, 255, 0), thickness=2)
		    img= cv2.fillPoly(img, np.expand_dims(polygon_coord, 0), color=(255, 0, 0, 0.5))
		
		plt.imshow(img)
		plt.title("Image with Polygons")
		plt.axis('off')
	
	##构建yolo数据集
	training_anno.info()
	if os.path.exists('yolo_seg_dataset'):
	    shutil.rmtree('yolo_seg_dataset')
	
	os.makedirs('yolo_seg_dataset/train')
	os.makedirs('yolo_seg_dataset/valid')
	def normalize_polygon(polygon, img_width, img_height):
	    return [(x / img_width, y / img_height) for x, y in polygon]
	
	for row in training_anno.iloc[:10000].iterrows():
	    shutil.copy(row[1].Path, 'yolo_seg_dataset/train')
	
	    img = cv2.imread(row[1].Path)
	    img_height, img_width = img.shape[:2]
	    txt_filename = os.path.join('yolo_seg_dataset/train/' + row[1].Path.split('/')[-1][:-4] + '.txt')
	    with open(txt_filename, 'w') as up:
	        for polygon in row[1].Polygons:
	            normalized_polygon = normalize_polygon(polygon, img_width, img_height)
	            normalized_coords = ' '.join([f'{coord[0]:.3f} {coord[1]:.3f}' for coord in normalized_polygon])
	            up.write(f'0 {normalized_coords}\n')
	for row in training_anno.iloc[10000:10150].iterrows():
	    shutil.copy(row[1].Path, 'yolo_seg_dataset/valid')
	
	    img = cv2.imread(row[1].Path)
	    img_height, img_width = img.shape[:2]
	    txt_filename = os.path.join('yolo_seg_dataset/valid/' + row[1].Path.split('/')[-1][:-4] + '.txt')
	    with open(txt_filename, 'w') as up:
	        for polygon in row[1].Polygons:
	            normalized_polygon = normalize_polygon(polygon, img_width, img_height)
	            normalized_coords = ' '.join([f'{coord[0]:.3f} {coord[1]:.3f}' for coord in normalized_polygon])
	            up.write(f'0 {normalized_coords}\n')
	with open('yolo_seg_dataset/data.yaml', 'w') as up:
	    data_root = os.path.abspath('yolo_seg_dataset/')
	    up.write(f'''
	path: {data_root}
	train: train
	val: valid
	
	names:
	    0: alter
	''')
	
	!mkdir -p /root/.config/Ultralytics/
	!wget http://mirror.coggle.club/yolo/Arial.ttf -O /root/.config/Ultralytics/Arial.ttf
	!wget http://mirror.coggle.club/yolo/yolov8n-v8.2.0.pt -O yolov8n.pt
	!wget http://mirror.coggle.club/yolo/yolov8n-seg-v8.2.0.pt -O yolov8n-seg.pt

##训练yolo

	from ultralytics import YOLO
	
	model = YOLO("./yolov8n-seg.pt")  
	results = model.train(data="./yolo_seg_dataset/data.yaml", epochs=10, imgsz=640)

##test测试集
	
	from ultralytics import YOLO
	import glob
	from tqdm import tqdm
	
	model = YOLO("./runs/segment/train/weights/best.pt")  
	
	test_imgs = glob.glob('./test_set_A_rename/*/*')
	
	Polygon = []
	for path in tqdm(test_imgs[:10000]):
	    results = model(path, verbose=False)
	    result = results[0]
	    if result.masks is None:
	        Polygon.append([])
	    else:
	        Polygon.append([mask.astype(int).tolist() for mask in result.masks.xy])
	
	import pandas as pd
	submit = pd.DataFrame({
	    'Path': [x.split('/')[-1] for x in test_imgs[:10000]],
	    'Polygon': Polygon
	})
	
	submit = pd.merge(submit, pd.DataFrame({'Path': [x.split('/')[-1] for x in test_imgs[:]]}), on='Path', how='right')
	
	submit = submit.fillna('[]')
	
	submit.to_csv('track2_submit.csv', index=None)

最后生成csv文件进行模型的提交