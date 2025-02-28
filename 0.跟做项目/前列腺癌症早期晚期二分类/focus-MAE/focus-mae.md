# Foucs-MAE

## 生成掩码的部分

	   args.window_size = (args.num_frames // args.tubelet_size,
	                        args.input_size // patch_size[0],
	                        args.input_size // patch_size[1])

设置了每个小windows的大小

然后在run_class_pretrain.py中

 	 transform = DataAugmentationForCandidateRegionVideomae(args, test_mode)

进入获取masked区域的图片视频

下面dataset中

	            if args.decoder_mask_type == 'run_cell':
                self.decoder_mask_map_generator = RunningCellMaskingGenerator(
                    args.window_size, args.decoder_mask_ratio)

				self.candidate_json_path = candidate_region_path

之后RunningCellMaskingGenerator
	
	class RunningCellMaskingGenerator:
	
	    def __init__(self, input_size, mask_ratio=0.5):
	        self.frames, self.height, self.width = input_size
	        self.mask_ratio = mask_ratio
	
	        num_masks_per_cell = int(4 * self.mask_ratio)# masks_per_cell = 2
	        assert 0 < num_masks_per_cell < 4
	        num_patches_per_cell = 4 - num_masks_per_cell # num_patches_per_cell = 2
	
	        self.cell = Cell(num_masks_per_cell, num_patches_per_cell)
	        self.cell_size = self.cell.size
	
	        mask_list = []
	        for ptr_pos in range(self.cell_size):
	            self.cell.set_ptr(ptr_pos)
	            mask = []
	            for _ in range(self.frames):
	                self.cell.run_cell()
	                mask_unit = self.cell.get_cell().reshape(2, 2)
	                mask_map = np.tile(mask_unit,
	                                   [self.height // 2, self.width // 2])
	                mask.append(mask_map.flatten())
	            mask = np.stack(mask, axis=0)
	            mask_list.append(mask)
	        self.all_mask_maps = np.stack(mask_list, axis=0)

在这里拿到ROI

    def get_candidate_region(self, frame_fname):

        vid = frame_fname.split("/")[-2]
        frame = frame_fname.split("/")[-1]
        jsonfile = os.path.join(self.candidate_json_path, f"res_frcnn_{vid}.json")
        try:
            with open(jsonfile, 'r') as f:
                data = json.load(f)
            framebboxes = data['results']
            for frames in framebboxes:
                
                if frames['image_id'] == frame:
                    return frames['Boxes']
        except:
            return []