import h5py
import numpy as np
import cv2
import random
import os
import math
import shutil
import argparse

	
#mat_file: COFW_train.mat, COFW_test.mat
#img_token: 'IsTr', 'IsT'
#bbox_token: 'bboxesTr', 'bboxesT'
#phis_token: 'phisTr', 'phisT'
def mat_to_files(mat_file, img_token, bbox_token, phis_token, img_dir, gt_txt_file):
	train_mat = h5py.File(mat_file, 'r')
	tr_imgs_obj = train_mat[img_token][:]
	total_num = tr_imgs_obj.shape[1]
	# print(total_num)
	
	with open(gt_txt_file, "w+") as trf:
		for i in range(total_num):
			img = train_mat[tr_imgs_obj[0][i]][:]
			bbox = train_mat[bbox_token][:]
			bbox = np.transpose(bbox)[i]
			
			img = np.transpose(img)
			if not os.path.exists(img_dir):
				os.mkdir(img_dir)
				
			cv2.imwrite(img_dir + "/{}.jpg".format(i), img)
			gt = train_mat[phis_token][:]
			gt = np.transpose(gt)[i]
			
			content = img_dir + "/{}.jpg,".format(i)
			for k in range(bbox.shape[0]):
				content = content + bbox[k].astype(str) + ' '
			content += ','
			for k in range(gt.shape[0]):
				content = content + gt[k].astype(str) + ' '
			content += '\n'
			trf.write(content)


def move_test_to_train(test_gt_txt, train_gt_txt, new_test_txt, new_train_txt, test_num):
	shutil.copy(train_gt_txt, new_train_txt)
	with open(test_gt_txt, 'r') as t_fp:
		test_lines = t_fp.readlines()
		with open(new_test_txt, 'w+') as new_t_fp:
			with open(new_train_txt, 'a+') as new_tr_fp:
				num = 0
				for line in test_lines:
					num += 1
					if num <= test_num:
						new_t_fp.write(line)
					else:
						new_tr_fp.write(line)
				

def crop_face(gt_txt, face_img_dir, show=False):
	if not os.path.exists(face_img_dir):
		os.mkdir(face_img_dir)
	img_num = 1
	with open(gt_txt, 'r') as gt_fp:
		line = gt_fp.readline()
		while line:
			img_path, bbox, phis = line.split(',')
			# print(img_path)
			
			img = cv2.imread(img_path)
			
			phis = phis.strip('\n').strip(' ').split(' ')
			phis = [int(float(x)) for x in phis]

			xarr = phis[:29]
			yarr = phis[30:58]
			min_x = np.min(xarr)
			max_x = np.max(xarr)
			min_y = np.min(yarr)
			max_y = np.max(yarr)
			#print(min_x, max_x, min_y, max_y)
			Lmax = np.max([max_x - min_x, max_y - min_y]) * 1.15

			delta = Lmax // 2
			center_x = (max_x + min_x) // 2
			center_y = (max_y + min_y) // 2
			x = int(center_x - delta)
			y = int(center_y - 0.98 * delta)
			endx = int(center_x + delta)
			endy = int(center_y + 1.02 * delta)

			if x < 0: x = 0
			if y < 0: y = 0

			if endx > img.shape[1]: endx = img.shape[1]
			if endy > img.shape[0]: endy = img.shape[0]
			
			face = img[y: endy, x: endx]
			
			if show:
				cv2.imshow("face", face)
				cv2.waitKey(0)
				
			cv2.imwrite(face_img_dir + "{}.jpg".format(img_num), face)
			
			line = gt_fp.readline()
			img_num += 1


def face_label(gt_txt, face_img_dir, face_txt, show=False):
	img_num = 1
	with open(face_txt, "w+") as face_txt_fp:
		with open(gt_txt, 'r') as gt_fp:
			line = gt_fp.readline()
			while line:
				img_path, bbox, phis = line.split(',')
				
				phis = phis.strip('\n').strip(' ').split(' ')
				phis = [int(float(x)) for x in phis]
				# print(phis)
				
				if show:
					for i in range(29):
						cv2.circle(img, (phis[i], phis[i + 29]), 2, (0, 255, 255))
						cv2.putText(img, str(i), (phis[i], phis[i + 29]), cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,255),1)
						cv2.imshow("img", img)
						cv2.waitKey(0)
				
				slot = phis[58:]
				label = [1, 0, 0, 0, 0, 0]
				# if slot[0] and slot[2] and slot[4] and slot[5]:
				# label[1] = 1  # left eyebrow
				# label[0] = 0
				if slot[16]:  # slot[10] or slot[12] or slot[13] or slot[16] or slot[8]:
					label[1] = 1  # left eye
					label[0] = 0
				# if slot[1] and slot[3] and slot[6] and slot[7]:
				# label[3] = 1  # right eyebrow
				# label[0] = 0
				if slot[17]:  # slot[11] or slot[14] or slot[15] or slot[17] or slot[9]:
					label[2] = 1  # right eye
					label[0] = 0
				if slot[20]:  # slot[18] or slot[19] or slot[20] or slot[21]:
					label[3] = 1  # nose
					label[0] = 0
				if slot[22] or slot[23] or slot[25] or slot[26] or slot[27]:  # or slot[24]
					label[4] = 1  # mouth
					label[0] = 0
				if slot[28]:
					label[5] = 1  # chin
					label[0] = 0
				
				lab_str = ''
				for x in label:
					lab_str += str(x) + ' '
				
				content = face_img_dir + "{}.jpg".format(img_num) + ',' + lab_str.rstrip(' ')
				content += '\n'
				# print(content)
				face_txt_fp.write(content)
				
				line = gt_fp.readline()
				img_num += 1
			
def img_shift(img_size, delta, shift_dir, orig_txt, shift_txt, show=False):
	with open(shift_txt, "w+") as shift_fp:
		with open(orig_txt, 'r') as orig_fp:
			if not os.path.exists(shift_dir):
				os.mkdir(shift_dir)
			
			line = orig_fp.readline()
			while line:
				# print(line)
				
				shift_fp.write(line)
				
				img_path, label = line.split(',')
				# print(img_path, label)
				
				img_name = os.path.basename(img_path)
				
				img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
				img = cv2.resize(img, (img_size, img_size))
				
				crop_size = img_size - delta
				shift_type = ['tl_', 'tr_', 'bl_', 'br_', 'ct_']
				
				for shift in shift_type:
					crop_img = img
					if shift == 'tl_':
						crop_img = img[0:crop_size, 0:crop_size]
					elif shift == 'tr_':
						crop_img = img[0:crop_size, delta - 1:-1]
					elif shift == 'bl_':
						crop_img = img[delta - 1:-1, 0:crop_size]
					elif shift == 'br_':
						crop_img = img[delta - 1:-1, delta - 1:-1]
					elif shift == 'ct_':
						crop_img = img[delta // 2:crop_size // 2, delta // 2:crop_size // 2]
					
					shift_img_path = shift_dir + shift + img_name
					cv2.imwrite(shift_img_path, crop_img)
					shift_fp.write(shift_img_path + "," + label)
					if show:
						cv2.imshow(shift, crop_img)
						cv2.waitKey(0)
				
				line = orig_fp.readline()
			


def add_block_and_crop_face(gt_txt, block_txt, block_dir, rand_num, show=False):
	with open(block_txt, "w+") as block_txt_fp:
		if not os.path.exists(block_dir):
			os.mkdir(block_dir)
		num = 1
		for i in range(rand_num):
			with open(gt_txt, 'r') as gt:
				line = gt.readline()
				while line:
					img_path, bbox, phis = line.split(',')
					# print(img_path)
					img = cv2.imread(img_path)
					
					phis = phis.strip('\n').strip(' ').split(' ')
					phis = [int(float(x)) for x in phis]
					# print(phis)
					
					slot = phis[58:]
					label = [1, 0, 0, 0, 0, 0]
					# if slot[0] and slot[2] and slot[4] and slot[5]:
					# label[1] = 1  # left eyebrow
					# label[0] = 0
					if slot[16]:  # slot[10] or slot[12] or slot[13] or slot[16]:  # slot[8] outer
						label[1] = 1  # left eye
						label[0] = 0
					# if slot[1] and slot[3] and slot[6] and slot[7]:
					# label[3] = 1  # right eyebrow
					# label[0] = 0
					if slot[17]:  # slot[11] or slot[14] or slot[15] or slot[17]:  # slot[9] outer
						label[2] = 1  # right eye
						label[0] = 0
					if slot[20]:  # slot[18] or slot[19] or slot[20] or slot[21]:
						label[3] = 1  # nose
						label[0] = 0
					if slot[22] or slot[23] or slot[25] or slot[26] or slot[27]:  # or slot[24]
						label[4] = 1  # mouth
						label[0] = 0
					if slot[28]:
						label[5] = 1  # chin
						label[0] = 0
					
					block_img = img
					area_num = random.randint(1, 6)
					
					for i in range(area_num):
						value = random.randint(0, 255)
						area_idx = random.randint(1, 6)
						# print("area_idx:{}".format(area_idx))
						if area_idx == 1:
							if label[area_idx] == 0:
								label[area_idx] = 1
								label[0] = 0
								block_img[phis[29 + 12] - 10: phis[29 + 13] + 8, phis[8] - 10: phis[10] + 10] = value
						elif area_idx == 2:
							if label[area_idx] == 0:
								label[area_idx] = 1
								label[0] = 0
								block_img[phis[29 + 14] - 10: phis[29 + 15] + 8, phis[11] - 10: phis[9] + 10] = value
						elif area_idx == 3:
							if label[area_idx] == 0:
								label[area_idx] = 1
								label[0] = 0
								block_img[phis[29 + 20] - 20: phis[29 + 21] + 1, phis[18] - 4: phis[19] + 5] = value
						elif area_idx == 4:
							if label[area_idx] == 0:
								label[area_idx] = 1
								label[0] = 0
								block_img[phis[29 + 24] - 1: phis[29 + 27] + 1, phis[22] - 2: phis[23] + 2] = value
						elif area_idx == 5:
							if label[area_idx] == 0:
								label[area_idx] = 1
								label[0] = 0
								block_img[phis[29 + 28] - 12: phis[29 + 28] + 15,
								phis[28] - 25: phis[28] + 25] = random.randint(160, 255)
						else:
							if label[1] == 0 and label[2] == 0:
								label[1] = 1
								label[0] = 0
								block_img[phis[29 + 12] - 10: phis[29 + 13] + 8,
								phis[8] - 10: phis[10] + 10] = random.randint(0, 100)
								label[2] = 1
								label[0] = 0
								block_img[phis[29 + 14] - 10: phis[29 + 15] + 8,
								phis[11] - 10: phis[9] + 10] = random.randint(0, 100)
					
					img_file = str(num) + ".jpg"
					lab_str = ''
					for x in label:
						lab_str += str(x) + ' '
					content = block_dir + img_file + ',' + lab_str.rstrip(' ')
					content += '\n'
					block_txt_fp.write(content)
					
					xarr = phis[:29]
					yarr = phis[30:58]
					min_x = np.min(xarr)
					max_x = np.max(xarr)
					min_y = np.min(yarr)
					max_y = np.max(yarr)
					# print(min_x, max_x, min_y, max_y)
					Lmax = np.max([max_x - min_x, max_y - min_y]) * 1.15

					delta = Lmax // 2
					center_x = (max_x + min_x) // 2
					center_y = (max_y + min_y) // 2
					x = int(center_x - delta)
					y = int(center_y - 0.98 * delta)
					endx = int(center_x + delta)
					endy = int(center_y + 1.02 * delta)
					
					if x < 0: x = 0
					if y < 0: y = 0
				
					# print(img.shape)
					if endx > block_img.shape[1]: endx = block_img.shape[1]
					if endy > block_img.shape[0]: endy = block_img.shape[0]
					
					face = block_img[y: endy, x: endx]
					
					cv2.imwrite(block_dir + img_file, face)
					if show:
						cv2.imshow("face", face)
						cv2.waitKey(0)
					
					line = gt.readline()
					num += 1
			
					
def label_spec_face(label_file, file_root):
	labels = '1 0 0 0 0 0'
	with open(label_file, 'w+') as tlf:
		for root, dirs, files in os.walk(file_root):
			for f in files:
				img_path = file_root + f
				content = img_path + ',' + labels + '\n'
				# print(content)
				tlf.write(content)
			
			
def merge_txt(txt1, txt2, merge_txt):
	with open(merge_txt, 'w+') as trf:
		with open(txt1, 'r') as fp:
			line = fp.readline()
			while line:
				trf.write(line)
				line = fp.readline()
		with open(txt2, 'r') as fp:
			line = fp.readline()
			while line:
				trf.write(line)
				line = fp.readline()


def split_train_file(merge_train_txt, train_txt, val_txt, train_ratio):
	with open(merge_train_txt, 'r') as fp:
		lines_list = fp.readlines()
		
		# random.shuffle(lines_list)
		total = len(lines_list)
		train_num = math.floor(total * train_ratio)
		val_num = total - train_num
		count = 0
		data = []
		for line in lines_list:
			# print(line)
			count += 1
			data.append(line)
			if count == train_num:
				with open(train_txt, "w+") as trainf:
					random.shuffle(data)
					for d in data:
						trainf.write(d)
				data = []
			
			if count == train_num + val_num:
				with open(val_txt, "w+") as valf:
					random.shuffle(data)
					for d in data:
						valf.write(d)
				data = []
		print("train_num:{}, val_num:{}".format(train_num, val_num))
		
		
def prepare_data(data_root):
	# 1. Get original txt
	print("1. Generate original data index file...")
	if not os.path.exists(data_root + "glass_face.txt"):
		label_spec_face(data_root + "glass_face.txt",
						data_root + "glass_face/")
	
	if not os.path.exists(data_root + "beard_face.txt"):
		label_spec_face(data_root + "beard_face.txt",
						data_root + "beard_face/")
	
	if not os.path.exists(data_root + "train_ground_true.txt"):
		mat_to_files(data_root + "COFW_train.mat",
					 'IsTr', 'bboxesTr', 'phisTr',
					 data_root + "train",
					 data_root + "train_ground_true.txt")
	
	if not os.path.exists(data_root + "test_ground_true.txt"):
		mat_to_files(data_root + "COFW_test.mat",
					 'IsT', 'bboxesT', 'phisT',
					 data_root + "test",
					 data_root + "test_ground_true.txt")

	if not os.path.exists(data_root + "train_gt.txt"):
		move_test_to_train(data_root + "test_ground_true.txt",
						   data_root + "train_ground_true.txt",
						   data_root + "test_gt.txt",
						   data_root + "train_gt.txt",
						   100)

	# 2. Crop face and get face txt
	print("2. Crop face and get face txt...")
	if not os.path.exists(data_root + "face_train/"):
		crop_face(data_root + "train_gt.txt",
				  data_root + "face_train/")
	if not os.path.exists(data_root + "face_train.txt"):
		face_label(data_root + "train_gt.txt",
				   data_root + "face_train/",
				   data_root + "face_train.txt")
		
	if not os.path.exists(data_root + "face_test/"):
		crop_face(data_root +"test_gt.txt",
				  data_root + "face_test/")
	if not os.path.exists(data_root + "face_test.txt"):
		face_label(data_root + "test_gt.txt",
				   data_root + "face_test/",
				   data_root + "face_test.txt")

	# 3. merge spec face txt with face train txt
	print("3. Merge spec face txt with face train txt...")
	if not os.path.exists(data_root + "spec_face.txt"):
		merge_txt(data_root + "glass_face.txt", data_root + "beard_face.txt", data_root + "spec_face.txt")
	if not os.path.exists(data_root + "merge_face.txt"):
		merge_txt(data_root + "spec_face.txt", data_root + "face_train.txt", data_root + "merge_face.txt")

	# 4. shift merge train face
	print("4. Generate shifted train face...")
	if not os.path.exists(data_root + "face_train_orig_shift.txt"):
		img_shift(120, 20,
				  data_root + 'face_train_shift/',
				  data_root + 'merge_face.txt',
				  data_root + 'face_train_orig_shift.txt')
	
	if not os.path.exists(data_root + "face_test_shift.txt"):
		img_shift(120, 20,
				  data_root + 'face_test_shift/',
				  data_root + 'face_test.txt',
				  data_root + 'face_test_shift.txt')
	
	# 5. add block to face and shift
	print("5. Add block to face and shift...")
	if not os.path.exists(data_root + "face_train_block.txt"):
		add_block_and_crop_face(data_root + "train_gt.txt",
								data_root + "face_train_block.txt",
								data_root + "face_train_block/",
								8)
	if not os.path.exists(data_root + 'face_train_block_shift.txt'):
		img_shift(120, 20,
				  data_root + 'face_train_block_shift/',
				  data_root + 'face_train_block.txt',
				  data_root + 'face_train_block_shift.txt')

	#6. merge all faces to merge_train.txt
	print("6. Merge all faces to merge_train.txt...")
	if not os.path.exists(data_root + 'merge_train.txt'):
		merge_txt(data_root + 'face_train_orig_shift.txt',
				  data_root + 'face_train_block_shift.txt',
				  data_root + 'merge_train.txt')
	
	#7. get train, val and test
	print("7. Generate train, val and test...")
	if not os.path.exists(data_root + "train.txt"):
		split_train_file(data_root + "merge_train.txt",
						 data_root + "train.txt",
						 data_root + "val.txt",
						 0.85)
	
	if not os.path.exists(data_root + "test.txt"):
		shutil.copy(data_root + 'face_test_shift.txt',
					data_root + 'test.txt')
	print("Done!")

def parse_args():
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda v: v.lower() == "true")
	
	parser.add_argument("--data_dir", type=str, default="./data/cofw/", help="Data root directory")
	return parser.parse_known_args()

if __name__ == '__main__':
	FLAGS, unparsed = parse_args()
	
	if not len(FLAGS.data_dir):
		raise Exception("Please set data root directory via --data_dir")
	
	prepare_data(FLAGS.data_dir)
