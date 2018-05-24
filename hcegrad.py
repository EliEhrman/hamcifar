"""
This module is descended from hc.py which attempts to learn hamming cnn using GA on cifar

descendent from hces.py at a point where ES was working on bit_to_int but producing poor results

The goal before starting is to investigate finding the gradient from diffs

"""

from __future__ import print_function

import enum
import numpy as np
import random
import sys
import copy
import bisect

fn = '../../data/hamcifar/data_batch_1'

c_bits_per_src_val = 8
c_min_bits = 1
c_max_bits = 4
c_img_x = 32
c_img_y = 32
c_rgb_dims = 3
c_fsize = 3 # filter size c_fsize * c_fsize. Expect it to stay at 3x3
c_num_c_input = c_rgb_dims * c_bits_per_src_val
c_l_num_channels = [c_num_c_input, 14, 12, 7]
# c_num_layers = 3
# c_num_c_l2 = 5
c_pad_size = int(c_fsize / 2)
c_rec_limit = 200
c_pct_train = 0.5
c_num_k = 10
c_num_Ws = 3
c_mid_score = c_num_Ws * 5 / 8
c_num_iters = 30000
c_rnd_asex = 0.95
c_rnd_sex = 0.99 # after asex selection
c_num_incr_muts = 1
c_num_change_muts = 1
c_change_mut_prob_change_len = 0.15 # 0.3
c_change_mut_num_change = 1
c_int_to_bit_mut_rate = 7.0 / (c_num_c_input * c_img_x * c_img_y)  # 0.0 -> 1.0. 1.0 means change them all, 0.0 means change none
c_asex_num_muts = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]

iui8 = np.iinfo(np.uint8)


def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

def conv2d(img, W, stride):
	n, c, w, h = img.shape
	num_o_channels = len(W)
	img_padded = np.zeros((n, c, w+(c_pad_size*2), h+(c_pad_size*2)), dtype=np.uint8)
	img_padded[:, :, 1:-1, 1:-1] = img
	new_layer = np.zeros((w/stride, h/stride, n, num_o_channels), dtype=np.uint8)
	for sx in range(0, w, stride):
		for sy in range(0, h, stride):
			slice = img_padded[:,:,sx:sx+c_fsize, sy:sy+c_fsize]
			slice_r = slice.reshape(-1, c*c_fsize*c_fsize)
			l_osum = []
			for iop in range(num_o_channels):
				new_pix = np.multiply(slice_r, W[iop][0])
				sum_pix = np.sum(new_pix, axis=1)
				l_osum.append(np.where(sum_pix >= W[iop][1], np.ones_like(sum_pix), np.zeros_like(sum_pix)))
			new_pix = np.stack(l_osum, axis=1)
			new_layer[sx/stride, sy/stride, :, :] = new_pix

	return np.transpose(new_layer, (2, 3, 0, 1))

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def maxpool(X):
	size, stride = 2, 2
	n, d, h, w = X.shape
	h_out, w_out = h/stride, w/stride
	# Let say our input X is 5x10x28x28
	# Our pooling parameter are: size = 2x2, stride = 2, padding = 0
	# i.e. result of 10 filters of 3x3 applied to 5 imgs of 28x28 with stride = 1 and padding = 1

	# First, reshape it to 50x1x28x28 to make im2col arranges it fully in column
	X_reshaped = X.reshape(n * d, 1, h, w)

	# The result will be 4x9800
	# Note if we apply im2col to our 5x10x28x28 input, the result won't be as nice: 40x980
	X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

	# Next, at each possible patch location, i.e. at each column, we're taking the max index
	max_idx = np.argmax(X_col, axis=0)

	# Finally, we get all the max value at each column
	# The result will be 1x9800
	out = X_col[max_idx, range(max_idx.size)]

	# Reshape to the output size: 14x14x5x10
	out = out.reshape(h_out, w_out, n, d)

	# Transpose to get 5x10x14x14 output
	return out.transpose(2, 3, 0, 1)

def avgpool(X):
	size, stride = 2, 2
	n, d, h, w = X.shape
	h_out, w_out = h/stride, w/stride
	# Let say our input X is 5x10x28x28
	# Our pooling parameter are: size = 2x2, stride = 2, padding = 0
	# i.e. result of 10 filters of 3x3 applied to 5 imgs of 28x28 with stride = 1 and padding = 1

	# First, reshape it to 50x1x28x28 to make im2col arranges it fully in column
	X_reshaped = X.reshape(n * d, 1, h, w)

	# The result will be 4x9800
	# Note if we apply im2col to our 5x10x28x28 input, the result won't be as nice: 40x980
	X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

	out = np.average(X_col, axis=0)

	# Reshape to the output size: 14x14x5x10
	out = out.reshape(h_out, w_out, n, d)

	# Transpose to get 5x10x14x14 output
	return out.transpose(2, 3, 0, 1)

def eval(bit_db, test_cases, test_labels):
	score_sum, score_eval = 0.0, 0.0
	for itest, test in enumerate(test_cases):
		# hd = np.sum(np.absolute(np.subtract(test, bit_db)), axis=1)
		hd = np.sum(np.where(np.not_equal(test, bit_db), np.ones_like(bit_db), np.zeros_like(bit_db)), axis=1)
		hd_winners = np.argpartition(hd, (c_num_k + 1))[:(c_num_k + 1)]
		hd_of_winners = hd[hd_winners]
		iwinners = np.argsort(hd_of_winners)
		hd_idx_sorted = hd_winners[iwinners]
		l_winner_labels = list(np.take(l_labels, hd_idx_sorted))
		nd_votes = np.zeros(10, dtype=np.float)
		for iilabel in range(10):
			scores = [float(c_num_k-ilabel)/float(c_num_k) if label == iilabel else 0.0 for ilabel, label in enumerate(l_winner_labels)]
			nd_votes[iilabel] += sum(scores)
		the_winner = np.argmax(nd_votes)
		if the_winner == test_labels[itest]:
			score_eval += 1.0

		scores = [float(c_num_k - ilabel) / float(c_num_k) if label == test_labels[itest] else 0.0 for ilabel, label in
				  enumerate(l_winner_labels)]
		if eval.counter == 0:
			eval.max_score = sum([float(c_num_k-ilabel)/float(c_num_k) for ilabel in range(c_num_k)])
			eval.counter += 1
		score_sum += sum(scores) / eval.max_score
		# label_winner = max(set(l_winner_labels), key=l_winner_labels.count)
		# if label_winner == test_labels[itest]:
		# 	sum +=1

	return float(score_sum) / float(len(test_cases)), float(score_eval) / float(len(test_cases))
eval.counter = 0

def eval_real(bit_db, test_cases, test_labels):
	score_sum = 0.0
	for itest, test in enumerate(test_cases):
		# hd = np.sum(np.absolute(np.subtract(test, bit_db)), axis=1)
		hd = np.sum(np.where(np.not_equal(test, bit_db), np.ones_like(bit_db), np.zeros_like(bit_db)), axis=1)
		hd_winners = np.argpartition(hd, (c_num_k + 1))[:(c_num_k + 1)]
		hd_of_winners = hd[hd_winners]
		iwinners = np.argsort(hd_of_winners)
		hd_idx_sorted = hd_winners[iwinners]
		l_winner_labels = list(np.take(l_labels, hd_idx_sorted))
		nd_votes = np.zeros(10, dtype=np.float)
		for iwin, wlabel in enumerate(l_winner_labels):
			nd_votes[wlabel] += 1.0 / float(iwin+2)
		the_winner = np.argmax(nd_votes)
		if the_winner == test_labels[itest]:
			score_sum += 1.0
		# label_winner = max(set(l_winner_labels), key=l_winner_labels.count)
		# if label_winner == test_labels[itest]:
		# 	sum +=1

	return float(score_sum) / float(len(test_cases))


def run_one_set(numrecs, nd_img_r_data, l_labels, train_limit, l_sms, b_real=False):
	# sm_int_to_bits, l_layer_Ws
	nd_img_bits = np.where(nd_img_r_data > np.repeat(np.expand_dims(l_sms[0], axis=0), numrecs, axis=0),
						   np.ones((numrecs, c_rgb_dims * c_bits_per_src_val, c_img_x, c_img_y), dtype=np.uint8),
						   np.zeros((numrecs, c_rgb_dims * c_bits_per_src_val, c_img_x, c_img_y), dtype=np.uint8))

	mp_layer = nd_img_bits
	for ilayer in range(len(c_l_num_channels) - 1):
		mp_layer = conv2d(mp_layer, l_sms[ilayer+1], 2)
		# mp_layer = avgpool(layer)

	all_db = mp_layer.reshape(numrecs, -1)

	bit_db = all_db[:train_limit, :]
	test_cases = all_db[train_limit:, :]
	test_labels = l_labels[train_limit:]

	if b_real:
		score = eval_real(bit_db, test_cases, test_labels)
		print('Eval score =', score)
		evalsc = score
	else:
		score, evalsc = eval(bit_db, test_cases, test_labels)
		print('score =', score, 'eval =' , evalsc)
	return score

def select_best(l_objs, iiter, l_record_scores, l_record_objs, l_best_sms, ilearning):
	min_score, max_score = sys.float_info.max, -sys.float_info.max
	num_objs, l_scores, l_sms, record_best, b_change = len(l_objs), [], list(l_best_sms), l_record_scores[0], False
	for iobj in range(num_objs):
		l_sms[ilearning] = l_objs[iobj]
		score =run_one_set(numrecs, nd_img_r_data, l_labels, train_limit, l_sms)
		l_scores.append(score)
		if score != record_best:
			b_change = True
		if score > max_score:
			max_score = score
		if score < min_score:
			min_score = score

	if not b_change:
		return

	# print('avg score:', np.mean(l_scores)) # , 'list', l_scores)
	print('iiter', iiter, 'avg score:', np.mean(l_scores), 'max score:', np.max(l_scores)) # , 'list', l_scores)
	if l_record_scores == [] or max_score > l_record_scores[0]:
		if l_record_scores == []:
			isorted = np.array(l_scores).argsort()
			best, second = l_objs[isorted[0]], l_objs[isorted[1]]
		else:
			best, second = l_objs[l_scores.index(max_score)], l_record_objs[0]

		l_record_scores.insert(0, max_score)
		l_record_objs.insert(0, l_objs[l_scores.index(max_score)])
	else:
		second, best = l_objs[l_scores.index(min_score)], l_record_objs[0]

	if ilearning == 0:
		diff = best.astype(np.int16) - second.astype(np.int16)
		inz = np.nonzero(diff)
		new_cand = np.clip(best.astype(np.int16) + (diff * 3), 0, iui8.max).astype(np.uint8)
		l_sms[ilearning] = new_cand
		new_score = run_one_set(numrecs, nd_img_r_data, l_labels, train_limit, l_sms)
		if new_score > l_record_scores[0]:
			l_record_scores.insert(0, new_score)
			l_record_objs.insert(0, new_cand)
		else:
			new_cand = best
		l_objs[:] = [copy.deepcopy(new_cand) for _ in xrange(num_objs)]
	else:
		raise ValueError('not coded yet')

	# 	l_objs[l_scores.index(min_score)] = l_record_objs[0]
	# 	l_scores[l_scores.index(min_score)] = l_record_scores[0]

	# mid_score = l_scores[np.array(l_scores).argsort()[c_mid_score]]
	# if max_score == min_score:
	# 		range_scores = max_score
	# 		nd_obj_scores = np.ones(len(l_scores), dtype=np.float32)
	# elif mid_score == max_score:
	# 	range_scores = max_score - min_score
	# 	nd_obj_scores = np.array([(score - min_score) / range_scores for score in l_scores])
	# else:
	# 	range_scores = max_score - mid_score
	# 	nd_obj_scores = np.array([(score - mid_score) / range_scores for score in l_scores])
	# nd_obj_scores = np.where(nd_obj_scores > 0.0, nd_obj_scores, np.zeros_like(nd_obj_scores))
	# sel_prob = nd_obj_scores/np.sum(nd_obj_scores)
	# if ilearning == 0:
	# 	nd_tr = np.transpose(np.stack(l_objs, axis=0), (1, 2, 3, 0))
	# 	nd_weighted = np.transpose((nd_tr * nd_obj_scores), (3, 0, 1, 2))
	# 	nd_new = (np.sum(nd_weighted, axis=0) / np.sum(nd_obj_scores)).astype(np.uint8)
	# 	l_objs[:] = [copy.deepcopy(nd_new) for _ in xrange(num_objs)]



def mutate_int_to_bits(l_sm_int_to_bits):
	# nc, nx, ny = l_sm_int_to_bits[0].shape
	for isel, sel_mat in enumerate(l_sm_int_to_bits):
		if random.random() < c_rnd_asex:
			# buf = sel_mat.flatten().astype(np.int16)
			# sel = np.random.randint(0, nc*nx*ny, 1)
			# buf[sel] += random.randint(-8, 9)
			# sel_mat[:, :, :] = (np.clip(np.reshape(buf, (nc, nx, ny)), 0, iui8.max)).astype(np.uint8)
			incr_vals = np.random.randint(-8, 9, (c_rgb_dims * c_bits_per_src_val, c_img_x, c_img_y), dtype=np.int16)
			incr_mask = np.random.randint(0, 29, (c_rgb_dims * c_bits_per_src_val, c_img_x, c_img_y), dtype=np.int16)
			sel_mat[:, :, :] = (np.clip((incr_vals * (incr_mask < 1)) + sel_mat, 0, iui8.max)).astype(np.uint8)
		elif random.random() < c_rnd_sex:
			for itry in range(100):
				partner_sel_mat = copy.deepcopy(random.choice(l_sm_int_to_bits))  # not the numpy function
				if not np.array_equal(partner_sel_mat, sel_mat):
					break
			new_mask = np.random.rand(c_num_c_input, c_img_x, c_img_y)
			sel_mat[:,:,:] = np.where(new_mask < 0.5, partner_sel_mat, sel_mat)

def mutate_sel_mats(l_sel_mats, ilayer):
	num_i_bits = c_l_num_channels[ilayer-1] * c_fsize * c_fsize
	num_o_bits = c_l_num_channels[ilayer]
	c_mut_options = ['incr_thresh', 'decr_thresh', 'incr_bitlist', 'decr_bitlist', 'chng_bitlist']
	num_muts = bisect.bisect(c_asex_num_muts, random.random())
	for isel, sel_mat in enumerate(l_sel_mats):
		if random.random() < c_rnd_asex:
			num_muts_succeeded = 0
			for imut in range(num_muts * 100):
				if num_muts_succeeded >= num_muts:
					break
				sel_mut = random.choice(c_mut_options)
				if sel_mut == 'incr_thresh':
					allele = random.randint(0, num_o_bits-1)
					num_bits = len(sel_mat[allele][0])
					if sel_mat[allele][1] < num_bits-2:
						sel_mat[allele][1] += 1
						num_muts_succeeded += 1
				elif sel_mut == 'decr_thresh':
					allele = random.randint(0, num_o_bits-1)
					if sel_mat[allele][1] > 1:
						sel_mat[allele][1] -= 1
						num_muts_succeeded += 1
				else:
					allele = random.randint(0, num_o_bits-1)
					bit_list = sel_mat[allele][0]

					if sel_mut == 'incr_bitlist':
						if len(bit_list) < c_max_bits:
							bit_list.append(random.randint(0, num_i_bits - 1))
							num_muts_succeeded += 1
					elif sel_mut == 'decr_bitlist':
						if len(bit_list) > c_min_bits:
							bit_list.pop(random.randrange(len(bit_list)))
							num_muts_succeeded += 1
							if sel_mat[allele][1] >= len(bit_list) - 1:
								sel_mat[allele][1] -= 1
					elif sel_mut == 'chng_bitlist':
						bit_list[random.randint(0, len(bit_list)-1)] = random.randint(0, num_i_bits - 1)
						num_muts_succeeded += 1
					else:
						raise ValueError('Invalid option. Coding error')
				# end else change bitlist
			# for imut
		elif random.random() < c_rnd_sex:
			for itry in range(100):
				partner_sel_mat = copy.deepcopy(random.choice(l_sel_mats)) # not the numpy function
				if partner_sel_mat != sel_mat:
					break
			for allele in range(num_o_bits):
				if random.random() < 0.5:
					sel_mat[allele] = list(partner_sel_mat[allele])
		#end rnd asex/sex

def init():
	d_img_data = unpickle(fn)

	nd_src = d_img_data['data'][:c_rec_limit,:]
	l_labels = d_img_data['labels'][:c_rec_limit]
	nd_img_data = nd_src.reshape(-1, c_rgb_dims, c_img_x, c_img_y)
	nd_img_r_data = np.repeat(nd_img_data, c_bits_per_src_val, axis=1)
	numrecs = nd_img_data.shape[0]
	train_limit = int(numrecs * c_pct_train)


	return numrecs, nd_img_r_data, l_labels, train_limit

# img_bits_padded = np.zeros((numrecs, c_rgb_dims * c_bits_per_src_val, c_img_x+2, c_img_y+2), dtype=np.uint8)
# img_bits_padded[:, :, 1:-1, 1:-1] = nd_img_bits
# nd_img_bits = np.zeros((3 * 8, 32, 32), np.bool)

numrecs, nd_img_r_data, l_labels, train_limit= init()

l_l_layer_Ws = [[] for _ in range(len(c_l_num_channels)-1)]
l_sm_int_to_bits = []

for iW in range(c_num_Ws):
	# l_layer_Ws = []
	for ilayer in range(len(c_l_num_channels)):
		if ilayer==0:
			continue
		W = []
		for iop in range(c_l_num_channels[ilayer]):
			num_input_bits = (c_l_num_channels[ilayer - 1] * c_fsize * c_fsize)
			wrow = [random.randint(1, 7) if random.random() < 0.2 else 0 for _ in range(num_input_bits)]
			W.append([wrow, sum(wrow) * 1 / 2])
		l_l_layer_Ws[ilayer-1].append(W)

	sm_int_to_bits = np.random.randint(0, 255, (c_num_c_input, c_img_x, c_img_y), dtype=np.uint8)
	l_sm_int_to_bits.append(sm_int_to_bits)

l_record_bitvals = [l_sm_int_to_bits[0]]
l_record_bitval_scores = [-1.0]
l_l_records_sel_mats = [[l_l_layer_Ws[ilayer][0]] for ilayer in range(len(c_l_num_channels)-1)]
l_l_record_sel_mat_scores = [[-1] for _ in range(len(c_l_num_channels)-1)]

for iiter in range(c_num_iters):

	for ilayer in  range(len(c_l_num_channels)*3):
		ilayer /= 3
			# continue
		# select_best(l_objs, iiter, l_record_scores, l_record_objs, l_best_sms, ilearning):
		l_best_sms = [l_record_bitvals[0]] + [l_records_sel_mats[0] for l_records_sel_mats in l_l_records_sel_mats]
		if iiter % 2 == 0 and ilayer==0:
			score = run_one_set(numrecs, nd_img_r_data, l_labels, train_limit, l_best_sms, b_real=True)
		if ilayer == 0:
			print('Scoring and learning for int to bits layer')
			select_best(l_sm_int_to_bits, iiter, l_record_bitval_scores, l_record_bitvals, l_best_sms, ilayer)
			mutate_int_to_bits(l_sm_int_to_bits)
		else:
			pass
			# print('Scoring and learning for conv layer', ilayer)
			# select_best(l_l_layer_Ws[ilayer-1], iiter, l_l_record_sel_mat_scores[ilayer-1],
			# 			l_l_records_sel_mats[ilayer-1], l_best_sms, ilayer)
			# mutate_sel_mats(l_l_layer_Ws[ilayer-1], ilayer)

	# run_one_set(numrecs, nd_img_r_data, l_labels, train_limit, sm_int_to_bits, l_layer_Ws)

