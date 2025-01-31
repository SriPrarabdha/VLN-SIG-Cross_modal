'''
R2R-style dataset
'''
import os
import json
import jsonlines
import numpy as np
import h5py
import math

import networkx as nx
import time
import random

from PIL import Image
import torch

# from utils.logger import LOGGER


IMGSIZE = 112

def angle_feature(heading, elevation, angle_feat_size):
    return np.array([math.sin(heading), math.cos(heading),
                     math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
                    dtype=np.float32)

def get_point_angle_feature(angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            heading = 0
            elevation = math.radians(-30)
        elif ix % 12 == 0:
            heading = 0
            elevation += math.radians(30)
        else:
            heading += math.radians(30)
        feature[ix, :] = angle_feature(heading - base_heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(angle_feat_size):
    return [get_point_angle_feature(angle_feat_size, baseViewId) for baseViewId in range(36)]

def get_point_rel_angles(baseViewId=0):
    rel_angles = np.zeros((36, 2), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            heading = 0
            elevation = math.radians(-30)
        elif ix % 12 == 0:
            heading = 0
            elevation += math.radians(30)
        else:
            heading += math.radians(30)
        rel_angles[ix, 0] = heading - base_heading
        rel_angles[ix, 1] = elevation #- base_elevation
    return rel_angles

def get_all_point_rel_angles():
    return [get_point_rel_angles(baseViewId) for baseViewId in range(36)]

def load_nav_graphs(connectivity_dir):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    scans = [x.strip() for x in open(os.path.join(connectivity_dir, 'scans.txt')).readlines()]
    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G

    shortest_distances = {}
    for scan, G in graphs.items():  # compute all shortest paths
        shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
    return graphs, shortest_distances

def softmax(logits, dim=1):
    # logits: (n, d)
    tmp = np.exp(logits)
    return tmp / np.sum(tmp, axis=dim, keepdims=True)

def read_img_features(feature_store):
    import csv
    import base64
    from tqdm import tqdm

    print("Start loading the image feature")
    start = time.time()

    views = 36

    tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
    features = {}
    with open(feature_store, "r") as tsv_in_file:     # Open the tsv file.
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
        for item in reader:
            long_id = item['scanId'] + "_" + item['viewpointId']
            features[long_id] = np.frombuffer(base64.decodestring(item['features'].encode('ascii')),
                                                   dtype=np.float32).reshape((views, -1))   # Feature of long_id is (36, 2048)

    print("Finish Loading the image feature from %s in %0.4f seconds" % (feature_store, time.time() - start))
    return features


class MultiStepNavBeitData(object):
    def __init__(
        self, traj_files, img_ft_file, blip_ft_file, dino_ft_file, scanvp_cands_file,  connectivity_dir,
        image_prob_size=1000, image_feat_size=2048, angle_feat_size=4,
        max_txt_len=80, max_act_len=100,
        hist_enc_pano=True, val_sample_num=None,
        in_memory=False, ob_cand_pano_view=False,
        img_aug_ft_file=None, img_db_file=None,
        preprocess=None
    ):
        self.traj_files = traj_files
        ####
        self.img_ft_file = img_ft_file
        self.dino_ft_file = dino_ft_file
        self.blip_ft_file = blip_ft_file
        #####
        self.img_db_file = img_db_file
        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.max_txt_len = max_txt_len
        self.max_act_len = min(30, max_act_len) # due to memory issue
        self.hist_enc_pano = hist_enc_pano
        self.ob_cand_pano_view = ob_cand_pano_view

        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}
            self._feature_store1 = {}
            self._feature_store2 = {}

        self.feature_index = 0

        self.img_transform = preprocess

        self.scanvp_cands = json.load(open(scanvp_cands_file))

        self.graphs, self.shortest_distances = load_nav_graphs(connectivity_dir)

        self.angle_features = get_all_point_angle_feature(angle_feat_size)
        self.rel_angles = get_all_point_rel_angles()

        self.traj_data = []
        self.traj_refer, self.traj_step_refer = [], []
        n = 0
        for traj_file in self.traj_files:
            with open(traj_file, 'r') as f:
                for item in jsonlines.Reader(f):
                    self.traj_data.append(item)
                    path_len = min(len(item['path']), self.max_act_len-1)
                    for j in range(len(item['instr_encodings'])):
                        self.traj_refer.append((n, j, path_len))
                        self.traj_step_refer.extend([(n, j, t) for t in range(path_len)])
                    n += 1

        if val_sample_num:
            # cannot evaluate all the samples as it takes too much time
            # sample K data for validation
            sel_idxs = np.random.permutation(len(self.traj_refer))[:val_sample_num]
            self.traj_refer = [self.traj_refer[sidx] for sidx in sel_idxs]
            sel_idxs = np.random.permutation(len(self.traj_step_refer))[:val_sample_num]
            self.traj_step_refer = [self.traj_step_refer[sidx] for sidx in sel_idxs]

    def get_input(
        self, i_path, j_instr, t_cur,
        return_ob=False, return_hist_img_probs=False,
        return_ob_action=False, return_ob_progress=False,
        ob_cand_pano_view=None, d_vae=False, return_ob_image=False, return_ob_pano_images=False, pano_mask_prob=0
    ):

        traj_data = self.traj_data[i_path]
        scan = traj_data['scan']
        path = traj_data['path'][:self.max_act_len-1]
        path_viewindex = traj_data['path_viewindex']
        action_viewindex = traj_data['action_viewindex']
        abs_pos_angles = traj_data['abs_pos_angles']
        rel_act_angles = traj_data['rel_act_angles']

        instr_id = traj_data['instr_ids'][j_instr]
        instr_encoding = traj_data['instr_encodings'][j_instr][:self.max_txt_len]

        hist_inputs = self.get_history_feature(
            scan, path, path_viewindex, rel_act_angles, t_cur,
            return_img_probs=return_hist_img_probs, d_vae=d_vae
        )

        outs = {
            'instr_id': instr_id,
            'instr_encoding': instr_encoding,
            'hist_img_fts1': hist_inputs[0],
            'hist_img_fts2': hist_inputs[1],
            'hist_ang_fts': hist_inputs[2],
            'hist_lens': t_cur,
        }
        if self.hist_enc_pano:
            outs['hist_pano_img_fts1'] = hist_inputs[3]
            outs['hist_pano_img_fts2'] = hist_inputs[4]
            outs['hist_pano_ang_fts'] = hist_inputs[5]
        if return_hist_img_probs:
            outs['hist_img_probs'] = hist_inputs[-1]
        if d_vae:
            outs['hist_images_dvae'] = hist_inputs[-1]

        if return_ob:
            if ob_cand_pano_view is None:
                ob_cand_pano_view = self.ob_cand_pano_view
            if ob_cand_pano_view:
                ob_img_feats1, ob_img_feats2, ob_ang_feats, ob_nav_types, gt_label, gt_angle = \
                    self.get_ob_cand_pano_view(scan, path, path_viewindex, action_viewindex, rel_act_angles, t_cur)
            else:
                ob_img_feats1,ob_img_feats2, ob_ang_feats, ob_nav_types, gt_label, gt_angle = \
                    self.get_ob_pano_view(scan, path, path_viewindex, action_viewindex, rel_act_angles, t_cur)

            outs.update({
                'ob_img_fts1': ob_img_feats1,
                'ob_img_fts2': ob_img_feats2,
                'ob_ang_fts': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
            })
            if return_ob_action:
                outs['ob_action_viewindex'] = gt_label
                outs['ob_action_angles'] = gt_angle
            if return_ob_progress:
                outs['ob_progress'] = self.get_progress(scan, path[0], path[t_cur], path[-1] if 'guide_path' not in traj_data else traj_data['guide_path'][-1])
            if return_ob_image:
                if action_viewindex[t_cur] == -1:
                    action_image = torch.zeros((3, IMGSIZE, IMGSIZE), dtype=torch.float)
                else:
                    action_image = self.get_image(scan, path[t_cur], action_viewindex[t_cur])
                outs['ob_action_image'] = action_image

            if return_ob_pano_images:
                ############### Masked Region Modeling ###############
                action_images = []
                img_mask = []
                for i in range(36):
                    if np.random.rand() < pano_mask_prob:
                        img_mask.append(True)
                        action_images.append(self.get_image(scan, path[t_cur], i))
                    else:
                        img_mask.append(False)
                        action_images.append(torch.zeros((3, IMGSIZE, IMGSIZE), dtype=torch.float))
                if not any(img_mask):
                    # at least mask 1
                    index = np.random.randint(36)
                    img_mask[index] = True
                    action_images[index] = self.get_image(scan, path[t_cur], index)
                img_mask.append(False)          # One more mask for STOP token appended at img features
                action_images.append(torch.zeros((3, IMGSIZE, IMGSIZE), dtype=torch.float))      # One more mask for STOP token appended at img features
                img_mask = torch.tensor(img_mask)
                outs['ob_pano_mask'] = img_mask
                action_images = np.stack(action_images, 0)
                outs['ob_pano_images'] = action_images

        return outs

    def get_ob_pano_view(self, scan, path, path_viewindex, action_viewindex, rel_act_angles, t_cur):
        #####
        ob_img_feats1 = self.get_image_feature1(scan, path[t_cur], pad_stop_token=True)[:, :self.image_feat_size]
        ob_img_feats2 = self.get_image_feature2(scan, path[t_cur], pad_stop_token=True)[:, :self.image_feat_size]
        #####
        ob_ang_feats = self.get_angle_feature(path_viewindex[t_cur], pad_stop_token=True)
        ob_nav_types = np.zeros((ob_img_feats1.shape[0], ), dtype=np.int64)
        ob_nav_types[-1] = 2   # 2 for [STOP]
        ob_nav_cands = self.scanvp_cands['%s_%s'%(scan, path[t_cur])]
        ob_nav_viewindexes = np.array([v[0] for v in ob_nav_cands.values()])
        ob_nav_types[ob_nav_viewindexes] = 1

        if action_viewindex[t_cur] != -1:
            gt_label = action_viewindex[t_cur]
            gt_angle = rel_act_angles[t_cur]
        else: # stop
            gt_label = ob_img_feats1.shape[0] - 1
            gt_angle = np.zeros((2, ), dtype=np.float32)

        return ob_img_feats1,ob_img_feats2, ob_ang_feats, ob_nav_types, gt_label, gt_angle

    def get_ob_cand_pano_view(self, scan, path, path_viewindex, action_viewindex, rel_act_angles, t_cur):
        # 36 pano views
        #######
        ob_img_feats1 = self.get_image_feature1(scan, path[t_cur], pad_stop_token=False)[:, :self.image_feat_size]
        ob_img_feats2 = self.get_image_feature2(scan, path[t_cur], pad_stop_token=False)[:, :self.image_feat_size]
        ########
        ob_ang_feats = self.get_angle_feature(path_viewindex[t_cur], pad_stop_token=False)

        # cand views
        ob_nav_cands = self.scanvp_cands['%s_%s'%(scan, path[t_cur])]
        cand_img_feats1, cand_img_feats2, cand_ang_feats = [], [], []
        non_cand_viewidxs = np.ones((36, ), dtype=np.bool)
        gt_label = None
        for k, v in ob_nav_cands.items():
            if t_cur < len(path) - 1 and k == path[t_cur + 1]: # non-stop
                gt_label = len(cand_img_feats1)
            non_cand_viewidxs[v[0]] = False
            cand_img_feats1.append(ob_img_feats1[v[0]])
            cand_img_feats2.append(ob_img_feats2[v[0]])
            tmp_angle = self.rel_angles[path_viewindex[t_cur]][v[0]]
            cand_ang_feats.append(
                angle_feature(tmp_angle[0] + v[2], tmp_angle[1] + v[3], self.angle_feat_size)
            )
        # cand_img_feats = np.stack(cand_img_feats, 0)
        cand_ang_feats = np.stack(cand_ang_feats, 0)

        # non cand views
        #######
        non_cand_img_feats1 = ob_img_feats1[non_cand_viewidxs]
        non_cand_img_feats2 = ob_img_feats2[non_cand_viewidxs]
        ######
        non_cand_ang_feats = ob_ang_feats[non_cand_viewidxs]

        # combine
        ob_nav_types = np.array(
            [1] * len(cand_img_feats1) + [2] + [0] * len(non_cand_img_feats1)
        )
        #####
        ob_img_feats1 = np.concatenate([cand_img_feats1, np.zeros((1, self.image_feat_size), dtype=np.float32), non_cand_img_feats1], 0)
        ob_img_feats2 = np.concatenate([cand_img_feats2, np.zeros((1, self.image_feat_size), dtype=np.float32), non_cand_img_feats2], 0)
        ######
        ob_ang_feats = np.concatenate([cand_ang_feats, np.zeros((1, self.angle_feat_size), dtype=np.float32), non_cand_ang_feats], 0)

        if gt_label is None:    # stop action
            gt_label = len(cand_img_feats1)
            gt_angle = np.zeros((2, ), dtype=np.float32)
        else:
            gt_angle = rel_act_angles[t_cur]

        return ob_img_feats1, ob_img_feats2, ob_ang_feats, ob_nav_types, gt_label, gt_angle


    def get_history_feature(
        self, scan, path, path_viewindex, rel_act_angles, t_cur,
        return_img_probs=False, d_vae=False
    ):
        # get history features before the step t_cur
        image_feats1, image_feats2, angle_feats, image_probs = [], [], [], []
        pano_image_feats1, pano_image_feats2, pano_angle_feats = [], [], []
        images_dvae = []
        ob_pano_image_feats, ob_pano_angle_feats = [], []

        for t in range(0, t_cur):
            vp = path[t]
            viewidx = path_viewindex[t]
            heading, elevation = rel_act_angles[t]

            if t == len(path) - 1: # STOP Action
                angle_feats.append(np.zeros((self.angle_feat_size, ), dtype=np.float32))
            else:
                angle_feats.append(angle_feature(heading, elevation, self.angle_feat_size))

            vp_fts1 = self.get_image_feature1(scan, vp, pad_stop_token=False)
            vp_fts2 = self.get_image_feature2(scan, vp, pad_stop_token=False)
            image_feats1.append(vp_fts1[viewidx, :self.image_feat_size])
            image_feats2.append(vp_fts2[viewidx, :self.image_feat_size])

            if self.hist_enc_pano:
                pano_image_feats1.append(vp_fts1[:, :self.image_feat_size])
                pano_image_feats2.append(vp_fts2[:, :self.image_feat_size])
                pano_angle_feats.append(self.angle_features[viewidx])

            if return_img_probs:
                image_probs.append(vp_fts1[viewidx, self.image_feat_size:])

            if d_vae:
                t_image_dvae = self.get_image(scan, vp, viewidx, d_vae)
                images_dvae.append(t_image_dvae)

        if t_cur > 0:
            image_feats1 = np.stack(image_feats1, 0)
            image_feats2 = np.stack(image_feats2, 0)
            angle_feats = np.stack(angle_feats)

            if self.hist_enc_pano:
                pano_image_feats1 = np.stack(pano_image_feats1, 0)
                pano_image_feats2 = np.stack(pano_image_feats2, 0)
                pano_angle_feats = np.stack(pano_angle_feats, 0)

            if return_img_probs:
                image_probs = np.stack(image_probs, 0)
                image_probs = softmax(image_probs)

            if d_vae:
                images_dvae = np.stack(images_dvae, 0)
        else:
            image_feats1 = np.zeros((0, self.image_feat_size), dtype=np.float32)
            image_feats2 = np.zeros((0, self.image_feat_size), dtype=np.float32)
            angle_feats = np.zeros((0, self.angle_feat_size), dtype=np.float32)
            if self.hist_enc_pano:
                pano_image_feats1 = np.zeros((0, 36, self.image_feat_size), dtype=np.float32)
                pano_image_feats2 = np.zeros((0, 36, self.image_feat_size), dtype=np.float32)
                pano_angle_feats = np.zeros((0, 36, self.angle_feat_size), dtype=np.float32)
            image_probs = np.zeros((0, self.image_prob_size), dtype=np.float32)

            if d_vae:
                images_dvae = np.zeros((0, 3, IMGSIZE, IMGSIZE), dtype=np.float32)

        if d_vae:
            return image_feats1, image_feats2, angle_feats, pano_image_feats1, pano_angle_feats, pano_angle_feats, images_dvae

        if return_img_probs:
            return image_feats1, image_feats2, angle_feats, pano_image_feats1,pano_image_feats2, pano_angle_feats, image_probs

        return image_feats1, image_feats2, angle_feats, pano_image_feats1,pano_image_feats2, pano_angle_feats


    def get_image_feature1(self, scan, viewpoint, pad_stop_token=False):
        key = '%s_%s' % (scan, viewpoint)
        feature_dict = self._feature_store1
        if self.in_memory and key in feature_dict:
            fts = feature_dict[key]
        else:
            with h5py.File(self.blip_ft_file, 'r') as f:
                fts = f[key][...].astype(np.float32)
            if self.in_memory:
                self._feature_store1[key] = fts

        if pad_stop_token:
            fts = np.vstack([fts, np.zeros((1, fts.shape[-1]), dtype=fts.dtype)])
        return fts
    def get_image_feature2(self, scan, viewpoint, pad_stop_token=False):
        key = '%s_%s' % (scan, viewpoint)
        feature_dict = self._feature_store2
        if self.in_memory and key in feature_dict:
            fts = feature_dict[key]
        else:
            with h5py.File(self.dino_ft_file, 'r') as f:
                fts = f[key][...].astype(np.float32)
            if self.in_memory:
                self._feature_store2[key] = fts

        if pad_stop_token:
            fts = np.vstack([fts, np.zeros((1, fts.shape[-1]), dtype=fts.dtype)])
        return fts

    def get_angle_feature(self, viewindex, pad_stop_token=False):
        fts = self.angle_features[viewindex]
        if pad_stop_token:
            fts = np.vstack([fts, np.zeros((1, fts.shape[-1]), dtype=fts.dtype)])
        return fts

    def get_progress(self, scan, start_vp, cur_vp, end_vp):
        if cur_vp == end_vp:
            return 1
        elif start_vp == cur_vp:
            return 0
        else:
            total_dist = self.shortest_distances[scan][start_vp][end_vp]
            remained_dist = self.shortest_distances[scan][cur_vp][end_vp]
            return 1 - remained_dist / max(total_dist, 0.1)

    def get_image(self, scan, viewpoint, viewindex, dvae=False):
        image_path = os.path.join(self.img_db_file, scan, viewpoint, str(viewindex) + ".jpg")

        im = Image.open(image_path)
        im = np.array(im)
        blob = Image.fromarray(im)
        p, t, _ = self.img_transform(blob)
        tokens = [t]

        tokens = torch.stack(tokens, 0).squeeze(0)
        return tokens