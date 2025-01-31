import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'
import sys
import json
import argparse
import time
from collections import defaultdict
from easydict import EasyDict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoTokenizer, PretrainedConfig
from transformers import AutoModel

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_dropout, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from utils.parser import load_parser, parse_with_config

from optim import get_lr_sched
from optim.misc import build_optimizer

from data import (
    MultiStepNavData,
    MlmDataset, mlm_collate,
    SapDataset, sap_collate,
    SarDataset, sar_collate,
    SprelDataset, sprel_collate,
    MrcDataset, mrc_collate,
    ItmDataset, itm_collate,
    MimImageDataset, mim_image_collate,
    ApwigImageDataset, apwig_collate,
    MppDataset,mpp_collate,
    MetaLoader, PrefetchLoader,
    build_dataloader)

from model.pretrain_cmt import MultiStepNavCMTPreTraining

from data.beit_data_utils import DataAugmentationForBEiT

from model.beit.modeling_discrete_vae import Dalle_VAE

from data.beit_data import MultiStepNavBeitData

import numpy as np
from pretrain_src.model.cross_modality import CrossModality


def create_dataloaders(
    data_cfg, nav_db, tok, is_train: bool, device: torch.device, opts
):
    dataloaders = {}
    for k, task_name in enumerate(data_cfg.tasks):
        if task_name == 'mlm':
            task_dataset = MlmDataset(nav_db, tok)
            task_collate_fn = mlm_collate
        elif task_name == 'sap':
            task_dataset = SapDataset(
                nav_db,
                tok,
                opts.ob_random_kill_v if is_train else 0,
                opts.ob_random_kill_a if is_train else 0
            )
            task_collate_fn = sap_collate
        elif task_name == 'sar':
            task_dataset = SarDataset(
                nav_db,
                tok,
                opts.ob_random_kill_v if is_train else 0,
                opts.ob_random_kill_a if is_train else 0
            )
            task_collate_fn = sar_collate
        elif task_name == 'sprel':
            task_dataset = SprelDataset(
                nav_db,
                tok,
                opts.ob_random_kill_v if is_train else 0,
                opts.ob_random_kill_a if is_train else 0
            )
            task_collate_fn = sprel_collate
        elif task_name == 'mrc':
            task_dataset = MrcDataset(nav_db, tok, opts.mrc_mask_prob)
            task_collate_fn = mrc_collate
        elif task_name == 'itm':
            task_dataset = ItmDataset(nav_db, tok)
            task_collate_fn = itm_collate
        elif task_name == 'mim':
            if opts.filter_dvae:
                count = np.load("visual_token_count.npy")
                count_dict = {}
                for i in range(len(count)):
                    count_dict[i] = count[i]
                sort_count = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
                sort_index = [key for key, value in sort_count]
                dvae_filter = sorted(sort_index[:opts.filter_dvae])
            else:
                dvae_filter = None
            task_dataset = MimImageDataset(nav_db, tok, opts.mim_mask_prob, dvae_filter=dvae_filter)
            task_collate_fn = mim_image_collate
        elif task_name == 'apwig':
            if opts.filter_dvae_apwig:
                count = np.load("visual_token_count.npy")
                count_dict = {}
                for i in range(len(count)):
                    count_dict[i] = count[i]
                sort_count = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
                sort_index = [key for key, value in sort_count]
                dvae_filter = sorted(sort_index[:opts.filter_dvae_apwig])
            else:
                dvae_filter = None
            task_dataset = ApwigImageDataset(nav_db, tok, dvae_filter)
            task_collate_fn = apwig_collate
        elif task_name == 'mpp':
            if opts.filter_dvae_mpp:
                count = np.load("visual_token_count.npy")
                count_dict = {}
                for i in range(len(count)):
                    count_dict[i] = count[i]
                sort_count = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
                sort_index = [key for key, value in sort_count]
                dvae_filter = sorted(sort_index[:opts.filter_dvae_mpp])
            else:
                dvae_filter = None
            task_dataset = MppDataset(nav_db, tok, opts.mpp_mask_prob, dvae_filter)
            task_collate_fn = mpp_collate
        elif task_name == 'mapwig':
            if opts.filter_dvae_apwig:
                count = np.load("visual_token_count.npy")
                count_dict = {}
                for i in range(len(count)):
                    count_dict[i] = count[i]
                sort_count = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
                sort_index = [key for key, value in sort_count]
                dvae_filter = sorted(sort_index[:opts.filter_dvae_apwig])
            else:
                dvae_filter = None
            task_dataset = ApwigImageDataset(nav_db, tok, dvae_filter)
            task_collate_fn = apwig_collate
        else:
            raise ValueError(f'Undefined task {task}')

        LOGGER.info(f"{task_name}: {len(task_dataset)} samples loaded")

        task_loader, pre_epoch = build_dataloader(
            task_name, task_dataset, task_collate_fn, is_train, opts
        )

        if is_train:
            ratio = data_cfg.mix_ratio[k]
            dataloaders[task_name] = (task_loader, ratio, pre_epoch)
        else:
            dataloaders[task_name] = PrefetchLoader(task_loader, device)
    return dataloaders


def main(opts):
    default_gpu, n_gpu, device = set_cuda(opts)

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}'.format(
                device, n_gpu, bool(opts.local_rank != -1), opts.fp16
            )
        )

    seed = opts.seed
    if opts.local_rank != -1:
        seed += opts.rank
    set_random_seed(seed)

    if default_gpu:
        save_training_meta(opts)
        TB_LOGGER.create(os.path.join(opts.output_dir, 'logs'))
        pbar = tqdm(initial=opts.start_step, total=opts.num_train_steps)
        model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # Model config
    model_config = PretrainedConfig.from_json_file(opts.model_config)
    model_config.pretrain_tasks = []
    for train_dataset_config in opts.train_datasets.values():
        model_config.pretrain_tasks.extend(train_dataset_config['tasks'])
    model_config.pretrain_tasks = set(model_config.pretrain_tasks)

    if opts.filter_dvae:
        model_config.dvae_classes = min(model_config.dvae_classes, opts.filter_dvae)
    if opts.filter_dvae_apwig:
        model_config.dvae_classes_apwig = min(model_config.dvae_classes_apwig, opts.filter_dvae_apwig)
    if opts.filter_dvae_mpp:
        model_config.dvae_classes_mpp = min(model_config.dvae_classes_mpp, opts.filter_dvae_mpp)

    tokenizer = AutoTokenizer.from_pretrained(model_config.lang_bert_name, cache_dir="cache")

    # Prepare model
    if opts.checkpoint:
        print("Load checkpoint:", opts.checkpoint)
        checkpoint = torch.load(opts.checkpoint, map_location=lambda storage, loc: storage)
    else:
        checkpoint = {}
        if opts.init_pretrained_bert:
            tmp = AutoModel.from_pretrained(model_config.lang_bert_name, cache_dir="cache")
            for param_name, param in tmp.named_parameters():
                checkpoint[param_name] = param
            if model_config.lang_bert_name == 'xlm-roberta-base':
                # embeddings.token_type_embeddings.weight (1 -> 2, the second is for image embedding)
                checkpoint['embeddings.token_type_embeddings.weight'] = torch.cat(
                    [checkpoint['embeddings.token_type_embeddings.weight']] * 2, 0
                )
            del tmp

    model = MultiStepNavCMTPreTraining.from_pretrained(
        pretrained_model_name_or_path=None, config=model_config, state_dict=checkpoint, cache_dir="cache"
    )
    cross_model = CrossModality()
    cross_model.train()
    model.train()
    set_dropout(model, opts.dropout)

    model = wrap_model(model, device, opts.local_rank)
    del checkpoint

    print("total_param", sum(p.numel() for p in model.parameters() if p.requires_grad))

    patch_size = (16, 16)
    print("Patch size = %s" % str(patch_size))
    opts.window_size = (opts.input_size // patch_size[0], opts.input_size // patch_size[1])
    opts.patch_size = patch_size

    preprocess = DataAugmentationForBEiT(opts)

    d_vae = Dalle_VAE(opts.second_input_size)
    d_vae.load_model(model_dir=opts.discrete_vae_weight_path, device=device)

    r2r_cfg = EasyDict(opts.train_datasets['R2R'])
    r2r_cfg['patch_nums'] = model_config.patch_nums

    img_db_file = r2r_cfg.img_db_file
    train_nav_db = MultiStepNavBeitData(
        r2r_cfg.train_traj_files, r2r_cfg.img_ft_file,
        r2r_cfg.blip_ft_file , r2r_cfg.dino_ft_file,
        r2r_cfg.scanvp_cands_file, r2r_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len, max_act_len=model_config.max_action_steps,
        hist_enc_pano=model_config.num_h_pano_layers > 0,
        ob_cand_pano_view=opts.ob_cand_pano_view,
        val_sample_num=None, in_memory=True,
        img_db_file=img_db_file,
        preprocess=preprocess
    )
    val_nav_db = MultiStepNavBeitData(
        r2r_cfg.val_seen_traj_files, r2r_cfg.img_ft_file,
        r2r_cfg.blip_ft_file , r2r_cfg.dino_ft_file,
        r2r_cfg.scanvp_cands_file, r2r_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len, max_act_len=model_config.max_action_steps,
        hist_enc_pano=model_config.num_h_pano_layers > 0,
        ob_cand_pano_view=opts.ob_cand_pano_view,
        val_sample_num=opts.val_sample_num, in_memory=True,
        img_db_file=img_db_file,
        preprocess=preprocess,
    )
    val2_nav_db = MultiStepNavBeitData(
        r2r_cfg.val_unseen_traj_files, r2r_cfg.img_ft_file,
        r2r_cfg.blip_ft_file , r2r_cfg.dino_ft_file,
        r2r_cfg.scanvp_cands_file, r2r_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len, max_act_len=model_config.max_action_steps,
        hist_enc_pano=model_config.num_h_pano_layers > 0,
        ob_cand_pano_view=opts.ob_cand_pano_view,
        val_sample_num=opts.val_sample_num, in_memory=True,
        img_db_file=img_db_file,
        preprocess=preprocess,
    )

    # Build data loaders
    train_dataloaders = create_dataloaders(
        r2r_cfg, train_nav_db, tokenizer, True, device, opts
    )
    val_dataloaders = create_dataloaders(
        r2r_cfg, val_nav_db, tokenizer, False, device, opts
    )
    val2_dataloaders = create_dataloaders(
        r2r_cfg, val2_nav_db, tokenizer, False, device, opts
    )
    meta_loader = MetaLoader(
        train_dataloaders,
        accum_steps=opts.gradient_accumulation_steps,
        distributed=opts.local_rank != -1,
        device=device
    )
    meta_loader = PrefetchLoader(meta_loader, device)

    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}

    global_step = opts.start_step
    TB_LOGGER._global_step = opts.start_step
    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.train_batch_size if opts.local_rank == -1 else opts.train_batch_size * opts.world_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)

    # to compute training statistics
    task2loss = {task: RunningMeter(f'loss/{task}')
                 for task in train_dataloaders.keys()}

    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)
    grad_norm = 0

    start_time = time.time()

    optimizer.zero_grad()
    optimizer.step()

    if opts.dynamic_filter_dvae:
        count = np.load("visual_token_count.npy")
        count_dict = {}
        for i in range(len(count)):
            count_dict[i] = count[i]
        sort_count = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        sort_index = [key for key, value in sort_count]
        if opts.filter_dvae:
            dynamic_dvae_filter = np.arange(opts.dynamic_filter_dvae)
        else:
            dynamic_dvae_filter = np.array(sorted(sort_index[:opts.dynamic_filter_dvae]))
    else:
        dynamic_dvae_filter = None

    dynamic_dvae_filter_score = None

    if opts.dynamic_filter_dvae_mpp:
        count = np.load("visual_token_count.npy")
        count_dict = {}
        for i in range(len(count)):
            count_dict[i] = count[i]
        sort_count = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        sort_index = [key for key, value in sort_count]
        if opts.filter_dvae_mpp:
            dynamic_dvae_filter_mpp = np.arange(opts.dynamic_filter_dvae_mpp)
        else:
            dynamic_dvae_filter_mpp = np.array(sorted(sort_index[:opts.dynamic_filter_dvae_mpp]))
    else:
        dynamic_dvae_filter_mpp = None

    dynamic_dvae_filter_score_mpp = None

    if opts.dynamic_filter_dvae_apwig:
        count = np.load("visual_token_count.npy")
        count_dict = {}
        for i in range(len(count)):
            count_dict[i] = count[i]
        sort_count = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        sort_index = [key for key, value in sort_count]
        if opts.filter_dvae_apwig:
            dynamic_dvae_filter_apwig = np.arange(opts.dynamic_filter_dvae_apwig)
        else:
            dynamic_dvae_filter_apwig = np.array(sorted(sort_index[:opts.dynamic_filter_dvae_apwig]))
    else:
        dynamic_dvae_filter_apwig = None

    dynamic_dvae_filter_score_apwig = None

    for step, (name, batch) in enumerate(meta_loader):
        # forward pass
        n_examples[name] += batch['txt_ids'].size(0)
        n_in_units[name] += (batch['txt_masks'] == 1).sum().item()
        task = name.split('_')[0]

        if task == "mim":
            loss = model(batch, task=task, compute_loss=True, d_vae=d_vae, dy_filter=dynamic_dvae_filter)
        elif task == "mpp":
            loss = model(batch, task=task, compute_loss=True, d_vae=d_vae, dy_filter=dynamic_dvae_filter_mpp)
        elif task == "mapwig" or task == "apwig":
            loss = model(batch, task=task, compute_loss=True, d_vae=d_vae, dy_filter=dynamic_dvae_filter_apwig)
        else:
            loss = model(batch, task=task, compute_loss=True, d_vae=d_vae)

        if task == "mim" and isinstance(loss, tuple):
            loss, target, difficulty = loss
            if opts.dynamic_filter_dvae:
                update_score = np.mean(target + opts.dynamic_filter_dvae_d_weight * difficulty, axis=0)
                # print(update_score.shape)
                if dynamic_dvae_filter_score is None:
                    dynamic_dvae_filter_score = update_score
                else:
                    dynamic_dvae_filter_score = dynamic_dvae_filter_score * opts.dynamic_filter_dvae_weight + (1 - opts.dynamic_filter_dvae_weight) * update_score

                dynamic_dvae_filter = dynamic_dvae_filter_score.argsort()[-opts.dynamic_filter_dvae:]

        if task == "mpp" and isinstance(loss, tuple):
            loss, target, difficulty = loss
            if opts.dynamic_filter_dvae_mpp:
                update_score = np.mean(target + opts.dynamic_filter_dvae_d_weight * difficulty, axis=0)
                # print(update_score.shape)
                if dynamic_dvae_filter_score_mpp is None:
                    dynamic_dvae_filter_score_mpp = update_score
                else:
                    dynamic_dvae_filter_score_mpp = dynamic_dvae_filter_score_mpp * opts.dynamic_filter_dvae_weight + (
                                1 - opts.dynamic_filter_dvae_weight) * update_score

                dynamic_dvae_filter_mpp = dynamic_dvae_filter_score_mpp.argsort()[-opts.dynamic_filter_dvae_mpp:]

        if (task == "mapwig" or task == "apwig") and isinstance(loss, tuple):
            loss, target, difficulty = loss
            if opts.dynamic_filter_dvae_apwig:
                update_score = np.mean(target + opts.dynamic_filter_dvae_d_weight * difficulty, axis=0)
                # print(update_score.shape)
                if dynamic_dvae_filter_score_apwig is None:
                    dynamic_dvae_filter_score_apwig = update_score
                else:
                    dynamic_dvae_filter_score_apwig = dynamic_dvae_filter_score_apwig * opts.dynamic_filter_dvae_weight + (
                            1 - opts.dynamic_filter_dvae_weight) * update_score

                dynamic_dvae_filter_apwig = dynamic_dvae_filter_score_apwig.argsort()[-opts.dynamic_filter_dvae_apwig:]

        n_loss_units[name] += loss.size(0)
        loss = loss.mean()  # loss is not normalized in model

        # backward pass
        if args.gradient_accumulation_steps > 1: # average loss
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        task2loss[name](loss.item())

        # optimizer update and logging
        if (step + 1) % opts.gradient_accumulation_steps == 0:
            global_step += 1

            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, opts)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
            # NOTE: not gathered across GPUs for efficiency
            TB_LOGGER.log_scalar_dict({ll.name: ll.val
                                       for ll in task2loss.values()
                                       if ll.val is not None})
            TB_LOGGER.step()

            # update model params
            if opts.grad_norm != -1:
                if opts.fp16:
                    grad_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opts.grad_norm
                )
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            if opts.fp16:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)

            if global_step % opts.log_steps == 0:
                # monitor training throughput
                LOGGER.info(f'==============Step {global_step}===============')
                for t in train_dataloaders.keys():
                    tot_ex = n_examples[t]
                    ex_per_sec = int(tot_ex / (time.time() - start_time))
                    tot_in = n_in_units[t]
                    in_per_sec = int(tot_in / (time.time() - start_time))
                    tot_l = n_loss_units[t]
                    l_per_sec = int(tot_l / (time.time() - start_time))
                    LOGGER.info(f'{t}: {tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar(f'perf/{t}_ex_per_s', ex_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_in_per_s', in_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_loss_per_s', l_per_sec,
                                         global_step)
                LOGGER.info('===============================================')

            if global_step % opts.valid_steps == 0:
                LOGGER.info(f'------Step {global_step}: start validation seen------')
                validate(model, val_dataloaders, setname='_seen', d_vae=d_vae, dynamic_dvae_filter=dynamic_dvae_filter, dynamic_dvae_filter_mpp = dynamic_dvae_filter_mpp, dynamic_dvae_filter_apwig = dynamic_dvae_filter_apwig)
                LOGGER.info(f'------Step {global_step}: start validation unseen------')
                validate(model, val2_dataloaders, setname='_unseen', d_vae=d_vae, dynamic_dvae_filter=dynamic_dvae_filter, dynamic_dvae_filter_mpp = dynamic_dvae_filter_mpp, dynamic_dvae_filter_apwig = dynamic_dvae_filter_apwig)
                model_saver.save(model, global_step, optimizer)
        if global_step >= opts.num_train_steps:
            break
    if global_step % opts.valid_steps != 0:
        LOGGER.info(f'------Step {global_step}: start validation seen------')
        validate(model, val_dataloaders, setname='_seen', d_vae=d_vae, dynamic_dvae_filter=dynamic_dvae_filter)
        LOGGER.info(f'------Step {global_step}: start validation unseen------')
        validate(model, val2_dataloaders, setname='_unseen', d_vae=d_vae, dynamic_dvae_filter=dynamic_dvae_filter)
        model_saver.save(model, global_step, optimizer)


def validate(model, val_dataloaders, setname='', d_vae=None, dynamic_dvae_filter=None, dynamic_dvae_filter_mpp=None, dynamic_dvae_filter_apwig=None):
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate val{setname} on {task} task")
        if task.startswith('mlm'):
            val_log = validate_mlm(model, loader)
        elif task == 'sap':
            val_log = validate_sap(model, loader)
        elif task.startswith('sar'):
            val_log = validate_sar(model, loader)
        elif task.startswith('sprel'):
            val_log = validate_sprel(model, loader)
        elif task.startswith('mrc'):
            val_log = validate_mrc(model, loader)
        elif task.startswith('itm'):
            val_log = validate_itm(model, loader)
        elif task.startswith('mim'):
            val_log = validate_mim(model, loader, d_vae, dynamic_dvae_filter)
        elif task.startswith('apwig'):
            val_log = validate_apwig(model, loader, d_vae, dynamic_dvae_filter_apwig)
        elif task.startswith('mpp'):
            val_log = validate_mpp(model, loader, d_vae, dynamic_dvae_filter_mpp)
        elif task.startswith('mapwig'):
            val_log = validate_mapwig(model, loader, d_vae, dynamic_dvae_filter_apwig)
        else:
            raise ValueError(f'Undefined task {task}')
        val_log = {f'val{setname}_{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scalar_dict(
            {f'valid{setname}_{task}/{k}': v for k, v in val_log.items()}
        )
    model.train()


@torch.no_grad()
def validate_mlm(model, val_loader):
    LOGGER.info("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='mlm', compute_loss=False)
        labels = batch['txt_labels']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
    val_loss = sum(all_gather(val_loss))
    n_correct = sum(all_gather(n_correct))
    n_word = sum(all_gather(n_word))
    tot_time = time.time()-st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log

@torch.no_grad()
def validate_sap(model, val_loader):
    LOGGER.info("start running SAP validation...")
    val_loss = 0
    n_correct = 0
    n_data = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='sap', compute_loss=False)
        labels = batch['ob_action_viewindex']
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_data += labels.numel()
    val_loss = sum(all_gather(val_loss))
    n_correct = sum(all_gather(n_correct))
    n_data = sum(all_gather(n_data))
    tot_time = time.time()-st
    val_loss /= n_data
    acc = n_correct / n_data
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_data/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log

@torch.no_grad()
def validate_sar(model, val_loader):
    LOGGER.info("start running SAR validation...")
    val_heading_loss, val_elevation_loss, val_progress_loss = 0, 0, 0
    n_data = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='sar', compute_loss=False)
        val_heading_loss += F.mse_loss(scores[:, 0], batch['ob_action_angles'][:, 0], reduction='sum').item()
        val_elevation_loss += F.mse_loss(scores[:, 1], batch['ob_action_angles'][:, 1], reduction='sum').item()
        val_progress_loss += F.mse_loss(scores[:, 2], batch['ob_progress'], reduction='sum').item()
        n_data += scores.size(0)
    val_heading_loss = sum(all_gather(val_heading_loss))
    val_elevation_loss = sum(all_gather(val_elevation_loss))
    val_progress_loss = sum(all_gather(val_progress_loss))
    n_data = sum(all_gather(n_data))
    tot_time = time.time()-st
    val_heading_loss /= n_data
    val_elevation_loss /= n_data
    val_progress_loss /= n_data
    val_log = {'heading_loss': val_heading_loss,
               'elevation_loss': val_elevation_loss,
               'progress_loss': val_progress_loss,
               'tok_per_s': n_data/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"heading_loss: {val_heading_loss:.4f}, "
                f"elevation_loss: {val_elevation_loss:.4f}, "
                f"progress_loss: {val_progress_loss:.4f}")
    return val_log

@torch.no_grad()
def validate_sprel(model, val_loader):
    LOGGER.info("start running SPREL validation...")
    val_heading_loss, val_elevation_loss = 0, 0
    n_data = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='sprel', compute_loss=False)
        val_heading_loss += F.mse_loss(scores[:, 0], batch['sp_targets'][:, 0], reduction='sum').item()
        val_elevation_loss += F.mse_loss(scores[:, 1], batch['sp_targets'][:, 1], reduction='sum').item()
        n_data += scores.size(0)
    val_heading_loss = sum(all_gather(val_heading_loss))
    val_elevation_loss = sum(all_gather(val_elevation_loss))
    n_data = sum(all_gather(n_data))
    tot_time = time.time()-st
    val_heading_loss /= n_data
    val_elevation_loss /= n_data
    val_log = {'heading_loss': val_heading_loss,
               'elevation_loss': val_elevation_loss,
               'tok_per_s': n_data/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"heading_loss: {val_heading_loss:.4f}, "
                f"elevation_loss: {val_elevation_loss:.4f}")
    return val_log

def compute_accuracy_for_soft_targets(out, labels):
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]  # argmax
    n_correct = (outputs == labels).sum().item()
    return n_correct

@torch.no_grad()
def validate_mrc(model, val_loader):
    LOGGER.info("start running MRC validation...")
    val_loss = 0
    n_feat = 0
    st = time.time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        prediction_soft_label, img_target_probs = model(batch, task='mrc', compute_loss=False)
        prediction_soft_label = F.log_softmax(prediction_soft_label, dim=-1)
        loss = F.kl_div(prediction_soft_label, img_target_probs, reduction='sum')
        tot_score += compute_accuracy_for_soft_targets(prediction_soft_label, img_target_probs)
        val_loss += loss.item()
        n_feat += batch['hist_mrc_masks'].sum().item()
    val_loss = sum(all_gather(val_loss))
    tot_score = sum(all_gather(tot_score))
    n_feat = sum(all_gather(n_feat))
    tot_time = time.time()-st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log

@torch.no_grad()
def validate_itm(model, val_loader):
    LOGGER.info("start running ITM validation...")
    val_loss = 0
    n_correct = 0
    n_data = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        scores, labels = model(batch, task='itm', compute_loss=False)
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_data += labels.numel()
    val_loss = sum(all_gather(val_loss))
    n_correct = sum(all_gather(n_correct))
    n_data = sum(all_gather(n_data))
    tot_time = time.time()-st
    val_loss /= n_data
    acc = n_correct / n_data
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_data/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log

@torch.no_grad()
def validate_mim(model, val_loader, d_vae, dynamic_dvae_filter):
    LOGGER.info("start running MIM validation...")
    val_loss = 0
    n_feat = 0
    st = time.time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        prediction_soft_label, img_target_probs = model(batch, task='mim', compute_loss=False, d_vae=d_vae, dy_filter=dynamic_dvae_filter)
        prediction_soft_label = F.log_softmax(prediction_soft_label, dim=-1)
        loss = F.kl_div(prediction_soft_label, img_target_probs, reduction='sum')
        tot_score += compute_accuracy_for_soft_targets(prediction_soft_label, img_target_probs)
        val_loss += loss.item()
        n_feat += batch['hist_mrc_masks'].sum().item()
    val_loss = sum(all_gather(val_loss))
    tot_score = sum(all_gather(tot_score))
    n_feat = sum(all_gather(n_feat))
    tot_time = time.time()-st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log


@torch.no_grad()
def validate_apwig(model, val_loader, d_vae, dynamic_dvae_filter):
    LOGGER.info("start running APWIG validation...")
    val_loss = 0
    n_feat = 0
    st = time.time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        prediction_soft_label, img_target_probs = model(batch, task='apwig', compute_loss=False, d_vae=d_vae, dy_filter=dynamic_dvae_filter)
        prediction_soft_label = F.log_softmax(prediction_soft_label, dim=-1)
        loss = F.kl_div(prediction_soft_label, img_target_probs, reduction='sum')
        tot_score += compute_accuracy_for_soft_targets(prediction_soft_label, img_target_probs)
        val_loss += loss.item()
        n_feat += prediction_soft_label.size(0)
    val_loss = sum(all_gather(val_loss))
    tot_score = sum(all_gather(tot_score))
    n_feat = sum(all_gather(n_feat))
    tot_time = time.time()-st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log

@torch.no_grad()
def validate_mpp(model, val_loader, d_vae, dynamic_dvae_filter):
    LOGGER.info("start running MPP validation...")
    val_loss = 0
    n_feat = 0
    st = time.time()
    tot_score = 0
    n_data = 0
    for i, batch in enumerate(val_loader):
        prediction_soft_label, img_target_probs = model(batch, task='mpp', compute_loss=False, d_vae=d_vae, dy_filter=dynamic_dvae_filter)
        prediction_soft_label = F.log_softmax(prediction_soft_label, dim=-1)
        loss = F.kl_div(prediction_soft_label, img_target_probs, reduction='sum')
        tot_score += compute_accuracy_for_soft_targets(prediction_soft_label, img_target_probs)
        val_loss += loss.item()
        n_feat += batch['ob_mpp_masks'].sum().item()
        if i > 10:
            break
    val_loss = sum(all_gather(val_loss))
    tot_score = sum(all_gather(tot_score))
    n_feat = sum(all_gather(n_feat))
    tot_time = time.time()-st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log


@torch.no_grad()
def validate_mapwig(model, val_loader, d_vae, dynamic_dvae_filter):
    LOGGER.info("start running MAPWIG validation...")
    val_loss = 0
    n_feat = 0
    st = time.time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        prediction_soft_label, img_target_probs = model(batch, task='mapwig', compute_loss=False, d_vae=d_vae, dy_filter=dynamic_dvae_filter)
        prediction_soft_label = F.log_softmax(prediction_soft_label, dim=-1)
        loss = F.kl_div(prediction_soft_label, img_target_probs, reduction='sum')
        tot_score += compute_accuracy_for_soft_targets(prediction_soft_label, img_target_probs)
        val_loss += loss.item()
        n_feat += prediction_soft_label.size(0)
    val_loss = sum(all_gather(val_loss))
    tot_score = sum(all_gather(tot_score))
    n_feat = sum(all_gather(n_feat))
    tot_time = time.time()-st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log


def build_args():
    parser = load_parser()

    opts = parse_with_config(parser)

    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir
            )
        )

    return opts

if __name__ == '__main__':
    args = build_args()
    main(args)
