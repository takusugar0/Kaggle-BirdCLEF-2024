import copy
from configs.common import common_cfg
from modules.augmentations import (
    CustomCompose,
    CustomOneOf,
    NoiseInjection,
    GaussianNoise,
    PinkNoise,
    AddGaussianNoise,
    AddGaussianSNR,
)
from audiomentations import Compose as amCompose
from audiomentations import OneOf as amOneOf
from audiomentations import AddBackgroundNoise, Gain, GainTransition, TimeStretch
import numpy as np

cfg = copy.deepcopy(common_cfg)
if not cfg.WANDB_API_KEY:
    print('WANDB_API_KEY is not set')
    raise NotImplementedError

cfg.exp = 'exp001_baseline'
# cfg.exp = 'exp002_rmv_dupfiles'
# cfg.exp = 'exp003_add_background_noise'

cfg.model_type = "sed"
cfg.model_name = "tf_efficientnet_b3_ns"

cfg.secondary_label = 0.9
cfg.secondary_label_weight = 0.5


cfg.batch_size = 70
cfg.PRECISION = 32
cfg.seed = {
    "pretrain_ce": 20201121,
    "pretrain_bce": 20200503,
    "train_ce": 20201019,
    "train_bce": 2020101911,
    "finetune": 2020101921,
}
cfg.DURATION_TRAIN = 10
cfg.DURATION_FINETUNE = 30
cfg.freeze = False
cfg.mixup = True
cfg.mixup2 = True
cfg.mixup_prob = 0.6
cfg.mixup_double = 0.5
cfg.mixup2_prob = 0.3
cfg.mix_beta = 5
cfg.mix_beta2 = 2
cfg.in_chans = 3
cfg.epochs = {
    "pretrain_ce": 75,
    "pretrain_bce": 45,
    "train_ce": 70,
    "train_bce": 20,
    "finetune": 10,
}
cfg.lr = {
    "pretrain_ce": 3e-4,
    "pretrain_bce": 1e-3,
    "train_ce": 1e-3,
    "train_bce": 6e-3,
    "finetune": 6e-4,
}

cfg.model_ckpt = {
    "pretrain_ce": None,
    "pretrain_bce": "outputs/sed_b3ns/pytorch/pretrain_ce/last.ckpt",
    # "train_ce": f"outputs/sed_b3ns/pytorch/pretrain_ce/{cfg.exp}/last.ckpt",
    "train_ce": None,
    "train_bce": f"outputs/sed_b3ns/pytorch/train_bce/{cfg.exp}/v2/ckpt_epoch_epoch=44_val_loss=40.47.ckpt",
    # "train_bce": None,
    "finetune": "outputs/sed_b3ns/pytorch/train_bce/last.ckpt",
}

cfg.output_path = {
    "pretrain_ce": "outputs/sed_b3nspytorch/pretrain_ce",
    "pretrain_bce": "outputs/sed_b3ns/pytorch/pretrain_bce",
    "train_ce": f"outputs/sed_b3ns/pytorch/train_ce/{cfg.exp}",
    "train_bce": f"outputs/sed_b3ns/pytorch/train_bce/{cfg.exp}/v3",
    "finetune": "outputs/sed_b3ns/pytorch/finetune",
}

cfg.final_model_path = f"outputs/sed_b3ns/pytorch/train_ce/{cfg.exp}/last.ckpt"
cfg.onnx_path = f"outputs/sed_b3ns/onnx/{cfg.exp}"
cfg.openvino_path = f"outputs/sed_b3ns/openvino/{cfg.exp}"

cfg.loss = {
    "pretrain_ce": "ce",
    "pretrain_bce": "bce",
    "train_ce": "ce",
    "train_bce": "bce",
    "finetune": "bce",
}

cfg.img_size = 300
cfg.n_mels = 128
cfg.n_fft = 1024
cfg.f_min = 50
cfg.f_max = 14000

cfg.valid_part = int(cfg.valid_duration / cfg.infer_duration)
cfg.hop_length = cfg.infer_duration * cfg.SR // (cfg.img_size - 1)

cfg.normal = 255

cfg.tta_delta = 3

cfg.am_audio_transforms = amCompose(
    [
        # sed
        # AddBackgroundNoise(
        #     cfg.birdclef2021_nocall,
        #     min_snr_in_db=0,
        #     max_snr_in_db=3,
        #     p=0.5,
        # ),
        # AddBackgroundNoise(
        #     cfg.freefield + cfg.warblrb + cfg.birdvox,
        #     min_snr_in_db=0,
        #     max_snr_in_db=3,
        #     p=0.25,
        # ),
        # AddBackgroundNoise(
        #     cfg.rainforest, min_snr_in_db=0, max_snr_in_db=3, p=0.35
        # ),
        amOneOf(
            [
                Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
                GainTransition(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
            ],
        ),
    ]
)


cfg.np_audio_transforms = CustomCompose(
    [
        CustomOneOf(
            [
                NoiseInjection(p=1, max_noise_level=0.04),
                GaussianNoise(p=1, min_snr=5, max_snr=20),
                PinkNoise(p=1, min_snr=5, max_snr=20),
                AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.03, p=0.5),
                AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=0.5),
            ],
            p=0.3,
        ),
    ]
)

cfg.input_shape = (48,cfg.in_chans,cfg.n_mels,300)
cfg.input_names = [ "x",'tta_delta' ]
cfg.output_names = [ "y" ]
cfg.opset_version = None

basic_cfg = cfg
