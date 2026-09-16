# Training

## Published recipes

All four releases were trained in SimpleTuner on 4 x NVIDIA L40S with PyTorch DDP, bfloat16
autocast, seed 42, 20 epochs, batch 16 per rank (global 64), no gradient accumulation, AdamW
under muP, learning rate 3e-4, weight decay 0.01, gradient norm limit 1.0, random 128 frame
training crops, deterministic 128 frame validation windows, validation and checkpoints every
500 steps, teacher KL weight 0.25 at temperature 1.0, exact alignment records only.

| Release | Width | Heads | Depth decoder | Schedule | Steps | Packaged checkpoint | Extra |
|---|---:|---:|---|---|---:|---|---|
| v1 | 512 | 8 | no | cosine, half period 500, no warm up | 17,640 | checkpoint-17500 | |
| v2 | 1,088 | 17 | no | linear, warm up 500, end 1e-7 | 17,660 | final | |
| v3 | 1,088 | 17 | no | linear, warm up 500, end 1e-7 | 17,660 | final | MERT alignment, weight 0.5 to step 0.7 T, zero by 0.9 T |
| v4 | 1,088 | 17 | 512 wide, 2 layers, 8 heads | linear, warm up 500, end 1e-7 | 17,660 | final | |

The v1 schedule reheats fully every 1,000 steps; the v2 to v4 schedule never reheats. The v1
replay cosine was measured on its final checkpoint while the packaged v1 is step 17,500; v2 to v4
replay and packaging both use final. v3's MERT projection is training only and is not exported,
so v2 and v3 have the same architecture and parameter count.

## Equivalent commands in this repository

Build the latent cache once (train and holdout), then train:

    uv run rvq-ae cache --split train --split holdout --revision 5029b1e7f1bbfbf028b76b38564fecccda94a111
    uv run torchrun --nproc_per_node 4 -m rvq_ae train --config configs/v4_169m.json
    uv run torchrun --nproc_per_node 4 -m rvq_ae train --config configs/v1_41m.json

| Config | Matches | Notes |
|---|---|---|
| configs/v1_41m.json | v1 | schedule cosine, warmup 500 is the half period |
| configs/v4_169m.json | v4 | schedule linear |
| configs/v4_169m_single_gpu.json | v4 on one 24 GB GPU | batch 16 x accumulation 4 keeps the effective batch of 64 windows; 396 ms per step and 5.3 GiB on an RTX 3090, about 1.9 h of compute for the 17,660 steps (docs/benchmarks.md) |

v2 is configs/v4_169m.json with depth_decoder false; the v3 auxiliary term is not implemented
here (docs/formulas.md section 7) because it changed replay cosine by 0.0004 and needs a MERT
dependency at training time.

Every TrainConfig field is documented in src/rvq_ae/train.py; the JSON keys are the field names.
precision takes fp32, bf16 or fp16 and becomes a torch dtype on load.

## Resume

A checkpoint folder holds the weights (rvq_encoder.safetensors), the config, trainer_state.json
and training_state.pt with the optimizer, the scheduler, the world size and the per rank CPU and
CUDA random state. Resuming with the same world size reproduces an uninterrupted run bitwise
(tests/test_train.py). With a different world size the run is reseeded from seed + rank + step
and continues from the same optimizer state.

    uv run torchrun --nproc_per_node 4 -m rvq_ae train --config configs/v4_169m.json --resume output/v4-169m/checkpoint-5000

Random crops are drawn from blake2b(seed, epoch, index), so a resumed epoch sees the same crops
as the original one.

## Two deliberate differences from the published inference adapter

1. The semantic readout and the depth context projection divide their input by the width
   multiplier m, as they did during training. The released adapter applies the same weights
   without that division. On a held out track this leaves semantic top 1 unchanged (43.6 against
   43.5 percent) and lowers free running acoustic top 1 from 7.1 to 5.8 percent, because the
   context projection feeds a nonlinear decoder. The published replay cosine was measured
   through the adapter.
2. Without a stitching table this repository uses the nominal timeline of docs/formulas.md
   section 2.1, the rule the training alignment converges to; the adapter uses a fixed 441 / 128
   stride.

## Evaluation of a run

    uv run rvq-ae evaluate --run output/v4-169m --split holdout

scores every checkpoint folder of the run on the cached holdout and writes metrics.json and
metrics.csv, with the free running and teacher forced columns of docs/evaluation.md.
