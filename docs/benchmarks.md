# Benchmarks

Generated numbers live in benchmarks/benchmarks.json and benchmarks/benchmarks.md; this page
records how they are measured and what they mean. Reproduce with

    uv run rvq-ae bench --device cuda --suite all

Every suite except audio builds its own weights from the released configurations and needs no
network; audio loads the released v4 encoder and the DAV encoder from the Hub.

## Machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, sm_86, 23.6 GiB, driver 580.173.02 |
| software | Python 3.13.15, torch 2.14.0+cu130, CUDA 13.0 |
| settings | bf16 autocast for model suites, tf32 for float32 matmul and convolution |

The published runs used 4 x NVIDIA L40S; the comparison in the training suite converts the
published step count to compute time on this card.

## Method (src/rvq_ae/bench/harness.py)

Each case runs a fixed warm up, then a fixed number of repetitions timed with CUDA events so the
figure is device time rather than launch time. The reported latency is the median, which
rejects the occasional allocator growth or clock excursion without hiding a regression. Peak
memory is torch.cuda.max_memory_allocated after a reset at the start of the measured region, so
it covers only that region. Host side cases use the wall clock. Suites are in
src/rvq_ae/bench/suites.py and each takes explicit inputs, so a case can be rerun in isolation.

## Suites and their reading

<!-- table: benchmarks -->
#### attention

| Case | Variant | ms | Peak MiB | Note |
|---|---|---:|---:|---|
| temporal (b16, 17h, len 128) forward | reference | 0.238 | 73.9 |  |
| temporal (b16, 17h, len 128) backward | reference | 0.880 | 118.0 |  |
| temporal (b16, 17h, len 128) forward | fused sdpa | 0.066 | 54.6 | max abs difference 1.6e-02 |
| temporal (b16, 17h, len 128) backward | fused sdpa | 0.309 | 71.8 | max abs difference 1.6e-02 |
| depth (b2048, 8h, len 8) forward | reference | 0.388 | 116.3 |  |
| depth (b2048, 8h, len 8) backward | reference | 0.995 | 176.3 |  |
| depth (b2048, 8h, len 8) forward | fused sdpa | 1.132 | 160.8 | max abs difference 1.6e-02 |
| depth (b2048, 8h, len 8) backward | fused sdpa | 3.791 | 713.3 | max abs difference 1.6e-02 |

#### pooling

| Case | Variant | ms | Peak MiB | Note |
|---|---|---:|---:|---|
| b16 x 128 frames x 441 latents x 1088 | dense bmm | 0.122 | 55.0 |  |
| b16 x 128 frames x 441 latents x 1088 | segment mean | 0.124 | 63.6 | max abs difference 1.5e-03 |
| b64 x 128 frames x 441 latents x 1088 | dense bmm | 0.353 | 168.7 |  |
| b64 x 128 frames x 441 latents x 1088 | segment mean | 0.375 | 203.0 | max abs difference 1.6e-03 |
| b16 x 512 frames x 1765 latents x 1088 | dense bmm | 1.219 | 171.9 |  |
| b16 x 512 frames x 1765 latents x 1088 | segment mean | 0.373 | 206.0 | max abs difference 1.5e-03 |
| collate b16 x 128 frames x 441 latents | dense matrix | 19.808 | 0.0 | 903,168 elements built per batch |
| collate b16 x 128 frames x 441 latents | segment index | 1.757 | 0.0 | 7,056 elements built per batch |
| collate b64 x 128 frames x 441 latents | dense matrix | 152.901 | 0.0 | 3,612,672 elements built per batch |
| collate b64 x 128 frames x 441 latents | segment index | 8.458 | 0.0 | 28,224 elements built per batch |

#### encoder

| Case | Variant | ms | Peak MiB | frames/s | Note |
|---|---|---:|---:|---:|---|
| v1 41M batch 8 | forward, teacher forced | 5.107 | 306.2 | 200,525.1 |  |
| v1 41M batch 8 | forward and backward | 22.466 | 856.9 | 45,580.9 |  |
| v1 41M batch 8 | forward and backward, checkpointed | 41.229 | 689.8 | 24,837.0 |  |
| v1 41M batch 16 | forward, teacher forced | 7.805 | 522.5 | 262,382.7 |  |
| v1 41M batch 16 | forward and backward | 27.725 | 1302.8 | 73,867.5 |  |
| v1 41M batch 16 | forward and backward, checkpointed | 41.530 | 966.5 | 49,313.3 |  |
| v1 41M batch 32 | forward, teacher forced | 13.159 | 625.9 | 311,259.8 |  |
| v1 41M batch 32 | forward and backward | 44.045 | 2344.1 | 92,995.1 |  |
| v1 41M batch 32 | forward and backward, checkpointed | 50.689 | 1621.3 | 80,806.6 |  |
| v4 169M batch 8 | forward, teacher forced | 15.474 | 1147.9 | 66,174.4 |  |
| v4 169M batch 8 | forward and backward | 48.365 | 2781.9 | 21,172.3 |  |
| v4 169M batch 8 | forward and backward, checkpointed | 60.785 | 2405.8 | 16,846.4 |  |
| v4 169M batch 16 | forward, teacher forced | 27.005 | 1985.4 | 75,838.4 |  |
| v4 169M batch 16 | forward and backward | 84.472 | 3928.4 | 24,244.8 |  |
| v4 169M batch 16 | forward and backward, checkpointed | 96.110 | 3185.4 | 21,309.0 |  |
| v4 169M batch 32 | forward, teacher forced | 49.636 | 2344.6 | 82,520.2 |  |
| v4 169M batch 32 | forward and backward | 156.053 | 6172.5 | 26,247.5 |  |
| v4 169M batch 32 | forward and backward, checkpointed | 177.390 | 4725.8 | 23,090.4 |  |

#### depth decoder

| Case | Variant | ms | Peak MiB | frames/s | Note |
|---|---|---:|---:|---:|---|
| v4 169M batch 16 x 128 frames | teacher forced | 7.413 | 977.0 | 276,262.2 |  |
| v4 169M batch 16 x 128 frames | free running | 30.582 | 1037.0 | 66,966.9 |  |

#### training

| Case | Variant | ms | Peak MiB | frames/s | Note |
|---|---|---:|---:|---:|---|
| v4 169M single GPU (64 windows per step) | batch 16 x accumulation 4 | 395.545 | 5467.0 | 20,710.7 | 1.9 h for the published 17,660 steps |

#### audio

| Case | Variant | ms | Peak MiB | x real time | Note |
|---|---|---:|---:|---:|---|
| 30 s stereo track | dav encoder | 344.614 | 4005.8 | 87.1 |  |
| 30 s stereo track | rvq encoder | 100.004 | 818.2 | 300.0 |  |
| 30 s stereo track | end to end | 398.878 | 4005.8 | 75.2 |  |
| 180 s stereo track | dav encoder | 2214.803 | 4332.6 | 81.3 |  |
| 180 s stereo track | rvq encoder | 652.836 | 824.2 | 275.7 |  |
| 180 s stereo track | end to end | 2803.223 | 4332.6 | 64.2 |  |
| 360 s stereo track | dav encoder | 4448.245 | 4475.7 | 80.9 |  |
| 360 s stereo track | rvq encoder | 1239.706 | 832.2 | 290.4 |  |
| 360 s stereo track | end to end | 5601.924 | 4475.7 | 64.3 |  |
<!-- end table -->

### Attention

Two shapes, the ones the model actually runs: the temporal stack (batch 16, 17 heads, length
128) and the depth decoder (batch 2,048 = 16 x 128 frames, 8 heads, length 8). The fused kernel
is 3.6x faster forward and 2.8x faster backward on the temporal shape, and 2.9x and 3.8x slower
on the depth shape, where sequences are too short for the tiled kernels to amortise their set
up. The code follows the measurement: the temporal stack uses the fused kernel, the depth
decoder the unfused reference. The bf16 difference between the two kernels is 1.6e-2, the
resolution of the format.

### Pooling

On device at the 128 frame training window the dense product and the segment mean are
equivalent, because the dense matrix is shared across the batch in this benchmark and the shape
is small; the segment form is 3.3x faster at a 512 frame window. The decisive figure is on the
host: assembling the operator for a batch of 64 windows costs 152.9 ms as dense matrices and
8.5 ms as index vectors, 3.6 million elements against 28 thousand, which is what the dataloader
pays per batch. The float32 difference of 1.5e-3 between the two device paths is the tf32
rounding of the dense matmul; the segment path accumulates in float32.

### Encoder

Forward (teacher forced) and forward plus backward time and peak memory for v1 and v4 at batch
8, 16 and 32 over 128 frame windows, with and without gradient checkpointing. Checkpointing
costs about 14 percent of step time and returns about 23 percent of peak memory at batch 32 on
v4, which is what leaves room for that batch on a 24 GB card.

### Depth decoder

Teacher forced decoding is one pass over the depth sequence; free running is seven sequential
passes with greedy feedback. At batch 16 x 128 frames the two cost 7.4 ms and 30.6 ms, so free
running decoding is the largest single cost of inference after the DAV encoder.

### Training

One optimizer step of configs/v4_169m_single_gpu.json (batch 16, accumulation 4, the KL term
included with synthetic top 50 teacher tensors) takes 396 ms and peaks at 5.3 GiB. The published
17,660 steps are therefore about 1.9 hours of compute on one RTX 3090, excluding data loading
and validation.

### Audio

End to end encoding of stereo 44.1 kHz audio through the released weights. The DAV encoder is
four fifths of the cost; its segmented evaluation (30 s segments, 1 s margins) holds peak memory
at about 4.4 GiB for any track length, so a 6 minute reference encodes in 5.6 s, 64x real time,
and a 30 s reference in 0.4 s, 75x real time.

## Reproducibility notes

Random inputs are seeded (seed 0) and the model suites use randomly initialised weights, so
every number except the audio suite is independent of the Hub. Numbers move by a few percent
between runs on the same card (thermal state, other processes); ratios between variants of the
same case are stable. When the numbers are regenerated, the README benchmarks section and the
paper tables are regenerated from the JSON with paper/tables.py, so no figure is typed by hand.
