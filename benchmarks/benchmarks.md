# Benchmarks

| python: 3.13.15 | torch: 2.14.0+cu130 | device: cuda | gpu: NVIDIA GeForce RTX 3090 | compute_capability: 8.6 | memory_gib: 23.6 | cuda: 13.0 | driver: 580.173.02 |

## attention

| case | variant | ms | peak MiB | note |
|---|---|---|---|---|
| temporal (b16, 17h, len 128) forward | reference | 0.238 | 73.9 |  |
| temporal (b16, 17h, len 128) backward | reference | 0.880 | 118.0 |  |
| temporal (b16, 17h, len 128) forward | fused sdpa | 0.066 | 54.6 | max abs difference 1.6e-02 |
| temporal (b16, 17h, len 128) backward | fused sdpa | 0.309 | 71.8 | max abs difference 1.6e-02 |
| depth (b2048, 8h, len 8) forward | reference | 0.388 | 116.3 |  |
| depth (b2048, 8h, len 8) backward | reference | 0.995 | 176.3 |  |
| depth (b2048, 8h, len 8) forward | fused sdpa | 1.132 | 160.8 | max abs difference 1.6e-02 |
| depth (b2048, 8h, len 8) backward | fused sdpa | 3.791 | 713.3 | max abs difference 1.6e-02 |

## pooling

| case | variant | ms | peak MiB | note |
|---|---|---|---|---|
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

## encoder

| case | variant | ms | peak MiB | frames/s |
|---|---|---|---|---|
| v1 41M batch 8 | forward, teacher forced | 5.107 | 306.2 | 200,525.1 |
| v1 41M batch 8 | forward and backward | 22.466 | 856.9 | 45,580.9 |
| v1 41M batch 8 | forward and backward, checkpointed | 41.229 | 689.8 | 24,837.0 |
| v1 41M batch 16 | forward, teacher forced | 7.805 | 522.5 | 262,382.7 |
| v1 41M batch 16 | forward and backward | 27.725 | 1302.8 | 73,867.5 |
| v1 41M batch 16 | forward and backward, checkpointed | 41.530 | 966.5 | 49,313.3 |
| v1 41M batch 32 | forward, teacher forced | 13.159 | 625.9 | 311,259.8 |
| v1 41M batch 32 | forward and backward | 44.045 | 2344.1 | 92,995.1 |
| v1 41M batch 32 | forward and backward, checkpointed | 50.689 | 1621.3 | 80,806.6 |
| v4 169M batch 8 | forward, teacher forced | 15.474 | 1147.9 | 66,174.4 |
| v4 169M batch 8 | forward and backward | 48.365 | 2781.9 | 21,172.3 |
| v4 169M batch 8 | forward and backward, checkpointed | 60.785 | 2405.8 | 16,846.4 |
| v4 169M batch 16 | forward, teacher forced | 27.005 | 1985.4 | 75,838.4 |
| v4 169M batch 16 | forward and backward | 84.472 | 3928.4 | 24,244.8 |
| v4 169M batch 16 | forward and backward, checkpointed | 96.110 | 3185.4 | 21,309.0 |
| v4 169M batch 32 | forward, teacher forced | 49.636 | 2344.6 | 82,520.2 |
| v4 169M batch 32 | forward and backward | 156.053 | 6172.5 | 26,247.5 |
| v4 169M batch 32 | forward and backward, checkpointed | 177.390 | 4725.8 | 23,090.4 |

## depth decoder

| case | variant | ms | peak MiB | frames/s |
|---|---|---|---|---|
| v4 169M batch 16 x 128 frames | teacher forced | 7.413 | 977.0 | 276,262.2 |
| v4 169M batch 16 x 128 frames | free running | 30.582 | 1037.0 | 66,966.9 |

## training

| case | variant | ms | peak MiB | frames/s | note |
|---|---|---|---|---|---|
| v4 169M single GPU (64 windows per step) | batch 16 x accumulation 4 | 395.545 | 5467.0 | 20,710.7 | 1.9 h for the published 17,660 steps |

## audio

| case | variant | ms | peak MiB | x real time |
|---|---|---|---|---|
| 30 s stereo track | dav encoder | 344.614 | 4005.8 | 87.1 |
| 30 s stereo track | rvq encoder | 100.004 | 818.2 | 300.0 |
| 30 s stereo track | end to end | 398.878 | 4005.8 | 75.2 |
| 180 s stereo track | dav encoder | 2214.803 | 4332.6 | 81.3 |
| 180 s stereo track | rvq encoder | 652.836 | 824.2 | 275.7 |
| 180 s stereo track | end to end | 2803.223 | 4332.6 | 64.2 |
| 360 s stereo track | dav encoder | 4448.245 | 4475.7 | 80.9 |
| 360 s stereo track | rvq encoder | 1239.706 | 832.2 | 290.4 |
| 360 s stereo track | end to end | 5601.924 | 4475.7 | 64.3 |
