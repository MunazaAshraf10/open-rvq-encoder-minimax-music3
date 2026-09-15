# Evaluation

Two protocols. Token agreement is computed by this repository on the cached holdout and is
reproduced below with the released weights. Condition replay cosine, the downstream metric that
decides which encoder is useful, needs the official MiniMax Music 3 language model and is a
published result whose provenance is recorded at the end of this page. Every table on this page
is written by paper/tables.py from the JSON under results/; nothing is typed by hand.

## Protocol 1: exact token agreement

Holdout split of the reverse distillation corpus, exact alignment records only: 130 tracks,
2,768 deterministic 128 frame windows with stride 128. For every window the encoder predicts
eight code distributions per frame; top 1 and top 5 count the frames whose sampled code is the
argmax or among the five largest logits, per codebook and pooled over the seven acoustic books.
Loss is the training objective at KL weight 0.25 and temperature 1. Models with a depth decoder
are scored free running (greedy feedback, the inference condition) and teacher forced (true
earlier codes, the training condition). Top k agreement measures inclusion of one sampled token
tuple; it does not measure equivalence of code tuples downstream, which is why protocol 2 exists.

## Protocol 2: condition replay cosine

For each of the 130 tracks the encoder's free running argmax codes are teacher forced through
the official language model and RVQ depth decoder, the hidden states pass through the official
condition encoder with the recorded chunk stitching, and the reconstructed condition embedding
is compared with the stored one, mean cosine over stitched condition frames, then mean over
tracks. True sampled codes replayed through the same path give 0.9999 and are the control. The
test stops before the diffusion transformer and the DAV decoder: it is not a waveform, STFT,
lyric identity or listening score.

## Published results

### Condition replay cosine, 130 holdout tracks

<!-- table: replay -->
| Model | Parameters | Mean | Std | 5th pct | 95th pct | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Serveurperso v1 (community) | 40,978,944 | 0.6633 | 0.0222 | 0.6281 | 0.6963 | 0.6026 | 0.7290 |
| v1, 41M | 40,978,944 | 0.7624 | 0.0195 | 0.7345 | 0.7904 | 0.7123 | 0.8286 |
| v2, 155M | 154,736,064 | 0.7698 | 0.0191 | 0.7430 | 0.7986 | 0.7206 | 0.8361 |
| v3, 155M + MERT | 154,736,064 | 0.7703 | 0.0193 | 0.7416 | 0.8005 | 0.7237 | 0.8356 |
| v4, 169M + depth | 169,008,576 | 0.8748 | 0.0157 | 0.8453 | 0.8938 | 0.8091 | 0.8977 |
| true codes (control) |  | 0.9999 | 0.0002 | 0.9998 | 1.0000 | 0.9984 | 1.0000 |
<!-- end table -->

v4 adds 0.1045 over v3, and its 5th percentile track exceeds the best v3 track. Width (v2)
added 0.0074 over v1 and MERT alignment (v3) 0.0004 over v2. The community 41M encoder by
Serveurperso, trained independently, is the external baseline.

### Matched comparison at step 17,500

<!-- table: matched -->
| Model | Loss | CE | KL | Sem. top 1 | Sem. top 5 | Ac. top 1 | Ac. top 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1, 41M | 5.3379 | 4.6962 | 2.5664 | 41.03 | 78.38 | 7.17 | 20.94 |
| v2, 155M | 5.2646 | 4.6333 | 2.5252 | 42.86 | 80.17 | 7.62 | 21.98 |
| v3, 155M + MERT | 5.2602 | 4.6294 | 2.5232 | 43.03 | 80.48 | 7.65 | 22.02 |
| v4, 169M + depth | 3.9882 | 3.5152 | 1.8921 | 43.16 | 80.51 | 7.30 | 19.88 |
<!-- end table -->

The v4 loss is conditional on the true earlier codes and is not comparable with the independent
head loss of v1 to v3. Free running v4 loses 0.35 points of acoustic top 1 against v3 while
gaining 0.10 of replay cosine: code tuple compatibility, not exact token agreement, is what the
downstream model rewards.

### Free running against teacher forced, v4

<!-- table: forcing -->
| Metric | Free running | Teacher forced |
|---|---:|---:|
| semantic top 1 | 43.17 | 43.17 |
| semantic top 5 | 80.51 | 80.51 |
| acoustic top 1 | 7.29 | 18.42 |
| acoustic top 5 | 19.88 | 44.55 |
<!-- end table -->

The decoder learned strong conditional distributions (18.4 percent teacher forced acoustic
top 1 against 7.7 percent for the best independent head model), and greedy feedback of its own
earlier predictions costs most of that gain in exact agreement. No beam search, scheduled
sampling or differentiable depth sampling was tried.

### Accuracy by codebook depth at step 17,500

<!-- table: depth -->
| Codebook | v1 | v2 | v3 | v4 free running | v4 teacher forced |
|---|---:|---:|---:|---:|---:|
| 0 | 41.03 | 42.86 | 43.03 | 43.16 | 43.16 |
| 1 | 11.27 | 11.96 | 12.04 | 14.72 | 27.04 |
| 2 | 11.92 | 12.56 | 12.62 | 13.02 | 24.49 |
| 3 | 9.34 | 10.00 | 9.98 | 9.46 | 19.32 |
| 4 | 7.54 | 8.07 | 8.17 | 6.56 | 16.66 |
| 5 | 4.45 | 4.71 | 4.74 | 3.41 | 15.46 |
| 6 | 3.08 | 3.29 | 3.26 | 2.16 | 14.10 |
| 7 | 2.58 | 2.77 | 2.77 | 1.76 | 11.88 |
<!-- end table -->

Every independent head model shows a monotone decline with codebook depth: later residual
books depend on earlier ones and an independent head cannot see which code was chosen. The
teacher forced v4 column removes most of the decline; the free running column shows how much
of it returns under greedy feedback.

### Packaged checkpoints

<!-- table: final -->
| Model | Checkpoint | Step | Loss | Sem. top 1 | Sem. top 5 | Ac. top 1 | Ac. top 5 |
|---|---|---:|---:|---:|---:|---:|---:|
| v1, 41M | checkpoint-17500 | 17,500 | 5.3379 | 41.03 | 78.38 | 7.17 | 20.94 |
| v2, 155M | final | 17,660 | 5.2644 | 42.86 | 80.18 | 7.62 | 21.98 |
| v3, 155M + MERT | final | 17,660 | 5.2599 | 43.03 | 80.49 | 7.66 | 22.03 |
| v4, 169M + depth | final | 17,660 | 3.9874 | 43.17 | 80.51 | 7.29 | 19.88 |
<!-- end table -->

v1 is packaged at step 17,500, the best checkpoint of its run by loss and top 1; its replay
cosine above was measured on its final checkpoint (step 17,640). v2 to v4 are packaged and
replayed at final.

### Independent evidence

Serveurperso's encoder, a separate implementation and training stack on a 550 track corpus,
reported held out STFT similarity 0.83 to 0.87, exact code replay STFT similarity 0.998, and
acoustic exact token match of 3 to 6 percent, with music and lyrics still identifiable after
predicted code replay, and overfitting by epoch 17. Those numbers are for that encoder, not for
any checkpoint here; they established that exact token agreement is not required for useful
replay, which the replay cosine table confirms at scale.

## Reproduced with this implementation

The commands, on one RTX 3090 (docs/benchmarks.md lists the machine):

    uv run rvq-ae cache --split holdout --revision 5029b1e7f1bbfbf028b76b38564fecccda94a111 --device cuda
    uv run rvq-ae evaluate --split holdout --variant v1 --precision bf16 --out results/reproduced/v1
    uv run rvq-ae evaluate --split holdout --variant v2 --precision bf16 --out results/reproduced/v2
    uv run rvq-ae evaluate --split holdout --variant v3 --precision bf16 --out results/reproduced/v3
    uv run rvq-ae evaluate --split holdout --variant v4 --precision bf16 --out results/reproduced/v4
    uv run rvq-ae evaluate --split holdout --variant v4 --precision fp32 --out results/reproduced/v4_fp32

The dataset loader yields 135 holdout records, 130 of them exact, and the window dataset yields
2,768 windows, the published count. The published evaluation ran under bfloat16 autocast on four
ranks; the bf16 rows below are the like for like comparison and the fp32 row shows the residual
effect of autocast.

<!-- table: reproduced -->
| Model | Source | Loss | Sem. top 1 | Sem. top 5 | Ac. top 1 | Ac. top 5 | TF ac. top 1 |
|---|---|---:|---:|---:|---:|---:|---:|
| v1, 41M | published, bf16, 4 ranks | 5.3379 | 41.03 | 78.38 | 7.17 | 20.94 |  |
| v1, 41M | this repository, bf16, 1 GPU | 5.3353 | 41.08 | 78.44 | 7.18 | 20.97 |  |
| v2, 155M | published, bf16, 4 ranks | 5.2644 | 42.86 | 80.18 | 7.62 | 21.98 |  |
| v2, 155M | this repository, bf16, 1 GPU | 5.2644 | 42.86 | 80.18 | 7.62 | 21.98 |  |
| v3, 155M + MERT | published, bf16, 4 ranks | 5.2599 | 43.03 | 80.49 | 7.66 | 22.03 |  |
| v3, 155M + MERT | this repository, bf16, 1 GPU | 5.2574 | 43.07 | 80.55 | 7.67 | 22.06 |  |
| v4, 169M + depth | published, bf16, 4 ranks | 3.9874 | 43.17 | 80.51 | 7.29 | 19.88 | 18.42 |
| v4, 169M + depth | this repository, bf16, 1 GPU | 3.9860 | 43.24 | 80.56 | 7.30 | 19.92 | 18.44 |
| v4, 169M + depth | this repository, fp32, 1 GPU | 3.9858 | 43.26 | 80.60 | 7.30 | 19.93 | 18.44 |
<!-- end table -->

## Provenance

<!-- table: provenance -->
| Version | Source repository | Revision | Checkpoint | Weights sha256 |
|---|---|---|---|---|
| v1 | SimpleTuner/open-rvq-encoder-minimax-music3-41m-v1 | e9fa3bba9b5b | checkpoint-17500 | 9683ba0a719fc8f4 |
| v2 | SimpleTuner/open-rvq-encoder-minimax-music3-155m-v2 | f317d72466b9 | final | 47dfffb7a76d9558 |
| v3 | SimpleTuner/open-rvq-encoder-minimax-music3-155m-v3 | 7c525705817d | final | 356e97fea65c486a |
| v4 | SimpleTuner/open-rvq-encoder-minimax-music3-169m-v4 | b9a9165b99d1 | final | e8fa93a7db2e5a09 |
| dataset | bghira/minimax-music3-rvq-reverse-distillation | 5029b1e7f1bb | holdout |  |
<!-- end table -->

Replay evaluation used the official components from MiniMaxAI/MiniMax-Music3 at revision
fbdf52fbaaca799592917417eb05f1899f1255ec under the MiniMax Music 3 Community License. The
Serveurperso checkpoint is best.pt from ServeurpersoCom/minimaxmusic.cpp at commit d19efe9f
(code MIT, checkpoint license not separately declared). Source files: results/published/
experiment-summary.json, condition-replay-aggregate.json, provenance.json, and per variant
evaluation-metrics.json and comparison-metrics.json copied from the four Hub model cards.
