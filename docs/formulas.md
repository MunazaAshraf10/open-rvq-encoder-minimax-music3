# Formulas

Every quantity the implementation computes, with the file and function that computes it. Symbols:
n frames of a track, t a frame index, l a latent index, d the encoder width, K = 8 codebooks,
V_k the size of codebook k (16,384 for k = 0, 1,024 otherwise), B a batch, T a window of frames, L
the latents a window covers.

## 1. Signal geometry (constants.py)

| Quantity | Value | Derivation |
|---|---:|---|
| sample rate | 44,100 Hz | DAV input |
| DAV hop | 512 samples | product of the encoder strides 2, 4, 8, 8 |
| latent rate | 86.1328125 Hz | 44,100 / 512 |
| latent channels | 128 | 64 posterior mean channels per stereo side, left first |
| frame rate | 25 Hz | one RVQ frame every 40 ms |
| latents per frame | 441 / 128 = 3.4453125 | (44,100 / 512) / 25 |
| rollout chunk | 200 frames, hop 100 | language model window and stride |
| stitched hop | 345 latents | floor(100 x 441 / 128) |
| ownership offset | 25 frames | warm up of every chunk after the first |
| encoder window | 128 frames = 5.12 s | max_position_embeddings |
| longest track | 9,000 frames = 6 min | language model limit |

## 2. Timeline contract (alignment.py)

Frame t covers the latent span [s_t, s_(t+1)) of the stitched timeline. The n + 1 boundaries
s_0 <= s_1 <= ... <= s_n are the whole contract; every consumer derives from them.

### 2.1 Nominal reconstruction, nominal_bounds

Used when a track carries no chunk stitching table (every inference call, and the training
alignment the exact rule converges to). With W = max(1, (n - 1) // 100) chunks, frame f belongs
to chunk c = clamp((f - 25) // 100, 0, W - 1); a chunk of cf = min(200, n - 100 c) frames yields
cl = floor(cf x 441 / 128) latents, and with local frame lf = f - 100 c

    s_f = 345 c + ceil(lf x cl / cf).

Invariants asserted by the tests: s_100 = 345, s_200 = 690, s_4500 = 15,524, and every span
s_(t+1) - s_t is 3 or 4.

### 2.2 Exact reconstruction, stitched_bounds

Used when the dataset records the chunk stitching table (the 130 exact alignment holdout
records and all training records). Chunk 0 owns frames [f_0, f_1 + 25); chunk i > 0 owns
[f_i + 25, f_(i+1) + 25); the last chunk owns through its own end, clamped to n. Inside an owner
with owned frames and kept latents [a, b), frame g maps to

    s_g = a + ceil((g - owner_start) x kept / owned),

and a later chunk rewrites the boundary it shares with the previous one. Non monotonic tables
are rejected.

### 2.3 Frame centres, frame_centers

    centre_t = (s_t + s_(t+1)) / 2 x 512 / 44,100 seconds.

## 3. Pooling operator (alignment.py, model.py)

Definition, pool_matrix:

    P[t, l] = 1 / (s_(t+1) - s_t)   for l in [s_t, s_(t+1)),   0 otherwise,

so every row sums to one and the pooled feature of a frame is the mean of the latent features it
was rendered from. Applied as a dense product this costs O(B T L d).

Computed form, Pool.apply. Because every row of P is one contiguous constant block, P h is a
segment mean. With frame[l] the owning frame of latent l and span[t] = s_(t+1) - s_t,

    sum[b, frame[b, l], :] += h[b, l, :]   for every l,      pooled = sum[:, :T] / span[:, :, None],

which costs O(B L d). Right padded latents carry frame index T, a sink row that is discarded.
The tests assert the two forms agree to 4e-16 in float64.

## 4. Encoder (model.py, layers.py)

Input: DAV posterior means h_0 in R^(B x L x 128) at the latent rate.

Stem, conv_in: Conv1d(128 -> d, kernel 7, padding 3).

Residual block, ResBlock, three of them with dilations 1, 3, 9:

    x + conv2(gelu(conv1(gelu(GroupNorm_1(x)))))

with conv1 a 3 tap dilated convolution (padding equal to the dilation) and conv2 a 1 tap
convolution. GroupNorm with one group normalises over channels and time jointly, so zero padded
latents take part in the statistics exactly as they did in training.

Pooling and positions: z = P h + E_pos[:T], with E_pos in R^(128 x d) learned, initialised
N(0, 0.02).

Transformer layer, Layer, eight of them, pre norm, bidirectional:

    x = x + Drop(Attn(LN_1(x))),      x = x + Drop(W_2 Drop(gelu(W_1 LN_2(x))))

with separate q, k, v, out projections, feed forward width 4d, dropout 0.1 in training.

Attention, attention: with q, k, v in R^(B x H x T x d_h), d_h = 64,

    Attn = softmax(q k^T x scale) v,      scale = 8 / d_h under muP,   d_h^(-1/2) otherwise.

At d_h = 64 both equal 1/8. The temporal stack calls the fused kernel; the depth decoder calls
attention_reference, the unfused form with the softmax in float32, because at sequence length 8
the fused kernel is slower (docs/benchmarks.md).

Output: features y = LN_out(x) in R^(B x T x d); semantic logits heads[0](y) in R^(B x T x 16,384).

Without a depth decoder (v1, v2, v3) the seven acoustic heads read y independently:
logits_k = heads[k](y).

## 5. Depth decoder (model.py, DepthDecoder), v4

Runs once per frame, so its batch is B x T. Sequence for one frame, with e_k the prior
embedding of codebook k and c_k its code:

    [W_ctx y_t,  e_0(c_0),  e_1(c_1),  ...,  e_6(c_6)] + E_depth,

E_depth in R^(8 x 512) learned. Two causal pre norm layers (width 512, 8 heads, feed forward
2,048, standard 1 / sqrt(d_h) scale) and a LayerNorm. Head j reads position j + 1 and predicts
acoustic codebook j + 1, so codebook k sees the frame context, the semantic code and codebooks
below k, never above.

Training teacher forces the sampled codes in one pass. Inference feeds back greedy predictions,
seven sequential passes:

    c_k = argmax head_(k-1)(decode([ctx, e_0(c_0), ..., e_(k-1)(c_(k-1))])[-1]).

The context projection W_ctx is a muP readout (1,088 -> 512, no bias); IGNORE targets borrow
token 0 in the embedding and are excluded from the loss.

## 6. Objective (losses.py, rvq_loss)

With hard targets c_(t,k) and, per frame and codebook, the generator's stored top 50 ids i and
teacher logits z_T:

    L = (1 / K) sum_k CE_k + w_KL (1 / K) sum_k KL_k,      w_KL = 0.25,  tau = 1.

Cross entropy, codebook_ce: CE_k = mean over frames with c_(t,k) != IGNORE of
    minus log softmax(z_S)[c_(t,k)], with IGNORE = minus 100.

Top k distillation, topk_kl, following Hinton et al. (2015):

    p_T(i) = softmax(z_T / tau) over the valid stored ids,
    log p_S(i) = log softmax(z_S / tau) over the whole vocabulary, gathered at those ids,
    KL_k = tau^2 sum_(i valid) p_T(i) (log p_T(i) minus log p_S(i)),

averaged over frames that have at least one valid id and a target that is not IGNORE. An id is
valid when 0 <= i < V_k; the semantic end of track id 16,384 and any padding are therefore
dropped and the teacher mass renormalised over what remains. The student is never renormalised
over the top 50 subset. A batch with no valid frame contributes 0 x sum(z_S), which keeps the
graph connected.

For v4 the acoustic logits inside CE_k and KL_k are teacher forced on the true earlier codes,
so the v4 loss is conditional and not comparable with the v1 to v3 loss.

## 7. MERT alignment (v3 only, training only, published; not in this codebase)

An auxiliary term used in the v3 run and annealed to zero before the end:

    L_cos = 1 minus mean_(b,t) cos(W_proj y^(4)_(b,t), m^(9)_(b,t)),

with y^(4) the student hidden state after transformer layer 4, W_proj a 1,088 -> 768 muP
readout without bias, and m^(9) the frozen MERT v1 95M layer 9 feature (24 kHz, 75 Hz),
interpolated linearly onto the frame centres of section 2.3. Weight schedule with progress
p = step / total: w_0 = 0.5 for p <= 0.7, linear to 0 at p = 0.9, zero after. The projection is
not exported, so the v3 weights have the v2 architecture. Reported holdout MERT cosine at the
end of the run: 0.762.

## 8. Maximal update parametrisation (mup.py, config.py)

Width multiplier m = d / 128 (v1: 4, v2 to v4: 8.5), from Yang et al., Tensor Programs V
(2022). Three rules:

1. Readout, Readout.forward: y = output_mult x W (x / m) + b, for every map from the width to a
   fixed size (the semantic readout, the v1 to v3 acoustic readouts, the depth context
   projection). The division lives in the forward, so the state dict matches nn.Linear.
2. Attention scale, EncoderConfig.attention_scale: 8 / d_h instead of d_h^(-1/2).
3. Optimizer groups, param_groups: weights whose input and output dimensions both grow with d
   (q, k, v, out, linear1, linear2 of the temporal stack, conv1 and conv2 of the residual
   blocks) take learning rate lr / m and weight decay wd x m; every other parameter keeps lr and
   wd; biases, norms and embeddings take no decay.

Initialisation, rescale_init, applied once to a fresh model: biases with a width sized fan in
and non zero readout weights and biases are multiplied by sqrt(m). The semantic readout is zero
initialised (mup_readout_zero_init), so the initial semantic distribution is uniform. Learned
positions are N(0, 0.02). The base width is part of the config, so no shape files are needed.

## 9. Learning rate schedules (schedule.py)

Linear warm up and decay (v2, v3, v4), lr_lambda, with floor f = lr_end / lr:

    m(g) = g / warmup for g < warmup,      m(g) = (1 minus f)(1 minus t) + f,   t = (g minus warmup) / (total minus warmup),

then f. Published values: lr 3e-4, lr_end 1e-7, warmup 500, total 17,660.

Cosine (v1), cosine_lambda, half period P = 500, no warm up:

    m(g) = f + (1 minus f) (1 + cos(pi g / P)) / 2,

so m(0) = 1 and the minima fall at 500, 1,500, ..., 17,500, which is the published v1 trace.

## 10. Metrics (losses.py, evaluate.py)

Top k hit, topk_hits: a frame counts as a hit for codebook k when its target is among the k
largest logits; accuracy is hits over frames whose target is not IGNORE. Reported as semantic
(codebook 0), acoustic (codebooks 1 to 7 pooled) and per codebook, for k = 1 and 5. Models
with a depth decoder are scored twice: teacher forced (true earlier codes) and free running
(greedy earlier codes).

Condition replay cosine (published, computed outside this codebase): for one track, predicted
codes are teacher forced through the official language model and RVQ depth decoder, the hidden
states pass through the official condition encoder with the recorded chunk stitching, and the
result is compared with the stored condition embedding frame by frame:

    replay(track) = mean_t cos(c_hat_t, c_t),

reported as the mean over 130 exact alignment holdout tracks. True sampled codes score 0.9999
through the same path and serve as the control.

## 11. Parameter counts (model.py)

Per temporal layer: 4(d^2 + d) projections + 4d for two LayerNorms + (4d^2 + 4d) + (4d^2 + d)
for the feed forward = 12 d^2 + 13 d. Stem: 128 x 7 x d + d = 897 d. Each residual block:
2d + (3d^2 + d) + (d^2 + d) = 4 d^2 + 4 d. Positions: 128 d. Output norm: 2 d. A readout of size
V: d V + V.

| Model | d | Stem | Blocks | Positions | Layers | Norm | Readouts | Depth decoder | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 512 | 459,264 | 3,151,872 | 65,536 | 25,219,072 | 1,024 | 12,082,176 | 0 | 40,978,944 |
| v2, v3 | 1,088 | 975,936 | 14,217,984 | 139,264 | 113,752,576 | 2,176 | 25,648,128 | 0 | 154,736,064 |
| v4 | 1,088 | 975,936 | 14,217,984 | 139,264 | 113,752,576 | 2,176 | 17,842,176 | 22,078,464 | 169,008,576 |

Depth decoder of v4: context projection 1,088 x 512 = 557,056; prior embeddings 16,384 x 512 +
6 x 1,024 x 512 = 11,534,336; depth positions 8 x 512 = 4,096; two layers 2 x (12 x 512^2 +
13 x 512) = 6,304,768; norm 1,024; seven heads 7 x (512 x 1,024 + 1,024) = 3,677,184.
tests/test_model.py asserts these totals on the meta device against the counts recorded with the
released weights.
