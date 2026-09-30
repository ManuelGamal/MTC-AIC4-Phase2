# UAV Single-Object Tracking — AIC-4

**5th place at the AIC-4 international final**, out of 200+ teams, after three
qualifying phases. Team NewbieSquad; team lead Manuel Gamal.

A frozen, off-the-shelf transformer tracker (UETrack, ViT-Base) adapted to UAV
footage **entirely at inference time**: no new trainable parameters, no
fine-tuning on competition data.

| Result | Value |
|---|---|
| Public leaderboard score | **0.7822** |
| Normalized precision, 67 private sequences | **0.97** |
| Model size | ~22M parameters (budget: 50M) |
| Speed | ~40 FPS on a single CUDA GPU; TensorRT FP16 path for deployment |

System description: [`paper/system_description.pdf`](paper/system_description.pdf).

## Why inference-time only

UAV tracking stresses what ground-level benchmarks rarely do: tiny targets,
abrupt camera ego-motion, occlusions, and fast altitude-driven scale change.
Trackers tuned on LaSOT or TrackingNet carry priors that hurt here: a
multiplicative Hanning window suppresses targets pushed toward the edge of the
search region, and heavy box smoothing lags real scale changes. With a short
competition window and no budget to retrain, we changed the priors instead of
the weights.

## What we changed

1. **Channel-fused modality adaptation.** The released checkpoint is a
   vision-language tracker with a 6-channel patch embedding and a CLIP branch.
   We disabled the language branch (removing a CLIP image and text encoder from
   the active model) and folded the 6-channel projection into 3 RGB channels by
   weight summation. After reconciling the positional embedding for two
   templates, `load_state_dict` reports zero missing parameters.
2. **Dual-template memory.** Slot 0 keeps the first-frame crop and never
   changes, anchoring against drift; slot 1 refreshes every 25 frames from
   recent high-confidence crops, and never while the target is flagged as lost.
3. **Re-acquisition.** A confidence-gated absent-state machine detects loss.
   While lost, the search region follows a constant-velocity prediction of the
   target and expands up to 6.0x, covering 1.78x more area for the same number
   of tokens.
4. **Additive center prior.** `R = (1 − λ)·S + λ·H` replaces the multiplicative
   Hanning window, so strong off-center responses keep their full amplitude.
5. **Asymmetric smoothing.** Light smoothing on box size (α = 0.25) follows
   scale changes; heavy smoothing on velocity (α = 0.8) suppresses jitter.

The full configuration is in Table 1 of the system description.

## Run it

Model weights are too large for GitHub. `download.py` fetches them into
`checkpoints/` (the Docker build runs it automatically).

**Docker (the final-round submission):**

```bash
docker buildx build --platform linux/amd64 -f Dockerfile -t newbiesquad_phase3:latest --load .
docker run --rm --gpus all newbiesquad_phase3:latest \
  bash run_inference.sh test.json public_lb predictions.csv
```

`run_inference.sh` builds and uses a TensorRT FP16 engine when it can, and falls
back to PyTorch FP16 otherwise. Build details, verification commands and the
runtime order are in [`PHASE3_DOCKER_SUBMISSION.md`](PHASE3_DOCKER_SUBMISSION.md).

**Without Docker:**

```bash
pip install -r requirements.txt
python download.py
python inference.py test.json public_lb predictions.csv
python check_submission.py sample_submission.csv predictions.csv
```

Output is one box per frame, `id,x,y,w,h`.

**Tests:**

```bash
python -m unittest discover -s tests -p "test*.py" -v
```

## Layout

| Path | What it is |
|---|---|
| `predictor.py` | the tracker: checkpoint surgery, memory, re-acquisition, post-processing |
| `inference.py`, `run_inference.sh` | competition entry points |
| `UETrack/` | upstream UETrack code the checkpoint loads into |
| `tools/` | TensorRT export, latency benchmarks, submission checks |
| `tests/` | unit and submission-readiness tests |
| `paper/` | system description |

## Acknowledgements

Built on UETrack (Kang et al.,
[*UETrack: A Unified and Efficient Framework for Single Object Tracking*](https://arxiv.org/abs/2603.01412),
CVPR 2026) and its released checkpoint. MIT licensed; see [`LICENSE`](LICENSE).
