# Experiment history and evidence audit

Audit baseline: `d265c87` on 24 October 2026. Sources: repository commit history,
tracked reports, locally retained JSON/NetCDF artifacts, and the accessible local
project chats from 22–24 October. Chats supply intent and context; executable
artifacts and measured reports support results. No private chat transcripts or
credentials are copied into the repository. This is the history accessible on
this machine, not an assertion that every project conversation was available.

## What we tried

| Commits | Question and method | Finding and status |
|---|---|---|
| `d07a764`, `4d96209` | Can a reproducible Aurora/KNMI environment be installed? Pixi, native NetCDF dependencies, pinned Aurora | Installed and smoke-tested; dependency linkage fixed. Software foundation, not forecast evaluation. |
| `2c7dd9e` | Can regional Aurora 1.5 run on MPS with ERA5/ARCO inputs? | 48×64 eight-hour hindcast executed. One early case had temperature MAE 0.982°C vs persistence 0.527°C, rain MAE 0.292 mm vs dry 0.286 mm. Insufficient sample for skill claims. |
| `77ae2e4` | Can fresh IFS states drive a one-line NL forecast? | Live path, sea-ice supplement, locality lookup and caching implemented. Atmospheric source/grid differences remain hypotheses for poor local accuracy. |
| `d58a6d4` | Can measured radar condition local rainfall? | Small external CNN and native-grid preparation/training added. Real ingestion and backward smoke checks passed; no representative storm-trained checkpoint was established. The earlier six-hour adapter was superseded. |
| `f6bcf5c` | Can 72h stations, calendar context and 2h radar refine eight-hour forecasts? | Ridge correction, radar-only control and eight-hour CNN evaluation added. Tiny January/April/July station pilot: pressure improved, temperature/wind mixed. Existing surface-head approach removed. |
| `bf86d31` | Were rainfall comparisons using equal spatial support? | Corrected aggregation and paired finite masks. Earlier scores must retain their original support definition and must not be ranked against corrected scores. |
| `4c449cf` | Why does Leusden disagree with Buienradar? Audit units/timestamps, replay, larger crop, latest-observation station features | Exact replay reproduced values. No conversion defect established. A cache freshness defect was fixed but did not explain that case. Station v2 helped one development comparison but did not establish generalization. |
| `082282d` | Does still wider regional context help, and fit the laptop? | 128×192 float32 execution passed. Temperature MAE vs reference 1.179°C vs 1.108°C for 64×96 and 1.573°C for 48×64. Rain totals remained far below reference. No domain promoted as generally best. |
| `b11e231` | Can deterministic radar motion help without a checkpoint? | Persistence/FFT/pySTEPS/blend evaluation. First-hour improvement in one measured event, worsening later. Historical schema mismatch blocked complete independent splits; no blend qualified. |
| `e626dca` | Is internal adaptation feasible at all? | One rank-eight LoRA update passed on MPS, 2,850,816 trainable parameters; updated weights discarded. Constructor/tensor integration failures preceded the successful run. No skill evidence. |
| `86f1591`, `d265c87` | Can execution be cheaper and method provenance clearer? | Sparse/dense float32 equality passed on tested windows; explicit rainfall modes and diagnostics added. Reused engine ~2.12 s vs cold ~38.86 s on one cached case. These exclude fresh downloads. |

## Results that must not be overinterpreted

- In the Leusden reference window Aurora leads were **8–15**, not model leads
  1–8: the initialization was already old. The requested horizon and model age
  answer different questions.
- Buienradar's pasted issue time and accumulation convention were unknown. Its
  7.6 mm total is a prediction reference, not observed rainfall ground truth.
- Wider crops improved temperature agreement but predicted only 0.025/0.033 mm
  rain versus 0.228 mm on the original crop. Larger context did not fix rainfall.
- Station v1/v2 reused extremely small splits; subsequently inspected July results
  are development evidence. Newly acquired past observations cannot prove their
  availability at the original issuance.
- In the measured radar-motion case, motion MAE was 0.197 mm vs paired Aurora
  0.378 mm at hour one, but 0.810 vs 0.473 mm at hour eight. Motion coverage fell
  from 0.831 to 0.542. Full-domain and motion-support scores are not interchangeable.
- Historical radar filenames were available, but January downloads used
  `ACCUMULATION_[MM]` and `:03` boundaries, incompatible with the supported
  `PRECIP_[MM]` exact-five-minute decoder. About 146.6 MB total new pilot payload
  was retained; the archive was not silently shifted or reinterpreted.
- An initial float16 test used upstream CPU autocast and was invalid as an MPS
  precision comparison. Corrected explicit MPS autocast was numerically close
  but slower (~0.39–0.44 vs ~0.24–0.27 s/hour); float32 stayed the default.
- Peak process RSS and MPS driver allocations overlap; adding them does not
  measure total unified-memory use. Passing pytest does not prove weather skill.

## Evidence and reproduction

- [Leusden report](../experiments/leusden-20261024/report.md) contains the full
  hourly predictions, grids, strict/retrospective station distinction and commands.
- [Radar/adaptation report](../experiments/radar-methods-20261024/report.md)
  contains paired rain scores, acquisition failures and LoRA feasibility evidence.
- [Earlier technical design](design.md) preserves ingestion details, timings and
  station pilot measurements. Its historical conclusions are not new qualification.
- Local originals live in ignored `outputs/`, `data/`, and `checkpoints/`.
  `pixi run experiment-collect --root . --output outputs/evidence/new-audit`
  inventories their sizes, schemas and SHA-256 hashes. Only explicit
  `compare-reference` results become a development comparison CSV; other legacy
  JSON remains inventoried in its original schema. Missing originals are visible
  limitations. Reproduction may require reacquisition and provider access.
