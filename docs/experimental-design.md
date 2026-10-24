# Experimental design: local Aurora with Dutch observations

Protocol v1, 24 October 2026. Primary outcomes are hourly temperature and rainfall
for the next eight hours in the European Netherlands. The laptop is the execution
constraint, not an assumed source of forecasting error. The study asks where
methods help or fail, with evidence sufficient to distinguish data limitations,
software defects, representativeness and model limitations.

## Hypotheses and experimental matrix

| Question | Controlled comparison | Outcomes/context |
|---|---|---|
| Does regional context matter? | Frozen Aurora on existing 48×64, 64×96, 128×192 profiles; identical states, issue time and targets | Temperature/rain error, boundary distance, inference cost |
| Does initialization age matter? | Latest available pair vs earlier eligible pairs at identical issuance | Model lead vs issue lead, input availability, rain/temperature error |
| Does initialization source matter? | ERA5/ARCO vs operational IFS where both genuinely exist | Source sensitivity; retrospective findings remain distinct from operational skill |
| Can stations correct local bias? | Raw Aurora, persistence, ridge v1 historical evidence, ridge v2; remove latest value/residual/calendar features in staged ablations | Temperature, pressure, wind; station holdout, history completeness |
| Is longer station history useful? | 1/24/72h histories after baseline data coverage is audited | Cold starts, intraday gaps, runtime and incremental skill |
| Does radar help short leads? | Dry, persistence, FFT, Lucas–Kanade/pySTEPS, validation-selected blend | Leads 1–2 vs 3–8, wet events, motion support and growth/decay |
| Is more radar context useful? | 30/60/120min histories, keeping issue/target fixed | Rain regime, support and motion computation |
| Does learning add useful information? | Radar-only CNN vs Aurora-conditioned CNN; calendar ablation | Skill beyond motion and Aurora, fitting cost and held-out generalization |
| Is internal adaptation worthwhile? | Gated LoRA vs frozen Aurora and external corrections | Same untouched cases, finite updates, memory, time and rain skill |
| Can execution become cheaper? | Cold/warm/cache, sparse/dense, float32/explicit MPS float16 | Numerical equivalence and resource costs, separately from accuracy |

The common runner directly executes frozen Aurora, cached predictions, observation
persistence and dry rain. Existing station/CNN/radar workflows are explicit jobs,
with their own established feature/split/schema checks. Their full raw metrics
and spatial skill remain in job JSON. Common-table comparison requires normalized
per-method NetCDF predictions; raw job averages are not automatically mixed with
point scores. History-length ablations require explicitly prepared input manifests;
the runner does not silently alter model feature schemas. This keeps unsupported
comparisons blocked instead of manufacturing an experiment.

## Study population and splits

1. **Historical audit:** inventory previous evidence, identifying synthetic,
   retrospective, reference-agreement, feasibility and corrected-score artifacts.
   Previously inspected Leusden/July pilots are development cases permanently.
2. **Six-event real pilot:** two dry, two widespread-rain and two localized/intense
   events. Require all relevant inputs and eight complete target hours. The shipped
   six-event synthetic demo is a software test, not this real pilot.
3. **48-event seasonal screening:** select 12 independent events from each available
   meteorological season, spanning dry/ordinary/heavier rain. Sort all events by
   issuance; freeze the earliest 24 for training, next 12 validation, last 12 test.
   Whole 72h history + 8h target windows must have >30h separation. Missing events
   cannot be replaced after candidate errors are inspected.
4. **Prospective track:** capture inputs and public forecasts at a declared issuance,
   save acquisition/publication evidence, freeze predictions, then collect targets.
   No scheduler is installed automatically; invoke acquisition explicitly.

Case selection uses observations and metadata, not candidate forecast errors.
If the available archive cannot supply the seasonal counts under the event-window
constraints, publish the coverage gap and defer qualification. Chronological splits
can confound season and split; publish the season distribution and do not claim
all-season test skill. A complete annual study requires later data collection.

Use all eligible in-domain KNMI stations; retain De Bilt as the established whole
station holdout for learned station correction. Distinguish coastal and interior
stations, measured wet-event intensity, source, domain and initialization age.
The runner accepts a frozen case manifest; event curation and coastal/seasonal
stratification are explicit study preparation, not automatic meteorological labels.
Regime labels must be assigned before candidate errors are inspected.

No fitting or hyperparameter decisions use test observations. Existing station and
radar preparation enforce their own embargoes and training/validation selection.
A campaign TOML split is metadata for execution/scoring; it does not replace those
training safeguards. Train jobs must use the checked existing case manifests.

## Data roles, availability and benchmarks

- Aurora requires its matching pinned statics plus two 3D atmospheric states.
  Stations/radar supplement those inputs; they cannot replace upper-air context.
- ERA5/ARCO provides reproducible historical initialization but includes retrospective
  information. Label ARCO as a regridded mirror. Sea ice supplements IFS where the
  supported product lacks it; keep the existing freshness check.
- Operational stations and past measured RTCOR radar are candidate causal inputs.
  Later validated stations and later measured radar are verification targets.
  Gauge adjustment in radar limits independent station/radar rain verification.
- KNMI HARMONIE Cy43 P1 provides hourly temperature/rain predictions. Its public
  archive starts 8 January 2026; don't assume it covers older pilots.
  [Dataset](https://dataplatform.knmi.nl/dataset/harmonie-arome-cy43-p1-1-0),
  [fields and accumulation conventions](https://english.knmidata.nl/open-data/harmonie).
- KNMI's public pySTEPS nowcast covers +0–120 minutes in five-minute amounts.
  Exclude +0 analysis and score its first two hours only. Raw forecast HDF5 requires
  version-specific decoding; never send it through the measured RTCOR decoder.
  [Product documentation](https://dataplatform.knmi.nl/dataset/radar-forecast-2-0).
- IFS public forecasts are a comparator distinct from step-zero initialization.
  The bounded fetcher selects 2t/tp messages from explicit three-hour steps.
  Accumulated tp is differenced, keeping native three-hour intervals. Aggregate
  hourly local predictions and observations to those exact intervals, using actual
  native endpoints fully within the requested future window. Missing hourly bins
  invalidate their aggregate instead of joining nonconsecutive records. Temperature
  is compared only at available valid times. No hourly rainfall disaggregation.
  [ECMWF products and rolling archive](https://www.ecmwf.int/en/forecasts/datasets/open-data).
- A historical public forecast unavailable from the archive is a missing comparator,
  not a reason to substitute a present-day run or reanalysis. Archive prospective
  runs before expiry; record model version, cycle, acquisition and attribution.
- Buienradar pasted forecasts remain development agreement references because their
  issue time/rain interval are unverified. They are not observational truth.

Prospective cached forecasts require `available_utc <= issue`; a TOML label alone
is not evidence. Retrospective availability remains unknown where publication
latency cannot be reconstructed. Prospective Aurora requires operational IFS;
ERA5 initialization is rejected. Each selected prepared surface/pressure input
for both atmospheric states needs a checksummed sidecar with an aware
`available_utc <= issue`. New IFS acquisitions record when the complete prepared
file (including supplements) became available. Older files without that evidence
remain retrospective; timestamps are not backfilled. These manifests are retained
in execution provenance and checked again on reuse.
Measured history files must end by issuance.
The current runner additionally needs complete, normalized hourly observation
inputs. Existing operational acquisition/validation is reused rather than bypassed.

Microsoft cautions that input source and regridding affect performance, and its
model-specific guidance defines Aurora 1.5's fields, statics and hourly rollout.
These are hypotheses to audit before attributing failures to hardware.
[Input cautions](https://microsoft.github.io/aurora/beware.html),
[model specification](https://microsoft.github.io/aurora/models.html).
Radar history/decay/blending experiments are motivated by pySTEPS methods, not by
assuming advected rain stays predictable for eight hours.
[pySTEPS examples](https://pysteps.readthedocs.io/en/latest/auto_examples/plot_steps_nowcast.html).

## Pipeline and artifact contract

`experiment-plan` validates TOML and reports missing files without writing outputs
or accessing providers. It does not invent transfer/time estimates: measure those
in the pilot. `experiment-fetch` executes declared download jobs.
`experiment-run` executes declared research jobs and isolated prediction workers,
then observational scoring. A failed method remains visible; partial scores are
labelled and the command exits nonzero. Independent methods can still complete.

Job arguments are arrays for existing Python CLI subcommands, never shell strings.
All relative job arguments resolve from the campaign TOML directory. Commands must
supply `--config`; the runner owns reports. Existing commands retain their own
no-overwrite rules. Do not use a target-case evaluation result to tune a job.

- Campaign settings: strict TOML; reject unknown keys, duplicate identifiers,
  invalid timestamps, nonchronological/overlapping splits and unsafe identifiers.
- Prediction/observation boundary: NetCDF `time` is naive **UTC**, unique and sorted.
  Supported physical fields are `temperature_c`/`degC`, `precipitation_mm`/`mm`,
  `pressure_hpa`/`hPa`, `wind_u_ms` and `wind_v_ms`/`m/s`.
  Rain requires integer `interval_hours` (1/2/3/6), covering the preceding interval.
  Observational targets are hourly. Missing is NaN, never dry.
- Stations use `time,station`, station identifiers and `latitude(station)` /
  `longitude(station)`. Regular atmospheric grids may be sampled at stations:
  interpolate temperature/wind/pressure, nearest cell for rain, mask outside domain.
  Other grid-to-grid comparisons require explicitly preprocessed matching grids.
- Cached forecast datasets require `initialization_utc`; prospective inputs also
  require an aware `available_utc`. Public products carry `verification_role`.
- Manifest: campaign text, commit, hardware/Python, seed, source/input checksums.
  Per-method NetCDF, exclusive completion receipts and subprocess logs preserve
  completed outputs and failure evidence. Aurora execution metadata includes model
  preparation hashes and timings. Large input payloads remain outside Git.
- Scores: tidy CSV with campaign/track/split/case/regime, method/baseline, variable,
  issue lead, actual model lead, interval, metric, count and possible count.
  `baseline=own` describes own support; named baselines use exactly paired masks.
  `lead_hour=0` means a complete-horizon rainfall total.
- Every invocation publishes an immutable snapshot. Analysis loads the latest
  snapshot once per run, preventing repeated invocations from doubling samples.
  Changed declared inputs/source require a fresh output root; corrupt completed
  artifacts are rejected. Incomplete publications are blocked, not overwritten.
  Scores are keyed by hashes of their completed prediction set and observations.
  A recovered method creates a new immutable score artifact; old partial scores
  stay intact, and the latest snapshot includes the recovered comparisons.
- Legacy evidence inventory preserves original schemas; only the explicit
  compare-reference schema is converted to a separate development CSV. No claim
  is inferred from filenames, training loss or smoke-test success.

## Verification and decisions

Primary temperature metrics: MAE, RMSE, signed bias. Rain: MAE/RMSE/bias, total
amount error and CSI/POD/FAR at 0.1/1/5 mm per evaluated interval. Always state
interval duration: a 3h event threshold is not a 1h intensity threshold.
Existing radar evaluators retain native/coarse shared-support metrics and fractions
skill at 1/5/10 km. Scoring native-grid maps requires complete neighbourhoods;
unknown support is not a zero score. Record available sample counts per metric.

Report own support descriptively and pairwise-support differences for comparison.
Bootstrap 2,000 equal-weight independent event means, seed 42; no intervals for
one event. Do not bootstrap pixels or pretend the same storm at multiple stations
is multiple independent weather events. Descriptive regime slices with too few
events remain inconclusive. The analysis never combines tracks/splits/intervals.

Qualify only claims with paired untouched observational evidence and a defensible
coverage/context statement. A method may help short-lead rain while worsening
longer leads. Retain validation-selected zero correction where learning fails.
No universal method promotion or automatic default change follows this overhaul.

Resource defaults: sequential workers; float32 MPS; existing model memory guards;
pilot 1,800s/250,000,000 new payload bytes; 25 GiB retained-storage ceiling. A full
campaign explicitly sets 7,200s/2 GiB per invocation and resumes completed work.
Wall time is enforced by worker termination, not merely recorded afterward.
Native download paths enforce incremental payload limits. External SDK/CDS calls
whose transfers are not streamed by the bounded downloader must not be campaign
fetch jobs; prepare them explicitly with the existing commands and audit costs.
RSS and MPS counters are separate, not summed unified-memory peaks.

Before internal LoRA: resolve archive/schema blockers; complete frozen baselines;
pass one-step memory/timeout gate; freeze training/validation/test identities.
Declare training steps and budgets before execution. Ensemble forecasts, global
inference, remote compute and smaller replacement models are outside this revision.

## Acceptance and current limitations

Run `pixi run check`; test strict configuration, timing/availability leakage,
accumulation conversion, paired masks, missing support, immutable artifacts,
resumption, CLI exit codes and deterministic end-to-end synthetic execution.
Execute the notebook offline on synthetic results. Default checks need no API,
model weights or accelerator. Network/real-model integration remains opt-in.

The first deliverable is an executable research foundation and an honest audit,
not a completed seasonal study. Real-case curation, raw KNMI forecast HDF5 decoding,
unsupported older RTCOR schemas and representative training remain explicit
prerequisites. Do not mark these studies completed because the pipeline exists.
