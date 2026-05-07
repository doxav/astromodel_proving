I read Reviewer 2, your preprint, the prior computational-plan PDF, the old Optuna notebook, the post-Optuna filtering notebook, and the new ATF-analysis notebook. I also checked the uploaded zip structure directly.

The main point is this: Reviewer 2 is not only criticizing the model. They are criticizing the inference logic. Your new dataset gives you a credible way to fix that, because it lets you move from fitting one sweep at a time to fitting a **cell-specific six-sweep response fingerprint** under empirical variability.

From the uploaded zip and your feature-extraction logic, the new dataset contains **37 cells x 6 sweeps = 222 sweeps**, with **11 control, 13 MFA, 13 MFA+Ba** cells, and **19 DH / 18 VH** cells. The cells are **unpaired across conditions**, and there are **no animal/slice IDs**, so the valid statistical unit is **cell/file**, with sweep nested within cell. This can support population-level region x condition conclusions, but not paired pharmacology claims or animal-level inference.

A few data properties matter immediately for the fit design. Using your attached extraction notebook logic, plateau/return behavior is strongly condition dependent: `plateau_reached` is common in control, less common in MFA, and much less common in MFA+Ba; `return_slope` is also missing much more often in MFA+Ba. That means plateau-related terms should be **conditional or low-weight**, not treated as equally reliable targets in every condition. Also, `peak_depolarization` and `stim_end_depolarization` are nearly redundant in your extracted features, so they should not both be given full weight. Region should also be explicit: DH-VH separation is small in control but much larger under MFA and MFA+Ba, especially at the largest sweep.

## What has to change in the fitting logic

The current Optuna design is the main weakness.

First, the old notebook is effectively a **single-sweep fit**. It fits one current/sweep at a time. Reviewer 2’s point about non-identifiability is much stronger in that design, because many parameter combinations can match one waveform.

Second, the old objective often uses **whole-trace mean centering**. That throws away baseline information and makes different mechanisms look more similar than they are.

Third, the old search mixes **model-family choice** (`sigmoid` vs `tanh`) with **parameter estimation** in one study. That makes model comparison and parameter interpretation messy.

Fourth, several parameters are not sensible as independent inferential targets in the current model. In the model code, the gap-junction term enters as `P_kgap = d_gap * P_k`, so `d` and `pk` are not separately interpretable unless one is fixed. Likewise, `gama_t` and `gama_s` enter through combinations with `w_a`, and `w_a / w_o` appears as a ratio in extracellular coupling. Those are classic cases where you should fit **effective combinations** first, then decide later whether raw physiological interpretation is justified. That general strategy is exactly what identifiability workflows are meant to enforce: structural identifiability first, then practical identifiability, then prediction uncertainty. Sloppiness is related but distinct; it is diagnosed from broad sensitivity spectra and often means predictions are better constrained than raw parameters. ([PLOS][1])

Fifth, your `Filtered_basline_sweep` notebook is useful, but right now it acts as a **post hoc rescue step** for a weak primary inference design. The right move is to keep that acceptance logic, but make it part of the main pipeline rather than an afterthought.

## The improved Optuna experiment

### 1) Fit one cell across all 6 sweeps jointly

This is the single most important redesign.

For each file/cell, fit all six sweeps together with:

* one shared cell parameter set
* one ordered stimulus mapping across the six sweep levels
* one shared model family per study
* one composite loss over all six sweeps

That changes the question from “what fits one trace?” to “what mechanism reproduces the full ordered response of this cell?”

### 2) Reparameterize into effective parameters

Do **not** optimize raw parameters when the model only sees combinations. I would fit these as the primary inferential objects:

* `g_gap_eff = d_gap * P_k`
* `k_t_eff = gama_t * Sig_a / (w_a * F)`
* `k_s_eff = gama_s * Sig_a / (w_a * F)`
* `beta_io = w_a / w_o`

Then keep `g_kir`, `gl_a`, `epsilon`, `Va_s`, `Va_l`, and the gating parameters as separate fitted quantities only if they pass identifiability checks.

For strictly positive parameters, search in log space. That will make the search better behaved.

### 3) Separate model families cleanly

Run separate studies for:

* sigmoid gating
* Hill-type gating
* soft-threshold / tanh-like gating

Do not put `switching_function` as a categorical Optuna parameter inside one joint search. Model family selection should be a separate comparison after fitting, ideally using held-out prediction, not a mixed hyperparameter trick. Model comparison under limited noisy measurements should be separated from parameter estimation, and inference should be judged with diagnostics rather than best-fit error alone. ([PLOS][2])

### 4) Use a monotone stimulus mapping across the six sweeps

Your six sweeps are ordered stimulation levels. The model input should respect that order.

Do one of these two:

* If the nominal stimulation values are known and reliable, fix them and only fit a small calibration map from nominal stimulus to effective `K_bath_middle`.
* If not, fit six latent input amplitudes constrained to be monotone increasing, for example with a cumulative positive parameterization.

This is much better than fitting each sweep as an unrelated experiment.

### 5) Use a composite objective that reflects both waveform and physiology-facing features

I would replace the old loss with:

`L_total = L_trace + L_feature + L_binary + L_prior + L_fail`

Where:

`L_trace`
Use a **Huber loss** on `DeltaV(t) = V(t) - baseline_pre_stim`, not whole-trace mean centering. Baseline subtraction preserves the response shape while avoiding global offset artifacts.

`L_feature`
Use a **covariance-aware feature loss** on a reduced, nonredundant feature set. For example:

* one depolarization amplitude term: choose **either** `peak_depolarization` **or** `stim_end_depolarization`, not both at full weight
* `rise_tau`
* `decay_tau`
* `undershoot_magnitude`
* possibly low-weight `rise_slope` and `decay_slope`

Use empirical scale/covariance from the new data within matching `region x condition x sweep`, with shrinkage toward pooled estimates where sample size is small, especially VH control.

`L_binary`
Treat `plateau_reached` and `has_undershoot` as binary targets. This is important because those features are not consistently present in every condition.

`L_prior`
Use weak physiological/effective-parameter penalties. Keep these weak in the main fit. Then do a second constrained rerun with stronger pharmacology-aware priors as a sensitivity analysis.

`L_fail`
Use hard penalties for nonphysical or unstable behavior: negative concentrations, impossible ratios, solver failure, no recovery, grossly unrealistic hidden-state excursions.

A practical refinement: choose the loss weights so that, on a pilot subset, the median contribution of each non-penalty term is on the same scale. That prevents arbitrary domination by one term.

### 6) Make feature reliability explicit

This comes directly from your new data.

In your extracted dataset:

* `peak_depolarization` and `stim_end_depolarization` are almost the same signal
* `plateau_slope` is very noisy
* `return_slope` is conditionally missing, especially in MFA+Ba

So I would define three feature tiers.

**Primary targets**
`stim_end_depolarization` or `peak_depolarization`, `rise_tau`, `decay_tau`, `undershoot_magnitude`

**Secondary low-weight targets**
`rise_slope`, `decay_slope`

**Conditional targets**
`plateau_reached`, `has_undershoot`, and only then `plateau_slope` / `return_slope` when the corresponding state exists and the feature is reliable

This gives you an objective that actually reflects the data you now have.

### 7) Keep acceptance as an explicit second stage

After the scalar objective search, define an **accepted-fit ensemble** using explicit gates:

* good total loss
* no hard failures
* passes enough empirical feature bands
* matches binary features adequately
* passes held-out sweep prediction

This is where your existing filtering notebook becomes useful. Keep the idea, but upgrade it to multi-sweep and effective-parameter logic.

## Computational experiments you should run

Run them in this order.

### 1) Build the empirical uncertainty model from the new data

From `astro_atf_analysis_improved_sectioned`:

* export per-sweep features for every cell
* estimate per `region x condition x sweep` medians, IQRs, covariance, missingness
* flag high-noise / low-availability features
* compute region x condition summaries

This addresses Reviewer 2’s point on variability and uncertainty directly.

### 2) Re-screen your old `.db` studies with the new empirical thresholds

This is the fast first pass.

Use the filtering notebook to ask: after imposing empirical feature bands from the new dataset, how much of the old “degeneracy” survives?

But use this only as triage. It will not be enough for Reviewer 2 by itself, because the original optimization was still single-sweep.

### 3) Refit each cell jointly across all 6 sweeps

This is the main new inference experiment.

For each cell:

* shared cell parameters
* monotone ordered stimulus mapping
* one model family at a time
* composite loss
* accepted-fit ensemble, not only best fit

Then compare accepted ensembles across `DH/VH x control/MFA/MFA+Ba`.

### 4) Leave-one-sweep-out prediction

For each fitted cell:

* fit 5 sweeps, predict the 6th
* rotate across all 6 sweeps
* summarize held-out error by region and condition

Also do a harder test:

* fit lower sweeps, predict higher sweeps
* fit higher sweeps, predict lower sweeps

That directly answers the reviewer’s robustness criticism and is more convincing than showing only fitted traces.

### 5) Structural identifiability on the effective parameterization

Do a STRIKE-GOLDD-style structural identifiability screen, or the closest numerical equivalent you can implement, using `Vm(t)` as the observation and the six-sweep protocol as the input design. Structural identifiability should be checked before interpretation, because otherwise “degeneracy” can just be a disguised observability problem. ([PLOS][1])

Use this result to decide which raw parameters can no longer be claimed as individually interpretable.

### 6) Practical identifiability with profile likelihood

Do profile likelihood on representative cells, but profile **effective parameters first**, not raw ones.

A workable design is:

* one representative cell per `region x condition` group as the minimum
* two per group if compute budget allows

For each profiled direction, compute prediction intervals not only for `Vm`, but also for hidden outputs such as `K_o`-related quantities. Profile-wise analysis is useful here because it links parameter non-identifiability to prediction uncertainty. ([PLOS][3])

This is the cleanest direct response to the reviewer’s point that fitting `Vm` alone may not constrain ionic dynamics.

### 7) Sloppiness / FIM spectrum on all best fits or accepted centers

For each cell or representative fit center, compute the Fisher Information Matrix and its eigenvalue spectrum.

Use this to separate three cases:

* **structural non-identifiability**
* **practical non-identifiability**
* **sloppy-but-predictive directions**

That distinction matters because sloppiness is not the same thing as unidentifiability, and in sloppy models the right emphasis is often on predictions rather than raw parameters. ([PLOS][4])

### 8) Geometry of the accepted-fit set

Do not rely on UMAP of raw parameters as evidence.

Instead:

* remove duplicate/numerically trivial fits
* cluster accepted fits in **effective-parameter space**
* estimate whether the good-fit set is one continuous compensation manifold or separated modes
* quantify intrinsic dimensionality locally

This helps answer the reviewer’s “submanifold” criticism in a principled way. The identifiable-surrogate viewpoint is relevant here: when the raw model is non-identifiable, you should analyze the geometry in effective coordinates rather than overread raw parameters. ([PLOS][5])

### 9) Mechanistic decomposition of accepted regimes

For each accepted cluster/regime, compute:

* `I_Kir(t)`
* leak current
* gap-junction current
* hidden states / flux partitions
* functionals such as peak excess `K_o`, recovery time, integrated excess above baseline

Then show whether different accepted regimes are **mechanistically distinct** while preserving the same observed six-sweep phenotype.

This is the place where “degeneracy” becomes defensible.

### 10) Assumption-sensitivity experiments

Run three explicit model-comparison experiments.

**A. Gating form**
Sigmoid vs Hill vs soft-threshold/tanh

**B. Proxy test**
Current proxy relation for extracellular K vs a minimal explicit ECS variant

**C. State split test**
Two intracellular states (`local` + `syncytial`) vs one intracellular state

Fit each model family with the same multi-sweep objective and compare:

* held-out sweep prediction
* accepted-fit counts
* mechanistic cluster structure
* hidden-state uncertainty

This is the computational answer to the reviewer’s criticism of the sigmoid, proxy, and local/syncytial assumptions.

### 11) Parameter plausibility and constrained reruns

Do two versions:

* broad effective-parameter fit
* constrained physiologically tighter fit

Then compare:

* fit quality
* held-out prediction
* accepted-fit count
* cluster structure

For any parameter that still fails plausibility or identifiability, label it explicitly as an **effective parameter**, not as a directly measured physiological quantity.

### 12) Population-level posterior predictive checks

Using the accepted-fit ensemble, simulate the distribution of extracted features and compare it to the empirical cell distributions in each `region x condition x sweep` group.

This answers the reviewer’s concern about experimental variability more directly than showing only exemplar traces. It also aligns with the broader inference literature: diagnostics and predictive uncertainty are part of the result, not a supplement to the best fit. ([PLOS][6])

## Mapping the computational tasks to Reviewer 2

### 1) “Degeneracy is not distinguished from non-identifiability or sloppiness.”

Do:

* structural identifiability on effective parameters
* profile likelihood on representative cells
* FIM/sloppiness spectrum
* geometry of accepted-fit set in effective space

Claim after that:

* non-identifiable directions are not “degeneracy”
* sloppy directions are not automatically “degeneracy”
* degeneracy is reserved for **mechanistically distinct accepted regimes that also pass predictive checks**

### 2) “Experimental variability, noise, or uncertainty are not clarified.”

Do:

* full feature extraction on the 37-cell dataset
* group/sweep-specific variability estimates
* reliability-based feature weighting
* population predictive checks
* bootstrap uncertainty bands on group summaries

Claim after that:

* the fit is evaluated against measured variability, not against one arbitrary exemplar trace

### 3) “The sigmoid gate, intracellular-K proxy, and local/syncytial split are weakly justified.”

Do:

* separate model-family refits
* proxy sensitivity with explicit ECS variant
* one-state vs two-state intracellular comparison

Claim after that:

* either the conclusions are robust to those assumptions, or they are not; in both cases the reviewer’s criticism is answered

### 4) “Fitting membrane potential alone cannot constrain ionic dynamics; some parameters are out of range.”

Do:

* six-sweep joint fitting
* effective-parameter reparameterization
* profile-wise prediction intervals for hidden states
* constrained reruns
* plausibility tables

Claim after that:

* here is what `Vm` plus six sweeps constrains
* here is what remains unconstrained
* these quantities are effective, not directly physiological, unless identifiability supports otherwise

### 5) “There is no evidence that different degenerate regimes correspond to distinct phenotypes or buffering pathways.”

Do:

* cluster accepted regimes in effective-mechanism space
* current/flux decomposition
* enrichment by region and condition
* perturbation divergence tests

Claim after that:

* use the phrase **mechanistic regimes**
* avoid **phenotypes** unless you have external biological evidence

### 6) “Robustness beyond fitted conditions is not tested.”

Do:

* leave-one-sweep-out prediction
* low-to-high and high-to-low sweep prediction
* in silico perturbation suite

Claim after that:

* accepted regimes are predictive beyond the exact fitted sweep set, or they are not

### 7) “Extracellular K homeostasis may be imposed by the model structure.”

Do:

* explicit ECS sensitivity
* one-state/two-state sensitivity
* hidden-state prediction intervals for `K_o`-related functionals

Claim after that:

* only claim model-level functional robustness if those outputs remain constrained across the alternative formulations

### 8) “Membrane kinetics as a proxy for buffering efficiency may not hold generally.”

Do:

* show predictive validity only within this protocol family
* quantify hidden-state uncertainty
* test divergence under perturbation

Claim after that:

* membrane kinetics are a useful observable for this protocol, not universal proof of buffering efficiency across all conditions and scales

### 9) “Figures and presentation are unclear.”

Do:

* composite panels
* uncertainty bands instead of many overplotted traces
* consistent units/axes
* one figure for identifiability, one for predictive validation, one for mechanism clusters

That is not cosmetic. It is part of making the computational argument legible.

## What I would explicitly change in the manuscript language

I would not defend the old claim in its current form.

I would redefine the story into three layers:

1. **Identifiability layer**
   Which parameters or combinations are actually constrained by the six-sweep `Vm` data?

2. **Mechanistic multiplicity layer**
   Among the constrained effective descriptions, are there multiple mechanistically distinct accepted regimes?

3. **Predictive robustness layer**
   Do those regimes remain valid on held-out sweeps and perturbations, and do they preserve specific functional outputs?

That framing is much closer to what Reviewer 2 is asking for, and it is also more consistent with good inference practice in nonlinear dynamic models. ([PLOS][1])

## The minimum package I would treat as necessary for a serious resubmission

At minimum, I would do these six:

1. six-sweep joint refit per cell
2. effective-parameter reparameterization
3. uncertainty-aware composite objective
4. leave-one-sweep-out prediction
5. profile likelihood + FIM on representative cells
6. one explicit assumption-sensitivity comparison for the gating/proxy/state-split story

Everything beyond that strengthens the paper, but those six are the core response.

One final constraint: because you do not have paired cells across conditions and do not know animal/slice identity, the revised paper should make **population-level** claims by region and condition, not paired blocker-effect claims and not animal-level claims. That limitation should be stated directly.

That is the route that gives you a defensible computational response to Reviewer 2.

[1]: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1005153 "https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1005153"
[2]: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1000696 "https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1000696"
[3]: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1011515 "https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1011515"
[4]: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006694 "https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006694"
[5]: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1010844&utm_source=chatgpt.com "Geometric analysis enables biological insight from complex non-identifiable models using simple surrogates | PLOS Computational Biology"
[6]: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1011108&utm_source=chatgpt.com "Methods and considerations for estimating parameters in biophysically detailed neural models with simulation based inference | PLOS Computational Biology"
