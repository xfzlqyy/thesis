# Thesis Revision Checklist

This checklist tracks the writing and evidence issues found in the research-paper-writing review.
Use the checkbox only after the fix has been verified in the paper text, tables, figures, and, when needed, compiled PDF.

Status labels:

- `open`: not fixed yet.
- `patched`: text/table structure has been modified, but the result still needs verification.
- `blocked`: needs experiment numbers, figures, or external results before it can be closed.
- `verified`: checked and accepted.

## Claim-Evidence And Experiments

- [ ] **E1. Main experiment results are still incomplete.**  
  Status: `blocked`  
  Current evidence: `待填` cells remain in `body/undergraduate/final/2-body.tex`, currently concentrated in the real-capture SfM experiment.  
  Required fix: fill all PSNR/SSIM/LPIPS values and update result analysis.  
  Verification: run `rg -n "待填" docs/thesis/body/undergraduate/final` and confirm no result table placeholders remain.

- [ ] **E2. Fast SfM contribution needs direct downstream comparison.**  
  Status: `patched`  
  Current patch: experiment 2 now compares `增量式 SfM` and `本文快速 SfM` using PSNR/SSIM/LPIPS, and explains why reprojection error is not the primary metric in `body/undergraduate/final/2-body.tex`.  
  Required fix: fill per-scene numbers for both calibration methods on the real-capture `company` and `jazz` scenes.  
  Verification: check that each scene has both rows and that the result discussion states whether fast SfM improves downstream rendering.

- [x] **E3. LiDAR initialization and K-d tree downsampling need ablation evidence.**  
  Status: `verified`  
  Current fix: the ablation table now includes the K-d tree downsampling Gaussian initialization result, and the analysis treats LiDAR initialization as a geometry prior and initialization-scale control strategy rather than a guaranteed quality improvement.  
  Required fix: none.  
  Verification: the ablation discussion explicitly states that the average metric gain is small and weakens the LiDAR initialization claim accordingly.

- [x] **E4. Experiment protocol needs more detail.**  
  Status: `verified`  
  Current fix: the experiment setup states that `llffhold=8` is used; every adjacent 8 fisheye images contain 1 held-out test source and 7 training sources; the test image is a fixed-FoV small perspective view split from the center direction of the held-out fisheye image.  
  Required fix: none.  
  Verification: confirmed this text matches the actual data preparation and experiment protocol.

## Method Clarity

- [x] **M1. Pipeline figure is missing.**  
  Status: `verified`  
  Current fix: copied `figure/paper/method-overview.pdf` into the thesis figure directory, inserted it in the method overview, and referenced it with `图~\ref{fig:method-overview}` in `body/undergraduate/final/2-body.tex`.  
  Required fix: none.  
  Verification: `make` compiles successfully and `rg -n "Pipeline figure placeholder" docs/thesis/body/undergraduate/final/2-body.tex` returns no matches.

- [x] **M2. Virtual perspective split algorithm needed correction.**  
  Status: `verified`  
  Current fix: the method describes vertical FoV search, horizontal FoV search using low-resolution four-diagonal-view boundary testing, final eight-view split, and resolution search.  
  Required fix: none.  
  Verification: confirmed the text matches the implementation and does not imply final training uses only four views.

- [x] **M3. Camera pose convention was ambiguous.**  
  Status: `verified`  
  Current fix: the virtual-view pose propagation paragraph states the `camera-to-world` convention, and \(R_k\) is consistently defined as the rotation from the virtual perspective camera to the fisheye camera in Formula 3-13 and the pose propagation formula.  
  Required fix: none.  
  Verification: confirmed that the paper's \(R_k\) corresponds to the inverse/transpose of the implementation's OpenCV rectification rotation, while the exported COLMAP world-to-camera pose uses the equivalent left-multiplied form.

- [x] **M4. K-d tree density-preservation claim is still too strong.**  
  Status: `verified`  
  Current fix: removed the formula-style density-ratio claim and rewrote the method as a qualitative K-d tree downsampling and scale-aware Gaussian initialization strategy. The text now states that scanned regions tend to retain more representative points after adaptive partitioning, without claiming strict proportional preservation or guaranteed quality improvement.  
  Required fix: none.  
  Verification: the method and experiment text now frame LiDAR initialization as a geometry prior and initialization-scale control strategy.

## Related Work And Story

- [ ] **R1. Related work is still somewhat textbook-like.**  
  Status: `patched`  
  Current patch: gap summaries were added for traditional reconstruction, neural rendering/radiance fields, 3DGS extensions, fisheye 3DGS adaptation, and SLAM/calibration.  
  Required fix: after the final result claims are known, verify that related-work gaps still match the paper's actual contribution scope.  
  Verification: each related-work subsection should answer "why this is insufficient for our setting".

- [ ] **R2. Introduction claims must be revisited after results are filled.**  
  Status: `blocked`  
  Issue: contribution wording currently claims improved stability, effective initialization, and engineering feasibility before all results are present.  
  Required fix: after metrics are filled, weaken or strengthen claims to match actual evidence.  
  Verification: create a claim-evidence map for the introduction and ensure each claim points to a table, figure, or explicit experiment.

## Language, Terminology, And Polish

- [ ] **L1. Terminology and style need normalization.**  
  Status: `patched`  
  Current patch: first-person `我们`, strong informal degree words, and several spoken-style method descriptions were rewritten in the reviewed sections. Remaining `比较` hits are comparison-context usages such as "进行比较", not degree modifiers.  
  Required fix: do one final full-paper language pass after all experiment text is filled.  
  Verification: run `rg -n "我们|大大|极大|比较|较好" docs/thesis/body/undergraduate/final` and inspect any remaining hits in context.

- [x] **L2. Known typos remain.**  
  Status: `verified`  
  Current fix: corrected `基于先验位姿态`, `渲染渲染速度`, `NvtVLAD`, `非线形`, and `的的`.  
  Required fix: none for these known typo patterns.  
  Verification: `rg -n "基于先验位姿态|渲染渲染|NvtVLAD|非线形|的的" docs/thesis/body/undergraduate/final` returns no matches.

- [ ] **L3. Table readability and numeric formatting still need final pass.**  
  Status: `blocked`  
  Issue: result precision is inconsistent in some existing metric cells, and many values are placeholders.  
  Required fix: after filling results, standardize decimal places per metric and highlight best/second-best values if desired.  
  Verification: inspect all tables in compiled PDF for alignment, line breaks, and consistent precision.

## Visual Evidence

- [ ] **V1. Qualitative figures are insufficient for the claims.**  
  Status: `open`  
  Issue: there is currently a high-FoV artifact example, but not enough visual evidence for edge quality, pose refinement effects, LiDAR initialization, or K-d tree downsampling.  
  Required fix: add qualitative comparisons for the major modules, especially edge regions and weak-texture regions.  
  Verification: every major ablation row should have either quantitative evidence, qualitative evidence, or both.

- [x] **V2. Virtual split coverage visualization is missing.**  
  Status: `verified`  
  Current fix: the method overview figure `figure/paper/method-overview.pdf` now includes the virtual perspective split/coverage visualization, and the method overview references it as `图~\ref{fig:method-overview}`.  
  Required fix: none.  
  Verification: compile PDF and confirm the pipeline figure makes the eight-view virtual split understandable before the detailed algorithm description.
