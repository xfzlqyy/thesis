# Thesis Revision Checklist

This checklist tracks the writing and evidence issues found in the research-paper-writing review.
Use the checkbox only after the fix has been verified in the paper text, tables, figures, and, when needed, compiled PDF.

Status labels:

- `open`: not fixed yet.
- `patched`: text/table structure has been modified, but the result still needs verification.
- `blocked`: needs experiment numbers, figures, or external results before it can be closed.
- `verified`: checked and accepted.

## Claim-Evidence And Experiments

- [x] **E1. Main experiment results are still incomplete.**  
  Status: `verified`  
  Current fix: all main-result and real-capture SfM table values have been filled, and the result analysis has been rewritten to match the completed experiments.  
  Required fix: none.  
  Verification: `rg -n "待填" docs/thesis/body/undergraduate/final` returns no matches.

- [x] **E2. Fast SfM contribution needs direct downstream comparison.**  
  Status: `verified`  
  Current fix: experiment 2 compares `增量式 SfM` and `本文快速 SfM` on the real-capture `company` and `jazz` scenes using calibration time plus PSNR/SSIM/LPIPS. The discussion now states that the proposed pipeline improves calibration efficiency while achieving calibration effects comparable to the traditional SfM pipeline.  
  Required fix: none.  
  Verification: each scene has both calibration rows, and the analysis avoids claiming superior calibration accuracy.

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

- [x] **R1. Related work is still somewhat textbook-like.**  
  Status: `verified`  
  Current fix: related-work subsections now include gap summaries for traditional reconstruction, neural rendering/radiance fields, 3DGS extensions, fisheye 3DGS adaptation, and SLAM/calibration, and connect those gaps to the paper's fast SfM, virtual perspective view, and LiDAR initialization design choices.  
  Required fix: none.  
  Verification: each related-work subsection has been checked to state the remaining difficulty and, where appropriate, how this thesis addresses it.

- [x] **R2. Introduction claims must be revisited after results are filled.**  
  Status: `verified`  
  Current fix: the introduction contribution wording now matches the final evidence: fast SfM is described as improving calibration efficiency while achieving calibration effects comparable to traditional SfM, and LiDAR initialization is described as K-d tree downsampling plus local covariance based scale/rotation initialization rather than strict proportional density preservation.  
  Required fix: none.  
  Verification: checked the introduction contribution list against the completed experiment and ablation conclusions.

## Language, Terminology, And Polish

- [x] **L1. Terminology and style need normalization.**  
  Status: `verified`  
  Current fix: final terminology/style pass completed for the thesis text and author CV. First-person wording, strong informal degree words, and spoken-style phrases were removed from the checked sections. The acknowledgement file was intentionally left unchanged.  
  Required fix: none.  
  Verification: `rg -n "我们|大大|极大|比较|较好" docs/thesis/body/undergraduate/final/1-introduction.tex docs/thesis/body/undergraduate/final/2-body.tex docs/thesis/body/undergraduate/final/3-appendix.tex docs/thesis/body/undergraduate/final/4-cv.tex docs/thesis/body/undergraduate/final/abstract.tex` returns no matches.

- [x] **L2. Known typos remain.**  
  Status: `verified`  
  Current fix: corrected `基于先验位姿态`, `渲染渲染速度`, `NvtVLAD`, `非线形`, and `的的`.  
  Required fix: none for these known typo patterns.  
  Verification: `rg -n "基于先验位姿态|渲染渲染|NvtVLAD|非线形|的的" docs/thesis/body/undergraduate/final` returns no matches.

- [x] **L3. Table readability and numeric formatting still need final pass.**  
  Status: `verified`  
  Current fix: result precision has been standardized: PSNR uses two decimals, SSIM/LPIPS use three decimals, and image counts/SfM runtimes use integers.  
  Required fix: none.  
  Verification: inspected the main comparison table, SfM table, and ablation table in `body/undergraduate/final/2-body.tex`.

## Visual Evidence

- [x] **V1. Qualitative figures are insufficient for the claims.**  
  Status: `verified`  
  Current fix: added annotated qualitative comparison grids for ScanNet++ and Zip-NeRF scenes using GT, 3DGUT, 3DGEER, and the proposed method. The annotated regions highlight local differences in structure, edge regions, and texture details. Module-level claims are supported by the quantitative ablation table.  
  Required fix: none.  
  Verification: the placeholder qualitative figure has been replaced, all annotated images are stored under `figure/paper/eval_image/`, and `make` compiles successfully.

- [x] **V2. Virtual split coverage visualization is missing.**  
  Status: `verified`  
  Current fix: the method overview figure `figure/paper/method-overview.pdf` now includes the virtual perspective split/coverage visualization, and the method overview references it as `图~\ref{fig:method-overview}`.  
  Required fix: none.  
  Verification: compile PDF and confirm the pipeline figure makes the eight-view virtual split understandable before the detailed algorithm description.

- [ ] **V3. Qualitative comparison still needs the undistortion baseline.**  
  Status: `blocked`  
  Current fix: qualitative grids currently include GT, 3DGUT, 3DGEER, and the proposed method, using the available annotated images.  
  Required fix: add annotated qualitative images for `3DGS + 去畸变图像` and include them in the ScanNet++ and Zip-NeRF qualitative grids.  
  Verification: after the images are added, confirm the qualitative grids cover the same baseline set as Table `tab:exp-main-fisheye`.
