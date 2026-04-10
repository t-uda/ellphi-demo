# Application Experiment: Brain DTI — Anisotropic Topological Analysis of White Matter

**Objective**: Apply anisotropic persistent homology to real diffusion tensor imaging (DTI) data, demonstrating that ellipsoidal tangency distances naturally capture white-matter tract topology that isotropic Vietoris–Rips filtrations miss.

**Status**: Not started.

## 1. Scientific Motivation

In diffusion tensor imaging, each brain voxel carries a $3 \times 3$ symmetric positive-definite (SPD) diffusion tensor $D$ describing how water molecules diffuse locally. White-matter axon bundles constrain diffusion along the fiber direction, producing highly anisotropic tensors (prolate ellipsoids). Gray matter and CSF are nearly isotropic (spherical).

Standard persistent homology on voxel coordinates treats the brain as an isotropic point cloud, grouping voxels by Euclidean proximity alone. This fragments long, thin fiber tracts into many small clusters. **Anisotropic PH using the tangency distance** respects the directional coherence of diffusion: voxels sharing a common fiber orientation "see" each other as closer, even when spatially separated. This should:

- **H0**: Consolidate major fiber bundles (corpus callosum, internal capsule, corticospinal tract) into fewer, longer-lived connected components.
- **H1**: Detect loop-like tract structures (e.g., arcuate fasciculus, cingulum) as persistent 1-cycles that reflect the directional continuity of the tract.

## 2. Data

- **Dataset**: Stanford HARDI (High Angular Resolution Diffusion Imaging)
- **Source**: Bundled with DIPY; also at [purl.stanford.edu/ng782rw8378](https://purl.stanford.edu/ng782rw8378)
- **License**: CC BY
- **Acquisition**: 150 diffusion directions, $b = 2000$ s/mm², $1.5$ mm isotropic voxels
- **Access**: `dipy.data.get_fnames("stanford_hardi")` — automatic download (~87 MB)
- **Format**: NIfTI (.nii.gz) + bvals/bvecs text files

## 3. Preprocessing Pipeline

### 3.1 DTI Model Fit
```
GradientTable(bvals, bvecs) → TensorModel(gtab) → TensorFit
```
- Use `dipy.reconst.dti.TensorModel` with default least-squares fit.
- Extract per-voxel $3 \times 3$ diffusion tensor (`tenfit.quadratic_form`, shape `(X,Y,Z,3,3)`).
- Compute fractional anisotropy: `tenfit.fa` (shape `(X,Y,Z)`).

### 3.2 Slice Selection & White-Matter Masking
1. Select an **axial slice** $z = z_0$ rich in white matter (e.g., at the level of the corpus callosum body, approximately slice 40–50 in the Stanford dataset).
2. Apply a **white-matter mask**: FA $> 0.3$. This retains voxels with significant anisotropy while excluding gray matter and CSF.
3. Expected yield: ~2000–4000 voxels per slice.

### 3.3 Region of Interest (ROI) Extraction
- Random subsampling fragments the continuous fiber topology. Instead, extract a dense bounding box (e.g., $X \in [20, 60], Y \in [40, 75]$ for the corpus callosum region) to preserve local spatial continuity while limiting the point count to $n \approx 500\text{–}1000$.
- This ensures adjacent voxels along a tract remain structurally connected in the tangency graph.

### 3.4 Tensor Projection (3D → 2D)
Since we analyze a single axial slice, the in-plane diffusion is captured by the $2 \times 2$ sub-matrix:
$$
D_{2\mathrm{D}} = \begin{pmatrix} D_{xx} & D_{xy} \\ D_{xy} & D_{yy} \end{pmatrix}
$$
This is the upper-left block of the full $3 \times 3$ tensor in the $(x, y, z)$ voxel frame. Being a principal sub-matrix of an SPD matrix, $D_{2\mathrm{D}}$ is guaranteed SPD.

## 4. Ellipsoid Construction

- **Centers**: $(x_i, y_i) \in \mathbb{R}^2$ — voxel coordinates within the slice (in mm or voxel units).
- **Covariances**: $\Sigma_i = \beta \cdot \alpha \cdot D_{2\mathrm{D}, i}$ — the in-plane diffusion tensor.
  - First, normalize the typical ellipsoid scale using $\alpha$ to make it comparable to inter-voxel spacing (e.g., via `rescale("median")`).
  - Then, apply an explicit amplification factor $\beta$ (e.g., `SCALE_FACTOR = 3.0`) to strongly overlap tensors along the fiber orientation. This acts mathematically as extending the observation time $t$, allowing tangential topologies to override the orthogonal Euclidean spacing of the voxel grid.
- **EllPHi API**: `coefs = ellphi.coef_from_cov(centers, covs)`

## 5. Persistent Homology Computation

### 5.1 Anisotropic PH (EllPHi)
```python
coefs = ellphi.coef_from_cov(centers, covs)
dm_aniso = squareform(ellphi.pdist_tangency(coefs))
dgm_aniso = ripser(dm_aniso, distance_matrix=True, maxdim=1)['dgms']
```

### 5.2 Isotropic PH (Baseline)
```python
dm_iso = squareform(pdist(centers, metric='euclidean'))
dgm_iso = ripser(dm_iso, distance_matrix=True, maxdim=1)['dgms']
```

### 5.3 Optional: Riemannian Baseline
For a fairer comparison, one could also use the log-Euclidean or affine-invariant metric on SPD tensors as an alternative distance. This is out of scope for the initial notebook but noted for future work.

## 6. Visualization

### 6.1 Anatomical Context
- FA color map of the selected slice with the white-matter mask boundary overlaid.
- Ellipsoid glyphs (via `ellphi.ellipse_cloud` or manual `matplotlib.patches.Ellipse`) at each selected voxel, colored by the primary diffusion direction (standard DTI RGB convention: R=left-right, G=anterior-posterior, B=superior-inferior).

### 6.2 Persistence Comparison
- Side-by-side **persistence diagrams** (H0 and H1) for anisotropic vs isotropic.
- Side-by-side **barcode plots**.
- Table of summary statistics: number of bars with persistence $> \tau$ for several thresholds.

### 6.3 Cluster Map (H0)
- From the H0 barcode at a chosen persistence threshold, extract the connected components of the anisotropic filtration.
- Color voxels by component membership, overlaid on the FA map.
- Compare with the isotropic clustering at the same number of components.
- **Expected**: Anisotropic clusters align with anatomically known fiber bundles; isotropic clusters are spatially compact blobs that split bundles.

## 7. Expected Results

| Feature | Isotropic PH | Anisotropic PH (EllPHi) |
|---------|-------------|------------------------|
| H0 long-lived components | Many small spatial clusters | Few large components aligned with fiber bundles |
| H0 most persistent bar | Spans the whole brain (trivial) | Corpus callosum or largest tract |
| H1 persistent cycles | Reflect spatial holes (ventricles) | Reflect directional loops in tract architecture |
| Sensitivity to fiber direction | None | High — merges along-fiber before across-fiber |

## 8. Dependencies

- **Core**: `ellphi>=0.1.2`, `numpy`, `scipy`, `ripser`, `matplotlib`
- **New**: `dipy>=1.7.0` (DTI fitting, data download, tensor utilities)
- **Installation**: Add `dipy` to `pyproject.toml` as optional dependency group `[dti]`.

## 9. Output Artifacts

- `notebooks/brain_dti/brain_dti.py` — marimo notebook
- `public/brain_dti_fa_ellipsoids.pdf` — FA map with ellipsoid glyphs
- `public/brain_dti_persistence_comparison.pdf` — PD/barcode comparison
- `public/brain_dti_cluster_map.pdf` — H0 cluster comparison

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Too many voxels for $O(n^2)$ tangency computation | Subsample to $\leq 1000$ points; use spatial stride |
| Tensor scaling mismatch (diffusivity ~$10^{-3}$ mm²/s vs voxel spacing ~1.5 mm) | Normalize tensors by median eigenvalue |
| 2D projection loses through-plane fiber information | Acknowledge limitation; this is a proof-of-concept on a single slice |
| `dipy` download fails in restricted networks | Provide fallback: pre-cached tensor data as `.npz` |
