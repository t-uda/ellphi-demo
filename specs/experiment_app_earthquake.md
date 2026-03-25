# Application Experiment: Earthquake Moment Tensors — Anisotropic Topological Detection of Fault Structures

**Objective**: Apply anisotropic persistent homology to earthquake moment tensor data from the Global CMT catalog, demonstrating that ellipsoidal tangency distances reveal fault-aligned seismogenic structures and plate boundary topology that isotropic PH cannot resolve.

**Status**: Not started.

## 1. Scientific Motivation

Every earthquake is characterized by a **moment tensor** $M$, a $3 \times 3$ symmetric matrix encoding the geometry and magnitude of the fault rupture. The eigenstructure of $M$ defines the T (tension), P (pressure), and N (null) axes — the "beach ball" representation familiar in seismology. Earthquakes on the same fault system share aligned moment tensors, even when their epicenters are spatially separated.

Standard persistent homology on epicenter coordinates $(lon, lat)$ or $(x, y)$ groups earthquakes by spatial proximity alone, ignoring the mechanical similarity encoded in their moment tensors. **Anisotropic PH** assigns each earthquake an ellipsoid derived from its moment tensor, so that events sharing a common rupture mechanism are "closer" in the tangency metric. This should:

- **H0**: Merge seismically coherent fault segments earlier, revealing mechanically unified fault zones (e.g., the entire Japan Trench megathrust) as single connected components.
- **H1**: Detect persistent 1-cycles along plate boundaries — the seismicity "outlines" the subducting slab or aseismic regions (locked asperities) as topological loops.

## 2. Data

- **Dataset**: Global Centroid-Moment Tensor (GCMT) Catalog
- **Source**: [globalcmt.org/CMTfiles.html](https://www.globalcmt.org/CMTfiles.html)
- **License**: Public domain (academic citation requested)
- **Coverage**: 1976–present, $M_w \geq 5.0$ globally (~55,000 events total)
- **Format**: NDK (5-line ASCII per event); parseable with `obspy.read_events()`
- **Access**: Direct HTTP download of `.ndk` files; no authentication

### 2.1 Study Region: Japan Trench

- **Bounding box**: Longitude $135°\text{–}150°$E, Latitude $30°\text{–}45°$N
- **Depth filter**: $\leq 100$ km (shallow seismicity on the plate interface)
- **Magnitude filter**: $M_w \geq 5.0$
- **Expected yield**: ~500–1500 events (1976–2024)
- **Rationale**: The Japan Trench is one of the most seismically active subduction zones, with well-characterized fault geometry. The 2011 Tohoku $M_w$ 9.1 event and its aftershock sequence provide a dense, well-studied cluster.

## 3. Preprocessing Pipeline

### 3.1 Data Retrieval & Parsing
```python
from obspy import read_events
cat = read_events("jan76_dec20.ndk")
```
Filter by region, depth, and magnitude. Extract per-event:
- Centroid location: $(lon, lat, depth)$
- Moment tensor components: $M_{rr}, M_{\theta\theta}, M_{\phi\phi}, M_{r\theta}, M_{r\phi}, M_{\theta\phi}$
- Scalar moment $M_0$ and moment magnitude $M_w$

### 3.2 Coordinate Transformation
Convert $(lon, lat)$ to local Cartesian $(x_E, x_N)$ in km using a reference point (e.g., centroid of the study region):
$$
x_E = (lon - lon_0) \cdot \frac{\pi}{180} \cdot R \cos(lat_0), \quad
x_N = (lat - lat_0) \cdot \frac{\pi}{180} \cdot R
$$
where $R = 6371$ km.

### 3.3 Moment Tensor → 2D Ellipsoid

The full moment tensor in spherical coordinates $(r, \theta, \phi)$ is:
$$
M = \begin{pmatrix}
M_{rr} & M_{r\theta} & M_{r\phi} \\
M_{r\theta} & M_{\theta\theta} & M_{\theta\phi} \\
M_{r\phi} & M_{\theta\phi} & M_{\phi\phi}
\end{pmatrix}
$$

For a 2D (map-view) analysis, extract the horizontal sub-block:
$$
M_{2\mathrm{D}} = \begin{pmatrix}
M_{\theta\theta} & M_{\theta\phi} \\
M_{\theta\phi} & M_{\phi\phi}
\end{pmatrix}
$$

**SPD guarantee**: Moment tensors are trace-free and generally indefinite (not positive-definite). To construct a valid covariance (SPD) matrix, use:
$$
\Sigma_i = M_{2\mathrm{D},i}^T M_{2\mathrm{D},i} + \epsilon I
$$
or equivalently take the matrix of squared components. This preserves directional information (eigenvectors) while ensuring positive definiteness. The regularization $\epsilon I$ ($\epsilon \sim 10^{-6}$) prevents degeneracy.

**Alternative**: Use $|M_{2\mathrm{D}}|$ (replace eigenvalues with their absolute values while keeping eigenvectors). This is more physically interpretable as it preserves the orientation of the P and T axes.

### 3.4 Scaling

The moment tensor components have units of N·m ($\sim 10^{16}\text{–}10^{22}$) while spatial coordinates are in km ($\sim 10^2$). A normalization is essential:
$$
\Sigma_i^{\mathrm{norm}} = \frac{\Sigma_i}{\mathrm{median}(\lambda_{\max}(\Sigma_i))} \cdot s^2
$$
where $s$ is a characteristic spatial scale (e.g., median nearest-neighbor distance in km). This ensures the anisotropy shape is preserved while the ellipsoid size is commensurate with inter-event spacing.

## 4. Ellipsoid Construction

- **Centers**: $(x_{E,i}, x_{N,i}) \in \mathbb{R}^2$ — local Cartesian coordinates (km)
- **Covariances**: $\Sigma_i^{\mathrm{norm}}$ — normalized SPD tensor from §3.3–3.4
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

## 6. Visualization

### 6.1 Seismotectonic Context
- Map of epicenters with **beach ball** (focal mechanism) glyphs, using `obspy.imaging.beachball`.
- Overlay ellipsoid glyphs from the normalized $\Sigma_i$ at each epicenter.
- Plate boundary traces (from publicly available plate boundary datasets) for reference.

### 6.2 Persistence Comparison
- Side-by-side **persistence diagrams** (H0, H1) for anisotropic vs isotropic PH.
- Side-by-side **barcode plots**.
- Summary statistics table.

### 6.3 Fault Cluster Map (H0)
- At a chosen persistence threshold, color epicenters by H0 connected component.
- **Expected**: Anisotropic clusters correspond to mechanically distinct fault segments (e.g., outer-rise normal faulting vs megathrust reverse faulting); isotropic clusters are spatially compact.

### 6.4 Plate Boundary Detection (H1)
- Identify the most persistent H1 representative cycles.
- Plot the cycle edges on the map to see if they trace the plate boundary or outline locked asperities.

## 7. Expected Results

| Feature | Isotropic PH | Anisotropic PH (EllPHi) |
|---------|-------------|------------------------|
| H0 clusters | Spatial proximity groups | Fault-mechanism-aligned groups |
| Thrust vs normal faulting | Mixed in same cluster | Separated into distinct components |
| H1 persistent cycles | Reflect spatial gaps in seismicity | Reflect directional coherence along plate boundary |
| Tohoku aftershock sequence | Single dense blob | Differentiated by along-strike vs down-dip mechanism variation |

## 8. Dependencies

- **Core**: `ellphi>=0.1.2`, `numpy`, `scipy`, `ripser`, `matplotlib`
- **New**: `obspy>=1.4.0` (NDK parsing, focal mechanism plotting)
- **Installation**: Add `obspy` to `pyproject.toml` as optional dependency group `[seismo]`.

## 9. Output Artifacts

- `notebooks/earthquake_cmt/earthquake_cmt.py` — marimo notebook
- `public/earthquake_cmt_map.pdf` — Epicenter map with beach balls and ellipsoid glyphs
- `public/earthquake_cmt_persistence.pdf` — PD/barcode comparison
- `public/earthquake_cmt_clusters.pdf` — H0 fault cluster map

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Moment tensor indefiniteness → invalid covariance | Use $M^T M$ or absolute-eigenvalue construction (§3.3) |
| Extreme magnitude range ($M_w$ 5–9) → dominant large events | Normalize each tensor by its own $M_0$ before constructing $\Sigma$ |
| Sparse data in some subregions | Use full 1976–2024 catalog; lower $M_w$ threshold to 4.5 if needed |
| ObsPy NDK parsing memory for full catalog | Stream-process or pre-filter the NDK file by region before parsing |
| `obspy` is a heavy dependency (~200 MB) | Provide alternative: pre-extracted CSV with tensor components |

## 11. Future Extensions

- **3D analysis**: Include depth as a third coordinate and use full $3 \times 3$ moment tensor.
- **Temporal persistence**: Track how H0/H1 features evolve over time (e.g., before/after the 2011 Tohoku earthquake).
- **Inverse problem**: Optimize ellipsoid parameters to maximize H1 persistence along known plate boundaries — a geophysical metric learning problem.
