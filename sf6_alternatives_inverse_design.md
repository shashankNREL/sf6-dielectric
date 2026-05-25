# SF6 Alternatives: Multi-Property Pareto Optimisation for Inverse Molecular Design

A comprehensive reference covering group contribution methods, QSPR modelling, graph neural networks, and NSGA-II-based inverse design for discovering environmentally friendly replacements for SF6 in medium and high voltage switchgear.

---

## Table of Contents

1. [Background: Why Replace SF6?](#1-background-why-replace-sf6)
2. [Group Contribution Methods for Property Prediction](#2-group-contribution-methods-for-property-prediction)
3. [ICAS / ProCAMD Suite from DTU](#3-icas--procamd-suite-from-dtu)
4. [QSPR for Dielectric Strength](#4-qspr-for-dielectric-strength)
5. [Open Source Descriptor Tools](#5-open-source-descriptor-tools)
6. [Graph Neural Networks and Message Passing](#6-graph-neural-networks-and-message-passing)
7. [Inverse / Adjoint Molecular Design](#7-inverse--adjoint-molecular-design)
8. [Multi-Property Pareto Optimisation Pipeline](#8-multi-property-pareto-optimisation-pipeline)
9. [Algorithm Deep Dive: NSGA-II](#9-algorithm-deep-dive-nsga-ii)
10. [SELFIES Chemical Space Encoding](#10-selfies-chemical-space-encoding)
11. [Objective Functions and Constraints](#11-objective-functions-and-constraints)
12. [Hypervolume Indicator and Candidate Prioritisation](#12-hypervolume-indicator-and-candidate-prioritisation)
13. [Active Learning Loop](#13-active-learning-loop)
14. [Software Stack and Installation](#14-software-stack-and-installation)
15. [Full Pipeline Code Reference](#15-full-pipeline-code-reference)

---

## 1. Background: Why Replace SF6?

SF6 (sulfur hexafluoride) has been the dominant insulating and arc-quenching gas in medium and high voltage switchgear for over 50 years. Its key performance metrics are:

| Property | SF6 Value | Significance |
|---|---|---|
| Dielectric strength | ~89 kV/cm at 1 bar | Arc quenching, insulation |
| GWP (100-yr) | 23,500 | Primary driver for replacement |
| Boiling point | −63.9°C | Usable at low temperatures |
| Thermal conductivity | ~0.013 W/m·K | Arc cooling |
| Chemical stability | Very high | Long service life |

SF6's GWP of 23,500 — the highest of any industrial gas — has triggered regulatory action worldwide. California mandated a phase-out in gas-insulated switchgear by 2025. The EU F-Gas Regulation is progressively restricting use. This has driven intensive research into alternatives.

### Promising Alternative Families

- **Perfluoronitriles** (e.g. C4F7N — perfluoroisobutyronitrile): high electron affinity from C≡N group, GWP ~2090, used as ~10% blend with CO2 (3M Novec 4710 / ABB g³)
- **Perfluoroketones** (e.g. C5F10O): C=O group strongly captures electrons, very low GWP (~1), higher boiling points require buffer gas
- **Fluoroiodides** (CF3I, C2F5I): iodine's high polarizability gives excellent dielectric strength, very low GWP, but toxicity of decomposition products is a concern
- **Trifluoromethylsulfonyl fluoride** (CF3SO2F): dielectric strength 1.3–1.6× SF6, GWP 3678, liquefaction temperature −22°C

---

## 2. Group Contribution Methods for Property Prediction

Group contribution (GC) methods estimate thermophysical and environmental properties from molecular building blocks before synthesis, enabling rational screening of large candidate spaces.

### Core GC Frameworks

**Joback–Reid Method**
Workhorse for estimating boiling point, critical properties, heat of vaporisation, and ideal gas heat capacity from functional groups. Use for screening candidates for appropriate volatility and thermal stability windows.

**UNIFAC / Modified UNIFAC (Dortmund)**
Excellent for estimating activity coefficients and mixture VLE behaviour — critical for designing binary mixtures (e.g. fluoronitrile + CO2 or N2 buffer gas blends).

**Marrero–Gani Method**
A three-level GC approach (first-order → second-order → third-order groups) giving better accuracy for complex fluorinated molecules. Particularly useful for:
- Normal boiling point
- Melting point (critical for low-temperature operation)
- Gibbs energy of formation (thermal stability proxy)

**GC Methods for Dielectric Strength (Specialised)**
Dielectric strength lacks a universal GC method. Research groups at ETH Zurich and TU Graz have developed QSPR models using molecular descriptors that overlap significantly with GC approaches. Key molecular features correlating with high dielectric strength:
- High electron affinity (electron-capturing halogens: F > Cl)
- High molecular weight / polarizability
- C–F bond density
- Branching and spherical symmetry

**Environmental GC Models**
- GWP estimation: GC-based atmospheric lifetime models (tropospheric OH reactivity via Kwok–Atkinson method) combined with radiative forcing estimates from bond dipole moments
- ODP: Cl/Br count with stratospheric reactivity corrections

### Design Workflow

```
1. Define molecular family
       ↓
   Fluoroketones (C=O electron capture)
   Fluoronitriles (C≡N electron capture)
   Fluoroolefins (C=C + F)
   Hydrofluoroethers

2. Enumerate candidate structures
       ↓
   Fix carbon backbone length (C4–C6 sweet spot for BP)
   Apply GC property estimation at each step

3. Screen with GC methods
       ↓
   Joback/Marrero-Gani → BP, Tm, Tc, Pc
   UNIFAC → mixture VLE with CO2/N2
   Kwok-Atkinson → atmospheric lifetime → GWP proxy
   QSPR → dielectric strength rank

4. Rank candidates on Pareto front
       ↓
   Dielectric strength vs GWP vs boiling point

5. Validate top candidates
       ↓
   DFT (electron affinity, bond dissociation energies)
   Molecular dynamics (thermal arc behaviour)
   Lab synthesis + testing
```

### Key Limitations

- GC methods for dielectric breakdown are far less mature than for thermodynamic properties — QSPR or DFT is needed to fill this gap
- Decomposition product toxicity (e.g. HF, PFAS fragments) is hard to predict purely from GC; supplement with bond dissociation energy calculations (DFT)
- GC accuracy degrades for highly branched or strained molecules — exactly the interesting SF6-like geometries
- EU PFAS restrictions are progressively limiting the fluorocarbon design space regardless of GWP

---

## 3. ICAS / ProCAMD Suite from DTU

ICAS (Integrated Computer Aided System) is a suite of toolboxes from DTU's CAPEC group (Computer Aided Process-Product Engineering Center), developed primarily by Professor Rafiqul Gani. Now maintained under the KT Consortium.

### Core Components

**1. CAPEC Database**
Curated database of pure component properties used as the starting point for candidate searches. Recommended workflow begins with an initial database search to generate a shortlist before running the generative design step.

**2. ProPred — Property Prediction**
Handles pure component estimation using revised Marrero–Gani parameters with uncertainty estimates. Also covers mixture evaluation via UNIFAC-CI parameters and routines for equations of state (CPA, PC-SAFT).

**3. ProCAMD — Computer-Aided Molecular Design**
The design engine. Given a set of building blocks and specified target property ranges, it generates molecular structures that satisfy them. Four stages:
- Problem formulation — define target properties and acceptable ranges
- Initial search — scan the CAPEC database for existing candidates
- Generate and test — enumerate and evaluate new molecular structures using GC property models
- Verification — assess shortlisted candidates in their application context

**4. SolventPro**
Solvent selection tool. The same constrained molecular search methodology maps directly to dielectric gas design — both are multi-constraint property optimisation problems.

### Application to SF6 Replacement

**Step 1 — Problem formulation:**
```
Boiling point:    −50°C to +10°C
GWP:              < 1000
Electron affinity: > threshold
Gibbs energy:     negative (stable)
Melting point:    < −50°C
```

**Step 2 — Building block selection:**
Restrict to fluorinated families: CF3, CF2, C≡N, C=O, C=C

**Step 3 — Generate and test:**
ProCAMD narrows an enormous chemical space down to manageable shortlists.

**Step 4 — Verification:**
Use ICASSIM or external process simulation to model arc quenching behaviour, thermal recovery, and mixture VLE.

### Access
ICAS operates on a KT Consortium membership model. Contact: www.kt.dtu.dk/research/kt-consortium. Academic licenses have historically been available at low cost for research.

### Limitations for SF6 Work
- Dielectric strength is not a native property in ProPred's GC model set — must import or build a QSPR model externally
- Built primarily for solvent design; property library skews toward VLE and thermodynamic properties
- For arc physics and thermal conductivity under plasma conditions, supplement with DFT calculations (Gaussian, ORCA) or MD simulations

---

## 4. QSPR for Dielectric Strength

### Physical Basis

The fundamental physics of gas-phase breakdown is electron avalanche suppression. A good insulating gas must:
1. **Capture free electrons** — high electron affinity / electronegativity
2. **Resist ionisation** — high ionisation potential
3. **Present a large collision cross-section** — molecular size / polarizability
4. **Dissipate energy** at the molecular surface — electrostatic potential topology

### Descriptor Families

#### GIPF Descriptors (Generalized Interaction Properties Function)
Computed on the electron density isosurface (typically 0.001 a.u.):

| GIPF Descriptor | Symbol | Physical Meaning |
|---|---|---|
| Molecular surface area | S | Collision cross-section proxy |
| Variance of ESP | σ²_total | Electrostatic "roughness" |
| Positive surface area | S⁺ | Electrophilic exposure |
| Negative surface area | S⁻ | Nucleophilic exposure |
| Mean positive potential | V̄⁺_S | Electron donation sites |
| Mean negative potential | V̄⁻_S | Electron capture sites |
| Balance parameter | ν | Charge balance measure |

#### Reactivity Descriptors (Frontier Molecular Orbital Based)
- **Electron affinity (EA)** — E(neutral) − E(anion): direct measure of electron capture ability
- **Ionisation potential (IP)** — resistance to electron stripping
- **HOMO energy** — correlates with ionisation
- **LUMO energy** — correlates with electron capture
- **Chemical hardness** η = (IP − EA)/2
- **Electronegativity** χ = (IP + EA)/2

#### Local Property Descriptors
Computed pointwise on the electron density surface:
- **Local electron affinity** A(r) — identifies regions that capture electrons
- **Local ionisation energy** I(r) — identifies where ionisation is easiest
- **Fukui functions** f⁺(r), f⁻(r) — reactivity maps for electrophilic/nucleophilic attack

#### Topological / Whole-Molecule Descriptors
Using PM6 semiempirical method (MOPAC + Codessa): up to 951 descriptors covering constitutional, topological, geometrical, electrostatic, quantum-chemical, and thermodynamic descriptor classes. Faster than full DFT; enables screening of larger libraries.

### Model Architectures

**Multiple Linear Regression (MLR)**
Early QSPR models used 3–5 descriptors in linear equations. Interpretable but limited accuracy.

**Gradient Boosting / Random Forest**
Comparison of statistical vs ML models (73 molecules) showed gradient boosting decision trees perform best overall in accuracy and stability.

**Neural Networks**
A neural network QSPR model trained on 65 insulating gas molecules achieved R² = 0.9951 on the test set after genetic function algorithm feature selection reduced 951 raw descriptors to 12 key descriptors.

### Practical Workflow

```
Step 1: Build training dataset
  └─ ~50–100 molecules with experimental dielectric strength values
     (reduced E/N or E at 1 bar relative to SF6 = 1.0)

Step 2: Generate 3D molecular geometries
  └─ SMILES → 3D optimize with RDKit/Open Babel
     → DFT geometry optimization (B3LYP/6-311+G**)

Step 3: Calculate descriptors
  ├─ DFT route: Gaussian/ORCA → Multiwfn
  │   for GIPF, local EA, Fukui functions
  └─ Fast route: MOPAC (PM6/PM7) → Codessa
      for 900+ constitutional/topological/quantum-chemical descriptors

Step 4: Feature selection
  └─ Genetic algorithm, LASSO, or correlation analysis
     → reduce to 5–15 most predictive descriptors

Step 5: Train and validate model
  └─ Split 80/20 train/test
     Cross-validate with LOO or k-fold
     Report R², Q², RMSE, MAE

Step 6: Applicability domain
  └─ Williams plot or leverage statistics h* = 3p/n

Step 7: Screen candidates
  └─ Apply model to generated structures; rank by predicted DS
```

### Software for QSPR

| Tool | Role |
|---|---|
| Gaussian 16 / ORCA | DFT geometry optimisation and ESP calculation |
| Multiwfn | Post-processing: GIPF descriptors, local EA/IP, Fukui functions |
| MOPAC (PM6/PM7) | Fast semiempirical descriptors for large libraries |
| Codessa / Codessa Pro | Automated descriptor generation + MLR model building |
| Dragon (Talete) | Large commercial descriptor suite (2000+ descriptors) |
| RDKit | Open-source Python: 2D/3D descriptors, fingerprints, SMILES handling |
| scikit-learn | Random forest, gradient boosting, cross-validation in Python |

---

## 5. Open Source Descriptor Tools

### Core Stack

**RDKit**
The foundational open-source cheminformatics library. Handles SMILES parsing, 3D geometry generation (ETKDGv3), Gasteiger charges, Morgan fingerprints, EState indices, topological descriptors, TPSA, Labute ASA.

```bash
pip install rdkit
```

**Mordred**
Molecular descriptor calculator built on RDKit. Computes 1800+ two- and three-dimensional descriptors. BSD licensed (commercial and non-commercial use). Can be used from CLI, web UI, or Python library.

```bash
pip install mordred
```

Typical preprocessing pipeline:
- Generate `.mol` files via RDKit
- Compute all descriptors with Mordred
- Remove non-numeric features
- Apply collinearity threshold (≥95%) to eliminate redundant descriptors
- Reduces 1826 raw descriptors to ~1200 usable ones for a typical dataset

**DOPtools (2025)**
Newer Python library unifying RDKit fingerprints (Morgan, Avalon, atom pairs, topological torsion) and Mordred 2D descriptors under a single scikit-learn-compatible `Transformer` interface.

```bash
pip install doptools
```

**DFT-derived descriptors (open-source stack)**
- ORCA (free for academics): geometry optimisation and wavefunction calculation
- Multiwfn (free): post-processing of wavefunction files for GIPF, local EA/IP, Fukui functions

### Physics-Informed Feature Set for Dielectric Applications

The following RDKit-derived features are most predictive for dielectric strength:

| Feature | Descriptor | Physical Relevance |
|---|---|---|
| n_F, frac_F | Fluorine count / fraction | Primary electronegativity source |
| n_N, n_O | Heteroatom counts | Electron-withdrawing groups |
| min_charge | Min Gasteiger partial charge | Electron capture site |
| min_estate | Min EState index | LUMO energy proxy |
| mol_wt, labuteASA | Molecular weight, surface area | Collision cross-section |
| kappa1, kappa2 | Topological shape indices | Molecular geometry |
| Morgan FP (r=2) | Substructure bits | Local chemical environment |

---

## 6. Graph Neural Networks and Message Passing

### Why GNNs Over Hand-Crafted Descriptors

Molecules are naturally graphs: atoms are nodes, bonds are edges. GNNs learn the representation from data rather than requiring expert feature engineering. Until recently, ML methods for molecular properties relied on hand-crafted feature descriptors such as Coulomb matrices, bag-of-bonds, and fingerprints. With large molecular databases, graph-based message passing models that learn directly from structure have become dominant.

### Message Passing Mechanism

In a message passing neural network (MPNN), each atom aggregates information from bonded neighbours over N rounds:

1. **Initialise** — each node gets a feature vector: atom type, formal charge, degree, aromaticity, hybridisation. Each edge gets bond features: bond type, conjugation, ring membership.
2. **Message** — each node passes its current hidden state to neighbours through a learned function.
3. **Aggregate** — each node sums (or takes mean/max/attention-weighted combination of) all incoming messages.
4. **Update** — each node updates its hidden state with a GRU or MLP applied to the aggregated message.
5. **Readout** — after N rounds, all node states are pooled into a single graph-level vector, passed to a regressor for the property.

### Chemprop: Directed MPNN (D-MPNN)

Chemprop is the leading open-source directed MPNN framework for molecular property prediction. Architecture:

1. **Local features encoding** — atom and bond feature vectors
2. **D-MPNN** — directed message passing to learn atomic embeddings (each bond treated as two directed edges to capture directionality)
3. **Aggregation** — atomic embeddings → molecular embedding
4. **Feed-forward network (FFN)** — molecular embedding → target properties

Key capabilities:
- Multi-task learning: simultaneous prediction of DS, BP, GWP from one model
- Uncertainty quantification: MC dropout, deep ensembles
- Transfer learning from pretrained models
- Augmentation with external RDKit/Mordred descriptors concatenated to the molecular embedding

```bash
pip install chemprop
```

Training:
```bash
chemprop train \
  --data-path train.csv \
  --task-type regression \
  --target-columns ds_rel bp_c gwp100 \
  --save-dir model/ \
  --epochs 100 \
  --features-generator rdkit_2d_normalized
```

### Physics-Informed Node Features for Dielectric Applications

The most physically meaningful features to include as node attributes:
- Atom electronegativity
- Formal charge
- Partial charge (from GFN2-xTB pre-pass)
- HOMO/LUMO orbital coefficients per atom
- Bond polarity

These encode exactly the electron-capture physics that governs dielectric breakdown.

---

## 7. Inverse / Adjoint Molecular Design

Three routes exist for inverting a trained GNN property predictor to generate molecules with target properties.

### Route 1: Direct Gradient Ascent (Adjoint Method)

The most direct analogue of adjoint/inverse methods from PDE-constrained optimisation. By exploiting the differentiability of GNNs, gradient ascent can be performed with GNN weights frozen to directly optimise the molecular graph input toward a target property.

**Key trick:** The adjacency matrix is constructed from a weight vector containing N(N−1)/2 elements. These elements are squared and populated in an upper triangular matrix, then added to its transpose to produce a positive symmetric matrix — this keeps optimisation in valid graph space while remaining differentiable.

**Advantages:**
- No additional training required
- Produces the most diverse set of molecules
- DFT-verified performance comparable to generative models
- Most direct analogue to classical adjoint optimisation

### Route 2: VAE Latent Space Optimisation

Variational autoencoders convert discrete molecular representations into continuous latent spaces. When jointly trained with property predictors, they allow gradient-based optimisation in the latent space. Key implementations:
- **JT-VAE** — junction tree variational autoencoder
- **SELFIES-VAE** — VAE over SELFIES representation (100% valid output)
- **PASITHEA** — gradient-based method directly reversing the learning process using SELFIES

**Why SELFIES matters here:** With SMILES, large areas of the latent space correspond to invalid molecules. SELFIES produces 100% valid strings, overcoming this problem during generative model sampling.

### Route 3: Reinforcement Learning / Generative Models

A combined SELFIES-VAE + GNN + RL approach integrates generative AI, predictive GNN modelling, and reinforcement learning. The GNN handles property prediction (eliminating hand-crafted QSPR descriptors) while the RL policy learns to navigate molecular space toward target properties.

Key tools: **JANUS**, **REINVENT**, **MARS**, **GuacaMol**

---

## 8. Multi-Property Pareto Optimisation Pipeline

### Problem Structure

Three objectives must be optimised simultaneously — but they conflict:
- High dielectric strength ↔ tends to come with high boiling point
- Low GWP ↔ often correlates with weaker insulation
- Low boiling point ↔ may compromise molecular size/electronegativity

No single molecule can be globally optimal. Instead we seek the **Pareto front**: the set of molecules where no objective can improve without another getting worse.

### Pipeline Overview

```
Dataset (~80 fluorinated molecules)
          ↓
Feature generation (RDKit + Mordred)
          ↓
Surrogate training (GBR, one per objective)
          ↓ Cross-validate (5-fold)
NSGA-II optimisation over SELFIES space
          ↓ 100 generations, pop=80
Pareto front post-processing
          ↓ HV contribution ranking
Applicability domain filter
          ↓
Active learning: top-N for DFT validation
          ↓
Retrain surrogates (active learning loop)
```

### Surrogate Models

**Gradient Boosted Regression (small dataset < 200 molecules)**
Three independent GBR models (one per objective). Advantages over deep networks on small datasets:
- Better generalisation on sparse data
- No hyperparameter sensitivity
- Naturally provides feature importance

**Chemprop D-MPNN (larger dataset > 200 molecules)**
Multi-task learning over all three objectives simultaneously. Better generalisation to structurally novel molecules. Transfer learning from QM9 or other large molecular databases helps further.

**Applicability Domain**
Leverage-based check: h* = 3p/n where p = number of features, n = training set size.
- h_i = x_i^T (X^T X)^{−1} x_i
- Candidates with h > h* are flagged as out-of-domain and deprioritised

---

## 9. Algorithm Deep Dive: NSGA-II

### What NSGA-II Is

NSGA-II (Non-dominated Sorting Genetic Algorithm II, Deb et al. 2002) is an evolutionary algorithm for multi-objective optimisation. It maintains a population of candidate solutions and improves them over generations through selection, crossover, and mutation — exactly like a biological population evolving under selection pressure.

It is the standard approach for 2–3 objective problems with constraints. For 4+ objectives, NSGA-III (also in pymoo) with reference directions provides better coverage.

### Generation Loop

```
Initialise population P (size N)
          ↓
For each generation:
  ├─ Evaluate all candidates with surrogate models
  ├─ Non-dominated sorting → assign ranks 1,2,3,...
  ├─ Crowding distance within each rank
  ├─ Selection: rank-first, then crowding distance
  ├─ Crossover: SELFIES token tail swap
  ├─ Mutation: SELFIES token replace/insert/delete
  └─ New population Q (size N)
  Combine P ∪ Q → select best N → new P
```

### Non-Dominated Sorting

**Definition:** Molecule A *dominates* molecule B if:
- A is at least as good as B on **every** objective, AND
- A is strictly better than B on **at least one** objective

**Algorithm:**
1. Find all non-dominated molecules in the population → **Rank 1** (current Pareto front)
2. Remove rank-1 molecules; find non-dominated subset of remainder → **Rank 2**
3. Repeat until all molecules assigned a rank
4. Rank 1 always preferred over rank 2, rank 2 over rank 3, etc.

**Key property:** No rank-1 molecule can improve on any objective without worsening another.

### Crowding Distance

Within the same rank, NSGA-II prefers molecules that are spread out along the front, not clustered together.

**Computation:**
1. For each objective separately, sort the rank-1 molecules by their value
2. Each molecule's crowding distance = sum over all objectives of (right_neighbour_value − left_neighbour_value) / (objective_range)
3. Boundary molecules get infinite distance (they are always preferred)
4. Geometrically: measures the perimeter of the bounding box formed by a molecule's two nearest neighbours in objective space

**Why it matters:** Prevents the algorithm from converging on one corner of the Pareto front. Ensures broad, uniform coverage of all trade-offs.

### Selection Mechanism

Molecules are ranked by a binary tournament:
1. If different ranks: lower rank wins
2. If same rank: higher crowding distance wins

This ensures that:
- Pareto-optimal molecules are always carried forward
- Among equally good molecules, isolated ones (high CD) are preferred over clustered ones

### Genetic Operators on SELFIES

**Crossover (probability 0.9):**
```
Parent 1: [C][F][C][#N][F]
Parent 2: [C][=C][F][F][C]
                ↑ cut point at position 3
Child 1:  [C][F][C][F][C]   (P1[:3] + P2[3:])
Child 2:  [C][=C][F][#N][F] (P2[:3] + P1[3:])
```

**Mutation (probability 0.3):**
Three equally likely operations on a random token position:
- **Replace**: swap one token for a random alphabet token
- **Insert**: insert a new token (if below max length)
- **Delete**: remove a token (if above min length)

Because SELFIES guarantees every string decodes to a valid molecule, no fitness penalty for invalid molecules — every offspring can be evaluated immediately.

---

## 10. SELFIES Chemical Space Encoding

### Why SELFIES Instead of SMILES

With SMILES, large regions of the latent / gene space correspond to invalid molecules — strings that parse to no valid structure. SELFIES (Self-Referencing Embedded Strings) overcomes this: every valid SELFIES string decodes to a valid molecule, with 100% validity guaranteed by the encoding rules.

### Encoding Scheme

Each molecule is represented as an integer array of token indices (the gene vector). The alphabet is filtered to relevant atoms for this application:

```python
RELEVANT_ATOMS = {"C", "F", "N", "O", "S", "I", "H"}
# Fluorine-containing tokens oversampled 3× to bias toward high-DS candidates
```

Maximum sequence length: 20 tokens (appropriate for C4–C6 fluorinated molecules).

### SMILES → SELFIES → Gene Array

```
CF3C#N  (SMILES)
   ↓
[C][Branch1][Branch2][C][#N][F][F]  (SELFIES)
   ↓
[3, 8, 9, 3, 5, 2, 2]  (integer gene array, padded to max_len=20)
```

The integer gene array is what NSGA-II manipulates. The alphabet index lookup handles encoding/decoding.

---

## 11. Objective Functions and Constraints

### Formulation for pymoo (All Minimisation)

pymoo minimises all objectives. Three physical objectives are reformulated:

| Physical objective | pymoo formulation | Reason |
|---|---|---|
| Maximise dielectric strength | f1 = −DS | Negate to flip direction |
| Minimise boiling point | f2 = BP (°C) | Already minimisation |
| Minimise GWP | f3 = log₁₀(GWP) | Log scale: GWP spans 4 orders of magnitude |

### Inequality Constraints (g ≤ 0 = satisfied)

| Constraint | Formulation | Physical meaning |
|---|---|---|
| DS ≥ 0.7 × SF6 | g1 = 0.7 − DS | At least 70% of SF6 dielectric strength |
| BP ≤ 20°C | g2 = BP − 20 | Gaseous at ambient temperature |
| GWP ≤ 5000 | g3 = GWP − 5000 | Well below SF6's 23,500 |

pymoo ranks infeasible molecules (any g > 0) after all feasible ones — a penalty-free constraint handling approach built into NSGA-II.

### Surrogate Evaluation Inside `_evaluate()`

```python
def _evaluate(self, x, out, *args, **kwargs):
    smiles = self._genes_to_smiles(x)         # decode gene → SMILES
    feats  = self.feature_fn(smiles)           # RDKit/Mordred features
    preds  = self.surrogate.predict_single(x)  # GBR surrogate prediction
    ds, bp, gwp = preds[0], preds[1], preds[2]

    f1 = -float(ds)                        # negate DS
    f2 = float(bp)
    f3 = float(np.log10(max(gwp, 1.0)))

    g1 = 0.7 - ds          # DS constraint
    g2 = bp - 20.0         # BP constraint
    g3 = gwp - 5000.0      # GWP constraint

    out["F"] = [f1, f2, f3]
    out["G"] = [g1, g2, g3]
```

---

## 12. Hypervolume Indicator and Candidate Prioritisation

### Hypervolume Indicator

The hypervolume (HV) measures the volume of objective space that is dominated by the Pareto front, bounded by a reference point (typically set slightly beyond the worst values):

```
HV = volume of { y ∈ R^k : ∃x ∈ Pareto front, x dominates y, y ≤ ref_point }
```

A single scalar summarising the quality of the entire front. Larger HV = better front (closer to ideal, more spread).

### Hypervolume Contribution

Each molecule's contribution = how much total HV shrinks if that molecule is removed:

```
HV_contribution(i) = HV(front) - HV(front \ {i})
```

**Geometric interpretation:**
- **Corner molecules** (convex hull vertices) cover objective space no other molecule reaches → high contribution
- **Interior / clustered molecules** contribute near-zero — their dominated volume is already covered by their neighbours

### Pareto Front Visualisation (Three Projections)

The three-panel output plot from the pipeline:

1. **DS vs Boiling Point** (coloured by log GWP): shows the fundamental trade-off between insulation performance and liquefaction risk
2. **DS vs GWP** (coloured by BP): shows the environmental performance landscape
3. **DS vs BP** (sized/coloured by HV contribution): highlights which candidates are most unique

### Reference Points for Known Candidates

| Molecule | DS (rel. SF6) | BP (°C) | GWP | Notes |
|---|---|---|---|---|
| SF6 | 1.00 | −63.8 | 23,500 | Reference |
| C4F7N | 1.90 | 0.0 | 2,090 | Commercial (Novec 4710) |
| C5F10O | 1.75 | 27.0 | 1 | Novec 5110 type |
| CF3I | 1.80 | −22.5 | 1 | High DS, decomp. concerns |
| C2F5I | 2.10 | 13.0 | 1 | Best known DS, low GWP |
| CF3SO2F | 1.55 | −22.0 | 3,678 | Promising 2024 candidate |

---

## 13. Active Learning Loop

After DFT validation of the top Pareto candidates, the surrogate models are retrained with the new data, and NSGA-II re-runs. This closed loop progressively improves both the property models and the candidate quality.

### Acquisition Function

Combined score for selecting which Pareto candidates to send for DFT:

```
score(i) = 0.6 × HV_contribution_normalised(i)
         + 0.4 × exploration_bonus_normalised(i)
         × domain_penalty(i)

where:
  exploration_bonus(i) = min distance from candidate i to all training points
  domain_penalty(i) = 1 if inside applicability domain, 0 if outside
```

This balances:
- **Exploitation** (HV contribution): prioritise Pareto-unique candidates
- **Exploration** (distance from training data): prioritise candidates in unexplored chemical space
- **Safety** (AD filter): deprioritise candidates the model is unreliable for

### Loop Structure

```
Round 0: Seed dataset (80 molecules, literature values)
  ↓ Train surrogates → NSGA-II → Pareto front → top-5 HV contrib
  ↓ DFT calculation (ORCA + Multiwfn)

Round 1: 85 molecules
  ↓ Retrain surrogates → NSGA-II → updated Pareto front → top-5
  ↓ DFT calculation

Round N: convergence criterion (HV improvement < threshold)
  → Final candidate list for synthesis
```

---

## 14. Software Stack and Installation

### Full Installation

```bash
pip install rdkit mordred selfies pymoo chemprop torch \
            pandas numpy scikit-learn matplotlib seaborn tqdm joblib
```

### Tool Reference

| Tool | Role | Source |
|---|---|---|
| RDKit | Molecular handling, 2D/3D descriptors, fingerprints | `pip install rdkit` |
| Mordred | 1800+ 2D/3D descriptors | `pip install mordred` |
| SELFIES | 100%-valid molecular encoding | `pip install selfies` |
| Chemprop | D-MPNN property prediction | `pip install chemprop` |
| PyTorch Geometric | Graph ML framework | `pip install torch_geometric` |
| pymoo | NSGA-II/III multi-objective optimisation | `pip install pymoo` |
| ORCA | DFT geometry optimisation (free academic) | orca-forum.kofo.mpg.de |
| Multiwfn | Wavefunction analysis, GIPF descriptors | sobereva.com/multiwfn |
| MOPAC | Fast PM6/PM7 semiempirical descriptors | openmopac.net |
| scikit-learn | GBR surrogates, cross-validation | `pip install scikit-learn` |
| GuacaMol / REINVENT | RL-based generative molecular design | GitHub |

### Running the Pipeline

```bash
python sf6_pareto_design.py
```

Outputs to `output/`:
- `pareto_candidates.csv` — full decoded Pareto population with predicted properties
- `pareto_front.png` — three-panel Pareto front visualisation
- `dft_candidates.csv` — top-5 candidates selected by active learning acquisition function

### Upgrade Path by Dataset Size

| Dataset size | Recommended configuration |
|---|---|
| < 100 molecules | GBR surrogates + RDKit features + NSGA-II (as-is) |
| 100–300 molecules | Enable full Mordred features + MC dropout uncertainty |
| > 300 molecules | Switch to Chemprop D-MPNN + transfer learning from QM9 |
| Any size | Upgrade to NSGA-III for 4+ objectives (add toxicity, decomposition energy) |

---

## 15. Full Pipeline Code Reference

The complete code (`sf6_pareto_design.py`) is structured in 11 sections:

| Section | Content |
|---|---|
| 1 | Embedded seed dataset (80 fluorinated molecules, SMILES + DS/BP/GWP) |
| 2 | `smiles_to_rdkit_features()` — physics-informed RDKit descriptor generation |
| 2 | `build_mordred_features()` — full Mordred descriptor pipeline with pruning |
| 3 | `SurrogateEnsemble` — GBR models, cross-validation, applicability domain |
| 4 | SELFIES alphabet construction, mutation/crossover operators |
| 5 | pymoo `Sampling`, `Mutation`, `Crossover` classes for SELFIES space |
| 6 | `SF6ReplacementProblem` — ElementwiseProblem with 3 objectives + 3 constraints |
| 7 | `decode_population()`, `hypervolume_contribution()` |
| 8 | `plot_pareto_front()` — three-panel matplotlib visualisation |
| 9 | `main()` — full pipeline: load → features → train → optimise → post-process |
| 10 | Chemprop D-MPNN integration script (for larger datasets) |
| 11 | `active_learning_round()` — acquisition function for DFT candidate selection |

### Key Design Decisions

**Why GBR over deep networks for small data?**
Gradient boosting handles sparse, noisy datasets better than neural networks. A 5-fold CV R² > 0.85 on 80 molecules is achievable with GBR; a neural network of similar depth would overfit.

**Why SELFIES over SMILES for the gene?**
SMILES mutations frequently produce invalid molecules (ring closures broken, valence violated). SELFIES guarantees 100% validity — every offspring from crossover or mutation can be evaluated immediately with no fitness penalty.

**Why log₁₀(GWP) as the third objective?**
GWP spans four orders of magnitude (1 to 23,500). Using raw GWP makes the optimiser insensitive to differences at the low end (e.g. GWP=1 vs GWP=100 — a 100× difference) while oversensitive at the high end. Log-scale balances the objective across the full range.

**Why leverage-based applicability domain?**
The surrogate models are trained on ~80 molecules. Extrapolation beyond the training set distribution produces unreliable predictions. The Williams plot / leverage statistic h* = 3p/n is the standard QSPR applicability domain criterion and is straightforward to implement.

---

## References and Further Reading

- Deb, K. et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
- Yu, X., Hou, H., Wang, B. (2017). Prediction on dielectric strength and boiling point of gaseous molecules for replacement of SF6. *Journal of Computational Chemistry*, 38(10), 721–729.
- Sun, H. et al. (2020). Prediction of the electrical strength and boiling temperature of the substitutes for greenhouse gas SF6 using neural network and random forest. *IEEE Access*, 8, 124204–124216.
- Heid, E. et al. (2023). Chemprop: A Machine Learning Package for Chemical Property Prediction. *Journal of Chemical Information and Modeling*.
- Krenn, M. et al. (2020). Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation. *Machine Learning: Science and Technology*, 1(4), 045024.
- Gani, R. et al. Computer-aided molecular design using the Marrero–Gani group contribution method. *Fluid Phase Equilibria* (various).
- Shimakawa, H. et al. (2025). Computational Exploration and Experimental Verification for Designing SF6 Alternatives. *IEEE Transactions on Dielectrics and Electrical Insulation*, 32(2), 667–673.
- Luo, L. et al. (2025). A prediction model for electrical strength of gaseous medium based on molecular reactivity descriptors and machine learning method. *Journal of Molecular Modeling*, 31(2).

---

*Generated from a technical deep-dive conversation on SF6 alternatives design, covering group contribution methods through to full multi-objective inverse molecular design. Last updated May 2026.*
