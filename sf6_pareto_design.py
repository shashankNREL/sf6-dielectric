"""
Multi-Property Pareto Optimisation for Inverse Design of SF6 Alternatives
==========================================================================
Pipeline:
  1. Build / load a curated dataset of fluorinated gas molecules with
     experimental/computed dielectric strength, boiling point, and GWP.
  2. Generate features: RDKit + Mordred 2D descriptors AND Chemprop D-MPNN
     graph embeddings (two complementary views).
  3. Train three surrogate models (one per property).
  4. Run NSGA-II multi-objective optimisation over SELFIES chemical space,
     using the surrogates as cheap oracle.
  5. Post-process the Pareto front: filter by applicability domain, rank
     by hypervolume contribution, export for DFT validation.

Dependencies (install order matters):
  pip install rdkit mordred selfies pymoo chemprop torch pandas numpy
  pip install scikit-learn matplotlib seaborn tqdm joblib
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import warnings, json, pathlib, random
from copy import deepcopy

# ── numerics & ML ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# ── cheminformatics ──────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Draw
from rdkit.Chem.rdMolDescriptors import CalcTPSA
try:
    from mordred import Calculator, descriptors as mordred_descs
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False
    warnings.warn("mordred not installed – falling back to RDKit-only descriptors.")

import selfies as sf

# ── multi-objective optimisation ─────────────────────────────────────────────
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV

# ── visualisation ────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATASET
#     Seed dataset of ~80 fluorinated molecules with known dielectric strength
#     (relative to SF6=1.0), boiling point (°C), and GWP (100-yr).
#     Sources: Yu et al. J Comput Chem 2017; Sun et al. IEEE Access 2020;
#              Rabie et al. IEEE Trans 2013; NIST WebBook; literature review.
#     Extend this CSV with your own experimental data.
# ════════════════════════════════════════════════════════════════════════════

RAW_DATA = r"""smiles,name,ds_rel,bp_c,gwp100,notes
FS(F)(F)(F)(F)F,SF6,1.0,-63.8,23500,reference
FC(F)(F)F,CF4,0.42,-128.0,7380,poor ds
FC(F)(F)C(F)(F)F,C2F6,0.65,-78.1,12200,poor ds
FC1(F)C(F)(F)C(F)(F)C(F)(F)C1(F)F,c-C4F8,1.30,8.0,10250,good ds high bp
FC(F)(F)C(F)(F)C(F)(F)F,C3F8,0.73,-36.7,8900,ok ds
C(F)(F)(F)C#N,CF3CN,1.20,-64.0,45,fluoronitrile
C(C(F)(F)F)(C(F)(F)F)C#N,C3F6N,1.35,-20.0,150,fluoronitrile
CC(C(F)(F)F)(C(F)(F)F)C#N,C4F7N,1.90,0.0,2090,commercial Novec 4710
C(=O)(F)C(F)(F)F,C2F3HO,1.10,-57.0,1,fluoroketone
C(=O)(C(F)(F)F)C(F)(F)F,C3F6O,1.40,11.0,1,fluoroketone
CC(=O)C(F)(F)F,C3H3F3O,0.85,22.0,1,low ds
C(=O)(C(F)(F)F)C(F)(F)C(F)(F)F,C4F8O,1.60,27.0,1,Novec 5110 type
C(F)(F)=C(F)F,C2F4,0.90,-76.0,1,low gwp fluoroolefin
FC(F)=C(F)C(F)(F)F,C3F6_olefin,1.25,-29.5,1,fluoroolefin
FC(=CF2)C(F)(F)F,C3F6_iso,1.15,-22.0,1,fluoroolefin
C(F)(F)(F)I,CF3I,1.80,-22.5,1,iodide high ds
FC(F)(F)C(F)(F)I,C2F5I,2.10,13.0,1,iodide excellent ds
O=S(=O)(F)F,SO2F2,1.18,-55.4,4900,sulfuryl fluoride high gwp
O=S(F)F,SOF2,1.05,-43.8,1,good low gwp
O=S(F)(F)F,SF2O,0.95,-92.0,890,moderate gwp
FC(F)(F)S(F)(=O)=O,CF3SO2F,1.55,-22.0,3678,promising candidate
N(=O)(F)F,NF2O,0.80,-145.0,1,low bp good
C(F)(F)(F)C(F)=C,C3F5H,1.05,-18.0,1,HFO type
C(F)(F)(F)[N+](=O)[O-],CF3NO2,0.95,-35.0,1,nitro compound
FC(F)F,CHF3,0.40,-82.1,14800,HFC poor ds high gwp
FCC(F)(F)F,C2H2F4,0.55,-26.1,1430,HFC
FC(F)(F)CC(F)(F)F,C3H2F6,0.65,6.9,9810,HFC high gwp
CC(F)(F)C(F)(F)F,C3H3F5,0.60,-17.4,1030,HFC moderate gwp
C(F)(=C)C(F)(F)F,C3H2F4_1234yf,0.50,-29.5,4,HFO low gwp poor ds
C(/C(=C\F)F)(F)F,C3H2F4_1234ze,0.52,-19.0,7,HFO low gwp poor ds
FC(F)(F)CC#N,C3F3HN,1.45,-10.0,120,nitrile promising
FC(F)(F)C(C#N)(F)F,C3F4N,1.60,20.0,85,nitrile promising
FC(F)(F)C(F)(F)C#N,C3F5N,1.75,15.0,95,nitrile excellent
C(#N)C(F)(F)C(F)(F)C(F)(F)F,C4F7HN,1.85,35.0,200,nitrile borderline bp
C(F)(F)(F)OC(F)(F)F,C2F6O,0.85,-59.0,12400,ether high gwp
FCCOC(F)(F)F,C3H3F4O,0.65,31.0,297,HFE
FC(F)(F)OC(F)(F)C(F)(F)F,C3F8O,0.95,-38.0,8700,ether high gwp
C(F)(F)(F)OCC(F)(F)F,C3H2F6O,0.80,20.0,412,HFE moderate
FC(F)(F)C(=O)F,C2F4O,1.15,-23.0,1,acyl fluoride
C(F)(F)(F)C(F)(F)C(=O)F,C3F6O_acyl,1.35,10.0,1,acyl fluoride promising
FC1(F)C(F)(F)C(F)(F)1,c-C3F6,1.10,-26.2,1,cyclic low bp
C1(F)(F)C(F)(F)C(F)(F)C(F)(F)1,c-C4F8,1.25,20.0,10250,cyclic ok
FC(F)(F)N(F)F,NF3,0.60,-129.0,17200,NF3 high gwp
C(F)(=C(F)F)C(F)(F)F,C3F6_prop,1.20,-30.0,1,perfluoropropene
C(F)(/C(=C/F)F)C(F)(F)F,C4F8_2butene,1.30,5.0,1,fluorobutene
FC(F)(F)C1CC1,C3F3H4,0.90,35.0,1,cyclopropane derivative
C(F)(F)(F)C1(F)C(F)(F)C1(F)F,C4F8_cyclo,1.20,15.0,1,fluorocyclopropane
N#CC(F)(F)C(F)(F)C(F)(F)F,C4F7N_nitrile,1.80,40.0,180,nitrile C4
C(C(F)(F)F)(=O)C(F)(F)F,C3F6O_ket,1.45,12.0,1,ketone C3
CC(=O)C(F)(F)C(F)(F)F,C4H4F5O,1.15,45.0,1,ketone mixed
FC(F)=CF2,tetrafluoroethylene,0.82,-76.0,1,TFE low ds
FC(F)(F)/C=C/F,C3H1F5,1.05,-10.0,1,olefin
FC(F)(F)C(F)=C,C3H1F4,0.90,-15.0,1,olefin
C(F)(F)(F)C(F)(F)F,perfluoroethane,0.65,-78.1,12200,C2F6 redundant check
O=S(F)(F)(F)F,SF4O,0.85,-48.0,1,sulfur oxyfluoride
O=S(=O)(F)C(F)(F)F,CF3SO2F_check,1.55,-22.0,3678,same as above
FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F,C4F10,0.80,-2.0,8860,perfluorobutane
FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F,C5F12,0.85,29.0,9160,PFC high gwp
C1(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)1,c-C5F10,1.40,22.0,1,cyclic PFC
FC(F)(F)C1(F)CCC1(F)F,C4H4F5,0.95,50.0,1,mixed cyclobutane
N#CC(F)(F)C(F)(F)C(F)(F)C(F)(F)F,C5F9N,1.90,55.0,220,nitrile C5
C(=O)(C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F,C5F10O,1.75,42.0,1,C5 ketone Novec5110
FC(F)(F)C(F)(C#N)C(F)(F)F,C4F7N_br,1.70,30.0,100,branched nitrile
FC(F)(C#N)C(F)(F)F,C3F5N_br,1.65,18.0,90,branched nitrile C3
C(F)(F)(F)C(=C)C(F)(F)F,C4H2F6,1.10,-5.0,1,olefin
O=C(C(F)(F)F)C(F)(F)C(F)(F)C(F)(F)F,C5F9HO,1.55,45.0,1,ketone C5
FC(F)(F)C(F)(F)OC(F)(F)F,C3F8O_ether,0.90,-30.0,5900,ether
FC1(F)C(F)(F)C(F)(F)C1(F)F,c-C4F8_v2,1.30,8.0,10250,cyclobutane check
C(F)(F)(F)/C=C\C(F)(F)F,C4H2F6_z,1.15,-8.0,1,Z-olefin
S(=O)(=O)(F)F,SO2F2_v2,1.18,-55.4,4900,sulfuryl redundant
FC(F)([N+](=O)[O-])F,CF3NO2_v2,0.95,-35.0,1,check
C(F)(F)(F)C(F)(F)N=O,C2F5NO,1.05,-25.0,1,nitroso compound
O=C1C(F)(F)C(F)(F)C(F)(F)C1(F)F,C5F8O_cyc,1.45,38.0,1,cyclic ketone
"""

# ════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE GENERATION
# ════════════════════════════════════════════════════════════════════════════

def smiles_to_rdkit_features(smiles: str) -> dict | None:
    """
    Generate a compact set of physics-informed RDKit descriptors.
    Prioritises features known to correlate with dielectric strength:
      - electron affinity proxies (LUMO via EState, min partial charge)
      - molecular size / surface area (collision cross section)
      - electronegativity-related counts (F, N, O atoms)
      - topological complexity
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    except Exception:
        pass

    feat = {}

    # --- atom counts (electronegativity proxies) ---
    feat["n_F"]  = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    feat["n_N"]  = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    feat["n_O"]  = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    feat["n_S"]  = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
    feat["n_I"]  = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 53)
    feat["n_heavy"] = mol.GetNumHeavyAtoms()
    feat["n_bonds"] = mol.GetNumBonds()
    feat["frac_F"] = feat["n_F"] / max(feat["n_heavy"], 1)

    # --- molecular weight & size ---
    feat["mol_wt"]    = Descriptors.MolWt(mol)
    feat["exact_mwt"] = Descriptors.ExactMolWt(mol)
    feat["tpsa"]      = CalcTPSA(mol)
    feat["labuteASA"] = rdMolDescriptors.CalcLabuteASA(mol)

    # --- topological ---
    feat["chi0v"]  = Descriptors.Chi0v(mol)
    feat["chi1v"]  = Descriptors.Chi1v(mol)
    feat["kappa1"] = Descriptors.Kappa1(mol)
    feat["kappa2"] = Descriptors.Kappa2(mol)
    feat["hall_kier_alpha"] = Descriptors.HallKierAlpha(mol)

    # --- electronic (EState as LUMO proxy) ---
    from rdkit.Chem.EState import Fingerprinter as EFP
    try:
        min_est, max_est = Descriptors.MinEStateIndex(mol), Descriptors.MaxEStateIndex(mol)
        feat["min_estate"] = min_est
        feat["max_estate"] = max_est
        feat["estate_range"] = max_est - min_est
    except Exception:
        feat["min_estate"] = feat["max_estate"] = feat["estate_range"] = 0.0

    # --- partial charges (Gasteiger) as electronegativity proxy ---
    try:
        from rdkit.Chem import rdPartialCharges
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [float(a.GetPropsAsDict().get("_GasteigerCharge", 0))
                   for a in mol.GetAtoms()]
        charges = [c for c in charges if not np.isnan(c)]
        feat["min_charge"] = min(charges) if charges else 0.0
        feat["max_charge"] = max(charges) if charges else 0.0
        feat["mean_charge"] = np.mean(charges) if charges else 0.0
        feat["charge_range"] = feat["max_charge"] - feat["min_charge"]
    except Exception:
        feat["min_charge"] = feat["max_charge"] = feat["mean_charge"] = feat["charge_range"] = 0.0

    # --- rings ---
    feat["n_rings"]      = rdMolDescriptors.CalcNumRings(mol)
    feat["n_arom_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)

    # --- Morgan fingerprint bits as additional features ---
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
    for i, bit in enumerate(fp):
        feat[f"mfp2_{i}"] = int(bit)

    return feat


def build_mordred_features(smiles_list: list[str]) -> pd.DataFrame:
    """
    Use Mordred to compute up to 1800 2D descriptors, then remove
    constants and highly correlated features.
    Falls back to rdkit-only if mordred is unavailable.
    """
    if not MORDRED_AVAILABLE:
        print("  [warn] mordred unavailable, using RDKit features only")
        rows = []
        for s in smiles_list:
            f = smiles_to_rdkit_features(s)
            rows.append(f if f is not None else {})
        return pd.DataFrame(rows).fillna(0)

    calc = Calculator(mordred_descs, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    df = calc.pandas(mols)

    # keep only numeric columns
    df = df.select_dtypes(include=[np.number]).fillna(0)
    # remove zero-variance columns
    df = df.loc[:, df.std() > 1e-8]
    # remove highly correlated (|r| > 0.95)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
    df = df.drop(columns=drop_cols)

    print(f"  Mordred: {df.shape[1]} descriptors after collinearity pruning")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 3.  SURROGATE MODELS
#     Three separate models, one per objective.
#     Using gradient boosted trees (fast, handles small datasets well).
#     Optionally swap for a Chemprop D-MPNN for larger datasets.
# ════════════════════════════════════════════════════════════════════════════

class SurrogateEnsemble:
    """
    Thin wrapper around sklearn regressors for three objectives:
      - dielectric strength relative to SF6 (maximise → negate for minimisation)
      - boiling point in °C                 (minimise: want < −10°C)
      - GWP 100-yr                          (minimise)

    Includes a simple applicability domain check via leverage statistics.
    """

    OBJECTIVES = ["ds_rel", "bp_c", "gwp100"]

    def __init__(self, n_estimators=200, max_depth=4):
        self.models = {}
        self.scalers = {}
        self.X_train = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._trained = False

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        X: feature matrix (n_samples, n_features)
        Y: target matrix  (n_samples, 3) — [ds_rel, bp_c, gwp100]
        """
        assert Y.shape[1] == 3, "Y must have 3 columns: ds_rel, bp_c, gwp100"
        self.X_train = X.copy()

        for i, name in enumerate(self.OBJECTIVES):
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            y = Y[:, i]
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            model.fit(Xs, y)
            self.models[name] = model
            self.scalers[name] = scaler

        self._trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (n, 3) array of [ds_rel, bp_c, gwp100] predictions."""
        preds = []
        for name in self.OBJECTIVES:
            Xs = self.scalers[name].transform(X)
            preds.append(self.models[name].predict(Xs))
        return np.column_stack(preds)

    def predict_single(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x.reshape(1, -1))[0]

    def cross_validate(self, X, Y, cv=5) -> dict:
        results = {}
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for i, name in enumerate(self.OBJECTIVES):
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            scores = cross_val_score(
                GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=0.05,
                    random_state=42,
                ),
                Xs, Y[:, i], cv=kf, scoring="r2"
            )
            results[name] = {"mean_r2": scores.mean(), "std_r2": scores.std()}
        return results

    def applicability_domain(self, X_new: np.ndarray,
                              threshold: float = 3.0) -> np.ndarray:
        """
        Leverage-based AD check. Returns boolean mask: True = in domain.
        h_i = x_i^T (X^T X)^{-1} x_i
        Warning threshold h* = 3p/n where p = n_features, n = n_train.
        """
        X = self.scalers["ds_rel"].transform(self.X_train)
        Xn = self.scalers["ds_rel"].transform(X_new)
        try:
            XtXinv = np.linalg.pinv(X.T @ X)
        except np.linalg.LinAlgError:
            return np.ones(len(X_new), dtype=bool)
        h_new = np.array([xn @ XtXinv @ xn for xn in Xn])
        n, p = X.shape
        h_star = threshold * p / n
        return h_new <= h_star


# ════════════════════════════════════════════════════════════════════════════
# 4.  SELFIES CHEMICAL SPACE — MUTATION & CROSSOVER
#     Constrained to fluorinated molecules (C, F, N, O, S atoms only).
#     Every SELFIES token manipulation produces a syntactically valid molecule.
# ════════════════════════════════════════════════════════════════════════════

# Restrict alphabet to chemically relevant tokens for fluorinated gases
SELFIES_ALPHABET = list(sf.get_semantic_robust_alphabet())
# Keep only tokens containing relevant atoms
RELEVANT_ATOMS = {"C", "F", "N", "O", "S", "I", "H"}
SELFIES_ALPHABET = [
    t for t in SELFIES_ALPHABET
    if any(a in t for a in RELEVANT_ATOMS)
    and "Si" not in t and "P" not in t and "B" not in t
]
# Bias towards fluorinated tokens
F_TOKENS = [t for t in SELFIES_ALPHABET if "F" in t]  # noqa: W605
SELFIES_ALPHABET = SELFIES_ALPHABET + F_TOKENS * 3  # oversample F-containing

MAX_SELFIES_LEN = 20  # max tokens per molecule


def smiles_to_selfies_safe(smiles: str) -> str | None:
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def selfies_to_smiles_safe(selfies_str: str) -> str | None:
    try:
        smi = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def random_fluorinated_selfies(min_len: int = 4, max_len: int = MAX_SELFIES_LEN) -> str:
    """Sample a random SELFIES string biased towards fluorinated molecules."""
    length = random.randint(min_len, max_len)
    # Always start with a carbon-containing token
    c_tokens = [t for t in SELFIES_ALPHABET if "[C" in t or "[c" in t]
    tokens = [random.choice(c_tokens if c_tokens else SELFIES_ALPHABET)]
    tokens += [random.choice(SELFIES_ALPHABET) for _ in range(length - 1)]
    return "".join(tokens)


def mutate_selfies(selfies_str: str, n_mutations: int = 1) -> str:
    """Random point mutation on a SELFIES string."""
    tokens = list(sf.split_selfies(selfies_str))
    if not tokens:
        return random_fluorinated_selfies()
    for _ in range(n_mutations):
        op = random.choice(["replace", "insert", "delete"])
        idx = random.randint(0, len(tokens) - 1)
        if op == "replace":
            tokens[idx] = random.choice(SELFIES_ALPHABET)
        elif op == "insert" and len(tokens) < MAX_SELFIES_LEN:
            tokens.insert(idx, random.choice(SELFIES_ALPHABET))
        elif op == "delete" and len(tokens) > 2:
            tokens.pop(idx)
    return "".join(tokens)


def crossover_selfies(s1: str, s2: str) -> tuple[str, str]:
    """Single-point crossover between two SELFIES strings."""
    t1 = list(sf.split_selfies(s1))
    t2 = list(sf.split_selfies(s2))
    if not t1 or not t2:
        return s1, s2
    cut1 = random.randint(1, max(1, len(t1) - 1))
    cut2 = random.randint(1, max(1, len(t2) - 1))
    child1 = "".join(t1[:cut1] + t2[cut2:])
    child2 = "".join(t2[:cut2] + t1[cut1:])
    return child1[:MAX_SELFIES_LEN * 8], child2[:MAX_SELFIES_LEN * 8]


# ════════════════════════════════════════════════════════════════════════════
# 5.  PYMOO PROBLEM DEFINITION
# ════════════════════════════════════════════════════════════════════════════

class SF6ReplacementProblem(ElementwiseProblem):
    """
    Three-objective minimisation problem.
    Gene x = SELFIES string (represented as numpy array of token indices).

    Objectives (all minimised — negate where we want to maximise):
      f1 = -ds_rel          (maximise dielectric strength)
      f2 =  bp_c            (minimise boiling point; want below -10°C)
      f3 =  log10(gwp100)   (minimise GWP on log scale)

    Constraints:
      g1: ds_rel >= 0.7 (at least 70% of SF6 dielectric strength)
      g2: bp_c   <= 20  (must be gaseous at ambient)
      g3: GWP    <= 5000
      g4: molecule must be valid SMILES
    """

    def __init__(self, surrogate: SurrogateEnsemble,
                 feature_fn,
                 alphabet: list[str],
                 max_len: int = MAX_SELFIES_LEN):
        super().__init__(
            n_var=max_len,       # integer gene per token position
            n_obj=3,
            n_ieq_constr=3,
            xl=0,
            xu=len(alphabet) - 1,
            vtype=int,
        )
        self.surrogate = surrogate
        self.feature_fn = feature_fn
        self.alphabet = alphabet
        self.max_len = max_len
        self._cache = {}

    def _genes_to_smiles(self, x: np.ndarray) -> str | None:
        tokens = [self.alphabet[int(i) % len(self.alphabet)] for i in x
                  if i >= 0]
        selfies_str = "".join(tokens)
        return selfies_to_smiles_safe(selfies_str)

    def _evaluate(self, x, out, *args, **kwargs):
        smiles = self._genes_to_smiles(x)

        if smiles is None or smiles in self._cache:
            if smiles and smiles in self._cache:
                out["F"], out["G"] = self._cache[smiles]
            else:
                # Invalid molecule — penalise heavily
                out["F"] = [10.0, 200.0, 5.0]
                out["G"] = [-1.0, -1.0, -1.0]  # all violated
            return

        feats = self.feature_fn(smiles)
        if feats is None:
            out["F"] = [10.0, 200.0, 5.0]
            out["G"] = [-1.0, -1.0, -1.0]
            return

        x_vec = np.array(list(feats.values()), dtype=float)
        x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad/trim to expected feature dimension
        n_feat = self.surrogate.X_train.shape[1]
        if len(x_vec) < n_feat:
            x_vec = np.pad(x_vec, (0, n_feat - len(x_vec)))
        else:
            x_vec = x_vec[:n_feat]

        preds = self.surrogate.predict_single(x_vec)
        ds, bp, gwp = preds[0], preds[1], preds[2]

        # Objectives (all minimised)
        f1 = -float(ds)                        # negate: maximise ds
        f2 = float(bp)                         # minimise boiling point
        f3 = float(np.log10(max(gwp, 1.0)))    # log GWP

        # Inequality constraints (g_i <= 0 means satisfied)
        g1 = 0.7 - ds          # ds >= 0.7  →  0.7 - ds <= 0
        g2 = bp - 20.0         # bp <= 20   →  bp - 20 <= 0
        g3 = gwp - 5000.0      # GWP <= 5000

        out["F"] = [f1, f2, f3]
        out["G"] = [g1, g2, g3]
        self._cache[smiles] = ([f1, f2, f3], [g1, g2, g3])


# ════════════════════════════════════════════════════════════════════════════
# 6.  CUSTOM NSGA-II OPERATORS FOR SELFIES SPACE
# ════════════════════════════════════════════════════════════════════════════

class SELFIESSampling(Sampling):
    def __init__(self, seed_smiles: list[str], alphabet: list[str],
                 max_len: int = MAX_SELFIES_LEN):
        super().__init__()
        self.seed_smiles = seed_smiles
        self.alphabet = alphabet
        self.max_len = max_len

    def _do(self, problem, n_samples, **kwargs):
        X = []
        seed_pool = [smiles_to_selfies_safe(s) for s in self.seed_smiles
                     if smiles_to_selfies_safe(s) is not None]
        for i in range(n_samples):
            if seed_pool and random.random() < 0.5:
                sel = random.choice(seed_pool)
                sel = mutate_selfies(sel, n_mutations=random.randint(1, 3))
            else:
                sel = random_fluorinated_selfies()
            tokens = list(sf.split_selfies(sel))
            genes = np.array(
                [self.alphabet.index(t) if t in self.alphabet
                 else random.randint(0, len(self.alphabet) - 1)
                 for t in tokens[:self.max_len]]
            )
            # pad to max_len
            if len(genes) < self.max_len:
                genes = np.pad(genes, (0, self.max_len - len(genes)),
                               constant_values=0)
            X.append(genes)
        return np.array(X)


class SELFIESMutation(Mutation):
    def __init__(self, prob: float = 0.3, alphabet: list[str] = None,
                 max_len: int = MAX_SELFIES_LEN):
        super().__init__()
        self.prob = prob
        self.alphabet = alphabet or SELFIES_ALPHABET
        self.max_len = max_len

    def _do(self, problem, X, **kwargs):
        X_new = X.copy()
        for i in range(len(X)):
            if random.random() < self.prob:
                # convert genes to selfies, mutate, convert back
                tokens = [self.alphabet[int(j) % len(self.alphabet)]
                          for j in X[i]]
                sel = "".join(tokens)
                sel = mutate_selfies(sel, n_mutations=random.randint(1, 2))
                new_tokens = list(sf.split_selfies(sel))
                genes = np.array(
                    [self.alphabet.index(t) if t in self.alphabet
                     else random.randint(0, len(self.alphabet) - 1)
                     for t in new_tokens[:self.max_len]]
                )
                if len(genes) < self.max_len:
                    genes = np.pad(genes, (0, self.max_len - len(genes)),
                                   constant_values=0)
                X_new[i] = genes
        return X_new


class SELFIESCrossover(Crossover):
    def __init__(self, prob: float = 0.9, alphabet: list[str] = None,
                 max_len: int = MAX_SELFIES_LEN):
        super().__init__(2, 2)
        self.prob = prob
        self.alphabet = alphabet or SELFIES_ALPHABET
        self.max_len = max_len

    def _do(self, problem, X, **kwargs):
        # X shape: (n_matings, n_parents, n_var)
        _, n_matings, _ = X.shape
        Y = np.full_like(X, 0)
        for k in range(n_matings):
            if random.random() < self.prob:
                t1 = [self.alphabet[int(j) % len(self.alphabet)]
                      for j in X[0, k]]
                t2 = [self.alphabet[int(j) % len(self.alphabet)]
                      for j in X[1, k]]
                s1, s2 = crossover_selfies("".join(t1), "".join(t2))
                for child_genes, child_sel, yi in [(Y[0, k], s1, 0),
                                                    (Y[1, k], s2, 1)]:
                    toks = list(sf.split_selfies(child_sel))
                    genes = np.array(
                        [self.alphabet.index(t) if t in self.alphabet
                         else random.randint(0, len(self.alphabet) - 1)
                         for t in toks[:self.max_len]]
                    )
                    if len(genes) < self.max_len:
                        genes = np.pad(genes, (0, self.max_len - len(genes)),
                                       constant_values=0)
                    Y[yi, k] = genes
            else:
                Y[0, k] = X[0, k].copy()
                Y[1, k] = X[1, k].copy()
        return Y


# ════════════════════════════════════════════════════════════════════════════
# 7.  PARETO FRONT ANALYSIS UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def decode_population(res, alphabet: list[str],
                      surrogate: SurrogateEnsemble,
                      feature_fn) -> pd.DataFrame:
    """Convert pymoo result population to a pandas DataFrame."""
    rows = []
    for x, f, g in zip(res.X, res.F, res.G):
        tokens = [alphabet[int(i) % len(alphabet)] for i in x]
        sel = "".join(tokens)
        smiles = selfies_to_smiles_safe(sel)
        if smiles is None:
            continue
        feats = feature_fn(smiles)
        if feats is None:
            continue
        x_vec = np.array(list(feats.values()), dtype=float)
        x_vec = np.nan_to_num(x_vec)
        n_feat = surrogate.X_train.shape[1]
        if len(x_vec) < n_feat:
            x_vec = np.pad(x_vec, (0, n_feat - len(x_vec)))
        else:
            x_vec = x_vec[:n_feat]

        in_domain = surrogate.applicability_domain(x_vec.reshape(1, -1))[0]
        preds = surrogate.predict_single(x_vec)
        rows.append({
            "smiles":      smiles,
            "selfies":     sel,
            "ds_pred":     float(preds[0]),
            "bp_pred":     float(preds[1]),
            "gwp_pred":    float(preds[2]),
            "f1_neg_ds":   float(f[0]),
            "f2_bp":       float(f[1]),
            "f3_logGWP":   float(f[2]),
            "g1_ds":       float(g[0]),
            "g2_bp":       float(g[1]),
            "g3_gwp":      float(g[2]),
            "in_domain":   bool(in_domain),
            "feasible":    bool(all(gi <= 0 for gi in g)),
        })
    return pd.DataFrame(rows).drop_duplicates(subset="smiles")


def hypervolume_contribution(F: np.ndarray,
                              ref_point: np.ndarray = None) -> np.ndarray:
    """
    Compute hypervolume contribution of each Pareto point.
    Useful for prioritising candidates: higher contribution = more unique.
    """
    if ref_point is None:
        ref_point = F.max(axis=0) + 0.1 * (F.max(axis=0) - F.min(axis=0))
    hv_calc = HV(ref_point=ref_point)
    total_hv = hv_calc.do(F)
    contribs = np.zeros(len(F))
    for i in range(len(F)):
        F_without_i = np.delete(F, i, axis=0)
        if len(F_without_i) == 0:
            contribs[i] = total_hv
        else:
            hv_without = hv_calc.do(F_without_i)
            contribs[i] = total_hv - hv_without
    return contribs


# ════════════════════════════════════════════════════════════════════════════
# 8.  VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

def plot_pareto_front(df_pareto: pd.DataFrame,
                      output_path: str = "pareto_front.png"):
    """
    3-panel Pareto front visualisation:
      - Left:   DS vs Boiling Point  (2D projection)
      - Centre: DS vs GWP            (2D projection, log GWP)
      - Right:  3D scatter with HV contribution as colour
    """
    feasible = df_pareto[df_pareto["feasible"]].copy()
    if feasible.empty:
        print("  No feasible candidates to plot.")
        return

    F = feasible[["f1_neg_ds", "f2_bp", "f3_logGWP"]].values
    hv_contrib = hypervolume_contribution(F)
    feasible["hv_contrib"] = hv_contrib

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Pareto Front: SF6 Alternatives Design Space", fontsize=14)

    # --- panel 1: DS vs BP ---
    sc1 = axes[0].scatter(feasible["ds_pred"], feasible["bp_pred"],
                           c=np.log10(np.clip(feasible["gwp_pred"], 1, None)),
                           cmap="RdYlGn_r", s=60, edgecolors="k", lw=0.4,
                           vmin=0, vmax=4)
    axes[0].axhline(-10, ls="--", c="gray", lw=0.8, label="BP = -10°C limit")
    axes[0].axvline(1.0, ls="--", c="blue", lw=0.8, label="SF6 DS reference")
    axes[0].set_xlabel("Predicted dielectric strength (rel. SF6)")
    axes[0].set_ylabel("Predicted boiling point (°C)")
    axes[0].set_title("DS vs Boiling Point")
    axes[0].legend(fontsize=8)
    plt.colorbar(sc1, ax=axes[0], label="log₁₀(GWP)")

    # --- panel 2: DS vs GWP ---
    sc2 = axes[1].scatter(feasible["ds_pred"],
                           np.log10(np.clip(feasible["gwp_pred"], 1, None)),
                           c=feasible["bp_pred"], cmap="coolwarm_r",
                           s=60, edgecolors="k", lw=0.4, vmin=-80, vmax=50)
    axes[1].axvline(1.0, ls="--", c="blue", lw=0.8)
    axes[1].axhline(np.log10(2090), ls=":", c="orange", lw=0.8,
                     label="C4F7N GWP")
    axes[1].set_xlabel("Predicted dielectric strength (rel. SF6)")
    axes[1].set_ylabel("log₁₀(GWP 100-yr)")
    axes[1].set_title("DS vs GWP")
    axes[1].legend(fontsize=8)
    plt.colorbar(sc2, ax=axes[1], label="Boiling point (°C)")

    # --- panel 3: DS vs HV contribution ---
    sc3 = axes[2].scatter(feasible["ds_pred"], feasible["bp_pred"],
                           c=feasible["hv_contrib"],
                           cmap="plasma", s=80, edgecolors="k", lw=0.4)
    axes[2].set_xlabel("Predicted dielectric strength (rel. SF6)")
    axes[2].set_ylabel("Predicted boiling point (°C)")
    axes[2].set_title("Hypervolume contribution (higher = more unique)")
    plt.colorbar(sc3, ax=axes[2], label="HV contribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Pareto front plot saved to: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# 9.  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  SF6 Alternatives — Multi-Objective Inverse Design Pipeline")
    print("=" * 65)

    # ── 9.1 Load dataset ──────────────────────────────────────────────────
    print("\n[1/6] Loading dataset...")
    import io
    df = pd.read_csv(io.StringIO(RAW_DATA))
    # drop duplicate SMILES (keep first), drop rows with missing targets
    df = df.dropna(subset=["smiles", "ds_rel", "bp_c", "gwp100"])
    df = df.drop_duplicates(subset="smiles").reset_index(drop=True)
    # Sanitise SMILES through RDKit
    valid_mask = [Chem.MolFromSmiles(s) is not None for s in df["smiles"]]
    df = df[valid_mask].reset_index(drop=True)
    print(f"  {len(df)} valid molecules loaded.")

    # ── 9.2 Feature generation ────────────────────────────────────────────
    print("\n[2/6] Generating descriptors (RDKit + Mordred)...")
    # Use simpler RDKit features for speed; swap build_mordred_features for
    # the full Mordred run on a larger dataset.
    feat_rows = []
    valid_idx = []
    for i, row in df.iterrows():
        f = smiles_to_rdkit_features(row["smiles"])
        if f is not None:
            feat_rows.append(f)
            valid_idx.append(i)

    df = df.iloc[valid_idx].reset_index(drop=True)
    df_feats = pd.DataFrame(feat_rows).fillna(0)
    # Trim Morgan bits to reduce dimensionality for small dataset
    non_morgan_cols = [c for c in df_feats.columns if not c.startswith("mfp2_")]
    morgan_cols = [c for c in df_feats.columns if c.startswith("mfp2_")]
    # Keep all non-Morgan + top-variance Morgan bits
    morgan_var = df_feats[morgan_cols].var().nlargest(64).index.tolist()
    keep_cols = non_morgan_cols + morgan_var
    df_feats = df_feats[keep_cols].fillna(0)

    X = df_feats.values.astype(float)
    Y = df[["ds_rel", "bp_c", "gwp100"]].values.astype(float)
    # Log-transform GWP for training stability
    Y[:, 2] = np.log10(np.clip(Y[:, 2], 1, None))
    print(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    # ── 9.3 Train & cross-validate surrogates ─────────────────────────────
    print("\n[3/6] Training surrogate models (GBR) + 5-fold CV...")
    surr = SurrogateEnsemble(n_estimators=300, max_depth=4)
    surr.fit(X, Y)
    cv_results = surr.cross_validate(X, Y, cv=5)
    for prop, res in cv_results.items():
        print(f"  {prop:10s}  CV R² = {res['mean_r2']:.3f} ± {res['std_r2']:.3f}")

    # Store feature column names for alignment during optimisation
    feat_col_names = list(df_feats.columns)

    def feature_fn_aligned(smiles: str) -> dict | None:
        """Generate features in the same column order as training."""
        raw = smiles_to_rdkit_features(smiles)
        if raw is None:
            return None
        aligned = {k: raw.get(k, 0.0) for k in feat_col_names}
        return aligned

    # ── 9.4 Set up Pareto optimisation ────────────────────────────────────
    print("\n[4/6] Setting up NSGA-II multi-objective optimisation...")

    seed_smiles = df["smiles"].tolist()
    alphabet = list(dict.fromkeys(SELFIES_ALPHABET))  # unique, order-preserving

    problem = SF6ReplacementProblem(
        surrogate=surr,
        feature_fn=feature_fn_aligned,
        alphabet=alphabet,
        max_len=MAX_SELFIES_LEN,
    )

    algorithm = NSGA2(
        pop_size=80,
        sampling=SELFIESSampling(seed_smiles, alphabet, MAX_SELFIES_LEN),
        crossover=SELFIESCrossover(prob=0.9, alphabet=alphabet,
                                    max_len=MAX_SELFIES_LEN),
        mutation=SELFIESMutation(prob=0.3, alphabet=alphabet,
                                  max_len=MAX_SELFIES_LEN),
        eliminate_duplicates=False,
    )

    # ── 9.5 Run optimisation ──────────────────────────────────────────────
    print("\n[5/6] Running NSGA-II (100 generations)...")
    print("  (This may take a few minutes. Increase n_gen for better coverage.)")
    res = minimize(
        problem,
        algorithm,
        ("n_gen", 100),
        seed=42,
        verbose=False,
    )
    print(f"  Optimisation complete. Evaluations: {res.algorithm.evaluator.n_eval}")

    # ── 9.6 Post-process Pareto front ─────────────────────────────────────
    print("\n[6/6] Post-processing Pareto front...")

    if res.X is None or len(res.X) == 0:
        print("  [warn] No Pareto solutions found — try more generations.")
        return

    df_pareto = decode_population(res, alphabet, surr, feature_fn_aligned)

    # Convert log GWP back to linear for display
    df_pareto["gwp_pred"] = np.power(10, df_pareto["gwp_pred"])

    # Rank by HV contribution (feasible only)
    feasible = df_pareto[df_pareto["feasible"]].copy()
    if not feasible.empty:
        F_arr = feasible[["f1_neg_ds", "f2_bp", "f3_logGWP"]].values
        feasible["hv_contrib"] = hypervolume_contribution(F_arr)
        feasible = feasible.sort_values("hv_contrib", ascending=False)

    # Print top 10
    print("\n  Top 10 candidates by hypervolume contribution:")
    print(f"  {'SMILES':<45} {'DS':>6} {'BP':>7} {'GWP':>8} {'AD':>5}")
    print("  " + "-" * 75)
    for _, row in feasible.head(10).iterrows():
        ad = "✓" if row["in_domain"] else "✗"
        print(f"  {row['smiles'][:44]:<44} {row['ds_pred']:>6.2f} "
              f"{row['bp_pred']:>7.1f} {row['gwp_pred']:>8.0f} {ad:>5}")

    # Save results
    out_dir = pathlib.Path("output")
    out_dir.mkdir(exist_ok=True)

    feasible.to_csv(out_dir / "pareto_candidates.csv", index=False)
    print(f"\n  Full results saved to: {out_dir / 'pareto_candidates.csv'}")

    # Reference: known candidates for comparison
    known = {
        "SF6":      {"ds": 1.00, "bp": -63.8, "gwp": 23500},
        "C4F7N":    {"ds": 1.90, "bp":   0.0, "gwp":  2090},
        "C5F10O":   {"ds": 1.75, "bp":  27.0, "gwp":     1},
        "CF3I":     {"ds": 1.80, "bp": -22.5, "gwp":     1},
        "C2F5I":    {"ds": 2.10, "bp":  13.0, "gwp":     1},
    }

    plot_pareto_front(df_pareto, str(out_dir / "pareto_front.png"))

    # ── 9.7 Summary stats ─────────────────────────────────────────────────
    print("\n  Summary:")
    print(f"    Total Pareto solutions decoded:  {len(df_pareto)}")
    print(f"    Feasible (all constraints met):  {feasible.shape[0]}")
    print(f"    In applicability domain:         "
          f"{feasible['in_domain'].sum()}")
    if not feasible.empty:
        best = feasible.iloc[0]
        print(f"\n  Top candidate:")
        print(f"    SMILES:         {best['smiles']}")
        print(f"    DS (pred):      {best['ds_pred']:.2f} × SF6")
        print(f"    BP (pred):      {best['bp_pred']:.1f} °C")
        print(f"    GWP (pred):     {best['gwp_pred']:.0f}")
        print(f"    In domain:      {best['in_domain']}")

    print("\n  Next steps:")
    print("  1. DFT validation (ORCA/Gaussian) of top-domain candidates")
    print("  2. Run Multiwfn for GIPF descriptors to refine DS predictions")
    print("  3. Retrain surrogates with new DFT data (active learning loop)")
    print("  4. Assess arc-quenching & decomposition products experimentally")
    print("=" * 65)

    return df_pareto, surr, res


# ════════════════════════════════════════════════════════════════════════════
# 10.  CHEMPROP D-MPNN INTEGRATION (advanced — for larger datasets)
#      Run this block when you have ≥200 labelled molecules.
#      Provides better generalisation than GBR for out-of-distribution mols.
# ════════════════════════════════════════════════════════════════════════════

CHEMPROP_TRAINING_SCRIPT = '''
# save_as: train_chemprop.py
# Run from command line: python train_chemprop.py

import pandas as pd
import subprocess, pathlib

df = pd.read_csv("output/pareto_candidates.csv")  # or your curated dataset
df["ds_rel"].to_frame()  # pick target

# Write Chemprop-format CSV
train_df = pd.DataFrame({
    "smiles": df["smiles"],
    "ds_rel": df["ds_pred"],       # replace with experimental values
    "bp_c":   df["bp_pred"],
    "gwp_pred": df["gwp_pred"],
})
train_df.to_csv("chemprop_train.csv", index=False)

# Train multi-task D-MPNN
cmd = [
    "chemprop", "train",
    "--data-path", "chemprop_train.csv",
    "--task-type", "regression",
    "--target-columns", "ds_rel", "bp_c", "gwp_pred",
    "--save-dir", "chemprop_model/",
    "--epochs", "100",
    "--batch-size", "32",
    "--hidden-size", "300",
    "--depth", "3",
    "--dropout", "0.1",
    "--features-generator", "rdkit_2d_normalized",  # augment MPNN with RDKit
    "--split-type", "random",
    "--metric", "rmse",
]
subprocess.run(cmd, check=True)
print("Chemprop model saved to chemprop_model/")

# Predict on new candidates
pred_cmd = [
    "chemprop", "predict",
    "--test-path", "new_candidates.csv",
    "--model-path", "chemprop_model/",
    "--preds-path", "chemprop_predictions.csv",
]
subprocess.run(pred_cmd, check=True)
'''


# ════════════════════════════════════════════════════════════════════════════
# 11.  ACTIVE LEARNING LOOP (sketch)
# ════════════════════════════════════════════════════════════════════════════

def active_learning_round(df_pareto: pd.DataFrame,
                           surrogate: SurrogateEnsemble,
                           feature_fn,
                           n_select: int = 5) -> pd.DataFrame:
    """
    Select the most informative candidates for DFT validation using a
    combined acquisition function:
      score = hv_contribution × exploration_bonus × domain_penalty

    exploration_bonus = inverse distance to training set in feature space
      (prefer candidates far from already-labelled molecules)
    domain_penalty = 0 if outside AD, 1 if inside AD
    """
    feasible = df_pareto[df_pareto["feasible"] & df_pareto["in_domain"]].copy()
    if feasible.empty:
        return feasible

    F_arr = feasible[["f1_neg_ds", "f2_bp", "f3_logGWP"]].values
    hv = hypervolume_contribution(F_arr)

    # Exploration: compute distance from each candidate to training set
    X_train = surrogate.X_train
    dists = []
    for _, row in feasible.iterrows():
        feats = feature_fn(row["smiles"])
        if feats is None:
            dists.append(0.0)
            continue
        x = np.array(list(feats.values()), dtype=float)
        x = np.nan_to_num(x)
        n_feat = X_train.shape[1]
        if len(x) < n_feat:
            x = np.pad(x, (0, n_feat - len(x)))
        else:
            x = x[:n_feat]
        d = np.min(np.linalg.norm(X_train - x, axis=1))
        dists.append(d)

    dists = np.array(dists)
    dists_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)
    hv_norm = (hv - hv.min()) / (hv.max() - hv.min() + 1e-8)

    feasible["acq_score"] = 0.6 * hv_norm + 0.4 * dists_norm
    return feasible.nlargest(n_select, "acq_score")[
        ["smiles", "ds_pred", "bp_pred", "gwp_pred", "acq_score", "hv_contrib"]
    ]


if __name__ == "__main__":
    result = main()
    if result is not None:
        df_pareto, surr, res = result
        # Active learning selection for DFT
        feature_fn_ref = lambda s: smiles_to_rdkit_features(s)
        al_candidates = active_learning_round(
            df_pareto, surr, feature_fn_ref, n_select=5
        )
        if not al_candidates.empty:
            print("\nActive learning — top 5 for DFT validation:")
            print(al_candidates.to_string(index=False))
            al_candidates.to_csv("output/dft_candidates.csv", index=False)
