#!/usr/bin/env python3
"""
TDC SOTA Integration Layer for ADDIE Service
=============================================

Integrates the 22 TDC benchmark-validated SOTA models into the existing
ADDIE prediction service. For each endpoint, the best reproducibly-validated
model is selected from MapLight, MapLight+GNN, or ADMET-AI.

Architecture:
  Novel molecule → Feature computation → Route to best model per endpoint
  Known molecule → Lookup pre-computed column (same models, batch-predicted)

Sources:
  - MapLight-TDC (MIT): github.com/maplightrx/MapLight-TDC
  - ADMET-AI (MIT): github.com/swansonk14/admet_ai
  - CaliciBoost (MIT): github.com/Calici/CaliciBoost

Validated by: "Critical Assessment of ML models for ADMET Prediction in
TDC leaderboards" (bioRxiv, Feb 2026) — only CaliciBoost, MapLight, and
MapLight+GNN pass all reproducibility checks.

Usage:
  predictor = TDCSOTAPredictor()
  predictor.load_models("/path/to/trained_models/")
  results = predictor.predict("CCO")
  results = predictor.predict_batch(["CCO", "c1ccccc1", ...])
"""

import os
import json
import logging
import hashlib
import warnings
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration: 22 TDC Endpoints
# =============================================================================

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Metric(Enum):
    AUROC = "auroc"
    AUPRC = "auprc"
    MAE = "mae"
    SPEARMAN = "spearman"


class ModelSource(Enum):
    MAPLIGHT = "maplight"           # CatBoost + ECFP/Avalon/ErG/RDKit
    MAPLIGHT_GNN = "maplight_gnn"   # CatBoost + ECFP/Avalon/ErG/RDKit + GIN
    ADMET_AI = "admet_ai"           # Chemprop-RDKit GNN
    CHEMPROP = "chemprop"           # Chemprop v2 MPNN (standalone .pt checkpoints)


@dataclass
class TDCEndpoint:
    """Definition of a single TDC ADMET benchmark endpoint"""
    tdc_name: str               # e.g. "caco2_wang"
    display_name: str           # e.g. "Caco-2 Permeability"
    category: str               # absorption / distribution / metabolism / excretion / toxicity
    task_type: TaskType
    metric: Metric
    dataset_size: int
    unit: str
    best_model: ModelSource     # Which model wins this endpoint
    best_score: float           # The benchmark score of the best model
    addie_field: str            # Maps to existing ADDIE standardizer field name
    db_column: str              # Column name in the compound database


# Master configuration: 22 TDC endpoints with best model per endpoint
# Based on reproducibility audit (bioRxiv Feb 2026): only MapLight, MapLight+GNN,
# CaliciBoost, and ADMET-AI pass all checks.
#
# Model selection: Run all 3 (MapLight, MapLight+GNN, ADMET-AI) on TDC splits,
# pick winner per endpoint. Below is the expected assignment based on published
# results. UPDATE these after running your own validation.

TDC_ENDPOINTS: Dict[str, TDCEndpoint] = {
    # -------------------------------------------------------------------------
    # ABSORPTION (6 endpoints)
    # -------------------------------------------------------------------------
    "caco2_wang": TDCEndpoint(
        tdc_name="caco2_wang",
        display_name="Caco-2 Permeability",
        category="absorption",
        task_type=TaskType.REGRESSION,
        metric=Metric.MAE,
        dataset_size=906,
        unit="cm/s",
        best_model=ModelSource.MAPLIGHT,
        best_score=0.276,  # MAE (lower is better)
        addie_field="caco2_permeability",
        db_column="tdc_caco2_wang_mae",
    ),
    "hia_hou": TDCEndpoint(
        tdc_name="hia_hou",
        display_name="Human Intestinal Absorption",
        category="absorption",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUROC,
        dataset_size=578,
        unit="%",
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.989,
        addie_field="hia_probability",
        db_column="tdc_hia_hou_auroc",
    ),
    "pgp_broccatelli": TDCEndpoint(
        tdc_name="pgp_broccatelli",
        display_name="P-gp Inhibition",
        category="absorption",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUROC,
        dataset_size=1212,
        unit="%",
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.940,
        addie_field="pgp_inhibitor_probability",
        db_column="tdc_pgp_broccatelli_auroc",
    ),
    # P-gp Substrate — Wang+Esposito merged dataset (NOT from TDC)
    # Trained: Chemprop MPNN on 1,995 compounds (1,202 substrates / 793 non-substrates)
    # Published SOTA: AUC 0.848 (AttentiveFP, Briefings in Bioinformatics 2025)
    "pgp_substrate_wang": TDCEndpoint(
        tdc_name="pgp_substrate_wang",  # custom, not TDC-native
        display_name="P-gp Substrate (Efflux)",
        category="absorption",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUROC,
        dataset_size=1995,
        unit="%",
        best_model=ModelSource.CHEMPROP,
        best_score=0.848,
        addie_field="pgp_substrate_probability",
        db_column="tdc_pgp_substrate_wang_auroc",
    ),
    "bioavailability_ma": TDCEndpoint(
        tdc_name="bioavailability_ma",
        display_name="Oral Bioavailability",
        category="absorption",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUROC,
        dataset_size=640,
        unit="%",
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.748,
        addie_field="bioavailability_probability",
        db_column="tdc_bioavailability_ma_auroc",
    ),
    "lipophilicity_astrazeneca": TDCEndpoint(
        tdc_name="lipophilicity_astrazeneca",
        display_name="Lipophilicity",
        category="absorption",
        task_type=TaskType.REGRESSION,
        metric=Metric.MAE,
        dataset_size=4200,
        unit="log-ratio",
        best_model=ModelSource.CHEMPROP,  # Chemprop winner beats MapLight CatBoost
        best_score=0.467,
        addie_field="lipophilicity_log_ratio",
        db_column="tdc_lipo_az_mae",
    ),
    "solubility_aqsoldb": TDCEndpoint(
        tdc_name="solubility_aqsoldb",
        display_name="Aqueous Solubility",
        category="absorption",
        task_type=TaskType.REGRESSION,
        metric=Metric.MAE,
        dataset_size=9982,
        unit="log mol/L",
        best_model=ModelSource.MAPLIGHT,
        best_score=0.761,
        addie_field="aqueous_solubility_log_mol_L",
        db_column="tdc_solubility_aqsoldb_mae",
    ),

    # -------------------------------------------------------------------------
    # DISTRIBUTION (3 endpoints)
    # -------------------------------------------------------------------------
    "bbb_martins": TDCEndpoint(
        tdc_name="bbb_martins",
        display_name="Blood-Brain Barrier Penetration",
        category="distribution",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUROC,
        dataset_size=1975,
        unit="%",
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.916,
        addie_field="bbb_penetration_probability",
        db_column="tdc_bbb_martins_auroc",
    ),
    "ppbr_az": TDCEndpoint(
        tdc_name="ppbr_az",
        display_name="Plasma Protein Binding Rate",
        category="distribution",
        task_type=TaskType.REGRESSION,
        metric=Metric.MAE,
        dataset_size=1797,
        unit="%",
        best_model=ModelSource.ADMET_AI,
        best_score=7.526,
        addie_field="ppbr_percent",
        db_column="tdc_ppbr_az_mae",
    ),
    "vdss_lombardo": TDCEndpoint(
        tdc_name="vdss_lombardo",
        display_name="Volume of Distribution at Steady State",
        category="distribution",
        task_type=TaskType.REGRESSION,
        metric=Metric.SPEARMAN,
        dataset_size=1130,
        unit="L/kg",
        best_model=ModelSource.ADMET_AI,
        best_score=0.713,
        addie_field="vdss_L_kg",
        db_column="tdc_vdss_lombardo_spearman",
    ),

    # -------------------------------------------------------------------------
    # METABOLISM (6 endpoints)
    # -------------------------------------------------------------------------
    "cyp2c9_veith": TDCEndpoint(
        tdc_name="cyp2c9_veith",
        display_name="CYP2C9 Inhibition",
        category="metabolism",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUPRC,
        dataset_size=12092,
        unit="%",
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.900,
        addie_field="cyp2c9_inhibitor_probability",
        db_column="tdc_cyp2c9_veith_auprc",
    ),
    "cyp2d6_veith": TDCEndpoint(
        tdc_name="cyp2d6_veith",
        display_name="CYP2D6 Inhibition",
        category="metabolism",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUPRC,
        dataset_size=13130,
        unit="%",
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.778,  # SOTA: NovoExpert-2 CatBoost+GIN retrained
        addie_field="cyp2d6_inhibitor_probability",
        db_column="tdc_cyp2d6_veith_auprc",
    ),
    "cyp3a4_veith": TDCEndpoint(
        tdc_name="cyp3a4_veith",
        display_name="CYP3A4 Inhibition",
        category="metabolism",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUPRC,
        dataset_size=12328,
        unit="%",
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.916,  # SOTA: NovoExpert-2 CatBoost+GIN retrained
        addie_field="cyp3a4_inhibitor_probability",
        db_column="tdc_cyp3a4_veith_auprc",
    ),
    "cyp2c9_substrate_carbonmangels": TDCEndpoint(
        tdc_name="cyp2c9_substrate_carbonmangels",
        display_name="CYP2C9 Substrate",
        category="metabolism",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUPRC,
        dataset_size=666,
        unit="%",
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.441,
        addie_field="cyp2c9_substrate_probability",
        db_column="tdc_cyp2c9_substrate_auprc",
    ),
    "cyp2d6_substrate_carbonmangels": TDCEndpoint(
        tdc_name="cyp2d6_substrate_carbonmangels",
        display_name="CYP2D6 Substrate",
        category="metabolism",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUPRC,
        dataset_size=664,
        unit="%",
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.736,
        addie_field="cyp2d6_substrate_probability",
        db_column="tdc_cyp2d6_substrate_auprc",
    ),
    "cyp3a4_substrate_carbonmangels": TDCEndpoint(
        tdc_name="cyp3a4_substrate_carbonmangels",
        display_name="CYP3A4 Substrate",
        category="metabolism",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUROC,
        dataset_size=667,
        unit="%",
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.660,  # SOTA: NovoExpert-2 CatBoost+GIN retrained
        addie_field="cyp3a4_substrate_probability",
        db_column="tdc_cyp3a4_substrate_auroc",
    ),

    # -------------------------------------------------------------------------
    # EXCRETION (3 endpoints)
    # -------------------------------------------------------------------------
    "half_life_obach": TDCEndpoint(
        tdc_name="half_life_obach",
        display_name="Half Life",
        category="excretion",
        task_type=TaskType.REGRESSION,
        metric=Metric.SPEARMAN,
        dataset_size=667,
        unit="hr",
        best_model=ModelSource.ADMET_AI,
        best_score=0.562,
        addie_field="half_life_hr",
        db_column="tdc_half_life_obach_spearman",
    ),
    "clearance_hepatocyte_az": TDCEndpoint(
        tdc_name="clearance_hepatocyte_az",
        display_name="Hepatocyte Clearance",
        category="excretion",
        task_type=TaskType.REGRESSION,
        metric=Metric.SPEARMAN,
        dataset_size=1020,
        unit="uL/min/1e6 cells",
        best_model=ModelSource.MAPLIGHT_GNN,  # SOTA: NovoExpert-2 CatBoost+GIN beats ADMET-AI
        best_score=0.511,
        addie_field="clearance_hepatocyte",
        db_column="tdc_clearance_hepatocyte_spearman",
    ),
    "clearance_microsome_az": TDCEndpoint(
        tdc_name="clearance_microsome_az",
        display_name="Microsomal Clearance",
        category="excretion",
        task_type=TaskType.REGRESSION,
        metric=Metric.SPEARMAN,
        dataset_size=1102,
        unit="mL/min/g",
        best_model=ModelSource.ADMET_AI,
        best_score=0.630,
        addie_field="clearance_microsome",
        db_column="tdc_clearance_microsome_spearman",
    ),

    # -------------------------------------------------------------------------
    # TOXICITY (4 endpoints)
    # -------------------------------------------------------------------------
    "ld50_zhu": TDCEndpoint(
        tdc_name="ld50_zhu",
        display_name="Acute Toxicity LD50",
        category="toxicity",
        task_type=TaskType.REGRESSION,
        metric=Metric.MAE,
        dataset_size=7385,
        unit="log(1/(mol/kg))",
        best_model=ModelSource.CHEMPROP,  # Chemprop winner beats MapLight CatBoost
        best_score=0.573,
        addie_field="ld50_log_mol_kg",
        db_column="tdc_ld50_zhu_mae",
    ),
    "herg": TDCEndpoint(
        tdc_name="herg",
        display_name="hERG Blockers",
        category="toxicity",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUROC,
        dataset_size=648,
        unit="%",
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.880,
        addie_field="herg_blocker_probability",
        db_column="tdc_herg_auroc",
    ),
    "ames": TDCEndpoint(
        tdc_name="ames",
        display_name="AMES Mutagenicity",
        category="toxicity",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUROC,
        dataset_size=7255,
        unit="%",
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.871,
        addie_field="ames_mutagenicity_probability",
        db_column="tdc_ames_auroc",
    ),
    "dili": TDCEndpoint(
        tdc_name="dili",
        display_name="Drug-Induced Liver Injury",
        category="toxicity",
        task_type=TaskType.CLASSIFICATION,
        metric=Metric.AUROC,
        dataset_size=1111,
        unit="%",
        # RETRAINED 2026-06-23 on the FDA-curated DILI gold standard (Seal lab, 1111
        # clinical labels). Replaces the prior TDC-DILI Chemprop, which scored 0.92 on
        # TDC's tiny 475-cpd test but was clinically INVERTED on real drugs (acetaminophen
        # 0.31, troglitazone below sildenafil). DILI is the hardest ADMET endpoint —
        # structure-only ceiling ~0.65 — so the win is correct RANKING, not high AUROC.
        # CatBoost+GIN 5-seed + isotonic calibrator (ECE 0.056). 99% recall @ thr 0.35.
        best_model=ModelSource.MAPLIGHT_GNN,
        best_score=0.654,
        addie_field="hepatotoxicity_probability",
        db_column="tdc_dili_auroc",
    ),
    # -------------------------------------------------------------------------
    # TOX21 PANEL (12 endpoints) — retrained 2026-06-23 (WS4) to replace the
    # constant ~0.50 base-ADDIE Tox21 heads. CatBoost 5-seed ensembles on the
    # MapLight+GIN featurizer (same as the other GNN winners). Test-split AUROC.
    # -------------------------------------------------------------------------
    "tox21_nr_ar": TDCEndpoint(
        tdc_name="tox21_nr_ar", display_name="NR-AR (Androgen Receptor) Agonism",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=7197, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.766,
        addie_field="nr_ar_agonist_probability", db_column="tdc_tox21_nr_ar_auroc",
    ),
    "tox21_nr_ar_lbd": TDCEndpoint(
        tdc_name="tox21_nr_ar_lbd", display_name="NR-AR-LBD Agonism",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=8003, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.802,
        addie_field="nr_ar_lbd_agonist_probability", db_column="tdc_tox21_nr_ar_lbd_auroc",
    ),
    "tox21_nr_ahr": TDCEndpoint(
        tdc_name="tox21_nr_ahr", display_name="NR-AhR (Aryl Hydrocarbon Receptor) Agonism",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=8169, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.919,
        addie_field="nr_ahr_agonist_probability", db_column="tdc_tox21_nr_ahr_auroc",
    ),
    "tox21_nr_aromatase": TDCEndpoint(
        tdc_name="tox21_nr_aromatase", display_name="NR-Aromatase Inhibition",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=7226, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.885,
        addie_field="nr_aromatase_inhibitor_probability", db_column="tdc_tox21_nr_aromatase_auroc",
    ),
    "tox21_nr_er": TDCEndpoint(
        tdc_name="tox21_nr_er", display_name="NR-ER (Estrogen Receptor) Agonism",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=7697, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.768,
        addie_field="nr_er_agonist_probability", db_column="tdc_tox21_nr_er_auroc",
    ),
    "tox21_nr_er_lbd": TDCEndpoint(
        tdc_name="tox21_nr_er_lbd", display_name="NR-ER-LBD Agonism",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=8753, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.840,
        addie_field="nr_er_lbd_agonist_probability", db_column="tdc_tox21_nr_er_lbd_auroc",
    ),
    "tox21_nr_ppar_gamma": TDCEndpoint(
        tdc_name="tox21_nr_ppar_gamma", display_name="NR-PPAR-gamma Agonism",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=6450, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.877,
        addie_field="nr_ppar_gamma_agonist_probability", db_column="tdc_tox21_nr_ppar_gamma_auroc",
    ),
    "tox21_sr_are": TDCEndpoint(
        tdc_name="tox21_sr_are", display_name="SR-ARE (Antioxidant Response) Activation",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=5832, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.863,
        addie_field="sr_are_activation_probability", db_column="tdc_tox21_sr_are_auroc",
    ),
    "tox21_sr_atad5": TDCEndpoint(
        tdc_name="tox21_sr_atad5", display_name="SR-ATAD5 (Genotoxicity) Activation",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=7073, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.936,
        addie_field="sr_atad5_activation_probability", db_column="tdc_tox21_sr_atad5_auroc",
    ),
    "tox21_sr_hse": TDCEndpoint(
        tdc_name="tox21_sr_hse", display_name="SR-HSE (Heat Shock) Activation",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=6467, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.867,
        addie_field="sr_hse_activation_probability", db_column="tdc_tox21_sr_hse_auroc",
    ),
    "tox21_sr_mmp": TDCEndpoint(
        tdc_name="tox21_sr_mmp", display_name="SR-MMP (Mitochondrial Membrane Potential) Disruption",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=5811, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.935,
        addie_field="sr_mmp_activation_probability", db_column="tdc_tox21_sr_mmp_auroc",
    ),
    "tox21_sr_p53": TDCEndpoint(
        tdc_name="tox21_sr_p53", display_name="SR-p53 (DNA Damage) Activation",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=6775, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.931,
        addie_field="sr_p53_activation_probability", db_column="tdc_tox21_sr_p53_auroc",
    ),
    # Clinical trial toxicity (TDC ClinTox) — WS4 follow-up; replaces the constant
    # ~0.50 base-ADDIE clinical_toxicity head.
    "clintox": TDCEndpoint(
        tdc_name="clintox", display_name="Clinical Trial Toxicity",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=1478, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.910,
        addie_field="clinical_toxicity_probability", db_column="tdc_clintox_auroc",
    ),
    # DICTrank clinical cardiotoxicity (FDA Liu-2023). NEW field (no base head to
    # replace). Structure-only is a hard endpoint (AUROC ~0.75) so it ships isotonic-
    # CALIBRATED (raw ECE 0.17 -> 0.05 via calibrator.json) as a recall-tuned broad-
    # cardiotox screen complementing hERG. best_score = calibrated test AUROC.
    "dictrank": TDCEndpoint(
        tdc_name="dictrank", display_name="Clinical Cardiotoxicity (DICTrank)",
        category="toxicity", task_type=TaskType.CLASSIFICATION, metric=Metric.AUROC,
        dataset_size=1020, unit="%", best_model=ModelSource.MAPLIGHT_GNN, best_score=0.750,
        addie_field="cardiotoxicity_dict_probability", db_column="tdc_dictrank_auroc",
    ),
}


# =============================================================================
# Feature Computation (MapLight-style)
# =============================================================================

class MolecularFeaturizer:
    """
    Computes MapLight-style concatenated molecular features.
    
    Feature vector (with GIN): ~3,163 dims
      - Morgan ECFP counts: 1024 (radius=2, hashed)
      - Avalon counts: 1024
      - ErG fingerprints: 315
      - RDKit 2D descriptors: ~200
      - GIN supervised masking: 300 (optional)
    
    Feature vector (without GIN): ~2,563 dims
    
    Mirrors: maplightrx/MapLight-TDC/maplight.py
    """

    def __init__(self, use_gin: bool = True, cache_dir: Optional[str] = None):
        self.use_gin = use_gin
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._gin_transformer = None
        self._descriptor_names = None
        
        # Lazy imports validated at init
        self._validate_dependencies()

    def _validate_dependencies(self):
        """Check that all required packages are available"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors, rdReducedGraphs
            from rdkit.Avalon import pyAvalonTools
            logger.info("RDKit dependencies validated")
        except ImportError as e:
            raise ImportError(f"RDKit is required for feature computation: {e}")

        if self.use_gin:
            try:
                import sys, types
                # Stub dgl.graphbolt to avoid missing .so dependency (same as benchmark)
                if 'dgl.graphbolt' not in sys.modules:
                    sys.modules['dgl.graphbolt'] = types.ModuleType('dgl.graphbolt')
                    sys.modules['dgl.graphbolt'].__path__ = []
                import dgl  # noqa: F401
                from dgllife.model import load_pretrained  # noqa: F401
                from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer  # noqa: F401
                logger.info("DGL-Life GIN dependency validated")
            except ImportError:
                logger.warning("dgllife not available, disabling GIN embeddings")
                self.use_gin = False

    def featurize(self, smiles: str) -> Optional[np.ndarray]:
        """
        Compute full feature vector for a single SMILES.
        Returns None for invalid SMILES.
        """
        # Check cache first
        if self.cache_dir:
            cached = self._load_cached(smiles)
            if cached is not None:
                return cached

        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles[:50]}...")
            return None

        features = []

        # 1. Morgan ECFP counts (1024)
        features.append(self._morgan_counts(mol))

        # 2. Avalon counts (1024)
        features.append(self._avalon_counts(mol))

        # 3. ErG fingerprints (315)
        features.append(self._erg_fingerprint(mol))

        # 4. RDKit 2D descriptors (~200)
        features.append(self._rdkit_descriptors(mol))

        # 5. GIN supervised masking embeddings (300, optional)
        if self.use_gin:
            gin_emb = self._gin_embedding(smiles)
            if gin_emb is not None:
                features.append(gin_emb)

        result = np.concatenate(features)

        # Cache result
        if self.cache_dir:
            self._save_cached(smiles, result)

        return result

    def featurize_batch(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        Featurize a batch of SMILES. Returns (feature_matrix, valid_indices).
        Invalid SMILES are skipped.
        """
        features = []
        valid_indices = []

        for i, smi in enumerate(smiles_list):
            feat = self.featurize(smi)
            if feat is not None:
                features.append(feat)
                valid_indices.append(i)

        if not features:
            return np.array([]), []

        return np.vstack(features), valid_indices

    # -- Individual feature extractors --

    def _morgan_counts(self, mol, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
        """Morgan ECFP count fingerprint"""
        from rdkit.Chem import AllChem
        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        for idx, count in fp.GetNonzeroElements().items():
            arr[idx] = count
        return arr

    def _avalon_counts(self, mol, n_bits: int = 1024) -> np.ndarray:
        """Avalon count fingerprint"""
        from rdkit.Avalon import pyAvalonTools
        fp = pyAvalonTools.GetAvalonCountFP(mol, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        for idx, count in fp.GetNonzeroElements().items():
            arr[idx] = count
        return arr

    def _erg_fingerprint(self, mol) -> np.ndarray:
        """Extended Reduced Graph fingerprint (315 dims)"""
        from rdkit.Chem import rdReducedGraphs
        try:
            fp = rdReducedGraphs.GetErGFingerprint(mol)
            return np.array(fp, dtype=np.float32)
        except Exception:
            return np.zeros(315, dtype=np.float32)

    def _rdkit_descriptors(self, mol) -> np.ndarray:
        """RDKit 2D molecular descriptors (~200)"""
        from rdkit.Chem import Descriptors
        from rdkit.ML.Descriptors import MoleculeDescriptors

        if self._descriptor_names is None:
            self._descriptor_names = [name for name, _ in Descriptors.descList]

        calc = MoleculeDescriptors.MolecularDescriptorCalculator(self._descriptor_names)
        try:
            values = calc.CalcDescriptors(mol)
            arr = np.array(values, dtype=np.float32)
            # Replace NaN/Inf with 0
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr
        except Exception:
            return np.zeros(len(self._descriptor_names), dtype=np.float32)

    def _gin_embedding(self, smiles: str) -> Optional[np.ndarray]:
        """GIN supervised masking embedding (300 dims) via DGL-Life.

        Uses the same pretrained model and featurization as the benchmark
        (novoexpert1-tdc-benchmark/run_benchmark.py:compute_gin_embeddings).
        """
        import torch
        from rdkit import Chem

        if self._gin_transformer is None:
            try:
                from dgllife.model import load_pretrained
                self._gin_transformer = load_pretrained('gin_supervised_masking')
                self._gin_transformer.eval()
                logger.info("Loaded GIN supervised masking model (DGL-Life)")
            except Exception as e:
                logger.warning(f"Failed to load GIN model: {e}")
                self.use_gin = False
                return None

        try:
            from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(300, dtype=np.float32)

            g = mol_to_bigraph(
                mol,
                node_featurizer=PretrainAtomFeaturizer(),
                edge_featurizer=PretrainBondFeaturizer(),
                add_self_loop=True,
            )
            with torch.no_grad():
                node_feats = [g.ndata[k] for k in ['atomic_number', 'chirality_type']]
                edge_feats = [g.edata[k] for k in ['bond_type', 'bond_direction_type']]
                node_repr = self._gin_transformer(g, node_feats, edge_feats)
                # Mean pool over nodes → 300-dim graph embedding
                embedding = node_repr.mean(dim=0).numpy()
            return embedding.astype(np.float32)
        except Exception:
            return np.zeros(300, dtype=np.float32)

    # -- Caching --

    def _cache_key(self, smiles: str) -> str:
        """Generate cache key from canonical SMILES"""
        from rdkit import Chem
        # Canonicalize first to avoid cache misses on equivalent molecules
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol)
        else:
            canonical = smiles
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _load_cached(self, smiles: str) -> Optional[np.ndarray]:
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{self._cache_key(smiles)}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        return None

    def _save_cached(self, smiles: str, features: np.ndarray):
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{self._cache_key(smiles)}.npy"
        np.save(cache_file, features)


# =============================================================================
# Model Registry & Predictor
# =============================================================================

class TDCSOTAPredictor:
    """
    Unified predictor for all 22 TDC ADMET endpoints.
    
    Loads pre-trained models and routes each endpoint to its best model.
    Integrates with existing ADDIE service as a drop-in replacement for
    overlapping endpoints.
    
    Directory structure expected:
      models/
        maplight/
          caco2_wang/model.cbm           # CatBoost single model
          solubility_aqsoldb/model.cbm
        maplight_gnn/
          hia_hou/model.cbm
          cyp2d6_veith/seed_0.cbm..seed_4.cbm   # SOTA 5-seed ensemble
          cyp3a4_veith/seed_0.cbm..seed_4.cbm
          cyp3a4_substrate_carbonmangels/seed_0.cbm..seed_4.cbm
          clearance_hepatocyte_az/seed_0.cbm..seed_4.cbm
          ... (single model.cbm for non-SOTA endpoints)
        chemprop/
          dili/chemprop_seed_0.ckpt..chemprop_seed_4.ckpt  # SOTA Chemprop ensemble
          ld50_zhu/model.pt              # Chemprop winner
          lipophilicity_astrazeneca/model.pt
          pgp_substrate_wang/model.pt    # P-gp substrate (Wang+Esposito)
        admet_ai/
          ppbr_az/model.cbm
          vdss_lombardo/model.cbm
          half_life_obach/model.cbm
          clearance_microsome_az/model.cbm
        provenance.json
    """

    def __init__(self, use_gin: bool = True, cache_dir: Optional[str] = None):
        self.featurizer = MolecularFeaturizer(use_gin=use_gin, cache_dir=cache_dir)
        self.models: Dict[str, Any] = {}  # endpoint_name -> loaded model (or list for ensemble)
        # Optional per-endpoint post-hoc probability calibrators (isotonic), loaded
        # from a calibrator.json next to the seeds. tdc_name -> (x_knots, y_knots).
        # Only set for endpoints whose raw output is poorly calibrated (e.g. dictrank).
        self.calibrators: Dict[str, Any] = {}
        self.provenance: Dict[str, Dict] = {}
        self._admet_ai_model = None
        self._chemprop_trainer = None  # Cached Lightning Trainer for Chemprop inference
        # process_batch runs process_molecule in a ThreadPoolExecutor, so several
        # molecules call _predict_chemprop concurrently on the shared cached
        # Trainer. Lightning's Trainer.predict is not thread-safe — concurrent
        # use corrupted its state and threw, surfacing as intermittent null
        # hepatotoxicity under batch load. Serialize Chemprop inference.
        self._chemprop_lock = threading.Lock()
        logger.info(f"TDCSOTAPredictor initialized ({len(TDC_ENDPOINTS)} endpoints)")

    def load_models(self, models_dir: str):
        """Load all trained models from directory"""
        models_path = Path(models_dir)
        loaded = 0

        for endpoint_name, endpoint in TDC_ENDPOINTS.items():
            try:
                if endpoint.best_model in (ModelSource.MAPLIGHT, ModelSource.MAPLIGHT_GNN):
                    self._load_catboost_model(models_path, endpoint)
                elif endpoint.best_model == ModelSource.CHEMPROP:
                    self._load_chemprop_model(models_path, endpoint)
                elif endpoint.best_model == ModelSource.ADMET_AI:
                    self._load_admet_ai_model(models_path, endpoint)
                loaded += 1
            except Exception as e:
                logger.error(f"Failed to load model for {endpoint_name}: {e}")

        # Load provenance
        prov_file = models_path / "provenance.json"
        if prov_file.exists():
            with open(prov_file) as f:
                self.provenance = json.load(f)

        logger.info(f"Loaded {loaded}/{len(TDC_ENDPOINTS)} TDC SOTA models")

    def _load_calibrator(self, model_dir: Path, tdc_name: str):
        """Load an optional isotonic calibrator (calibrator.json) next to the seeds."""
        import json as _json
        cf = model_dir / "calibrator.json"
        if cf.exists():
            try:
                d = _json.loads(cf.read_text())
                self.calibrators[tdc_name] = (np.asarray(d["x"], dtype=float), np.asarray(d["y"], dtype=float))
                logger.info(f"Loaded isotonic calibrator: {tdc_name}")
            except Exception as e:
                logger.warning(f"Failed to load calibrator for {tdc_name}: {e}")

    def _apply_calibration(self, tdc_name: str, p):
        """Apply the per-endpoint isotonic calibrator (np.interp) if one exists; else
        return p unchanged. Works on a scalar or a numpy array."""
        cal = self.calibrators.get(tdc_name)
        if cal is None:
            return p
        x, y = cal
        return np.interp(p, x, y)

    def _load_catboost_model(self, models_path: Path, endpoint: TDCEndpoint):
        """Load CatBoost model(s) for a MapLight/MapLight+GNN endpoint.

        Supports both single model.cbm and 5-seed ensembles (seed_0.cbm..seed_4.cbm).
        Ensemble predictions are averaged at inference time.
        """
        from catboost import CatBoostClassifier, CatBoostRegressor

        source_dir = endpoint.best_model.value  # "maplight" or "maplight_gnn"
        model_dir = models_path / source_dir / endpoint.tdc_name

        # Check for ensemble (seed_*.cbm)
        seed_files = sorted(model_dir.glob("seed_*.cbm"))
        if seed_files:
            models = []
            for sf in seed_files:
                if endpoint.task_type == TaskType.CLASSIFICATION:
                    m = CatBoostClassifier()
                else:
                    m = CatBoostRegressor()
                m.load_model(str(sf))
                models.append(m)
            self.models[endpoint.tdc_name] = models  # list = ensemble
            self._load_calibrator(model_dir, endpoint.tdc_name)
            logger.info(f"Loaded CatBoost ensemble ({len(models)} seeds): {endpoint.tdc_name}")
            return

        # Fallback: single model.cbm
        model_file = model_dir / "model.cbm"
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_dir} (no seed_*.cbm or model.cbm)")

        if endpoint.task_type == TaskType.CLASSIFICATION:
            model = CatBoostClassifier()
        else:
            model = CatBoostRegressor()

        model.load_model(str(model_file))
        self.models[endpoint.tdc_name] = model  # single model
        logger.info(f"Loaded CatBoost model: {endpoint.tdc_name} ({source_dir})")

    def _load_chemprop_model(self, models_path: Path, endpoint: TDCEndpoint):
        """Load Chemprop model(s) — v2 .ckpt ensemble, v2 single .pt, or v1 legacy .pt.

        Supports (in priority order):
          1. Ensemble: chemprop/{endpoint}/chemprop_seed_*.ckpt  (Chemprop v2 Lightning)
          2. Single:   chemprop/{endpoint}/model.pt              (try v2 first, fall back to v1)
        """
        import torch

        model_dir = models_path / "chemprop" / endpoint.tdc_name

        # Check for ensemble (chemprop_seed_*.ckpt) — Chemprop v2 Lightning format
        seed_files = sorted(model_dir.glob("chemprop_seed_*.ckpt"))
        if seed_files:
            from chemprop.models import MPNN
            models = []
            for sf in seed_files:
                m = MPNN.load_from_checkpoint(str(sf))
                m.eval()
                models.append(m)
            self.models[endpoint.tdc_name] = models  # list = ensemble
            logger.info(f"Loaded Chemprop v2 ensemble ({len(models)} seeds): {endpoint.tdc_name}")
            return

        # Single model.pt
        model_file = model_dir / "model.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Chemprop model not found: {model_dir}")

        # Try Chemprop v2 format first
        try:
            from chemprop.models import MPNN
            model = MPNN.load_from_file(str(model_file))
            model.eval()
            self.models[endpoint.tdc_name] = model
            logger.info(f"Loaded Chemprop v2 model: {endpoint.tdc_name}")
            return
        except Exception as v2_err:
            logger.info(f"Chemprop v2 load failed for {endpoint.tdc_name} ({v2_err}), trying v1 format")

        # Fall back to Chemprop v1 format: {model_state_dict, hparams, scaler}
        ckpt = torch.load(str(model_file), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format for {endpoint.tdc_name}")

        self.models[endpoint.tdc_name] = {
            "_chemprop_v1": True,
            "state_dict": ckpt["model_state_dict"],
            "hparams": ckpt.get("hparams", {}),
            "scaler": ckpt.get("scaler"),
        }
        logger.info(
            f"Loaded Chemprop v1 model: {endpoint.tdc_name} "
            f"(hparams={ckpt.get('hparams', {})})"
        )

    def _load_admet_ai_model(self, models_path: Path, endpoint: TDCEndpoint):
        """Load an ADMET-AI endpoint — try CatBoost first, fall back to ADMET-AI.

        In production all 22 endpoints are deployed as CatBoost .cbm files
        regardless of which method originally won the benchmark. This method
        checks for a CatBoost model file under admet_ai/{endpoint}/model.cbm
        before falling back to the native ADMET-AI multi-task model.
        """
        # Try CatBoost .cbm first (preferred in production)
        cbm_file = models_path / "admet_ai" / endpoint.tdc_name / "model.cbm"
        if cbm_file.exists():
            try:
                from catboost import CatBoostClassifier, CatBoostRegressor
                if endpoint.task_type == TaskType.CLASSIFICATION:
                    model = CatBoostClassifier()
                else:
                    model = CatBoostRegressor()
                model.load_model(str(cbm_file))
                self.models[endpoint.tdc_name] = model
                # Override best_model so predict() uses CatBoost path
                endpoint.best_model = ModelSource.MAPLIGHT
                logger.info(f"Loaded CatBoost model for ADMET-AI endpoint: {endpoint.tdc_name}")
                return
            except Exception as e:
                logger.warning(f"CatBoost fallback failed for {endpoint.tdc_name}: {e}")

        # Fall back to ADMET-AI multi-task model
        if self._admet_ai_model is None:
            try:
                from admet_ai import ADMETModel
                self._admet_ai_model = ADMETModel()
                logger.info("Loaded ADMET-AI multi-task model")
            except ImportError:
                # Fall back to loading individual checkpoints
                model_dir = models_path / "admet_ai" / endpoint.tdc_name
                if model_dir.exists():
                    logger.info(f"Loaded ADMET-AI checkpoint: {endpoint.tdc_name}")
                else:
                    raise FileNotFoundError(f"ADMET-AI model not found: {model_dir}")

        self.models[endpoint.tdc_name] = self._admet_ai_model

    def predict(self, smiles: str) -> Dict[str, Any]:
        """
        Predict all 22 TDC ADMET endpoints for a single molecule.
        
        Returns dict with standardized field names matching ADDIE convention:
        {
            "herg_blocker_probability": 0.23,
            "ames_mutagenicity_probability": 0.15,
            "cyp2c9_inhibitor_probability": 0.67,
            ...
            "_provenance": {
                "herg": {"model": "maplight_gnn", "tdc_score": 0.880, "metric": "auroc"},
                ...
            }
        }
        """
        results = {}
        provenance = {}
        errors = {}  # per-endpoint failure reasons, surfaced so a null isn't silent

        # Compute features once (shared by all MapLight/MapLight+GNN models)
        features = self.featurizer.featurize(smiles)
        if features is None:
            logger.error(f"Could not featurize SMILES: {smiles[:50]}")
            return {"error": "Invalid SMILES", "smiles": smiles}

        features_2d = features.reshape(1, -1)

        for endpoint_name, endpoint in TDC_ENDPOINTS.items():
            if endpoint_name not in self.models:
                continue

            try:
                model = self.models[endpoint_name]

                if endpoint.best_model in (ModelSource.MAPLIGHT, ModelSource.MAPLIGHT_GNN):
                    # CatBoost prediction — single or ensemble
                    if isinstance(model, list):
                        # Ensemble: average predictions across seeds
                        preds = []
                        for m in model:
                            if endpoint.task_type == TaskType.CLASSIFICATION:
                                preds.append(m.predict_proba(features_2d)[0][1])
                            else:
                                preds.append(m.predict(features_2d)[0])
                        pred = float(np.mean(preds))
                    else:
                        if endpoint.task_type == TaskType.CLASSIFICATION:
                            pred = model.predict_proba(features_2d)[0][1]
                        else:
                            pred = model.predict(features_2d)[0]

                elif endpoint.best_model == ModelSource.CHEMPROP:
                    # Chemprop — v2 ensemble, v2 single, or v1 legacy
                    if isinstance(model, dict) and model.get("_chemprop_v1"):
                        pred = self._predict_chemprop_v1(model, smiles, endpoint)
                    elif isinstance(model, list):
                        preds = [self._predict_chemprop(m, smiles, endpoint) for m in model]
                        pred = float(np.mean(preds))
                    else:
                        pred = self._predict_chemprop(model, smiles, endpoint)

                elif endpoint.best_model == ModelSource.ADMET_AI:
                    # ADMET-AI prediction (uses SMILES directly)
                    if hasattr(model, 'predict'):
                        admet_preds = model.predict(smiles=smiles)
                        # Map ADMET-AI field name to our value
                        pred = self._extract_admet_ai_prediction(
                            admet_preds, endpoint
                        )
                    else:
                        continue

                # Store with ADDIE-compatible field name
                results[endpoint.addie_field] = float(self._apply_calibration(endpoint.tdc_name, pred))

                # Track provenance
                provenance[endpoint_name] = {
                    "model": endpoint.best_model.value,
                    "tdc_benchmark_score": endpoint.best_score,
                    "metric": endpoint.metric.value,
                }

            except Exception as e:
                logger.error(f"Prediction failed for {endpoint_name}: {e}")
                results[endpoint.addie_field] = None
                errors[endpoint_name] = f"{type(e).__name__}: {e}"

        results["_provenance"] = provenance
        if errors:
            results["_errors"] = errors
        return results

    def predict_batch(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """Predict all 22 endpoints for a batch of molecules"""
        # For CatBoost models, batch featurization is more efficient
        feature_matrix, valid_indices = self.featurizer.featurize_batch(smiles_list)

        all_results = [{"error": "Invalid SMILES"} for _ in smiles_list]

        if len(valid_indices) == 0:
            return all_results

        for endpoint_name, endpoint in TDC_ENDPOINTS.items():
            if endpoint_name not in self.models:
                continue

            model = self.models[endpoint_name]

            try:
                if endpoint.best_model in (ModelSource.MAPLIGHT, ModelSource.MAPLIGHT_GNN):
                    # Average across the seed ensemble (SOTA winners are 5-seed
                    # lists); a single model is treated as a 1-seed ensemble. The
                    # previous code assumed a bare model and threw on the list,
                    # silently dropping ensemble endpoints from batch output.
                    cb_models = model if isinstance(model, list) else [model]
                    if endpoint.task_type == TaskType.CLASSIFICATION:
                        preds = np.mean([m.predict_proba(feature_matrix)[:, 1] for m in cb_models], axis=0)
                        preds = self._apply_calibration(endpoint.tdc_name, preds)
                    else:
                        preds = np.mean([m.predict(feature_matrix) for m in cb_models], axis=0)

                    for batch_idx, original_idx in enumerate(valid_indices):
                        if "error" in all_results[original_idx]:
                            all_results[original_idx] = {}
                        all_results[original_idx][endpoint.addie_field] = float(preds[batch_idx])

                elif endpoint.best_model == ModelSource.ADMET_AI:
                    # ADMET-AI handles its own batching
                    valid_smiles = [smiles_list[i] for i in valid_indices]
                    if hasattr(model, 'predict'):
                        admet_preds = model.predict(smiles=valid_smiles)
                        for batch_idx, original_idx in enumerate(valid_indices):
                            if "error" in all_results[original_idx]:
                                all_results[original_idx] = {}
                            pred = self._extract_admet_ai_prediction_batch(
                                admet_preds, endpoint, batch_idx
                            )
                            all_results[original_idx][endpoint.addie_field] = pred

                elif endpoint.best_model == ModelSource.CHEMPROP:
                    # Batched Chemprop: one Trainer.predict per seed across the
                    # WHOLE batch (5 calls for 50 molecules, not 250). v1 legacy
                    # checkpoints have no batch path — fall back per-molecule.
                    from rdkit import Chem
                    if isinstance(model, dict) and model.get("_chemprop_v1"):
                        batch_preds = [
                            self._predict_chemprop_v1(model, s, endpoint) if Chem.MolFromSmiles(s) else None
                            for s in smiles_list
                        ]
                    else:
                        batch_preds = self._predict_chemprop_batch(model, smiles_list, endpoint)
                    for original_idx, val in enumerate(batch_preds):
                        if val is None:
                            continue
                        if "error" in all_results[original_idx]:
                            all_results[original_idx] = {}
                        all_results[original_idx][endpoint.addie_field] = float(val)

            except Exception as e:
                logger.error(f"Batch prediction failed for {endpoint_name}: {e}")

        return all_results

    def _predict_chemprop(self, model, smiles: str, endpoint: TDCEndpoint) -> float:
        """Run Chemprop v2 inference on a single SMILES."""
        import torch
        import chemprop
        from scipy.special import expit
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.5 if endpoint.task_type == TaskType.CLASSIFICATION else 0.0

        dp = chemprop.data.MoleculeDatapoint(mol, np.array([0.0]))
        dataset = chemprop.data.MoleculeDataset([dp])
        loader = chemprop.data.build_dataloader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )

        # Lazy-create and reuse one Lightning Trainer, but serialize all use of
        # it: Trainer.predict is not thread-safe and process_molecule runs in a
        # thread pool, so concurrent calls raced and threw under batch load
        # (intermittent null hepatotoxicity). The lock makes Chemprop inference
        # correct-under-concurrency; CatBoost endpoints are unaffected.
        with self._chemprop_lock:
            if self._chemprop_trainer is None:
                from lightning.pytorch import Trainer
                self._chemprop_trainer = Trainer(
                    logger=False, enable_progress_bar=False,
                    accelerator="cpu", devices=1,
                )
            preds = self._chemprop_trainer.predict(model, loader)
        raw = float(torch.cat(preds).numpy().flatten()[0])

        # Chemprop v2's predict_step already applies the output transform — sigmoid
        # for classification — so trainer.predict() returns a probability, not a
        # logit. The previous `expit(raw)` here applied sigmoid a SECOND time, which
        # compressed every classification probability into [0.5, 0.73] (the flat
        # hepatotoxicity band reported in the 50-drug screen). Return the model
        # output directly, matching train_chemprop_tdc.py, which feeds predict()
        # straight into AUROC with no extra activation.
        return raw

    def _predict_chemprop_batch(self, model, smiles_list: List[str], endpoint: TDCEndpoint) -> List[Optional[float]]:
        """Batched Chemprop v2 inference — the core latency fix.

        Runs ONE Trainer.predict per seed across the whole batch instead of one
        call per (molecule, seed): 50 molecules × 5 seeds becomes 5 predict()
        calls, not 250. Numerics are identical to _predict_chemprop because an
        MPNN scores each molecular graph independently of batch composition.

        Returns a list aligned to smiles_list (None where the SMILES is invalid).
        Mirrors _predict_chemprop's output transform exactly — returns the model's
        already-sigmoided probability for classification, with NO second sigmoid.
        """
        import torch
        import chemprop
        from rdkit import Chem

        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        valid_idx = [i for i, m in enumerate(mols) if m is not None]
        if not valid_idx:
            return [None] * len(smiles_list)

        dps = [chemprop.data.MoleculeDatapoint(mols[i], np.array([0.0])) for i in valid_idx]
        dataset = chemprop.data.MoleculeDataset(dps)
        # Bounded batch_size keeps memory flat on large libraries; the dataloader
        # chunks and trainer.predict concatenates across chunks in order.
        loader = chemprop.data.build_dataloader(
            dataset, batch_size=min(len(dps), 64), shuffle=False, num_workers=0
        )

        seed_models = model if isinstance(model, list) else [model]

        # Serialize Trainer use (not thread-safe) exactly as _predict_chemprop does.
        with self._chemprop_lock:
            if self._chemprop_trainer is None:
                from lightning.pytorch import Trainer
                self._chemprop_trainer = Trainer(
                    logger=False, enable_progress_bar=False,
                    accelerator="cpu", devices=1,
                )
            seed_sum = None
            for m in seed_models:
                preds = self._chemprop_trainer.predict(m, loader)
                vals = torch.cat(preds).numpy().flatten()  # aligned to valid_idx order
                seed_sum = vals if seed_sum is None else seed_sum + vals
        mean_vals = seed_sum / len(seed_models)

        out: List[Optional[float]] = [None] * len(smiles_list)
        for pos, orig_i in enumerate(valid_idx):
            out[orig_i] = float(mean_vals[pos])
        return out

    def _predict_chemprop_v1(self, model_dict: Dict, smiles: str, endpoint: TDCEndpoint) -> float:
        """Run Chemprop v1 inference using raw PyTorch forward pass.

        v1 checkpoints store {model_state_dict, hparams, scaler}. We reconstruct
        the MPNN from chemprop v1's MoleculeModel class, load the state dict, and
        run a manual forward pass with chemprop v1-style featurization.
        """
        import torch
        from scipy.special import expit
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.5 if endpoint.task_type == TaskType.CLASSIFICATION else 0.0

        state_dict = model_dict["state_dict"]
        hparams = model_dict["hparams"]
        scaler = model_dict.get("scaler")

        # Lazy-build and cache the model on first call
        cache_key = f"_v1_model_{endpoint.tdc_name}"
        if not hasattr(self, cache_key):
            from chemprop.nn import BondMessagePassing, BinaryClassificationFFN, RegressionFFN, MeanAggregation
            from chemprop.models import MPNN

            hidden_dim = hparams.get("hidden_dim", 300)
            depth = hparams.get("depth", 3)
            dropout = hparams.get("dropout", 0.0)

            mp = BondMessagePassing(d_h=hidden_dim, depth=depth, dropout=dropout)

            if endpoint.task_type == TaskType.CLASSIFICATION:
                ffn = BinaryClassificationFFN(input_dim=hidden_dim)
            else:
                ffn = RegressionFFN(input_dim=hidden_dim)

            agg = MeanAggregation()
            built_model = MPNN(mp, agg, ffn)

            # Map v1 state_dict keys to v2 structure
            v2_state = {}
            for k, v in state_dict.items():
                # v1: message_passing.X → v2: message_passing.X (same)
                # v1: bn.X → v2: message_passing.bn.X
                # v1: predictor.X → v2: predictor.X (same)
                if k.startswith("bn."):
                    v2_state[f"message_passing.{k}"] = v
                elif k in ("metrics.0.task_weights", "metrics.1.task_weights",
                           "predictor.criterion.task_weights"):
                    continue  # Skip metric weights, not needed for inference
                else:
                    v2_state[k] = v

            try:
                built_model.load_state_dict(v2_state, strict=False)
            except Exception as e:
                logger.warning(f"Partial state_dict load for {endpoint.tdc_name}: {e}")

            built_model.eval()
            setattr(self, cache_key, built_model)

            # Cache scaler too
            if scaler:
                setattr(self, f"_v1_scaler_{endpoint.tdc_name}", scaler)

        cached_model = getattr(self, cache_key)
        cached_scaler = getattr(self, f"_v1_scaler_{endpoint.tdc_name}", None)

        # Use v2 prediction path with the rebuilt model
        raw = self._predict_chemprop(cached_model, smiles, endpoint)

        # Apply inverse scaler for regression (v1 models trained on normalized targets)
        if endpoint.task_type == TaskType.REGRESSION and cached_scaler:
            mean = float(cached_scaler["mean"][0])
            scale = float(cached_scaler["scale"][0])
            raw = raw * scale + mean

        return raw

    def _extract_admet_ai_prediction(self, preds: Dict, endpoint: TDCEndpoint) -> float:
        """Extract the relevant prediction from ADMET-AI's output"""
        # ADMET-AI uses its own naming convention, map to TDC endpoint
        ADMET_AI_FIELD_MAP = {
            "ppbr_az": "PPBR_AZ",
            "vdss_lombardo": "VDss_Lombardo",
            "half_life_obach": "Half_Life_Obach",
            "clearance_hepatocyte_az": "Clearance_Hepatocyte_AZ",
            "clearance_microsome_az": "Clearance_Microsome_AZ",
        }
        field = ADMET_AI_FIELD_MAP.get(endpoint.tdc_name, endpoint.tdc_name)
        return float(preds.get(field, 0.0))

    def _extract_admet_ai_prediction_batch(
        self, preds, endpoint: TDCEndpoint, idx: int
    ) -> float:
        """Extract prediction from ADMET-AI batch output (DataFrame)"""
        ADMET_AI_FIELD_MAP = {
            "ppbr_az": "PPBR_AZ",
            "vdss_lombardo": "VDss_Lombardo",
            "half_life_obach": "Half_Life_Obach",
            "clearance_hepatocyte_az": "Clearance_Hepatocyte_AZ",
            "clearance_microsome_az": "Clearance_Microsome_AZ",
        }
        field = ADMET_AI_FIELD_MAP.get(endpoint.tdc_name, endpoint.tdc_name)
        try:
            return float(preds.iloc[idx][field])
        except Exception:
            return 0.0

    def get_endpoint_summary(self) -> Dict[str, Any]:
        """Return summary of loaded models and their benchmark scores"""
        summary = {
            "total_endpoints": len(TDC_ENDPOINTS),
            "loaded_models": len(self.models),
            "by_source": {},
            "by_category": {},
            "endpoints": {},
        }

        for name, ep in TDC_ENDPOINTS.items():
            source = ep.best_model.value
            summary["by_source"][source] = summary["by_source"].get(source, 0) + 1
            summary["by_category"][ep.category] = summary["by_category"].get(ep.category, 0) + 1
            summary["endpoints"][name] = {
                "display_name": ep.display_name,
                "category": ep.category,
                "task": ep.task_type.value,
                "metric": ep.metric.value,
                "best_model": source,
                "benchmark_score": ep.best_score,
                "loaded": name in self.models,
                "addie_field": ep.addie_field,
            }

        return summary


# =============================================================================
# Integration with Existing ADDIE Standardizer
# =============================================================================

# Mapping: TDC endpoint -> existing ADDIE model it replaces
# Only endpoints with direct ADDIE equivalents are listed.
# ADDIE models without TDC equivalents continue to run unchanged.

ADDIE_REPLACEMENT_MAP = {
    # TDC endpoint            -> ADDIE model key it replaces (must match MODEL_LIST keys)
    "dili":                     "hepatotoxicity",
    "ames":                     "ames_mutagenicity",
    "herg":                     "cardiotox_general",
    "cyp2c9_veith":             "cyp2c9",
    "cyp2d6_veith":             "cyp2d6",
    "cyp3a4_veith":             "cyp3a4",
    # Tox21 panel (WS4 retrain) — replace the constant ~0.50 base-ADDIE heads.
    # Values are MODELS dict keys (underscores, the first MODEL_LIST tuple element),
    # NOT the dash-form S3 paths — a dash here silently fails to skip the base head.
    "tox21_nr_ar":              "nr_ar",
    "tox21_nr_ar_lbd":          "nr_ar_lbd",
    "tox21_nr_ahr":             "nr_ahr",
    "tox21_nr_aromatase":       "nr_aromatase",
    "tox21_nr_er":              "nr_er",
    "tox21_nr_er_lbd":          "nr_er_lbd",
    "tox21_nr_ppar_gamma":      "nr_ppar_gamma",
    "tox21_sr_are":             "sr_are",
    "tox21_sr_atad5":           "sr_atad5",
    "tox21_sr_hse":             "sr_hse",
    "tox21_sr_mmp":             "sr_mmp",
    "tox21_sr_p53":             "sr_p53",
    "clintox":                  "clinical_toxicity",
}

# ADDIE models that continue running (no TDC equivalent):
ADDIE_EXCLUSIVE_MODELS = [
    # Cardiotoxicity timepoints (TDC only has binary hERG)
    "cardiotox-1", "cardiotox-5", "cardiotox-10", "cardiotox-30",
    # Toxicity endpoints not in TDC 22
    "carcinogenicity", "clinical-toxicity", "respiratory-toxicity",
    "developmental-toxicity", "reproductive-toxicity",
    "eye-corrosion", "eye-irritation",
    # Nuclear receptors (Tox21, not in TDC 22)
    "nr-ahr", "nr-ar", "nr-ar-lbd", "nr-aromatase",
    "nr-er", "nr-er-lbd", "nr-ppar-gamma",
    # Stress response (Tox21, not in TDC 22)
    "sr-are", "sr-atad5", "sr-hse", "sr-mmp", "sr-p53",
    # CYP1A2 and CYP2C19 (in ADDIE but not TDC 22)
    "cyp1a2", "cyp2c19",
    # Affinity (not in TDC)
    "affinity-suite",
]

# NEW endpoints added by TDC (not previously in ADDIE):
TDC_NEW_ENDPOINTS = [
    "caco2_wang",               # Caco-2 permeability (regression)
    "hia_hou",                  # Human intestinal absorption
    "bioavailability_ma",       # Oral bioavailability
    "lipophilicity_astrazeneca",# Lipophilicity
    "solubility_aqsoldb",       # Aqueous solubility
    "bbb_martins",              # BBB penetration (ML, complements BOILED-Egg)
    "ppbr_az",                  # Plasma protein binding rate
    "vdss_lombardo",            # Volume of distribution
    "cyp2c9_substrate_carbonmangels",  # CYP2C9 substrate
    "cyp2d6_substrate_carbonmangels",  # CYP2D6 substrate
    "cyp3a4_substrate_carbonmangels",  # CYP3A4 substrate
    "half_life_obach",          # Half life
    "clearance_hepatocyte_az",  # Hepatocyte clearance
    "clearance_microsome_az",   # Microsomal clearance
    "ld50_zhu",                 # Acute toxicity LD50
]
