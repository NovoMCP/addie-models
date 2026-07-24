#!/usr/bin/env python3
"""
ADDIE Response Standardizer - Fixed Version
Maps all 31 deployed ADDIE models to standardized field names
Fixes:
1. Missing toxicity model mappings
2. Cardiotoxicity timepoint preservation
3. Complete stress-response model coverage
4. Nuclear receptor standardization
5. CYP inhibition mappings
"""

import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ADDIEStandardizer:
    """Standardizes ADDIE model outputs to consistent field names"""

    # Complete mapping of all 31 deployed models
    MODEL_MAPPINGS = {
        # Toxicity Models (9 models)
        'toxicity': {
            'hepatotoxicity': 'hepatotoxicity_probability',
            'ames-mutagenicity': 'ames_mutagenicity_probability',
            'carcinogenicity': 'carcinogenicity_probability',
            'clinical-toxicity': 'clinical_toxicity_probability',
            'respiratory-toxicity': 'respiratory_toxicity_probability',
            'developmental-toxicity': 'developmental_toxicity_probability',
            'reproductive-toxicity': 'reproductive_toxicity_probability',
            'eye-corrosion': 'eye_corrosion_probability',
            'eye-irritation': 'eye_irritation_probability',
        },

        # Cardiotoxicity Models with Timepoints (5 models)
        'cardiotoxicity': {
            'cardiotox-1': 'cardiotoxicity_1d_probability',
            'cardiotox-5': 'cardiotoxicity_5d_probability',
            'cardiotox-10': 'cardiotoxicity_10d_probability',
            'cardiotox-30': 'cardiotoxicity_30d_probability',
            'cardiotox-general': 'cardiotoxicity_probability',
        },

        # Nuclear Receptor Models (7 models)
        'nuclear_receptors': {
            'nr-ahr': 'nr_ahr_agonist_probability',
            'nr-ar': 'nr_ar_agonist_probability',
            'nr-ar-lbd': 'nr_ar_lbd_agonist_probability',
            'nr-aromatase': 'nr_aromatase_inhibitor_probability',
            'nr-er': 'nr_er_agonist_probability',
            'nr-er-lbd': 'nr_er_lbd_agonist_probability',
            'nr-ppar-gamma': 'nr_ppar_gamma_agonist_probability',
        },

        # Stress Response Models (5 models)
        'stress_response': {
            'sr-are': 'sr_are_activation_probability',
            'sr-atad5': 'sr_atad5_activation_probability',
            'sr-hse': 'sr_hse_activation_probability',
            'sr-mmp': 'sr_mmp_activation_probability',
            'sr-p53': 'sr_p53_activation_probability',
        },

        # CYP450 Inhibition Models (5 models, not in original 31 but often deployed)
        'cyp_inhibition': {
            'cyp1a2': 'cyp1a2_inhibitor_probability',
            'cyp2c19': 'cyp2c19_inhibitor_probability',
            'cyp2c9': 'cyp2c9_inhibitor_probability',
            'cyp2d6': 'cyp2d6_inhibitor_probability',
            'cyp3a4': 'cyp3a4_inhibitor_probability',
        },

        # Affinity Model (1 model)
        'affinity': {
            'affinity-suite': 'binding_affinity_score',
            'affinity-all': 'binding_affinity_score',
            'affinity': 'binding_affinity_score',
            'binding_affinity': 'binding_affinity_score',  # Fix: Add missing mapping
        },

        # TDC SOTA: Absorption — classification endpoints (0-1 probability)
        'absorption': {
            'hia_hou': 'hia_probability',
            'pgp_broccatelli': 'pgp_inhibitor_probability',
            'pgp_substrate_wang': 'pgp_substrate_probability',
            'bioavailability_ma': 'bioavailability_probability',
        },

        # TDC SOTA: Absorption — regression endpoints (continuous values, native units)
        'absorption_properties': {
            'caco2_wang': 'caco2_permeability',              # log cm/s
            'lipophilicity_astrazeneca': 'lipophilicity_log_ratio',  # log-ratio
            'solubility_aqsoldb': 'aqueous_solubility_log_mol_L',    # log mol/L
        },

        # TDC SOTA: Distribution — classification endpoints (0-1 probability)
        'distribution': {
            'bbb_martins': 'bbb_penetration_probability',
        },

        # TDC SOTA: Distribution — regression endpoints (continuous values, native units)
        'distribution_properties': {
            'ppbr_az': 'ppbr_percent',       # 0-100%
            'vdss_lombardo': 'vdss_L_kg',    # L/kg
        },

        # TDC SOTA: Excretion — all regression endpoints (continuous values, native units)
        'excretion': {
            'half_life_obach': 'half_life_hr',              # hours
            'clearance_hepatocyte_az': 'clearance_hepatocyte',  # uL/min/1e6 cells
            'clearance_microsome_az': 'clearance_microsome',    # mL/min/g
        },

        # TDC SOTA: CYP Substrates (3 endpoints, 0-1 probability)
        'cyp_substrates': {
            'cyp2c9_substrate_carbonmangels': 'cyp2c9_substrate_probability',
            'cyp2d6_substrate_carbonmangels': 'cyp2d6_substrate_probability',
            'cyp3a4_substrate_carbonmangels': 'cyp3a4_substrate_probability',
        },

        # TDC SOTA: Acute Toxicity — regression endpoint (continuous value)
        'acute_toxicity': {
            'ld50_zhu': 'ld50_log_mol_kg',   # log(1/(mol/kg))
        },
    }

    # Additional standardized fields for ADMET properties
    ADMET_FIELDS = {
        # Basic molecular descriptors
        'molecular_weight': ['MW', 'mw', 'molecular_weight'],
        'logp': ['LogP', 'logp', 'log_p', 'alogp'],
        'logd_74': ['LogD', 'logd', 'log_d', 'logd_7.4'],
        'tpsa': ['TPSA', 'tpsa', 'topological_polar_surface_area'],
        'hba': ['HBA', 'hba', 'h_bond_acceptors', 'num_h_acceptors'],
        'hbd': ['HBD', 'hbd', 'h_bond_donors', 'num_h_donors'],
        'rotatable_bonds': ['rotatable_bonds', 'num_rotatable_bonds', 'n_rotatable'],

        # Solubility
        'kinetic_solubility_uM': ['solubility', 'kinetic_solubility', 'ksol'],
        'thermodynamic_solubility_uM': ['thermodynamic_solubility', 'tsol'],

        # Permeability
        'caco2_permeability': ['caco2', 'caco2_perm', 'caco2_permeability'],
        'pampa_permeability': ['pampa', 'pampa_perm', 'pampa_permeability'],

        # Clearance (if models available)
        'cyp2d6_clearance_uL_min_mg': ['cyp2d6_clearance', 'cyp2d6_clr'],
        'cyp3a4_clearance_uL_min_mg': ['cyp3a4_clearance', 'cyp3a4_clr'],

        # Substrate predictions
        'cyp2d6_substrate_probability': ['cyp2d6_substrate', 'is_cyp2d6_substrate'],
        'cyp3a4_substrate_probability': ['cyp3a4_substrate', 'is_cyp3a4_substrate'],

        # Binding affinity fields
        'binding_affinity_pic50': ['pIC50', 'pic50', 'binding_pic50'],
        'binding_affinity_ic50_nM': ['IC50_nM', 'ic50_nm', 'binding_ic50'],
        'binding_affinity_ki_nM': ['Ki_nM', 'ki_nm', 'binding_ki'],
    }

    def __init__(self):
        """Initialize the standardizer"""
        self.field_lookup = self._build_field_lookup()
        logger.info(f"Initialized ADDIEResponseStandardizer with {len(self.field_lookup)} field mappings")

    def _build_field_lookup(self) -> Dict[str, str]:
        """Build reverse lookup dictionary for field mapping"""
        lookup = {}

        # Add model mappings
        for category, models in self.MODEL_MAPPINGS.items():
            for model_name, standard_field in models.items():
                lookup[model_name] = standard_field
                # Add variations
                lookup[model_name.replace('-', '_')] = standard_field
                lookup[model_name.replace('-', '')] = standard_field

        # Add ADMET field mappings
        for standard_field, variations in self.ADMET_FIELDS.items():
            for variation in variations:
                lookup[variation.lower()] = standard_field

        return lookup

    def standardize_predictions(self, raw_predictions: Dict[str, float], smiles: str) -> Dict[str, Any]:
        """
        Convert raw model predictions to industry-standard outputs

        Args:
            raw_predictions: Dictionary with 31 raw model outputs
            smiles: SMILES string for the molecule

        Returns:
            Dictionary with industry-standard fields and units
        """
        standardized = {}
        predictions_data = raw_predictions

        # Map all fields
        for field_name, value in predictions_data.items():
            if field_name in ['id', 'smiles', 'error', 'confidence']:
                continue

            # Try to find standard field name
            standard_field = self._find_standard_field(field_name)

            if standard_field:
                standardized[standard_field] = value
                logger.debug(f"Mapped {field_name} -> {standard_field}")
            else:
                # Keep unrecognized fields with original names
                standardized[field_name] = value
                logger.warning(f"No mapping found for field: {field_name}")

        # Add molecular descriptors if SMILES provided
        if smiles:
            standardized.update(self._calculate_descriptors(smiles))

        # Calculate aggregate scores
        standardized.update(self._calculate_aggregates(standardized))

        return standardized

    def _find_standard_field(self, field_name: str) -> Optional[str]:
        """Find the standard field name for a given input field"""
        # Direct lookup
        field_lower = field_name.lower()
        if field_lower in self.field_lookup:
            return self.field_lookup[field_lower]

        # Try removing common prefixes
        prefixes = ['tox_', 'nr_', 'sr_', 'cyp_', 'addie_']
        for prefix in prefixes:
            if field_lower.startswith(prefix):
                stripped = field_lower[len(prefix):]
                if stripped in self.field_lookup:
                    return self.field_lookup[stripped]

        # Try partial matching for cardiotoxicity timepoints
        if 'cardiotox' in field_lower:
            for timepoint in ['1', '5', '10', '30']:
                if timepoint in field_lower:
                    return f'cardiotoxicity_{timepoint}d_probability'

        return None

    # Fields that are continuous values (NOT 0-1 probabilities) — excluded from category score averaging
    NON_PROBABILITY_FIELDS = {
        'caco2_permeability',           # log cm/s
        'lipophilicity_log_ratio',      # log-ratio
        'aqueous_solubility_log_mol_L', # log mol/L
        'ppbr_percent',                 # 0-100%
        'vdss_L_kg',                    # L/kg
        'half_life_hr',                 # hours
        'clearance_hepatocyte',         # uL/min/1e6 cells
        'clearance_microsome',          # mL/min/g
        'ld50_log_mol_kg',             # log(1/(mol/kg))
        'binding_affinity_score',       # arbitrary scale
    }

    def _calculate_aggregates(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate scores from individual predictions.

        IMPORTANT: Only 0-1 probability fields are used for category score
        averaging.  Continuous-valued fields (log units, percentages, hours,
        clearance rates) are kept as raw values but excluded from averages
        to prevent absurd category scores like -222% or 3226%.
        """
        aggregates = {}

        # Helper: get float values for a list of fields, filtering out non-numeric and None
        def _values(field_names):
            return [predictions[f] for f in field_names
                    if f in predictions and isinstance(predictions[f], (int, float))]

        # --- ADMET Category Scores (only 0-1 probability fields) ---

        # Absorption score: HIA, P-gp inhibition, bioavailability (all 0-1 classification)
        absorption_prob_fields = ['hia_probability', 'pgp_inhibitor_probability', 'bioavailability_probability']
        absorption_vals = _values(absorption_prob_fields)
        if absorption_vals:
            aggregates['absorption_score'] = sum(absorption_vals) / len(absorption_vals)

        # Distribution score: BBB penetration (only classification endpoint in distribution)
        distribution_prob_fields = ['bbb_penetration_probability']
        distribution_vals = _values(distribution_prob_fields)
        if distribution_vals:
            aggregates['distribution_score'] = sum(distribution_vals) / len(distribution_vals)

        # CNS penetration assessment: BBB + P-gp substrate + P-gp inhibitor
        bbb = predictions.get('bbb_penetration_probability')
        pgp_sub = predictions.get('pgp_substrate_probability')  # efflux risk
        pgp_inh = predictions.get('pgp_inhibitor_probability')  # interim proxy if substrate unavailable
        if isinstance(bbb, (int, float)):
            cns = {'bbb_penetration': round(bbb, 3)}
            # Use substrate model if available, fall back to inhibitor as proxy
            efflux_prob = pgp_sub if isinstance(pgp_sub, (int, float)) else pgp_inh
            efflux_source = 'pgp_substrate' if isinstance(pgp_sub, (int, float)) else 'pgp_inhibitor_proxy'
            if isinstance(efflux_prob, (int, float)):
                cns['efflux_probability'] = round(efflux_prob, 3)
                cns['efflux_source'] = efflux_source
                # Classification: high BBB + high efflux = warning
                if bbb > 0.7 and efflux_prob > 0.5:
                    cns['cns_risk'] = 'efflux_warning'
                    cns['cns_note'] = 'Predicted CNS-penetrant but likely cleared by P-gp efflux'
                elif bbb > 0.7 and efflux_prob <= 0.5:
                    cns['cns_risk'] = 'favorable'
                    cns['cns_note'] = 'Predicted CNS-penetrant with low efflux risk'
                elif bbb <= 0.7:
                    cns['cns_risk'] = 'low_penetration'
                    cns['cns_note'] = 'Low predicted BBB penetration'
            aggregates['cns_assessment'] = cns

        # Metabolism score: CYP inhibitor + substrate probabilities (all 0-1 classification)
        metabolism_fields = [f for f in predictions.keys()
                           if 'cyp' in f and ('inhibitor_probability' in f or 'substrate_probability' in f)]
        metabolism_vals = _values(metabolism_fields)
        if metabolism_vals:
            aggregates['metabolism_score'] = sum(metabolism_vals) / len(metabolism_vals)

        # Excretion: all regression endpoints (half-life, clearance) — no probability score
        # Raw values (half_life_hr, clearance_hepatocyte, clearance_microsome) are returned as-is

        # Overall toxicity score (average of all toxicity probabilities)
        tox_fields = [f for f in predictions.keys()
                     if 'toxicity_probability' in f or f in (
                         'herg_blocker_probability', 'ames_mutagenicity_probability',
                         'carcinogenicity_probability', 'clinical_toxicity_probability',
                         'respiratory_toxicity_probability', 'developmental_toxicity_probability',
                         'reproductive_toxicity_probability', 'eye_corrosion_probability',
                         'eye_irritation_probability')]
        tox_vals = _values(tox_fields)
        if tox_vals:
            aggregates['overall_toxicity_score'] = sum(tox_vals) / len(tox_vals)

        # --- Existing sub-category aggregates ---

        # Cardiotoxicity aggregate (max across timepoints)
        cardio_fields = [f for f in predictions.keys() if 'cardiotoxicity' in f and 'd_probability' in f]
        cardio_vals = _values(cardio_fields)
        if cardio_vals:
            aggregates['cardiotoxicity_max_probability'] = max(cardio_vals)

        # Nuclear receptor activity score
        nr_fields = [f for f in predictions.keys() if f.startswith('nr_')]
        nr_vals = _values(nr_fields)
        if nr_vals:
            aggregates['nuclear_receptor_activity_score'] = sum(nr_vals) / len(nr_vals)

        # Stress response activity score
        sr_fields = [f for f in predictions.keys() if f.startswith('sr_')]
        sr_vals = _values(sr_fields)
        if sr_vals:
            aggregates['stress_response_activity_score'] = sum(sr_vals) / len(sr_vals)

        # CYP inhibition risk score (max across CYP inhibitor probabilities)
        cyp_inh_fields = [f for f in predictions.keys() if 'cyp' in f and 'inhibitor_probability' in f]
        cyp_inh_vals = _values(cyp_inh_fields)
        if cyp_inh_vals:
            aggregates['cyp_inhibition_risk_score'] = max(cyp_inh_vals)

        # CYP substrate max probability (across 3 TDC substrate endpoints)
        cyp_sub_fields = [f for f in predictions.keys() if 'cyp' in f and 'substrate_probability' in f]
        cyp_sub_vals = _values(cyp_sub_fields)
        if cyp_sub_vals:
            aggregates['cyp_substrate_max_probability'] = max(cyp_sub_vals)

        return aggregates

    def batch_standardize(self, raw_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize a batch of responses"""
        # Fix: Use the correct method name
        standardized = []
        for response in raw_responses:
            # Extract molecule info
            mol_id = response.get('id', 'unknown')
            smiles = response.get('smiles', '')

            # Get predictions
            if 'predictions' in response:
                predictions = response['predictions']
            else:
                predictions = {k: v for k, v in response.items() if k not in ['id', 'smiles']}

            # Standardize predictions
            std_predictions = self.standardize_predictions(predictions, smiles)

            # Build standardized response
            standardized.append({
                'id': mol_id,
                'smiles': smiles,
                'predictions': std_predictions
            })

        return standardized

    def get_model_coverage(self) -> Dict[str, List[str]]:
        """Return the complete list of models covered by the standardizer"""
        coverage = {}
        for category, models in self.MODEL_MAPPINGS.items():
            coverage[category] = list(models.keys())
        return coverage

    # TDC categories that are optional — don't block health when TDC models aren't deployed
    TDC_OPTIONAL_CATEGORIES = {
        'absorption', 'absorption_properties',
        'distribution', 'distribution_properties',
        'excretion', 'cyp_substrates', 'acute_toxicity',
    }

    def validate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that a response contains expected fields"""
        validation = {
            'valid': True,
            'missing_categories': [],
            'coverage_stats': {}
        }

        predictions = response.get('predictions', {})

        for category, models in self.MODEL_MAPPINGS.items():
            expected_fields = list(models.values())
            found_fields = [f for f in expected_fields if f in predictions]

            coverage = len(found_fields) / len(expected_fields) if expected_fields else 0
            validation['coverage_stats'][category] = {
                'expected': len(expected_fields),
                'found': len(found_fields),
                'coverage': coverage,
                'missing': [f for f in expected_fields if f not in predictions]
            }

            # TDC-only categories are optional — don't invalidate when TDC isn't deployed
            if coverage < 0.5 and category not in self.TDC_OPTIONAL_CATEGORIES:
                validation['missing_categories'].append(category)

        if validation['missing_categories']:
            validation['valid'] = False

        return validation


    def _calculate_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular descriptors from SMILES"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return {
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'hba': Descriptors.NumHAcceptors(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol)
                }
        except:
            pass
        return {}

def main():
    """Test the standardizer with example data"""
    standardizer = ADDIEStandardizer()

    # Example raw response (simulating actual ADDIE output)
    raw_response = {
        'id': 'mol_001',
        'smiles': 'CC(=O)Oc1ccccc1C(=O)O',
        'hepatotoxicity': 0.23,
        'ames-mutagenicity': 0.15,
        'cardiotox-1': 0.18,
        'cardiotox-5': 0.22,
        'cardiotox-10': 0.31,
        'cardiotox-30': 0.45,
        'nr-ahr': 0.12,
        'sr-are': 0.08,
        'sr-atad5': 0.11,
        'sr-hse': 0.09,
        'sr-mmp': 0.14,
        'sr-p53': 0.21,
        'affinity-suite': 7.8,
        'MW': 180.16,
        'LogP': 1.19,
        'TPSA': 63.6,
        'HBA': 4,
        'HBD': 1
    }

    # Standardize - Fix: Use the correct method name
    # Extract predictions from raw response
    predictions = {k: v for k, v in raw_response.items() if k not in ['id', 'smiles']}
    smiles = raw_response.get('smiles', '')
    standardized_predictions = standardizer.standardize_predictions(predictions, smiles)

    # Build full response
    standardized = {
        'id': raw_response.get('id', 'unknown'),
        'smiles': smiles,
        'predictions': standardized_predictions
    }

    # Print results
    print("Standardized Response:")
    print(json.dumps(standardized, indent=2))

    # Validate coverage
    validation = standardizer.validate_response(standardized)
    print("\nValidation Results:")
    print(json.dumps(validation, indent=2))

    # Show model coverage
    print("\nModel Coverage:")
    coverage = standardizer.get_model_coverage()
    for category, models in coverage.items():
        print(f"  {category}: {len(models)} models")
        for model in models[:3]:  # Show first 3
            print(f"    - {model}")
        if len(models) > 3:
            print(f"    ... and {len(models)-3} more")


if __name__ == "__main__":
    main()