#!/usr/bin/env python3
"""
ADDIE Models Service - FastAPI application for 31 ML models
Provides HTTPS API endpoint for molecular property predictions
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from collections import OrderedDict
import logging
import os
import sys
import json
import tarfile
import torch
import numpy as np
import boto3
import pickle
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
from addie_standardized import ADDIEStandardizer
try:
    import onnxruntime as ort  # Optional, used for ONNX hepatotoxicity model
except Exception:
    ort = None

# TDC SOTA model support
try:
    from tdc_sota_integration import TDCSOTAPredictor, TDC_ENDPOINTS, ADDIE_REPLACEMENT_MAP
    TDC_AVAILABLE = True
except ImportError:
    TDC_AVAILABLE = False
    TDC_ENDPOINTS = {}
    ADDIE_REPLACEMENT_MAP = {}

# TDC globals
TDC_PREDICTOR = None
TDC_MODEL_ERRORS: Dict[str, str] = {}
TDC_MODEL_PREFIX = os.environ.get('TDC_MODEL_PREFIX', 'tdc-sota/')
# Root under which the TDC v2/chemprop weight trees live. On the default
# Hugging Face backend the repo has `models/...` at its root, so the prefix is
# empty. An S3 backend that stores weights under an `addie-models/` key prefix
# should set TDC_SOTA_KEY_PREFIX to match, or every TDC download 404s and the
# TDC-only endpoints silently fall out of the response.
TDC_SOTA_KEY_PREFIX = os.environ.get(
    'TDC_SOTA_KEY_PREFIX',
    '' if os.environ.get('STORAGE_BACKEND', 'HF').upper() == 'HF' else 'addie-models/'
)
# Default ON: several TDC winners (Tox21 nr/sr, CYP/clearance) were trained with
# the 300-dim GIN embedding (2874-dim features). With GIN off they load but throw
# a feature-count mismatch at predict time and return null for those heads.
TDC_USE_GIN = os.environ.get('TDC_USE_GIN', 'true').lower() == 'true'

# Azure Blob Storage support
try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    BlobServiceClient = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ADDIE Models API",
    description="Molecular property prediction service with ADDIE + TDC SOTA models",
    version="1.0.0"
)
try:
    import sklearn
    logger.info(f"scikit-learn version: {getattr(sklearn, '__version__', 'unknown')}")
except Exception as e:
    logger.warning(f"Unable to import scikit-learn to log version: {e}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class Molecule(BaseModel):
    id: str
    smiles: str

class BatchRequest(BaseModel):
    molecules: List[Molecule]
    include_descriptors: bool = True
    include_confidence: bool = False
    tdc_only: bool = False  # Skip ADDIE models, only run TDC SOTA predictions

class MoleculeResult(BaseModel):
    id: str
    smiles: str
    predictions: Dict[str, Any]
    descriptors: Optional[Dict[str, float]]
    confidence: Optional[Dict[str, float]]
    error: Optional[str]

class BatchResponse(BaseModel):
    results: List[MoleculeResult]
    processing_time: float
    model_count: int

# Global variables for models
MODELS = {}
MODEL_CONFIGS = {}
# Track model-level load errors to surface in responses instead of silently dropping
MODEL_ERRORS: Dict[str, str] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get('EXECUTOR_WORKERS', '8')))

# Storage backend for model weights.
#   HF (default) — pull weights from a public Hugging Face repo; no cloud creds.
#   S3           — pull from an S3 bucket (set MODEL_BUCKET / MODEL_PREFIX).
#   AZURE        — legacy, inert.
STORAGE_BACKEND = os.environ.get('STORAGE_BACKEND', 'HF').upper()  # 'HF' | 'S3' | 'AZURE'

# S3 configuration
S3_CLIENT = None
if STORAGE_BACKEND == 'S3':
    S3_CLIENT = boto3.client('s3')

# Hugging Face Hub configuration — the default open-source path. Weights are
# pulled once from a public model repo into a local cache; model lookups then
# read from that directory. No cloud credentials required.
HF_MODEL_REPO = os.environ.get('HF_MODEL_REPO', 'NovoMCP/addie-models')
_HF_ROOT = None

def ensure_hf_root() -> str:
    """Download the public weights from Hugging Face once, return the local dir.

    Base heads land under `<root>/production/...` and the TDC overlay under
    `<root>/models/...`, mirroring the keys the loaders construct — so model
    lookups resolve as `<root>/<model_key>` with no path translation.
    """
    global _HF_ROOT
    if _HF_ROOT is None:
        from huggingface_hub import snapshot_download
        _HF_ROOT = snapshot_download(repo_id=HF_MODEL_REPO, repo_type='model')
        logging.info(f"Weights ready from Hugging Face {HF_MODEL_REPO} at {_HF_ROOT}")
    return _HF_ROOT

# Azure configuration
AZURE_BLOB_CLIENT = None
AZURE_STORAGE_ACCOUNT = os.environ.get('AZURE_STORAGE_ACCOUNT', '')
AZURE_STORAGE_KEY = os.environ.get('AZURE_STORAGE_KEY', '')
AZURE_CONTAINER = os.environ.get('AZURE_CONTAINER', 'addie-models')

if STORAGE_BACKEND == 'AZURE' and AZURE_AVAILABLE and AZURE_STORAGE_KEY:
    try:
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={AZURE_STORAGE_ACCOUNT};AccountKey={AZURE_STORAGE_KEY};EndpointSuffix=core.windows.net"
        AZURE_BLOB_CLIENT = BlobServiceClient.from_connection_string(connection_string)
        logging.info(f"Azure Blob client initialized successfully for {AZURE_STORAGE_ACCOUNT}/{AZURE_CONTAINER}")
    except Exception as e:
        logging.error(f"Failed to initialize Azure Blob client: {e}")

# Model configuration
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', '')  # S3 bucket (only used when STORAGE_BACKEND=S3)
MODEL_PREFIX = os.environ.get('MODEL_PREFIX', 'production/')  # Works for both S3 and Azure

# List of all 31 ADDIE models with their S3 paths
MODEL_LIST = [
    # Affinity/Binding
    ('binding_affinity', 'affinity/affinity-all/model.tar.gz'),

    # Cardiotoxicity models (4 time-window models + 1 aggregate computed downstream)
    # `cardiotox_general` was retired in favor of the TDC SOTA `herg` model
    # (AUROC 0.880, validated). See tdc_sota_integration.py:1178 +
    # models/tdc_sota/provenance.json. herg_blocker_probability now
    # populates the response slot the old cardiotoxicity_probability covered.
    ('cardiotox_1', 'cardiotoxicity/cardiotox-1/model.tar.gz'),
    ('cardiotox_5', 'cardiotoxicity/cardiotox-5/model.tar.gz'),
    ('cardiotox_10', 'cardiotoxicity/cardiotox-10/model.tar.gz'),
    ('cardiotox_30', 'cardiotoxicity/cardiotox-30/model.tar.gz'),

    # CYP450 models
    ('cyp1a2', 'cyp450/cyp1a2/model.tar.gz'),
    ('cyp2c19', 'cyp450/cyp2c19/model.tar.gz'),
    ('cyp2c9', 'cyp450/cyp2c9/model.tar.gz'),
    ('cyp2d6', 'cyp450/cyp2d6/model.tar.gz'),
    ('cyp3a4', 'cyp450/cyp3a4/model.tar.gz'),

    # Nuclear Receptor models
    ('nr_ahr', 'nuclear-receptors/nr-ahr/model.tar.gz'),
    ('nr_ar_lbd', 'nuclear-receptors/nr-ar-lbd/model.tar.gz'),
    ('nr_ar', 'nuclear-receptors/nr-ar/model.tar.gz'),
    ('nr_aromatase', 'nuclear-receptors/nr-aromatase/model.tar.gz'),
    ('nr_er_lbd', 'nuclear-receptors/nr-er-lbd/model.tar.gz'),
    ('nr_er', 'nuclear-receptors/nr-er/model.tar.gz'),
    ('nr_ppar_gamma', 'nuclear-receptors/nr-ppar-gamma/model.tar.gz'),

    # Stress Response models
    ('sr_are', 'stress-response/sr-are/model.tar.gz'),
    ('sr_atad5', 'stress-response/sr-atad5/model.tar.gz'),
    ('sr_hse', 'stress-response/sr-hse/model.tar.gz'),
    ('sr_mmp', 'stress-response/sr-mmp/model.tar.gz'),
    ('sr_p53', 'stress-response/sr-p53/model.tar.gz'),

    # Toxicity models
    ('ames_mutagenicity', 'toxicity/ames-mutagenicity/model.tar.gz'),
    ('carcinogenicity', 'toxicity/carcinogenicity/model.tar.gz'),
    ('clinical_toxicity', 'toxicity/clinical-toxicity/model.tar.gz'),
    ('developmental_toxicity', 'toxicity/developmental-toxicity/model.tar.gz'),
    ('eye_corrosion', 'toxicity/eye-corrosion/model.tar.gz'),
    ('eye_irritation', 'toxicity/eye-irritation/model.tar.gz'),
    ('hepatotoxicity', 'toxicity/hepatotoxicity/model.tar.gz'),
    ('reproductive_toxicity', 'toxicity/reproductive-toxicity/model.tar.gz'),
    ('respiratory_toxicity', 'toxicity/respiratory-toxicity/model.tar.gz'),
]

class ADDIEModel:
    """Wrapper for individual ADDIE model"""

    def __init__(self, model_name: str, model_path: str, model_type: str = 'pytorch'):
        self.name = model_name
        self.model = None
        self.model_type = model_type
        self.config = {}
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load model from file"""
        try:
            # Check if it's a sklearn model
            if self.model_type == 'sklearn' or model_path.endswith('.pkl') or model_path.endswith('.joblib'):
                # Load sklearn model
                try:
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                except:
                    # Try joblib if pickle fails
                    self.model = joblib.load(model_path)

                logger.info(f"Loaded sklearn model: {self.name}")
                self.model_type = 'sklearn'
                return

            # Otherwise load as PyTorch model
            checkpoint = torch.load(model_path, map_location='cpu')
            logger.info(f"Loaded checkpoint type: {type(checkpoint).__name__}")

            # Check if it's a full model or just state dict
            if isinstance(checkpoint, dict) and not isinstance(checkpoint, OrderedDict):
                # It's a checkpoint with state dict and possibly config
                if 'model' in checkpoint:
                    # Full model saved
                    self.model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # Need to reconstruct model from state dict
                    state_dict = checkpoint['state_dict']

                    # Try to infer architecture from state dict keys
                    layers = []
                    layer_keys = [k for k in state_dict.keys() if 'weight' in k]

                    for i, key in enumerate(layer_keys):
                        weight_shape = state_dict[key].shape
                        in_features = weight_shape[1]
                        out_features = weight_shape[0]

                        layers.append(torch.nn.Linear(in_features, out_features))

                        # Add activation except for last layer
                        if i < len(layer_keys) - 1:
                            layers.append(torch.nn.ReLU())
                            # Add dropout for intermediate layers
                            if i < len(layer_keys) - 2:
                                layers.append(torch.nn.Dropout(0.2))
                        else:
                            # Last layer - add sigmoid for binary classification
                            layers.append(torch.nn.Sigmoid())

                    self.model = torch.nn.Sequential(*layers)
                    self.model.load_state_dict(state_dict)

                elif 'model_state_dict' in checkpoint:
                    # Another common checkpoint format
                    state_dict = checkpoint['model_state_dict']

                    # Default architecture if we can't infer
                    input_dim = 2048  # Morgan fingerprint size
                    hidden_dim = 512
                    output_dim = 1

                    self.model = torch.nn.Sequential(
                        torch.nn.Linear(input_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.2),
                        torch.nn.Linear(hidden_dim, hidden_dim // 2),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.2),
                        torch.nn.Linear(hidden_dim // 2, output_dim),
                        torch.nn.Sigmoid()
                    )

                    try:
                        self.model.load_state_dict(state_dict)
                    except:
                        # If default doesn't work, try to build from state dict
                        logger.warning(f"Default architecture failed for {self.name}, attempting to infer...")
                        layers = []
                        for key in state_dict.keys():
                            if 'weight' in key:
                                weight = state_dict[key]
                                bias_key = key.replace('weight', 'bias')
                                if weight.dim() == 2:  # Linear layer
                                    layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
                                    layer.weight.data = weight
                                    if bias_key in state_dict:
                                        layer.bias.data = state_dict[bias_key]
                                    layers.append(layer)

                        if layers:
                            # Add activations between layers
                            model_layers = []
                            for i, layer in enumerate(layers):
                                model_layers.append(layer)
                                if i < len(layers) - 1:
                                    model_layers.append(torch.nn.ReLU())
                                else:
                                    model_layers.append(torch.nn.Sigmoid())
                            self.model = torch.nn.Sequential(*model_layers)

                # Store config if available
                if 'config' in checkpoint:
                    self.config = checkpoint['config']

            elif isinstance(checkpoint, torch.nn.Module):
                # Direct model object
                self.model = checkpoint
            elif isinstance(checkpoint, OrderedDict):
                # It's an OrderedDict state dict directly (common for ADDIE models)
                logger.info(f"Processing OrderedDict with {len(checkpoint)} keys")
                # Extract layer info from keys like 'model.0.weight', 'model.3.weight', etc.
                layers = []
                layer_indices = sorted(set(int(k.split('.')[1]) for k in checkpoint.keys() if '.' in k and k.split('.')[1].isdigit()))
                logger.info(f"Found layer indices: {layer_indices}")

                for i in range(len(layer_indices)):
                    idx = layer_indices[i]
                    weight_key = f'model.{idx}.weight'
                    bias_key = f'model.{idx}.bias'

                    if weight_key in checkpoint:
                        weight = checkpoint[weight_key]
                        in_features = weight.shape[1]
                        out_features = weight.shape[0]
                        logger.info(f"Layer {idx}: {in_features} -> {out_features}")

                        # Create layer
                        layer = torch.nn.Linear(in_features, out_features)
                        layer.weight.data = weight
                        if bias_key in checkpoint:
                            layer.bias.data = checkpoint[bias_key]

                        layers.append(layer)

                        # Add activation based on position
                        # Assuming ReLU for hidden layers and Sigmoid for output
                        if i < len(layer_indices) - 1:
                            # Not the last layer - add ReLU and Dropout
                            layers.append(torch.nn.ReLU())
                            layers.append(torch.nn.Dropout(0.2))
                        else:
                            # Last layer - add Sigmoid for binary classification
                            layers.append(torch.nn.Sigmoid())

                if layers:
                    self.model = torch.nn.Sequential(*layers)
                    logger.info(f"Built model with {len(layers)} layers")
                else:
                    # Fallback - no layers found
                    logger.warning(f"No layers found in OrderedDict for {self.name}")
                    self.model = None
            else:
                # Unknown checkpoint type
                logger.error(f"Unknown checkpoint type for {self.name}: {type(checkpoint)}")
                self.model = None

            if self.model is not None:
                self.model.eval()
                logger.info(f"Successfully loaded model: {self.name} with {sum(p.numel() for p in self.model.parameters())} parameters")
            else:
                logger.error(f"ERROR: Model {self.name} is None after loading - check logs above for details")

        except Exception as e:
            logger.error(f"Failed to load model {self.name}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't raise - allow service to start with partial models
            self.model = None

    def predict(self, fingerprint: np.ndarray) -> Dict[str, float]:
        """Make prediction with the model"""
        try:
            if self.model_type == 'sklearn':
                # Handle sklearn models
                # Ensure input is 2D
                if len(fingerprint.shape) == 1:
                    fingerprint = fingerprint.reshape(1, -1)

                # Get prediction probability
                if hasattr(self.model, 'predict_proba'):
                    # For classifiers with predict_proba
                    proba = self.model.predict_proba(fingerprint)[0]

                    # CRITICAL FIX: Get toxic probability (class 1)
                    # This was inverted - we were getting class 0 (non-toxic)
                    if len(proba) == 2:
                        prediction = proba[1]  # Get probability of toxic class
                    else:
                        prediction = proba[0]
                else:
                    # For regressors
                    prediction = self.model.predict(fingerprint)[0]

                # NOTE: Hepatotoxicity model now fixed with TF-IDF training
                # Model returns correct P(toxic) predictions - no inversion needed
                # if self.name == 'hepatotoxicity':
                #     prediction = 1.0 - prediction  # NO LONGER NEEDED - commented out

                confidence = abs(prediction - 0.5) * 2.0

                return {
                    'prediction': float(prediction),
                    'confidence': float(confidence),
                    'binary_class': int(prediction > 0.5)
                }
            else:
                # PyTorch models
                with torch.no_grad():
                    # Convert to tensor
                    x = torch.FloatTensor(fingerprint).unsqueeze(0)

                    # Get prediction
                    output = self.model(x)
                    prediction = output.squeeze().item()

                    # Calculate confidence (simplified)
                    confidence = abs(prediction - 0.5) * 2.0

                    return {
                        'prediction': prediction,
                        'confidence': confidence,
                        'binary_class': int(prediction > 0.5)
                    }

        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {str(e)}")
            return {
                'prediction': None,
                'confidence': 0.0,
                'error': str(e)
            }

def generate_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Generate Morgan fingerprint from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

        # Convert to numpy array
        arr = np.zeros((n_bits,))
        for i in range(n_bits):
            if fp.GetBit(i):
                arr[i] = 1

        return arr

    except Exception as e:
        logger.error(f"Fingerprint generation failed for {smiles}: {str(e)}")
        return None

def calculate_descriptors(smiles: str) -> Dict[str, float]:
    """Calculate molecular descriptors"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        descriptors = {
            'molecular_weight': Descriptors.ExactMolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'tpsa': Descriptors.TPSA(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol)
        }

        return descriptors

    except Exception as e:
        logger.error(f"Descriptor calculation failed for {smiles}: {str(e)}")
        return {}

def _parse_semver(v: str) -> List[int]:
    try:
        parts = v.strip().split('.')
        return [int(p) for p in parts[:3]] + [0] * (3 - len(parts))
    except Exception:
        return [0, 0, 0]


def download_model_file(model_key: str, local_path: str) -> bool:
    """Copy a model file into local_path from the configured backend (HF / S3 / Azure)."""
    if STORAGE_BACKEND == 'HF':
        import shutil
        src = os.path.join(ensure_hf_root(), model_key)
        if os.path.exists(src):
            shutil.copy(src, local_path)
            return True
        logger.error(f"HF weight not found: {model_key}")
        return False
    if STORAGE_BACKEND == 'AZURE' and AZURE_BLOB_CLIENT:
        try:
            blob_client = AZURE_BLOB_CLIENT.get_blob_client(container=AZURE_CONTAINER, blob=model_key)
            with open(local_path, 'wb') as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            return True
        except Exception as e:
            logger.error(f"Azure download failed for {model_key}: {e}")
            return False
    elif S3_CLIENT:
        try:
            S3_CLIENT.download_file(MODEL_BUCKET, model_key, local_path)
            return True
        except Exception as e:
            logger.error(f"S3 download failed for {model_key}: {e}")
            return False
    logger.error(f"No storage client available for {model_key}")
    return False


def check_blob_exists(model_key: str) -> bool:
    """Check whether a model file exists in the configured backend (HF / S3 / Azure)."""
    if STORAGE_BACKEND == 'HF':
        return os.path.exists(os.path.join(ensure_hf_root(), model_key))
    if STORAGE_BACKEND == 'AZURE' and AZURE_BLOB_CLIENT:
        try:
            blob_client = AZURE_BLOB_CLIENT.get_blob_client(container=AZURE_CONTAINER, blob=model_key)
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False
    elif S3_CLIENT:
        try:
            S3_CLIENT.head_object(Bucket=MODEL_BUCKET, Key=model_key)
            return True
        except Exception:
            return False
    return False


def load_models_from_storage() -> int:
    """Load all models from S3 or Azure Blob Storage at startup"""
    global MODELS

    if STORAGE_BACKEND == 'AZURE':
        logger.info(f"Loading models from Azure: {AZURE_STORAGE_ACCOUNT}/{AZURE_CONTAINER}/{MODEL_PREFIX}")
    else:
        logger.info(f"Loading models from S3: s3://{MODEL_BUCKET}/{MODEL_PREFIX}")

    loaded_count = 0
    for model_name, model_path in MODEL_LIST:
        try:
            # Download model archive (prefer ONNX variant if available)
            model_key = f"{MODEL_PREFIX}{model_path}"
            onnx_key = None
            if model_path.endswith('model.tar.gz'):
                onnx_key = f"{MODEL_PREFIX}{model_path[:-11]}model_onnx.tar.gz"
            archive_path = f"/tmp/{model_name}.tar.gz"
            extract_dir = f"/tmp/{model_name}_extracted"

            downloaded = False
            if onnx_key and check_blob_exists(onnx_key):
                logger.info(f"Found ONNX artifact, downloading {onnx_key}...")
                downloaded = download_model_file(onnx_key, archive_path)

            if not downloaded:
                logger.info(f"Downloading {model_key}...")
                downloaded = download_model_file(model_key, archive_path)

            if not downloaded:
                logger.error(f"Failed to download model {model_name}")
                MODEL_ERRORS[model_name] = "Download failed"
                continue

            # Extract tar.gz archive
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)

            # Find the model file (PyTorch or sklearn)
            model_file = None
            model_type = 'pytorch'  # default

            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    if file.endswith('.pth') or file.endswith('.pt'):
                        model_file = full_path
                        model_type = 'pytorch'
                        break
                    elif file.endswith('.pkl') or file.endswith('.joblib'):
                        model_file = full_path
                        model_type = 'sklearn'
                        break
                    # Also check for just "model" without extension
                    elif file == 'model':
                        model_file = full_path
                        model_type = 'pytorch'  # assume pytorch for backward compat
                        break
                if model_file:
                    break

            if not model_file:
                # Check for common names in extract directory
                for name in ['model.pkl', 'model.joblib', 'model.pth', 'model.pt', 'model', 'checkpoint.pth', 'checkpoint.pt']:
                    test_path = os.path.join(extract_dir, name)
                    if os.path.exists(test_path):
                        model_file = test_path
                        if name.endswith('.pkl') or name.endswith('.joblib'):
                            model_type = 'sklearn'
                        else:
                            model_type = 'pytorch'
                        break

            if model_file:
                logger.info(f"Found model file for {model_name}: {model_file} (type: {model_type})")
                # Try to read optional METADATA.json for artifact guidance
                metadata = {}
                metadata_path = os.path.join(extract_dir, 'METADATA.json')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as mf:
                            metadata = json.load(mf)
                        logger.info(f"Loaded artifact metadata for {model_name}: {metadata}")
                    except Exception as me:
                        logger.warning(f"Failed to parse METADATA.json for {model_name}: {me}")

                # Special handling for hepatotoxicity TF-IDF model
                if model_name == 'hepatotoxicity':
                    # Honor metadata: prefer framework-agnostic runtime when provided
                    artifact_framework = str(metadata.get('framework', '')).lower() if metadata else ''
                    required_sklearn = metadata.get('required_sklearn_version') or metadata.get('sklearn_version')
                    artifact_input = str(metadata.get('input', '')).lower() if metadata else ''

                    # If artifact indicates sklearn and version mismatch, flag early
                    if artifact_framework == 'sklearn' and required_sklearn:
                        try:
                            import sklearn as _sk
                            rt_ver = getattr(_sk, '__version__', '0.0.0')
                        except Exception:
                            rt_ver = '0.0.0'
                        if _parse_semver(rt_ver) < _parse_semver(str(required_sklearn)):
                            MODEL_ERRORS[model_name] = f"sklearn runtime {rt_ver} < artifact requires {required_sklearn}"
                            logger.error(f"{MODEL_ERRORS[model_name]}")
                            # Do not attempt to load incompatible pickle; try ONNX fallback if present
                    # Prefer ONNX pipeline if present (version-agnostic)
                    onnx_path = None
                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            if file.endswith('.onnx') and 'hepatotoxicity' in file.lower():
                                onnx_path = os.path.join(root, file)
                                break
                        if onnx_path:
                            break

                    if onnx_path and ort is not None:
                        logger.info(f"Loading ONNX hepatotoxicity model: {onnx_path}")

                        class HepatotoxicityONNXModel:
                            def __init__(self, onnx_file: str):
                                self.session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
                                self.input_name = self.session.get_inputs()[0].name
                                outs = self.session.get_outputs()
                                self.output_names = [o.name for o in outs]

                            def predict_from_smiles(self, smiles: str):
                                import numpy as np
                                # Most sklearn→ONNX pipelines with TfidfVectorizer accept string input
                                feed = {self.input_name: np.array([smiles])}
                                outputs = self.session.run(None, feed)
                                y = outputs[0]
                                # Handle shapes: (1,), (1,1), (1,2)
                                y = np.array(y)
                                if y.ndim == 0:
                                    prob = float(y)
                                elif y.ndim == 1:
                                    if y.shape[0] == 1:
                                        prob = float(y[0])
                                    elif y.shape[0] >= 2:
                                        prob = float(y[1])  # assume index 1 is P(toxic)
                                    else:
                                        prob = float(y.ravel()[0])
                                else:
                                    # Take last dimension index 1 if available, else first element
                                    if y.shape[-1] >= 2:
                                        prob = float(y.reshape(-1, y.shape[-1])[0, 1])
                                    else:
                                        prob = float(y.ravel()[0])
                                return [1.0 - prob, prob]

                        MODELS[model_name] = HepatotoxicityONNXModel(onnx_path)
                        loaded_count += 1
                        logger.info("Successfully loaded ONNX hepatotoxicity model")
                        # Skip legacy loading paths for this model
                        continue
                    elif onnx_path and ort is None:
                        MODEL_ERRORS[model_name] = "onnxruntime not available in runtime"
                        logger.error(f"{MODEL_ERRORS[model_name]}")
                    # If no ONNX file, try TF-IDF triplet (vectorizer + scaler + classifier)
                    # Check for vectorizer and scaler files
                    vectorizer_path = os.path.join(extract_dir, 'vectorizer.pkl')
                    scaler_path = os.path.join(extract_dir, 'scaler.pkl')

                    if os.path.exists(vectorizer_path) and os.path.exists(scaler_path):
                        logger.info(f"Loading TF-IDF hepatotoxicity model with vectorizer and scaler")
                        # Load all components
                        import pickle
                        with open(model_file, 'rb') as f:
                            actual_model = pickle.load(f)
                        with open(vectorizer_path, 'rb') as f:
                            vectorizer = pickle.load(f)
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)

                        # Verify we loaded the right objects (check for common mix-ups)
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from sklearn.preprocessing import StandardScaler

                        # If actual_model is actually a vectorizer, we have a file swap
                        if isinstance(actual_model, TfidfVectorizer):
                            logger.warning("Model file contains vectorizer! Swapping model and vectorizer...")
                            actual_model, vectorizer = vectorizer, actual_model

                        # Final check
                        if isinstance(actual_model, (TfidfVectorizer, StandardScaler)):
                            logger.error(f"Model is still wrong type: {type(actual_model)}")
                            raise ValueError(f"Model file contains {type(actual_model)} instead of classifier")

                        logger.info(f"Loaded model type: {type(actual_model)}, vectorizer type: {type(vectorizer)}, scaler type: {type(scaler)}")

                        # Create special wrapper for hepatotoxicity
                        class HepatotoxicityTFIDFModel:
                            def __init__(self, model, vectorizer, scaler):
                                self.model = model
                                self.vectorizer = vectorizer
                                self.scaler = scaler

                            def predict_from_smiles(self, smiles):
                                # Transform SMILES to features
                                X = self.vectorizer.transform([smiles])
                                X_scaled = self.scaler.transform(X.toarray())
                                # Get prediction
                                return self.model.predict_proba(X_scaled)[0]

                        # Create wrapper and store
                        wrapper = HepatotoxicityTFIDFModel(actual_model, vectorizer, scaler)
                        MODELS[model_name] = wrapper
                        loaded_count += 1
                        logger.info(f"Successfully loaded TF-IDF hepatotoxicity model")
                    else:
                        # Fall back to normal loading
                        model = ADDIEModel(model_name, model_file, model_type)
                        if model.model is not None:
                            MODELS[model_name] = model
                            loaded_count += 1
                            logger.info(f"Successfully loaded model: {model_name}")
                        else:
                            logger.error(f"Model {model_name} failed to load properly")
                            MODEL_ERRORS[model_name] = "model loaded as None"
                else:
                    # Load model with appropriate type
                    model = ADDIEModel(model_name, model_file, model_type)
                    # Only count as loaded if model is not None
                    if model.model is not None:
                        MODELS[model_name] = model
                        loaded_count += 1
                        logger.info(f"Successfully loaded model: {model_name}")
                    else:
                        logger.error(f"Model {model_name} failed to load properly")
                        MODEL_ERRORS[model_name] = "model loaded as None"
            else:
                logger.warning(f"No model file found in archive for {model_name}")
                # List files in archive for debugging
                logger.debug(f"Files in {extract_dir}: {os.listdir(extract_dir)}")
                MODEL_ERRORS[model_name] = "no model file found in archive"

            # Clean up temporary files
            os.remove(archive_path)
            import shutil
            shutil.rmtree(extract_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            MODEL_ERRORS[model_name] = f"exception during load: {type(e).__name__}: {str(e)}"
            # Continue loading other models

    logger.info(f"Loaded {loaded_count} out of {len(MODEL_LIST)} models")
    return loaded_count


def load_tdc_models_from_storage() -> int:
    """Load TDC SOTA models from blob storage.

    Handles three model layouts:
      1. SOTA CatBoost+GIN ensembles (5 seeds): tdc_sota_v2/winners/models/{endpoint}/seed_*.cbm
      2. Chemprop winners (single .pt): tdc_chemprop/chemprop/{endpoint}/model.pt
      3. Chemprop DILI (5 seeds): tdc_sota_v2/winners/models/dili/chemprop_seed_*.ckpt
      4. P-gp substrate: pgp_substrate/model.pt
      5. Legacy CatBoost (single .cbm): tdc-sota/{source_dir}/{endpoint}/model.cbm

    Downloads to /tmp/tdc_sota/{source_dir}/{endpoint}/ then loads via TDCSOTAPredictor.
    """
    global TDC_PREDICTOR

    if not TDC_AVAILABLE:
        logger.info("TDC SOTA integration module not available, skipping TDC model loading")
        return 0

    from tdc_sota_integration import ModelSource

    local_base = "/tmp/tdc_sota"
    os.makedirs(local_base, exist_ok=True)

    # Blob path registry: maps (endpoint_name, model_source) → download strategy
    # SOTA winners with 5-seed ensembles
    SOTA_ENSEMBLE_ENDPOINTS = {
        "cyp2d6_veith", "cyp3a4_veith", "cyp3a4_substrate_carbonmangels",
        "clearance_hepatocyte_az",
        # Tox21 panel (WS4 retrain) — 5-seed CatBoost+GIN ensembles
        "tox21_nr_ar", "tox21_nr_ar_lbd", "tox21_nr_ahr", "tox21_nr_aromatase",
        "tox21_nr_er", "tox21_nr_er_lbd", "tox21_nr_ppar_gamma",
        "tox21_sr_are", "tox21_sr_atad5", "tox21_sr_hse", "tox21_sr_mmp", "tox21_sr_p53",
        "clintox", "dictrank",
        # dili retrained 2026-06-23 (CatBoost+GIN on the DILI gold standard) — now a
        # SOTA ensemble + calibrator, NOT the old Chemprop. The SOTA_ENSEMBLE `if` branch
        # takes precedence over the DILI_CHEMPROP_ENDPOINT `elif`, so it downloads
        # seed_*.cbm + calibrator.json from winners/models/dili/ (the old chemprop ckpts
        # coexist there, ignored by the CatBoost loader).
        "dili",
    }
    # Chemprop DILI has 5-seed .ckpt ensemble
    DILI_CHEMPROP_ENDPOINT = "dili"
    # Chemprop single-model endpoints (relative to TDC_SOTA_KEY_PREFIX)
    CHEMPROP_SINGLE = {
        "ld50_zhu": f"{TDC_SOTA_KEY_PREFIX}models/tdc_chemprop/chemprop/ld50_zhu/model.pt",
        "lipophilicity_astrazeneca": f"{TDC_SOTA_KEY_PREFIX}models/tdc_chemprop/chemprop/lipophilicity_astrazeneca/model.pt",
        "pgp_substrate_wang": f"{TDC_SOTA_KEY_PREFIX}models/pgp_substrate/model.pt",
    }

    downloaded = 0
    for endpoint_name, endpoint in TDC_ENDPOINTS.items():
        source_dir = endpoint.best_model.value
        local_dir = os.path.join(local_base, source_dir, endpoint_name)
        os.makedirs(local_dir, exist_ok=True)

        try:
            if endpoint_name in SOTA_ENSEMBLE_ENDPOINTS:
                # Download 5-seed CatBoost ensemble from SOTA v2
                ok = True
                for seed in range(5):
                    remote_key = f"{TDC_SOTA_KEY_PREFIX}models/tdc_sota_v2/winners/models/{endpoint_name}/seed_{seed}.cbm"
                    local_path = os.path.join(local_dir, f"seed_{seed}.cbm")
                    if not os.path.exists(local_path):
                        if not download_model_file(remote_key, local_path):
                            ok = False
                            break
                if ok:
                    # Optional per-endpoint isotonic calibrator (only some endpoints
                    # have one, e.g. dictrank). Best-effort; absence is fine.
                    cal_key = f"{TDC_SOTA_KEY_PREFIX}models/tdc_sota_v2/winners/models/{endpoint_name}/calibrator.json"
                    if check_blob_exists(cal_key):
                        download_model_file(cal_key, os.path.join(local_dir, "calibrator.json"))
                    downloaded += 1
                    logger.info(f"Downloaded SOTA ensemble (5 seeds): {endpoint_name}")
                else:
                    TDC_MODEL_ERRORS[endpoint_name] = "ensemble download failed"

            elif endpoint_name == DILI_CHEMPROP_ENDPOINT:
                # Download 5-seed Chemprop DILI ensemble
                ok = True
                for seed in range(5):
                    remote_key = f"{TDC_SOTA_KEY_PREFIX}models/tdc_sota_v2/winners/models/dili/chemprop_seed_{seed}.ckpt"
                    local_path = os.path.join(local_dir, f"chemprop_seed_{seed}.ckpt")
                    if not os.path.exists(local_path):
                        if not download_model_file(remote_key, local_path):
                            ok = False
                            break
                if ok:
                    downloaded += 1
                    logger.info(f"Downloaded Chemprop DILI ensemble (5 seeds)")
                else:
                    TDC_MODEL_ERRORS[endpoint_name] = "dili ensemble download failed"

            elif endpoint_name in CHEMPROP_SINGLE:
                # Download single Chemprop .pt
                remote_key = CHEMPROP_SINGLE[endpoint_name]
                local_path = os.path.join(local_dir, "model.pt")
                if not os.path.exists(local_path):
                    if download_model_file(remote_key, local_path):
                        downloaded += 1
                        logger.info(f"Downloaded Chemprop model: {endpoint_name}")
                    else:
                        TDC_MODEL_ERRORS[endpoint_name] = "chemprop download failed"
                else:
                    downloaded += 1

            else:
                # Legacy: single .cbm from TDC_MODEL_PREFIX
                remote_key = f"{TDC_MODEL_PREFIX}{source_dir}/{endpoint_name}/model.cbm"
                local_path = os.path.join(local_dir, "model.cbm")
                if not os.path.exists(local_path):
                    if download_model_file(remote_key, local_path):
                        downloaded += 1
                        logger.info(f"Downloaded TDC model: {endpoint_name}")
                    else:
                        TDC_MODEL_ERRORS[endpoint_name] = "download failed"
                else:
                    downloaded += 1

        except Exception as e:
            logger.warning(f"Failed to download TDC model {endpoint_name}: {e}")
            TDC_MODEL_ERRORS[endpoint_name] = str(e)

    if downloaded == 0:
        logger.info("No TDC models downloaded — running with ADDIE models only")
        return 0

    try:
        predictor = TDCSOTAPredictor(use_gin=TDC_USE_GIN)
        predictor.load_models(local_base)
        loaded_count = len(predictor.models)
        if loaded_count > 0:
            TDC_PREDICTOR = predictor
            logger.info(f"TDC SOTA predictor ready with {loaded_count}/{len(TDC_ENDPOINTS)} models")
        else:
            logger.warning("TDC predictor loaded 0 models — running with ADDIE models only")
        return loaded_count
    except Exception as e:
        logger.error(f"Failed to initialize TDC predictor: {e}")
        return 0


def process_molecule(molecule: Molecule, include_descriptors: bool, include_confidence: bool, enhanced_mode: bool = True, tdc_only: bool = False, tdc_precomputed: Optional[Dict[str, Any]] = None) -> MoleculeResult:
    """Process a single molecule through all models (or TDC-only for reprocessing).

    tdc_precomputed: when supplied (by process_batch's batched TDC pass), use it
    instead of calling TDC_PREDICTOR.predict() per molecule. This is what lets the
    expensive Chemprop ensemble run once across the whole batch rather than 5×
    per molecule. None => compute TDC inline (single-molecule requests)."""

    start_time = time.time()

    # TDC-only fast path: skip ADDIE models entirely
    if tdc_only and TDC_PREDICTOR is not None:
        try:
            tdc_results = tdc_precomputed if tdc_precomputed is not None else TDC_PREDICTOR.predict(molecule.smiles)
            if 'error' in tdc_results:
                return MoleculeResult(
                    id=molecule.id, smiles=molecule.smiles,
                    predictions={}, descriptors=None, confidence=None,
                    error=f"TDC prediction failed: {tdc_results.get('error')}"
                )
            tdc_results.pop('_provenance', None)
            # Calculate aggregates
            standardizer = ADDIEStandardizer()
            tdc_results.update(standardizer._calculate_aggregates(tdc_results))
            return MoleculeResult(
                id=molecule.id, smiles=molecule.smiles,
                predictions=tdc_results, descriptors=None, confidence=None,
                error=None
            )
        except Exception as e:
            return MoleculeResult(
                id=molecule.id, smiles=molecule.smiles,
                predictions={}, descriptors=None, confidence=None,
                error=f"TDC error: {str(e)}"
            )
    logger.debug(f"Processing molecule {molecule.id} with SMILES: {molecule.smiles}")

    # Generate fingerprint
    fingerprint = generate_fingerprint(molecule.smiles)

    if fingerprint is None:
        return MoleculeResult(
            id=molecule.id,
            smiles=molecule.smiles,
            predictions={},
            descriptors=None,
            confidence=None,
            error="Invalid SMILES or fingerprint generation failed"
        )

    # Get base predictions from all models
    base_predictions = {}
    confidences = {}

    # Determine which ADDIE models are replaced by TDC when TDC is active.
    # Only skip an ADDIE model if its TDC replacement is *actually loaded* —
    # otherwise we lose the prediction entirely (e.g. hepatotoxicity when DILI
    # fails to load).
    tdc_replaced_addie_names = set()
    if TDC_PREDICTOR is not None:
        loaded_tdc_endpoints = set(getattr(TDC_PREDICTOR, "models", {}).keys())
        for tdc_endpoint, addie_name in ADDIE_REPLACEMENT_MAP.items():
            if tdc_endpoint in loaded_tdc_endpoints:
                tdc_replaced_addie_names.add(addie_name)

    logger.debug(f"Running predictions with {len(MODELS)} models (skipping {len(tdc_replaced_addie_names)} replaced by TDC)")

    per_model_errors: Dict[str, str] = {}
    for model_name, model in MODELS.items():
        # Skip ADDIE models that are replaced by TDC SOTA models
        if model_name in tdc_replaced_addie_names:
            logger.debug(f"Skipping ADDIE model {model_name} — replaced by TDC SOTA")
            continue

        try:
            # Special handling for hepatotoxicity TF-IDF model
            if model_name == 'hepatotoxicity' and hasattr(model, 'predict_from_smiles'):
                # This is the TF-IDF model that needs SMILES
                proba = model.predict_from_smiles(molecule.smiles)
                prediction = proba[1] if len(proba) == 2 else proba[0]  # Get toxic probability
                result = {
                    'prediction': float(prediction),
                    'confidence': abs(prediction - 0.5) * 2.0,
                    'binary_class': int(prediction > 0.5)
                }
            else:
                # Standard models using fingerprints
                result = model.predict(fingerprint)

            logger.debug(f"Model {model_name} prediction: {result}")

            # Store base prediction
            if result['prediction'] is not None:
                base_predictions[model_name] = result['prediction']  # Store raw probability
                if include_confidence:
                    confidences[model_name] = result['confidence']

        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            per_model_errors[model_name] = f"prediction failed: {type(e).__name__}: {str(e)}"
            # Continue with other models

    # Use standardized mode for industry-compliant outputs
    if enhanced_mode:
        try:
            # Standardize to industry-standard outputs
            standardizer = ADDIEStandardizer()
            standardized_results = standardizer.standardize_predictions(base_predictions, molecule.smiles)
            predictions = standardized_results
            descriptors = None  # Already included in standardized results
        except Exception as e:
            logger.error(f"Standardization failed: {e}, falling back to basic mode")
            # Fallback to basic predictions
            predictions = {}
            for model_name, prob in base_predictions.items():
                predictions[model_name] = int(prob > 0.5)  # Binary class
                predictions[f"{model_name}_score"] = prob
            descriptors = calculate_descriptors(molecule.smiles) if include_descriptors else None
    else:
        # Basic mode (shouldn't happen since we always pass True)
        predictions = {}
        for model_name, prob in base_predictions.items():
            predictions[model_name] = int(prob > 0.5)  # Binary class
            predictions[f"{model_name}_score"] = prob
        descriptors = calculate_descriptors(molecule.smiles) if include_descriptors else None

    # Merge TDC SOTA predictions (already use standardized field names)
    per_model_errors: Dict[str, str] = {}
    if TDC_PREDICTOR is not None:
        try:
            tdc_results = tdc_precomputed if tdc_precomputed is not None else TDC_PREDICTOR.predict(molecule.smiles)
            if 'error' not in tdc_results:
                # Remove provenance + per-endpoint error metadata before merging
                tdc_provenance = tdc_results.pop('_provenance', {})
                tdc_errors = tdc_results.pop('_errors', {})
                # Surface per-endpoint failures so a per-molecule model error
                # (e.g. dili → hepatotoxicity for aspirin) isn't silently dropped.
                for ep_name, reason in tdc_errors.items():
                    per_model_errors[ep_name] = reason
                if 'dili' in tdc_errors:
                    per_model_errors['hepatotoxicity'] = tdc_errors['dili']
                # Merge TDC predictions — these override any ADDIE predictions for replaced endpoints
                predictions.update(tdc_results)
                # Recalculate aggregates with the merged predictions
                standardizer = ADDIEStandardizer()
                predictions.update(standardizer._calculate_aggregates(predictions))
                logger.debug(f"Merged {len(tdc_results)} TDC predictions")
            else:
                logger.warning(f"TDC prediction failed for {molecule.smiles[:50]}: {tdc_results.get('error')}")
        except Exception as e:
            logger.error(f"TDC prediction error for {molecule.id}: {e}")

    # Ensure hepatotoxicity is surfaced; avoid silent drop
    try:
        std = ADDIEStandardizer()
        hepato_std_field = std.MODEL_MAPPINGS['toxicity']['hepatotoxicity']  # 'hepatotoxicity_probability'
    except Exception:
        hepato_std_field = 'hepatotoxicity_probability'

    error_message: Optional[str] = None
    # Covers both ABSENT and present-but-None. The per-endpoint exception handler
    # in TDCSOTAPredictor.predict sets hepatotoxicity_probability = None on a
    # per-molecule model failure (the aspirin/DILI case), so checking only for
    # absence missed it — the field was present with value None and the reason
    # was silently dropped.
    if predictions.get(hepato_std_field) is None:
        # If we computed a base prediction but it wasn't standardized for some reason, map it
        if isinstance(base_predictions.get('hepatotoxicity'), (int, float)):
            predictions[hepato_std_field] = base_predictions['hepatotoxicity']
        else:
            # Explicitly include the field as None and surface a non-null reason
            predictions[hepato_std_field] = None
            reason = MODEL_ERRORS.get('hepatotoxicity') or per_model_errors.get('hepatotoxicity') or 'unavailable or skipped'
            error_message = f"hepatotoxicity unavailable: {reason}"

    # Prepare confidence scores
    confidence = confidences if include_confidence and not enhanced_mode else None

    return MoleculeResult(
        id=molecule.id,
        smiles=molecule.smiles,
        predictions=predictions,
        descriptors=descriptors,
        confidence=confidence,
        error=error_message
    )

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting ADDIE Models Service...")
    logger.info(f"Storage backend: {STORAGE_BACKEND}")

    # Load models from S3 or Azure
    try:
        model_count = load_models_from_storage()
        if model_count == 0:
            logger.warning("No models loaded! Service running in demo mode.")
        else:
            logger.info(f"Service ready with {model_count} models")
    except Exception as e:
        logger.warning(f"Could not load models: {str(e)}. Service running in demo mode.")

    # Load TDC SOTA models (graceful — service works without them)
    try:
        tdc_count = load_tdc_models_from_storage()
        if tdc_count > 0:
            logger.info(f"TDC SOTA: {tdc_count}/{len(TDC_ENDPOINTS)} models loaded")
        else:
            logger.info("TDC SOTA: no models loaded — running with ADDIE models only")
    except Exception as e:
        logger.warning(f"TDC SOTA loading failed: {e} — running with ADDIE models only")

    # Service is ready regardless of model loading status
    logger.info("ADDIE Models Service is ready and healthy")

    # Quick self-test for hepatotoxicity availability to surface issues early
    try:
        if 'hepatotoxicity' in MODELS:
            test_mol = Molecule(id="selftest", smiles="CCO")
            res = process_molecule(test_mol, include_descriptors=False, include_confidence=False, enhanced_mode=True)
            val = res.predictions.get('hepatotoxicity_probability')
            logger.info(f"Hepatotoxicity self-test: value={val}, error={res.error}")
        else:
            logger.warning(f"Hepatotoxicity model not loaded: {MODEL_ERRORS.get('hepatotoxicity', 'unknown reason')}")
    except Exception as se:
        logger.error(f"Hepatotoxicity self-test failed: {se}")

@app.get("/")
async def root():
    """Root endpoint"""
    tdc_count = len(TDC_PREDICTOR.models) if TDC_PREDICTOR else 0
    return {
        "service": "ADDIE Models API",
        "version": "2.0.0",
        "models_loaded": len(MODELS) + tdc_count,
        "addie_models": len(MODELS),
        "tdc_models": tdc_count
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = len([m for m in MODELS.values() if m.model is not None])
    total_models = len(MODEL_LIST)

    # TDC model counts (not required for healthy status)
    tdc_models_loaded = len(TDC_PREDICTOR.models) if TDC_PREDICTOR else 0
    tdc_models_total = len(TDC_ENDPOINTS)

    # Return unhealthy if less than 50% of ADDIE models loaded (TDC is optional)
    if models_loaded < total_models * 0.5:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "models_loaded": models_loaded,
                "total_models": total_models,
                "tdc_models_loaded": tdc_models_loaded,
                "tdc_models_total": tdc_models_total,
                "service": "addie-models",
                "message": f"Only {models_loaded}/{total_models} models loaded"
            }
        )

    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "total_models": total_models,
        "tdc_models_loaded": tdc_models_loaded,
        "tdc_models_total": tdc_models_total,
        "service": "addie-models"
    }

@app.get("/models")
async def list_models():
    """List loaded models"""
    tdc_models = list(TDC_PREDICTOR.models.keys()) if TDC_PREDICTOR else []
    return {
        "addie_models": list(MODELS.keys()),
        "tdc_models": tdc_models,
        "count": len(MODELS) + len(tdc_models)
    }

@app.post("/addie/process", response_model=BatchResponse)
async def process_batch(request: BatchRequest):
    """Process batch of molecules"""

    logger.info(f"Processing request with {len(request.molecules)} molecules")
    logger.info(f"Models available: {len(MODELS)}")

    if len(MODELS) == 0 and not request.tdc_only:
        logger.error("No models loaded - returning 503")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models loaded"
        )

    if request.tdc_only and TDC_PREDICTOR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TDC models not loaded — cannot use tdc_only mode"
        )

    start_time = time.time()

    # Compute the TDC layer for the WHOLE batch in one pass so the Chemprop (DILI)
    # ensemble runs ONE Trainer.predict per seed across all molecules instead of
    # 5× per molecule serialized by the trainer lock. CatBoost endpoints are
    # vectorized too. Each molecule's slice is then injected into process_molecule,
    # which still does its per-molecule base-ADDIE work in the thread pool. On any
    # failure we fall back to per-molecule TDC (tdc_precomputed=None) so behavior
    # degrades gracefully rather than erroring the batch.
    tdc_batch: Optional[List[Dict[str, Any]]] = None
    if TDC_PREDICTOR is not None and len(request.molecules) > 1:
        try:
            tdc_batch = TDC_PREDICTOR.predict_batch([m.smiles for m in request.molecules])
        except Exception as e:
            logger.error(f"Batched TDC precompute failed, falling back to per-molecule: {e}")
            tdc_batch = None

    # Process molecules in parallel
    loop = asyncio.get_event_loop()
    tasks = []

    for idx, molecule in enumerate(request.molecules):
        task = loop.run_in_executor(
            EXECUTOR,
            process_molecule,
            molecule,
            request.include_descriptors,
            request.include_confidence,
            True,  # Always use enhanced mode for 330 columns
            request.tdc_only,
            tdc_batch[idx] if tdc_batch is not None else None,
        )
        tasks.append(task)

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)

    # Calculate processing time
    processing_time = time.time() - start_time

    logger.info(f"Processed {len(results)} molecules in {processing_time:.2f}s")
    logger.info(f"First result sample: {results[0].predictions if results else 'No results'}")

    tdc_count = len(TDC_PREDICTOR.models) if TDC_PREDICTOR else 0
    response = BatchResponse(
        results=results,
        processing_time=processing_time,
        model_count=len(MODELS) + tdc_count
    )

    logger.info(f"Returning response with {len(response.results)} results")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8025)
