"""Configuration module for the ACE Active Carver project.

This module defines the configuration data structures used throughout the application.
It uses Pydantic for definition and PyYAML for loading from files.
"""

import yaml
import numpy as np
import logging
from pathlib import Path
from typing import Any, Optional, List, Dict, Literal
from ase.data import atomic_numbers, covalent_radii
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator

logger = logging.getLogger(__name__)

CrystalType = Literal["metallic", "ionic", "covalent", "random"]

def generate_default_lj_params(elements: List[str]) -> Dict[str, float]:
    """
    Generates robust default Lennard-Jones parameters based on element physics.
    """
    if not elements:
        return {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}

    try:
        radii = []
        for el in elements:
            z = atomic_numbers.get(el)
            if z is None:
                raise ValueError(f"Unknown element symbol: {el}")
            radii.append(covalent_radii[z])

        avg_radius = np.mean(radii)

        # r_min = 2^(1/6) * sigma
        sigma = (2.0 * avg_radius) * 0.8909

        return {
            "epsilon": 1.0,
            "sigma": float(round(sigma, 3)),
            "cutoff": float(round(2.5 * sigma, 3))
        }
    except Exception as e:
        logger.warning(f"Could not auto-generate LJ params ({e}). Using safe defaults.")
        return {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}

class MetaConfig(BaseModel):
    """Environment-specific configuration."""
    dft: Dict[str, Any]
    lammps: Dict[str, Any]

    @property
    def dft_command(self) -> str:
        return self.dft.get("command", "pw.x")

    @property
    def pseudo_dir(self) -> Path:
        return Path(self.dft.get("pseudo_dir", "."))

    @property
    def sssp_json_path(self) -> Path:
        return Path(self.dft.get("sssp_json_path", "."))

    @property
    def lammps_command(self) -> str:
        return self.lammps.get("command", "lmp_serial")


class ExperimentConfig(BaseModel):
    """Experiment metadata and output settings."""
    name: str
    output_dir: Path


class ExplorationParams(BaseModel):
    """Parameters for choosing the exploration strategy."""
    strategy: str = "hybrid"


class MDParams(BaseModel):
    """Parameters for Molecular Dynamics simulations."""
    timestep: float = Field(..., ge=0.1, le=5.0, description="Timestep in fs")
    temperature: float = Field(..., ge=0, le=5000, description="Temperature in K")
    pressure: float = Field(..., ge=0)
    n_steps: int = Field(..., gt=0)
    elements: List[str] = Field(..., min_length=1)
    initial_structure: str
    masses: Dict[str, float]
    restart_freq: int = Field(1000, gt=0)
    dump_freq: int = Field(1000, gt=0)
    n_md_walkers: int = Field(1, gt=0)


class ALParams(BaseModel):
    """Parameters for Active Learning strategy."""
    gamma_threshold: float
    n_clusters: int = Field(..., gt=0)
    r_core: float = Field(..., gt=0)
    box_size: Optional[float] = Field(None, gt=0)
    initial_potential: str
    potential_yaml_path: str
    initial_dataset_path: Optional[str] = None
    initial_active_set_path: Optional[str] = None
    stoichiometry_tolerance: float = 0.1
    min_bond_distance: float = 1.5
    num_parallel_labeling: int = 4
    query_strategy: str = "uncertainty"
    sampling_strategy: str = "composite"
    outlier_energy_max: float = 10.0
    gamma_upper_bound: float = 2.0


class KMCParams(BaseModel):
    """Parameters for kMC simulations."""
    active: bool = False
    temperature: float = Field(300.0, gt=0)
    n_searches: int = Field(10, gt=0)
    search_radius: float = 0.1
    dimer_fmax: float = 0.05
    check_interval: int = 5
    prefactor: float = 1e12
    box_size: float = Field(12.0, gt=0)
    buffer_width: float = 2.0
    n_workers: int = 4
    active_region_mode: str = "surface_and_species"
    active_species: List[str] = Field(default_factory=lambda: ["Co", "Ti", "O"])
    active_z_cutoff: float = 10.0
    move_type: str = "cluster"
    cluster_radius: float = 3.0
    selection_bias: str = "coordination"
    bias_strength: float = 2.0
    adsorbate_cn_cutoff: int = 9
    cluster_connectivity_cutoff: float = 3.0


class DFTParams(BaseModel):
    """Parameters for Density Functional Theory calculations."""
    kpoint_density: float = Field(60.0, gt=0)
    auto_physics: bool = True


class LJParams(BaseModel):
    """Parameters for Lennard-Jones potential."""
    epsilon: float = Field(..., gt=0)
    sigma: float = Field(..., gt=0)
    cutoff: float = Field(..., gt=0)
    shift_energy: bool = True


class PreOptimizationParams(BaseModel):
    """Parameters for MACE pre-optimization."""
    enabled: bool = False
    model: str = "medium"
    fmax: float = 0.1
    steps: int = Field(50, gt=0)
    device: str = "cuda"

class SeedGenerationParams(BaseModel):
    """Parameters for the Seed Generation phase."""
    crystal_type: CrystalType = "random"
    types: Dict[str, Any] = Field(default_factory=dict)
    n_random_structures: int = Field(100, gt=0)
    exploration_temperatures: List[float] = Field(default_factory=lambda: [300.0, 1000.0])
    n_md_steps: int = Field(1000, gt=0)
    n_samples_for_dft: int = Field(20, gt=0)

class GenerationParams(BaseModel):
    """Parameters for scenario-driven generation."""
    pre_optimization: PreOptimizationParams = Field(default_factory=PreOptimizationParams)
    scenarios: List[Dict[str, Any]] = Field(default_factory=list)
    device: str = "cuda"

    @field_validator('scenarios')
    @classmethod
    def check_scenarios_not_empty(cls, v):
        if not v:
            # We allow empty scenarios if it's not the generation phase, but usually
            # if GenerationParams is instantiated, we expect scenarios or we should handle it.
            # However, the prompt specifically asks: "Add a custom validator that checks if generation.scenarios is not empty."
            # So I will raise a value error if it is empty.
            raise ValueError("generation.scenarios must not be empty.")
        return v


class ACEModelParams(BaseModel):
    """Parameters for Pacemaker potential model."""
    pacemaker_config: Dict[str, Any] = Field(default_factory=dict)
    initial_potentials: List[str] = Field(default_factory=list)
    delta_learning_mode: bool = True

    @field_validator('pacemaker_config')
    @classmethod
    def validate_pacemaker_cutoff(cls, v):
        if 'cutoff' in v:
            cutoff = v['cutoff']
            if not (2.0 <= cutoff <= 15.0):
                raise ValueError(f"ace_model.pacemaker_config.cutoff must be between 2.0 and 15.0 Angstrom. Got {cutoff}")
        return v


class TrainingParams(BaseModel):
    """Parameters for Active Learning Training strategy."""
    replay_ratio: float = 1.0
    global_dataset_path: str = "data/global_dataset.pckl"


class ExplorationStage(BaseModel):
    """Parameters for a single thermodynamic exploration stage."""
    iter_start: int = Field(..., ge=0)
    iter_end: int = Field(..., ge=0)
    temp: List[float]
    press: List[float]

class MonitoringParams(BaseModel):
    """Parameters for monitoring and observability (W&B, TensorBoard)."""
    enabled: bool = False
    project: str = "ace_active_carver"
    entity: Optional[str] = None
    use_wandb: bool = True
    use_tensorboard: bool = False

class Config(BaseModel):
    """Main configuration class aggregating all parameter sections."""
    meta: MetaConfig
    experiment: ExperimentConfig
    exploration: ExplorationParams
    md_params: MDParams
    al_params: ALParams
    dft_params: DFTParams
    lj_params: LJParams
    training_params: TrainingParams
    ace_model: ACEModelParams
    seed: int = 42
    kmc_params: KMCParams = Field(default_factory=KMCParams)
    generation_params: GenerationParams = Field(default_factory=GenerationParams)
    seed_generation: SeedGenerationParams = Field(default_factory=SeedGenerationParams)
    exploration_schedule: List[ExplorationStage] = Field(default_factory=list)
    monitoring: MonitoringParams = Field(default_factory=MonitoringParams)

    @model_validator(mode='after')
    def compute_defaults(self) -> "Config":
        """Compute smart defaults for optional parameters."""
        # Smart default for AL box_size
        if self.al_params.box_size is None:
            cutoff = self.ace_model.pacemaker_config.get('cutoff', 5.0)
            self.al_params.box_size = 2 * cutoff + 4.0
            logger.info(f"Auto-calculated AL box_size: {self.al_params.box_size} (cutoff={cutoff})")
        return self

    @classmethod
    def load_meta(cls, path: Path) -> MetaConfig:
        """Load environment configuration from meta_config.yaml."""
        if not path.exists():
             raise FileNotFoundError(f"Meta config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            meta_dict = yaml.safe_load(f) or {}

        return MetaConfig(
            dft=meta_dict.get("dft", {}),
            lammps=meta_dict.get("lammps", {})
        )

    @classmethod
    def load_experiment(cls, config_path: Path, meta_config: MetaConfig) -> "Config":
        """Load experiment configuration and combine with meta config."""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        # Handle constants inheritance if needed (legacy support)
        constant_path = path.parent / "constant.yaml"
        if constant_path.exists():
            with constant_path.open("r", encoding="utf-8") as f:
                constant_dict = yaml.safe_load(f) or {}

            merged_dict = constant_dict.copy()
            def update_recursive(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = update_recursive(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            update_recursive(merged_dict, config_dict)
            config_dict = merged_dict

        return cls.from_dict(config_dict, meta_config)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], meta_config: MetaConfig) -> "Config":
        """Create a Config instance from a dictionary."""
        # Preprocessing to match Pydantic structure

        # Special handling for LJ Params generation if missing
        md_dict = config_dict.get("md_params", {})
        lj_dict = config_dict.get("lj_params", {})
        if not lj_dict:
            elements = md_dict.get("elements", [])
            lj_dict = generate_default_lj_params(elements)

        # Special handling for DFT Params filtering
        dft_dict = config_dict.get("dft_params", {}).copy()
        allowed_dft_keys = {"kpoint_density", "auto_physics"}
        dft_dict = {k: v for k, v in dft_dict.items() if k in allowed_dft_keys}

        # Construct the full dictionary for Pydantic

        final_dict = config_dict.copy()
        final_dict["meta"] = meta_config
        final_dict["lj_params"] = lj_dict
        final_dict["dft_params"] = dft_dict

        # experiment config needs to be constructed as a dict or object
        exp_dict = config_dict.get("experiment", {})
        final_dict["experiment"] = {
            "name": exp_dict.get("name", "experiment"),
            "output_dir": exp_dict.get("output_dir", "output")
        }

        # Handle generation params mapping
        if "generation" in config_dict:
            final_dict["generation_params"] = config_dict["generation"]

        return cls(**final_dict)
