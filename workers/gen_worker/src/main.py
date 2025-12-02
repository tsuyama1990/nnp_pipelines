import argparse
import yaml
import sys
import logging
from pathlib import Path
from ase.io import read, write
from scenarios import ScenarioFactory
from filter import MACEFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_generate(args):
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file {config_path} not found.")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Config might be a list of scenarios or a single dict under 'scenarios' key
    # If root is dict and has 'scenarios', use that.
    # If root is list, iterate.
    # If root is dict but no 'scenarios', treat as single scenario config (if 'type' exists).

    if isinstance(config, dict):
        if "scenarios" in config:
            scenarios_conf = config["scenarios"]
        elif "type" in config:
            scenarios_conf = [config]
        else:
            # Maybe full config with generation params
            scenarios_conf = config.get("generation_params", {}).get("scenarios", [])
    elif isinstance(config, list):
        scenarios_conf = config
    else:
        scenarios_conf = []

    all_structures = []

    for scen_conf in scenarios_conf:
        try:
            generator = ScenarioFactory.create(scen_conf)
            structures = generator.generate()
            all_structures.extend(structures)
            logger.info(f"Generated {len(structures)} structures for scenario {scen_conf.get('type')}")
        except Exception as e:
            logger.error(f"Failed to generate for scenario {scen_conf}: {e}", exc_info=True)

    # Pre-optimization (MACE)
    # Check if we should run pre-optimization.
    gen_params = config.get("generation_params", {}) if isinstance(config, dict) else {}
    pre_opt_conf = gen_params.get("pre_optimization", {})

    # Defaults: medium model, check for cuda
    model_size = pre_opt_conf.get("model", "medium")
    import torch
    device_default = "cuda" if torch.cuda.is_available() else "cpu"
    device = pre_opt_conf.get("device", device_default)

    if all_structures:
        logger.info(f"Pre-relaxing {len(all_structures)} structures using MACE...")
        try:
            from optimizer import FoundationOptimizer
            optimizer = FoundationOptimizer(model=model_size, device=device)
            all_structures = optimizer.relax(all_structures)
            logger.info(f"Pre-relaxation complete. Remaining structures: {len(all_structures)}")
        except ImportError:
            logger.warning("MACE not available. Skipping pre-relaxation.")
        except Exception as e:
            logger.warning(f"Pre-relaxation failed: {e}. Proceeding with raw structures.")

    write(args.output, all_structures)
    logger.info(f"Wrote {len(all_structures)} structures to {args.output}")

def run_filter(args):
    # args: input, output, model, fmax
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file {input_path} not found.")
        sys.exit(1)

    structures = read(input_path, index=":")
    logger.info(f"Filtering {len(structures)} structures with MACE (model={args.model}, fmax={args.fmax})")

    mace_filter = MACEFilter(
        model_size=args.model,
        force_cutoff=args.fmax
    )

    valid_structures = mace_filter.filter(structures)

    write(args.output, valid_structures)
    logger.info(f"Wrote {len(valid_structures)} filtered structures to {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Scenario Generation Worker")
    subparsers = parser.add_subparsers(dest="task") # Optional for back-compat if we want default?
    # Actually better to enforce task or check args.

    # Generate Parser
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--config", required=True, help="Path to YAML config file")
    gen_parser.add_argument("--output", required=True, help="Output path for generated structures (.xyz)")

    # Filter Parser
    filter_parser = subparsers.add_parser("filter")
    filter_parser.add_argument("--input", required=True, help="Input XYZ file")
    filter_parser.add_argument("--output", required=True, help="Output XYZ file")
    filter_parser.add_argument("--model", default="medium", help="MACE model size")
    filter_parser.add_argument("--fmax", type=float, default=100.0, help="Force cutoff (eV/A)")

    args = parser.parse_args()

    # Backwards compatibility: if no task but config/output present, assume generate
    if args.task is None:
        if hasattr(args, "config") and hasattr(args, "output"):
            run_generate(args)
        else:
            parser.print_help()
            sys.exit(1)
    elif args.task == "generate":
        run_generate(args)
    elif args.task == "filter":
        run_filter(args)

if __name__ == "__main__":
    main()
