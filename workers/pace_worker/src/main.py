import argparse
import sys
from pathlib import Path
import logging
import json
from ase.io import read, write

sys.path.append("/app")

from shared.core.config import Config
from strategies.pacemaker import PacemakerTrainer
from sampling.strategies.max_gamma import MaxGammaSampler
from sampling.strategies.composite import CompositeSampler
from sampler import DirectSampler
from validation.pacemaker_validator import PacemakerValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pace_worker")

def run_train(args, config, meta_config):
    trainer = PacemakerTrainer(ace_model_params=config.ace_model, training_params=config.training_params)
    output_pot = trainer.train(
        dataset_path=args.dataset,
        initial_potential=args.initial_potential,
        potential_yaml_path=args.potential_yaml,
        asi_path=args.asi,
        iteration=args.iteration
    )
    print(f"Output Potential: {output_pot}")

def run_sample(args, config):
    candidates = read(args.candidates, index=":")
    strategy_name = config.al_params.sampling_strategy
    if strategy_name == "max_gamma":
        sampler = MaxGammaSampler()
    else:
        sampler = CompositeSampler()
    selected = sampler.sample(candidates, n_samples=args.n_samples)
    selected_atoms = [s[0] for s in selected]
    write(args.output, selected_atoms)
    print(f"Selected {len(selected_atoms)} structures written to {args.output}")

def run_direct_sample(args):
    structures = read(args.input, index=":")
    sampler = DirectSampler(n_clusters=args.n_clusters)
    selected = sampler.sample(structures)
    write(args.output, selected)
    print(f"Direct Sampled {len(selected)} structures written to {args.output}")

def run_validate(args):
    # args: potential, output (json)
    validator = PacemakerValidator()
    metrics = validator.validate(args.potential)
    with open(args.output, "w") as f:
        json.dump(metrics, f)
    print(f"Validation metrics written to {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Pacemaker Worker")
    subparsers = parser.add_subparsers(dest="task", required=True)

    # Train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--meta-config", required=True)
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--initial-potential")
    train_parser.add_argument("--potential-yaml")
    train_parser.add_argument("--asi")
    train_parser.add_argument("--iteration", type=int, default=0)

    # Sample
    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--config", required=True)
    sample_parser.add_argument("--meta-config", required=True)
    sample_parser.add_argument("--candidates", required=True)
    sample_parser.add_argument("--n_samples", type=int, required=True)
    sample_parser.add_argument("--output", required=True)

    # Direct Sample
    ds_parser = subparsers.add_parser("direct_sample")
    ds_parser.add_argument("--input", required=True)
    ds_parser.add_argument("--output", required=True)
    ds_parser.add_argument("--n_clusters", type=int, required=True)

    # Validate
    val_parser = subparsers.add_parser("validate")
    val_parser.add_argument("--potential", required=True)
    val_parser.add_argument("--output", required=True)

    args = parser.parse_args()

    try:
        if args.task in ["direct_sample", "validate"]:
            if args.task == "direct_sample":
                run_direct_sample(args)
            elif args.task == "validate":
                run_validate(args)
        else:
            meta_config = Config.load_meta(Path(args.meta_config))
            config = Config.load_experiment(Path(args.config), meta_config)

            if args.task == "train":
                run_train(args, config, meta_config)
            elif args.task == "sample":
                run_sample(args, config)

    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
