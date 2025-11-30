
import sys
import os
import shutil
import pytest
from pathlib import Path
from hypothesis import given, settings, strategies as st, HealthCheck

# Setup paths
worker_path = os.path.join(os.getcwd(), 'workers/al_md_kmc_worker')
if worker_path not in sys.path:
    sys.path.append(worker_path)
sys.path.append(os.getcwd())

from shared.core.config import (
    Config, MetaConfig, ExperimentConfig, ExplorationParams,
    MDParams, ALParams, DFTParams, LJParams, TrainingParams,
    ACEModelParams, KMCParams, SeedGenerationParams, GenerationParams,
    ExplorationStage, PreOptimizationParams
)
from src.state_manager import StateManager, OrchestratorState
from src.interfaces.explorer import ExplorationStatus
from src.workflows.active_learning_loop import ActiveLearningOrchestrator, HandledOrchestratorError
try:
    from tests.fakes import FakeExplorer, FakeALService
except ImportError:
    # If running from root, python path issues might occur if tests isn't a package
    import sys
    sys.path.append('tests')
    from fakes import FakeExplorer, FakeALService
from pydantic import ValidationError

# --- Strategies ---

def build_config_strategy():
    return st.builds(
        Config,
        meta=st.builds(MetaConfig, dft=st.dictionaries(st.text(), st.text()), lammps=st.dictionaries(st.text(), st.text())),
        experiment=st.builds(ExperimentConfig, name=st.text(min_size=1), output_dir=st.just(Path("output"))),
        exploration=st.builds(ExplorationParams),
        md_params=st.builds(
            MDParams,
            timestep=st.floats(min_value=0.1, max_value=10.0),
            temperature=st.floats(min_value=1.0, max_value=2000.0),
            pressure=st.floats(min_value=0.0, max_value=1000.0),
            n_steps=st.integers(min_value=10, max_value=1000),
            elements=st.lists(st.sampled_from(["Al", "Cu", "Ti", "O"]), min_size=1, max_size=3),
            initial_structure=st.just("initial.xyz"),
            masses=st.dictionaries(st.text(), st.floats(min_value=1.0, max_value=100.0)),
            restart_freq=st.integers(min_value=100, max_value=1000),
            dump_freq=st.integers(min_value=100, max_value=1000),
            n_md_walkers=st.integers(min_value=1, max_value=4)
        ),
        al_params=st.builds(
            ALParams,
            gamma_threshold=st.floats(min_value=0.1, max_value=1.0),
            n_clusters=st.integers(min_value=1, max_value=10),
            r_core=st.floats(min_value=1.0, max_value=5.0),
            box_size=st.floats(min_value=5.0, max_value=20.0),
            initial_potential=st.just("init.yace"),
            potential_yaml_path=st.just("pot.yaml"),
            initial_dataset_path=st.just("data.pckl"),
            initial_active_set_path=st.just("active.asi")
        ),
        dft_params=st.builds(DFTParams),
        lj_params=st.builds(
            LJParams,
            epsilon=st.floats(min_value=0.1, max_value=5.0),
            sigma=st.floats(min_value=1.0, max_value=5.0),
            cutoff=st.floats(min_value=2.0, max_value=10.0)
        ),
        training_params=st.builds(TrainingParams),
        ace_model=st.builds(ACEModelParams),
        seed=st.integers(),
        kmc_params=st.builds(KMCParams),
        generation_params=st.builds(GenerationParams),
        seed_generation=st.builds(SeedGenerationParams),
        exploration_schedule=st.lists(
            st.builds(
                ExplorationStage,
                iter_start=st.integers(min_value=0, max_value=10),
                iter_end=st.integers(min_value=11, max_value=20),
                temp=st.lists(st.floats(min_value=100, max_value=1000), min_size=1),
                press=st.lists(st.floats(min_value=0, max_value=100), min_size=1)
            ),
            max_size=2
        )
    )

def build_state_strategy():
    return st.builds(
        OrchestratorState,
        iteration=st.integers(min_value=0, max_value=10),
        current_structure=st.just("current.xyz"),
        current_potential_path=st.just("current.yace"),
        potential_history=st.lists(st.text(), max_size=3),
        uncertainty_history=st.lists(st.floats(), max_size=3),
        is_converged=st.booleans()
    )


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(config=build_config_strategy(), state=build_state_strategy())
def test_orchestrator_invariants(tmp_path, config, state):
    # Setup Paths
    # Ensure unique work dir for each iteration if needed, or clean up
    # tmp_path is shared across examples.
    # We should create a unique subdirectory for each run to avoid collision
    import uuid
    work_dir = tmp_path / f"work_{uuid.uuid4()}"
    work_dir.mkdir()

    # Adjust config output_dir to be within tmp_path to avoid filesystem side effects
    config.experiment.output_dir = work_dir / "output"

    # Create Dummy Files required by config/state
    (work_dir / "initial.xyz").touch()
    (work_dir / "init.yace").touch()
    (work_dir / "pot.yaml").touch()
    (work_dir / "data.pckl").touch()
    (work_dir / "active.asi").touch()
    (work_dir / "current.xyz").touch()
    (work_dir / "current.yace").touch()

    # Also need potential_path in state to exist
    if state.current_potential_path and not Path(state.current_potential_path).is_absolute():
        (work_dir / state.current_potential_path).touch()
        # Update state path to absolute or relative to where orchestrator runs?
        # The orchestrator usually expects absolute paths or relative to execution.
        # Let's make paths absolute to be safe
        state.current_potential_path = str(work_dir / state.current_potential_path)

    state.current_structure = str(work_dir / "current.xyz")
    config.md_params.initial_structure = str(work_dir / "initial.xyz")
    config.al_params.initial_potential = str(work_dir / "init.yace")
    config.al_params.potential_yaml_path = str(work_dir / "pot.yaml")

    # Setup Fakes
    # FakeExplorer needs to be robust. We limit max_calls to 1 to simulate a single step or short run.
    explorer = FakeExplorer(max_calls=1)
    al_service = FakeALService()
    state_manager = StateManager(work_dir)
    # StateManager.save expects a dict, but we are passing an OrchestratorState object
    state_manager.save(state.model_dump())

    # Initialize SUT
    # Note: orchestrator.run() runs a loop. We need to ensure it doesn't run forever in tests.
    # We can inject a customized explorer that returns FAILED after 1 step to break loop,
    # or rely on config to limit iterations.
    # Here config.exploration_schedule determines iterations if used.

    # To prevent infinite loops if config allows many iterations, we might need to mock loop condition or max_iterations
    # But for this test, we rely on FakeExplorer returning a result that leads to loop termination or just checking one step.
    # Wait, ActiveLearningOrchestrator.run() loops until max_generations or convergence.
    # The config has 'iter_end'.

    # Let's rely on FakeExplorer returning something that advances state, but also we can monkeypatch check_convergence
    # or just trust max_calls in FakeExplorer if it raises an exception?
    # No, we want to test invariants under normal execution.

    # Hack: Set iter_end to iteration + 1 to ensure loop finishes quickly
    if config.exploration_schedule:
        config.exploration_schedule[0].iter_start = state.iteration
        config.exploration_schedule[0].iter_end = state.iteration + 1
    else:
        # If no schedule, maybe it runs forever?
        # We should ensure config has a schedule or handle it.
        # Our strategy creates schedules.
        pass

    orch = ActiveLearningOrchestrator(config, al_service, explorer, state_manager)

    # Execution
    try:
        orch.run()
    except Exception as e:
        # The ONLY allowed exceptions are specific Pydantic ValidationErrors
        # or handled RuntimeErrors (HandledOrchestratorError).
        # Raw Python crashes (AttributeError, TypeError, etc.) are Failures.

        # Also allowed: FileNotFoundError if our random strings point to non-existent files that we didn't mock
        # But we tried to mock main ones.
        # If validation error happens during init, it's fine (invalid config caught).
        # But here we pass a valid config object. So validation error shouldn't happen unless runtime data is bad.

        if isinstance(e, (ValidationError, HandledOrchestratorError)):
             return # Pass

        # Allow normal exit (RuntimeError usually handled?)
        # If the orchestrator raises generic Exception for known failures, we check that.
        # But here we want to catch UNHANDLED exceptions.

        # Rethrow unhandled
        raise e

    # Verify Invariants (Post-Conditions)
    final_state = state_manager.load()
    # StateManager.load returns a dict (model_dump())
    assert isinstance(final_state['iteration'], int)
    assert final_state['iteration'] >= state.iteration
    assert Path(final_state['current_structure']).exists()
