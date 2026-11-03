"""Compatibility entry point for the shared CPU-only artifact validator."""
import os
from pathlib import Path
import sys

# Support both installed ROS deployments and standalone source-tree commands.
sys.path.append(os.environ.get(
    "DRONE3D_RUNTIME_PYTHON_ROOT",
    str(Path(__file__).resolve().parents[2] / "ROS_Compatibility_layer/src/drone3d_runtime/src"),
))
from reconstruction_runtime.result_validator import (  # noqa: E402
    ResultValidationError as ResultValidationError,
    ValidatedResult as ValidatedResult,
    validate_result_directory as validate_result_directory,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
