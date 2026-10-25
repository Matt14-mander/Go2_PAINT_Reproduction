"""Standard-library checks that must run before Kit changes Python search paths."""

import importlib.metadata as metadata
import os
import site
import sys


def get_rsl_version_before_kit():
    """Use real installed metadata; never guess a version or enable user packages."""
    try:
        result = metadata.version("rsl-rl-lib")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "RSL-RL metadata is not visible before Isaac/Kit startup. "
            f"Python={sys.executable}; ENABLE_USER_SITE={site.ENABLE_USER_SITE}; "
            f"PYTHONNOUSERSITE={os.environ.get('PYTHONNOUSERSITE')!r}; "
            f"user_site={site.getusersitepackages()}. "
            "Use the same limx interpreter/environment as training. Run "
            "`python -m pip show rsl-rl-lib` and "
            "`python scripts/paint/server_train.py diagnose` before reinstalling anything."
        ) from error
    print(f"[INFO] RSL-RL version before Kit startup: {result}", flush=True)
    return result
