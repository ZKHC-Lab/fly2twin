from pathlib import Path
from setuptools import setup
runtime = Path("src/reconstruction_runtime")
resources = [str(p.relative_to(runtime)) for p in runtime.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and "tests" not in p.parts
             and p.suffix != ".pyc" and (p.suffix != ".py" or "vendor" in p.parts)]
setup(name="drone3d_runtime", version="0.1.0",
      packages=["reconstruction_runtime", "drone3d_resources"],
      package_dir={"": "src"}, package_data={"reconstruction_runtime": resources})
