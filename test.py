import importlib.util
from pathlib import Path

module_path = Path("src/bci/2_preprocessing/__init__.py")
spec = importlib.util.spec_from_file_location("preprocessing", module_path)
preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing)

print("✓ Module loaded!")
print(f"Available functions: {preprocessing.__all__}")