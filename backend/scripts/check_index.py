import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, ".")
from app.services.vector_store import all_collection_info
info = all_collection_info()
for name, d in info.items():
    pts = d.get("points_count", "?")
    err = d.get("error", "")
    print(f"{name}: points={pts} {err}")
