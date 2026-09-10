"""The HTTP listener opens only after executor warmup completes."""

import os
import urllib.request


if __name__ == "__main__":
    url = os.environ.get("FLUXSERVE_HEALTH_URL", "http://127.0.0.1:8000/health")
    with urllib.request.urlopen(url, timeout=2) as response:
        if response.status != 200:
            raise SystemExit(1)
