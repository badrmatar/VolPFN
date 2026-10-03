"""Stop-only guard for this explicitly authorized $10/eight-hour operation.

Reuses the tested pod-local authentication and self-pod checks. The authority
constants are extended only in this module; the old frozen benchmark is intact.
This process is not a provider-enforced dollar cap. It never terminates a pod.
"""
from experiments.return_only_benchmark import cloud_stop
import os
from pathlib import Path


def inherit_container_identity():
    """Inherit native pod identity/auth in memory; never print or persist auth."""
    for item in Path('/proc/1/environ').read_bytes().split(b'\0'):
        if item.startswith((b'RUNPOD_POD_ID=', b'RUNPOD_API_KEY=')):
            name, value = item.split(b'=', 1)
            os.environ[name.decode()] = value.decode()
    os.environ['PATH'] = '/workspace/bin:' + os.environ.get('PATH', '')

if __name__ == '__main__':
    inherit_container_identity()
    cloud_stop.MAX_HOURS = 8.0
    cloud_stop.MAX_BUDGET_USD = 10.0
    cloud_stop.main()
