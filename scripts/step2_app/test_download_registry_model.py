import requests

DEFAULT_WORKSPACE="IFT6758_team4"
DEFAULT_PROJECT="milestone_2"
DEFAULT_MODEL="lr-angle"
DEFAULT_VERSION="v1"
DEFAULT_PORT=8000


payload = {
    "workspace": DEFAULT_WORKSPACE,
    "project": DEFAULT_PROJECT,
    "model": DEFAULT_MODEL,
    "version": DEFAULT_WORKSPACE
}

res = requests.post(f"http://0.0.0.0:{DEFAULT_PORT}/download_registry_model", json=payload)

print(res.status_code)
print(res.json())
