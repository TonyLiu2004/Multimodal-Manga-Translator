# Multimodal-Manga-Translator

A cross-platform manga reader that dynamically pulls content from provider APIs and uses an AI-powered backend to overlay real-time translations directly onto the source media.

The system leverages a FastAPI backend to perform automated bubble-text detection, OCR, and machine translation, delivering a structured JSON response to the frontend for low-latency and non-destructive rendering over the original images.

NOTES: 
- First build will take a long time because the base image is large.
- Docker images use Python 3.10 for compatibility with AI libraries (PyTorch/ROCm) and hardware drivers.
- Dockerfile.nvidia on the backend has not been tested

## Prerequisites
* Windows Users: Docker Desktop with the WSL 2 backend enabled.

* AMD GPU Users: Ensure you have the latest AMD Adrenalin Drivers installed on the host.


## Getting Started

1. Prepare Models & Fonts
Models and fonts are automatically downloaded into local directories (these sync to the container via volumes):

* backend/models

* backend/fonts/NotoSansCJK.ttc


2. Configuration
The system is configured via docker-compose.yml.

* For AMD GPU: Ensure dockerfile: backend/Dockerfile.amd is set.

* For CPU only: Change to dockerfile: backend/Dockerfile.cpu.

* For NVIDIA GPU: Not implemented yet


## Build and Start
Run these commands in the root directory:
### Build the image
```
docker compose build
```

### Start the application
```
docker compose up
```

Once the logs show Uvicorn running, access the interactive API documentation at:
http://localhost:8000/docs

Server runs at:
http://localhost:8000/

Frontend runs at:
http://localhost:8081/

## Testing hosted frontend 
Frontend is deployed at https://manglify-d6ebc.web.app/
- Browsers block public websites from reaching local addresses, so we use ngrok to tunnel our backend


In Terminal 1:
```
python main.py
```

In Terminal 2:
```
ngrok http 8000
```

Take the url provided by ngrok and paste it into the BACKEND_URL variable in frontend/app/index.tsx

### Deployment and redeployment
Run:
```
npx expo export --platform web # builds frontend website and puts it in the dist folder

firebase deploy --only hosting # deploys from dist folder

```

---

## Troubleshooting & Maintenance
<details>
<summary><b>Click to expand: Advanced Docker Commands</b></summary>

#### Completely rebuild images from scratch without using cached layers
```
docker-compose build --no-cache
```

#### Stop services and remove containers, networks, and old 'orphan' services
```
docker-compose down --remove-orphans
```

</details>
