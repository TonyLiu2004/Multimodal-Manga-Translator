# 1. Use a lightweight Python base image
FROM python:3.12-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /app

COPY requirements_deploy.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements_deploy.txt

# 7. Copy your project files
COPY . .

# This ensures they are part of the "permanent" image and not deleted on restart
RUN env PYTHONPATH=. python -c "from services.bubble_detector_kiuyha_service import Bubble_Detector_Kiuyha_Service; b = Bubble_Detector_Kiuyha_Service();"
RUN env PYTHONPATH=. python -c "from helpers import setup_fonts; setup_fonts()"
RUN env PYTHONPATH=. python -c "from manga_ocr import MangaOcr; MangaOcr()"

# 8. Expose and Start
EXPOSE 8000
CMD ["python", "main.py"]