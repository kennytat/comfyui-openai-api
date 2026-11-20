"""HTTP routers: images (generations), models (+ availability catalog), files, health."""

from app.routes import files, health, images, models

__all__ = ["files", "health", "images", "models"]
