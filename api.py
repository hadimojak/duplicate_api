from __future__ import annotations

import os

from app.main import app

__all__ = [
    "app",
]


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),
        reload=False,
        workers=1,
    )
