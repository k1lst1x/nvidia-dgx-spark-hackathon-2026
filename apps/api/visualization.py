"""Optional Rerun viewer for frames, masks, and object centers.

Disabled unless RERUN_ENABLED is true. A missing package or a viewer error
turns logging off for the rest of the process and leaves detection running.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("guidelens.rerun")


class RerunView:
    def __init__(self) -> None:
        self.enabled = os.getenv("RERUN_ENABLED", "false").lower() in {"1", "true", "yes"}
        self.spawn = os.getenv("RERUN_SPAWN", "false").lower() in {"1", "true", "yes"}
        self._started = False
        self._failed = False

    def status(self) -> dict[str, object]:
        return {"enabled": self.enabled, "active": self._started and not self._failed}

    def log(self, objects: list) -> None:
        if not self.enabled or self._failed:
            return
        try:
            import rerun as rr

            if not self._started:
                rr.init("visionmate", spawn=self.spawn)
                self._started = True
            labels = [getattr(item, "label", "") for item in objects]
            centers = []
            for item in objects:
                spatial = getattr(item, "spatial", None) or {}
                center = spatial.get("center_m") if isinstance(spatial, dict) else None
                if isinstance(center, list) and len(center) == 3:
                    centers.append(center)
            rr.log("perception/labels", rr.TextLog(", ".join(labels) or "none"))
            if centers:
                rr.log("perception/centers", rr.Points3D(centers))
        except Exception:
            self._failed = True
            logger.exception("rerun_disabled_after_error")
