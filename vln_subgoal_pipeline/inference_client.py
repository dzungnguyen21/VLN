import io
import logging
import multiprocessing as mp
from typing import Any, Dict, List, Optional

logger = logging.getLogger("InferenceServer")

def _worker_loop(request_queue: mp.Queue, result_queue: mp.Queue, use_mock: bool) -> None:
    import logging
    logging.basicConfig(level=logging.INFO)
    wlog = logging.getLogger("InferenceWorker")

    from PIL import Image
    from vln_subgoal_pipeline.models.cosmos3_reasoner import Cosmos3Reasoner
    from vln_subgoal_pipeline.models.locate_anything import LocateAnythingGrounder

    wlog.info("Worker: loading models...")
    reasoner = Cosmos3Reasoner(use_mock=use_mock)
    grounder = LocateAnythingGrounder(use_mock=use_mock)
    wlog.info("Worker: models ready.")

    result_queue.put({"status": "ready"})

    while True:
        try:
            request = request_queue.get(timeout=300)
        except Exception:
            wlog.warning("Worker: request queue timeout.")
            break

        if request is None:
            wlog.info("Worker: received shutdown signal.")
            break

        req_type = request.get("type")

        try:
            if req_type == "decompose":
                instruction = request["instruction"]
                image_bytes = request.get("image_bytes")
                image = None
                if image_bytes is not None:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                result = reasoner.decompose(instruction=instruction, image=image)
                result_queue.put({"type": "decompose", "result": result, "error": None})

            elif req_type == "ground":
                image_bytes = request["image_bytes"]
                target = request["target"]
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                gr = grounder.ground(image, target)
                if gr is not None:
                    result_queue.put({"type": "ground", "result": gr.to_dict(), "error": None})
                else:
                    result_queue.put({"type": "ground", "result": None, "error": None})

            else:
                result_queue.put({"type": req_type, "result": None, "error": f"Unknown request type: {req_type}"})

        except Exception as exc:
            wlog.error(f"Worker: error handling request {req_type}: {exc}")
            result_queue.put({"type": req_type, "result": None, "error": str(exc)})


class InferenceClient:
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self._request_queue: Optional[mp.Queue] = None
        self._result_queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None

    def start(self) -> None:
        ctx = mp.get_context("spawn")
        self._request_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_worker_loop,
            args=(self._request_queue, self._result_queue, self.use_mock),
            daemon=True,
        )
        self._process.start()
        logger.info("InferenceClient: subprocess started, waiting for models to load...")
        ready = self._result_queue.get(timeout=600)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Worker failed to start: {ready}")
        logger.info("InferenceClient: models ready in subprocess.")

    def stop(self) -> None:
        if self._request_queue is not None:
            try:
                self._request_queue.put(None)
            except Exception:
                pass
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
        self._process = None
        self._request_queue = None
        self._result_queue = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def decompose(self, instruction: str, image=None, timeout: float = 120.0) -> List[Dict[str, Any]]:
        image_bytes = None
        if image is not None:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
            image_bytes = buf.getvalue()

        self._request_queue.put({"type": "decompose", "instruction": instruction, "image_bytes": image_bytes})
        response = self._result_queue.get(timeout=timeout)
        if response.get("error"):
            logger.error(f"decompose error from worker: {response['error']}")
            return []
        return response.get("result") or []

    def ground(self, image, target: str, timeout: float = 120.0):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        image_bytes = buf.getvalue()

        self._request_queue.put({"type": "ground", "image_bytes": image_bytes, "target": target})
        response = self._result_queue.get(timeout=timeout)
        if response.get("error"):
            logger.error(f"ground error from worker: {response['error']}")
            return None
        res_dict = response.get("result")
        if res_dict:
            from vln_subgoal_pipeline.models.locate_anything import GroundingResult
            return GroundingResult(**res_dict)
        return None
