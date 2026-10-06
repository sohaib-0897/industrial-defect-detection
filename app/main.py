import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings
from app.inference import Predictor
from app.schemas import Prediction
from src.data import read_image

LOG = logging.getLogger("defect_api")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"level": record.levelname, "event": record.getMessage(), **getattr(record, "fields", {})})


class BodyLimit:
    """Bound multipart bodies before parsing, including chunked requests."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.limit:
                response = JSONResponse({"error": "Request body exceeds size limit"}, status_code=413)
                return await response(scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()
    if not LOG.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        LOG.addHandler(handler)
        LOG.setLevel(logging.INFO)
        LOG.propagate = False

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.predictor = None
        try:
            application.state.predictor = Predictor(config)
            LOG.info(
                "model_loaded", extra={"fields": {"backend": config.backend, "model_path": str(config.model_path)}}
            )
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError) as error:
            LOG.error("model_unavailable", extra={"fields": {"reason": str(error)}})
        yield
        application.state.predictor = None

    application = FastAPI(title="Manufacturing defect detector", version="1.0.0", lifespan=lifespan)
    application.add_middleware(BodyLimit, limit=config.max_upload_bytes + 64 * 1024)

    @application.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, error: StarletteHTTPException):
        return JSONResponse({"error": error.detail}, status_code=error.status_code)

    @application.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError):
        return JSONResponse({"error": "A multipart image field named file is required"}, status_code=422)

    @application.exception_handler(Exception)
    async def unexpected_error(_: Request, error: Exception):
        LOG.exception("unhandled_error")
        return JSONResponse({"error": "Internal inference error"}, status_code=500)

    @application.get("/health")
    def health(request: Request):
        predictor = request.app.state.predictor
        if predictor is None:
            return JSONResponse({"status": "unavailable", "reason": "Model artifact not loaded"}, status_code=503)
        return {"status": "ok", "backend": config.backend, "model_version": predictor.version}

    @application.post("/predict", response_model=Prediction)
    async def predict_image(request: Request, file: UploadFile = File(...)):
        try:
            if file.content_type not in {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}:
                raise HTTPException(415, "Supported types: JPEG, PNG, BMP, TIFF, WebP")
            contents = await file.read(config.max_upload_bytes + 1)
            if len(contents) > config.max_upload_bytes:
                raise HTTPException(413, "Image exceeds upload-size limit")
            try:
                image = await run_in_threadpool(read_image, contents)
            except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                raise HTTPException(422, "Unreadable, multi-frame, unsupported or oversized image") from None
            predictor = request.app.state.predictor
            if predictor is None:
                raise HTTPException(503, "Model artifact not loaded")
            result = await run_in_threadpool(predictor.predict, image)
            LOG.info("prediction", extra={"fields": result.model_dump()})
            return result
        finally:
            await file.close()

    return application


app = create_app()
