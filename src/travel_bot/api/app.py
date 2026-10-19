"""HTTP API. Session identity comes from the URL + bearer token, never from chat text."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.responses import FileResponse, JSONResponse

from travel_bot.bootstrap import build_chat_service
from travel_bot.config import Settings
from travel_bot.errors import (
    RateLimitedError,
    RequestValidationError,
    SessionExpiredError,
    SessionNotFoundError,
    SessionUnauthorizedError,
)
from travel_bot.schemas.api import (
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    EntitiesResponse,
    MessageView,
)
from travel_bot.models.model_factory import RoleModel
from travel_bot.services.chat_service import ChatService

STATIC = Path(__file__).parent / "static"


router = APIRouter()


def chat_service(request: Request) -> ChatService:
    return request.app.state.chat


async def authorized_session(
    session_id: str,
    x_session_token: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    svc: ChatService = Depends(chat_service),
) -> str:
    await svc.authorize(session_id, x_session_token, x_user_id)
    return session_id


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest | None = None,
    x_user_id: str | None = Header(default=None),
    svc: ChatService = Depends(chat_service),
) -> CreateSessionResponse:
    return await svc.create_session(
        user_id=x_user_id, preferred_language=body.preferred_language if body else None
    )


@router.post("/sessions/{session_id}/messages", response_model=ChatResponse)
async def post_message(
    session_id: str,
    body: ChatRequest,
    x_session_token: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    svc: ChatService = Depends(chat_service),
) -> ChatResponse:
    return await svc.handle_turn(
        session_id=session_id, session_token=x_session_token,
        message_id=body.message_id, content=body.content, user_id=x_user_id,
    )


@router.get("/sessions/{session_id}/entities", response_model=EntitiesResponse)
async def get_entities(
    session_id: str = Depends(authorized_session),
    svc: ChatService = Depends(chat_service),
) -> EntitiesResponse:
    return await svc.entities(session_id)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageView])
async def get_messages(
    session_id: str = Depends(authorized_session),
    svc: ChatService = Depends(chat_service),
) -> list[MessageView]:
    return await svc.messages(session_id)


def create_app(settings: Settings | None = None, models: dict[str, RoleModel] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with build_chat_service(settings, models=models) as service:
            app.state.chat = service
            yield

    app = FastAPI(title="Travel Intake Chatbot", version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    def error(status: int, detail: str):
        return lambda _r, exc: JSONResponse(
            {"detail": detail, **({"reason_code": exc.reason_code} if hasattr(exc, "reason_code") else {})},
            status_code=status,
        )

    # Unauthorized and not-found both avoid revealing details about other sessions.
    app.add_exception_handler(SessionNotFoundError, error(404, "session not found"))
    app.add_exception_handler(SessionUnauthorizedError, error(403, "not authorized for this session"))
    app.add_exception_handler(SessionExpiredError, error(410, "session expired; start a new session"))
    app.add_exception_handler(RequestValidationError, error(422, "invalid message"))
    app.add_exception_handler(RateLimitedError, error(429, "too many messages; please slow down"))
    return app


app = create_app()


def run() -> None:
    import uvicorn

    from travel_bot.config import get_settings

    settings = get_settings()
    uvicorn.run("travel_bot.api.app:app", host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":  # pragma: no cover
    run()
