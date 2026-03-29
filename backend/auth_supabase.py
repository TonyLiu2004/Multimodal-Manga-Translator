"""Verify Supabase user access_token: locally (JWT secret) or via Auth REST (no legacy secret)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Optional
from uuid import UUID

import httpx
import jwt
from fastapi import Depends, Header, HTTPException

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


@dataclass(frozen=True)
class AuthContext:
    user_id: UUID
    email: Optional[str]


def _auth_context_from_jwt_local(token: str, secret: str) -> AuthContext:
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except jwt.InvalidAudienceError:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token missing sub")
    try:
        user_id = UUID(str(sub))
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid user id in token")
    raw_email = payload.get("email")
    email: Optional[str] = (
        str(raw_email) if isinstance(raw_email, str) and raw_email.strip() else None
    )
    return AuthContext(user_id=user_id, email=email)


def _auth_context_from_supabase_remote(token: str, base_url: str, anon_key: str) -> AuthContext:
    url = f"{base_url.rstrip('/')}/auth/v1/user"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": anon_key,
                },
            )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach Supabase Auth: {e}",
        ) from e

    if r.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Supabase Auth returned {r.status_code}",
        )

    try:
        data: dict[str, Any] = r.json()
    except ValueError as e:
        raise HTTPException(status_code=502, detail="Invalid response from Supabase Auth") from e
    uid = data.get("id")
    if not uid:
        raise HTTPException(status_code=502, detail="Supabase Auth response missing user id")
    try:
        user_id = UUID(str(uid))
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid user id from Supabase Auth")
    raw_email = data.get("email")
    email: Optional[str] = (
        str(raw_email) if isinstance(raw_email, str) and raw_email.strip() else None
    )
    return AuthContext(user_id=user_id, email=email)


def auth_context_from_authorization(authorization: str | None) -> AuthContext:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )
    token = authorization[7:].strip()

    secret = (os.environ.get("SUPABASE_JWT_SECRET") or "").strip()
    if secret:
        return _auth_context_from_jwt_local(token, secret)

    base_url = (os.environ.get("SUPABASE_URL") or "").strip()
    anon_key = (
        os.environ.get("SUPABASE_ANON_KEY")
        or os.environ.get("SUPABASE_PUBLISHABLE_KEY")
        or ""
    ).strip()
    if base_url and anon_key:
        return _auth_context_from_supabase_remote(token, base_url, anon_key)

    raise HTTPException(
        status_code=503,
        detail=(
            "Server is not configured for Supabase auth: set SUPABASE_JWT_SECRET, "
            "or set SUPABASE_URL + SUPABASE_ANON_KEY (publishable/anon key) to verify tokens via Auth API."
        ),
    )


def user_id_from_authorization(authorization: str | None) -> UUID:
    return auth_context_from_authorization(authorization).user_id


def get_current_user_id(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> UUID:
    return user_id_from_authorization(authorization)


def get_auth_context(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> AuthContext:
    return auth_context_from_authorization(authorization)


CurrentUserId = Annotated[UUID, Depends(get_current_user_id)]
CurrentAuthContext = Annotated[AuthContext, Depends(get_auth_context)]
