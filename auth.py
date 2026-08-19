"""Sign-in (Google or email), signed session cookies, and the token-credit ledger.

Deliberately dependency-free: Google's own `tokeninfo` endpoint verifies the ID
token (signature, expiry, issuer) so there is no JWT library here, passwords are
hashed with stdlib scrypt so there is no bcrypt/passlib, and the session cookie is
`uid.expiry.hmac` signed with stdlib hmac so there is no session table.

api.py owns the cookie itself (set/read/clear); this module owns what goes in it
and what the database says about the user.

Only the name and email are kept — no avatar URL, and passwords are stored as
scrypt digests, never recoverable.

Env: GOOGLE_CLIENT_ID (required for the Google button), SESSION_SECRET (required in
     production — a random per-process key is used if unset, so restarting the
     API logs everyone out), FREE_TOKENS, ADMIN_EMAILS (comma-separated).
"""

import hashlib
import hmac
import os
import secrets
import time

import httpx
from sqlalchemy import text

from rag import engine, log

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
FREE_TOKENS = int(os.getenv("FREE_TOKENS", "50000"))
ADMIN_EMAILS = {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}
COOKIE = "auth"
SESSION_DAYS = 30

_SECRET = os.getenv("SESSION_SECRET", "").encode()
if not _SECRET:
    _SECRET = secrets.token_bytes(32)
    log.warning("SESSION_SECRET is not set - logins will not survive a restart. See .env.example")

TOKENINFO = "https://oauth2.googleapis.com/tokeninfo"


# --- Google ID token ---------------------------------------------------------


def verify_google(credential: str) -> dict:
    """Google's claims for an ID token, or ValueError. Google checks the signature
    and expiry; we check the token was minted for *this* app."""
    if not GOOGLE_CLIENT_ID:
        raise RuntimeError("GOOGLE_CLIENT_ID is not set - sign-in is unavailable.")
    r = httpx.get(TOKENINFO, params={"id_token": credential}, timeout=10)
    if r.status_code != 200:
        raise ValueError("That sign-in didn't verify — please try again.")
    claims = r.json()
    if claims.get("aud") != GOOGLE_CLIENT_ID:
        raise ValueError("That sign-in was issued for a different app.")
    if str(claims.get("email_verified", "")).lower() != "true":
        raise ValueError("That Google account has no verified email address.")
    if not claims.get("email"):
        raise ValueError("That sign-in carried no email address.")
    return claims


# --- email + password -------------------------------------------------------

# scrypt's cost *is* the protection: ~100ms per attempt, so a stolen table is
# expensive to crack and online guessing is slow.
# ponytail: no lockout and no rate limit — add one if this ever leaves portfolio use.
_SCRYPT = {"n": 1 << 14, "r": 8, "p": 1}
MIN_PASSWORD = 8


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        if kind != "scrypt":
            return False
        again = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32
        )
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(again.hex(), digest)


def _clean_email(email: str) -> str:
    email = (email or "").strip().lower()
    # not RFC-complete on purpose; it only has to reject obvious junk before storage
    if "@" not in email or "." not in email.partition("@")[2] or len(email) > 254:
        raise ValueError("That doesn't look like an email address.")
    return email


def signup(email: str, password: str, name: str = "") -> dict:
    """Create an email account with the free grant. ValueError if the email is taken."""
    email = _clean_email(email)
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"Please use a password of at least {MIN_PASSWORD} characters.")
    with engine().begin() as c:
        taken = c.execute(
            text("SELECT pw_hash FROM analytics.users WHERE email = :email"), {"email": email}
        ).first()
        if taken:
            raise ValueError(
                "That email already has an account — sign in instead."
                if taken[0]
                else "That email is registered with Google — use the Google button."
            )
        row = c.execute(
            text(
                "INSERT INTO analytics.users (email, name, pw_hash, tokens_left, is_admin) "
                "VALUES (:email, :name, :pw, :grant, :admin) "
                "RETURNING id, email, name, tokens_left, tokens_used, is_admin"
            ),
            {
                "email": email,
                "name": (name or "").strip() or email.split("@")[0],
                "pw": hash_password(password),
                "grant": FREE_TOKENS,
                "admin": email in ADMIN_EMAILS,
            },
        ).mappings().one()
    log.info("signed up | %s | %d tokens", email, row["tokens_left"])
    return dict(row)


def login_email(email: str, password: str) -> dict:
    """Verify a password. One message for every failure — never reveal which emails exist."""
    wrong = ValueError("Wrong email or password.")
    with engine().begin() as c:
        row = c.execute(
            text(
                "SELECT id, email, name, tokens_left, tokens_used, is_admin, pw_hash "
                "FROM analytics.users WHERE email = :email"
            ),
            {"email": (email or "").strip().lower()},
        ).mappings().first()
    if not row or not row["pw_hash"] or not check_password(password, row["pw_hash"]):
        raise wrong
    # is_admin follows ADMIN_EMAILS on every sign-in, exactly as the Google path does —
    # editing the env must be enough to grant or revoke the console
    admin = row["email"] in ADMIN_EMAILS
    with engine().begin() as c:
        c.execute(
            text("UPDATE analytics.users SET last_seen = now(), is_admin = :admin WHERE id = :uid"),
            {"admin": admin, "uid": row["id"]},
        )
    user = {k: v for k, v in row.items() if k != "pw_hash"} | {"is_admin": admin}
    log.info("signed in | %s | %d tokens left", user["email"], user["tokens_left"])
    return user


# --- session cookie ---------------------------------------------------------


def sign(uid: int) -> str:
    body = f"{uid}.{int(time.time()) + SESSION_DAYS * 86400}"
    return f"{body}.{hmac.new(_SECRET, body.encode(), hashlib.sha256).hexdigest()[:32]}"


def parse(cookie: str) -> int | None:
    """User id from a cookie we signed ourselves, else None (forged or expired)."""
    uid, _, rest = cookie.partition(".")
    exp, _, sig = rest.partition(".")
    if not (uid.isdigit() and exp.isdigit() and sig):
        return None
    good = hmac.new(_SECRET, f"{uid}.{exp}".encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(sig, good) or int(exp) < time.time():
        return None
    return int(uid)


# --- users ------------------------------------------------------------------


def login(claims: dict) -> dict:
    """Create the user on first sign-in (with the free grant) or refresh their profile."""
    email = claims["email"].lower()
    with engine().begin() as c:
        row = c.execute(
            text(
                "INSERT INTO analytics.users (email, name, tokens_left, is_admin) "
                "VALUES (:email, :name, :grant, :admin) "
                "ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name, "
                "is_admin = EXCLUDED.is_admin, last_seen = now() "
                "RETURNING id, email, name, tokens_left, tokens_used, is_admin"
            ),
            {
                "email": email,
                "name": claims.get("name") or email.split("@")[0],
                "grant": FREE_TOKENS,
                "admin": email in ADMIN_EMAILS,
            },
        ).mappings().one()
    log.info("signed in | %s | %d tokens left", email, row["tokens_left"])
    return dict(row)


def user(uid: int) -> dict | None:
    with engine().begin() as c:
        row = c.execute(
            text(
                "SELECT id, email, name, tokens_left, tokens_used, is_admin "
                "FROM analytics.users WHERE id = :uid"
            ),
            {"uid": uid},
        ).mappings().first()
    return dict(row) if row else None


def spend(uid: int, tokens: int) -> None:
    """Charge a finished request. Clamped at zero: the check happens before a turn
    starts, so the last turn is allowed to overshoot rather than be cut off."""
    if tokens <= 0:
        return
    with engine().begin() as c:
        c.execute(
            text(
                "UPDATE analytics.users SET tokens_left = greatest(tokens_left - :n, 0), "
                "tokens_used = tokens_used + :n WHERE id = :uid"
            ),
            {"n": tokens, "uid": uid},
        )


def list_users() -> list[dict]:
    with engine().begin() as c:
        rows = c.execute(
            text(
                "SELECT id, email, name, tokens_left, tokens_used, is_admin, "
                "created, last_seen FROM analytics.users ORDER BY created DESC"
            )
        ).mappings().all()
    return [dict(r) for r in rows]


def set_tokens(uid: int, tokens: int) -> dict | None:
    with engine().begin() as c:
        c.execute(
            text("UPDATE analytics.users SET tokens_left = :n WHERE id = :uid"),
            {"n": max(tokens, 0), "uid": uid},
        )
    return user(uid)


def charged(handler, text_len: int = 0) -> int:
    """Tokens to bill from a UsageMetadataCallbackHandler.

    Gateways and self-hosted routers often omit `usage`, which would make every
    request free. Fall back to the rough 4-chars-per-token rule so credits still
    move. ponytail: estimate, not accounting — swap in the gateway's own numbers
    if you ever bill money against this.
    """
    reported = sum(u.get("total_tokens", 0) for u in (handler.usage_metadata or {}).values())
    return reported or text_len // 4 + 200
