import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


def get_anon_client() -> Client:
    """Anon client — used for login/signup."""
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def get_service_client() -> Client:
    """Service role client — used for admin operations."""
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# =========================
# Sign Up
# =========================

def sign_up(email: str, password: str) -> dict:
    """Register a new user. Returns user data or raises on error."""
    client = get_anon_client()
    response = client.auth.sign_up({"email": email, "password": password})
    if response.user is None:
        raise ValueError("Sign up failed. Please try again.")
    return {
        "user_id": response.user.id,
        "email": response.user.email,
        "access_token": response.session.access_token if response.session else None,
        "refresh_token": response.session.refresh_token if response.session else None,
    }


# =========================
# Sign In
# =========================

def sign_in(email: str, password: str) -> dict:
    """Login existing user. Returns tokens or raises on error."""
    client = get_anon_client()
    response = client.auth.sign_in_with_password({"email": email, "password": password})
    if response.user is None:
        raise ValueError("Invalid email or password.")
    return {
        "user_id": response.user.id,
        "email": response.user.email,
        "access_token": response.session.access_token,
        "refresh_token": response.session.refresh_token,
    }


# =========================
# Verify Token
# =========================

def verify_token(access_token: str) -> dict:
    """Verify a JWT access token and return user info."""
    client = get_anon_client()
    response = client.auth.get_user(access_token)
    if response.user is None:
        raise ValueError("Invalid or expired token.")
    return {
        "user_id": response.user.id,
        "email": response.user.email,
    }


# =========================
# Sign Out
# =========================

def sign_out(access_token: str):
    """Invalidate the user's session."""
    try:
        client = get_anon_client()
        client.auth.sign_out()
    except Exception:
        pass  # Best effort


# =========================
# Register PDF for User
# =========================

def register_pdf_for_user(user_id: str, pdf_name: str):
    """Save a record that this user owns this PDF."""
    client = get_service_client()
    client.table("user_pdfs").insert({
        "user_id": user_id,
        "pdf_name": pdf_name
    }).execute()


# =========================
# Get PDFs for User
# =========================

def get_pdfs_for_user(user_id: str) -> list:
    """Return list of PDF names belonging to this user."""
    client = get_service_client()
    response = client.table("user_pdfs").select("pdf_name").eq("user_id", user_id).execute()
    return [row["pdf_name"] for row in response.data]


# =========================
# Delete PDF Record for User
# =========================

def delete_pdf_record(user_id: str, pdf_name: str):
    """Remove PDF ownership record from user_pdfs table."""
    client = get_service_client()
    client.table("user_pdfs").delete().eq("user_id", user_id).eq("pdf_name", pdf_name).execute()