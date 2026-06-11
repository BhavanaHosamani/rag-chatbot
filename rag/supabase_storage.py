import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BUCKET_NAME = "pdfs"


def get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def _user_path(user_id: str, filename: str) -> str:
    """PDFs stored as {user_id}/{filename} inside the bucket."""
    return f"{user_id}/{filename}"


# =========================
# Upload PDF
# =========================

def upload_pdf_to_supabase(user_id: str, filename: str, file_bytes: bytes) -> bool:
    """Upload PDF into the user's own folder in the bucket."""
    try:
        client = get_client()
        path = _user_path(user_id, filename)
        client.storage.from_(BUCKET_NAME).upload(
            path=path,
            file=file_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"}
        )
        print(f"✅ Uploaded {path} to Supabase Storage.")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise


# =========================
# Download PDF
# =========================

def download_pdf_from_supabase(user_id: str, filename: str, dest_path: str) -> bool:
    """Download a PDF from the user's folder to dest_path."""
    try:
        client = get_client()
        path = _user_path(user_id, filename)
        response = client.storage.from_(BUCKET_NAME).download(path)
        with open(dest_path, "wb") as f:
            f.write(response)
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise


# =========================
# Delete PDF
# =========================

def delete_pdf_from_supabase(user_id: str, filename: str) -> bool:
    """Delete a PDF from the user's folder."""
    try:
        client = get_client()
        path = _user_path(user_id, filename)
        client.storage.from_(BUCKET_NAME).remove([path])
        print(f"✅ Deleted {path} from Supabase Storage.")
        return True
    except Exception as e:
        print(f"❌ Delete failed: {e}")
        return False


# =========================
# List PDFs for a user
# =========================

def list_pdfs_from_supabase(user_id: str) -> list:
    """List all PDFs in this user's folder."""
    try:
        client = get_client()
        files = client.storage.from_(BUCKET_NAME).list(path=user_id)
        return [f["name"] for f in files if f["name"].endswith(".pdf")]
    except Exception as e:
        print(f"❌ List failed: {e}")
        return []