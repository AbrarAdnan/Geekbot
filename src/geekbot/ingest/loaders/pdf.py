"""PDF loader: one segment per page, with the page number a reader would see (step S1.1)."""

from pathlib import Path

from pypdf import PasswordType, PdfReader

from geekbot.ingest.loaders.common import clean_text
from geekbot.ingest.loaders.limits import check_page_limit
from geekbot.ingest.loaders.model import LoadedDocument, LoadError, Segment


def load_pdf(path: Path) -> LoadedDocument:
    """Extract the text layer of a PDF, page by page.

    Pages are numbered from 1 and a page with no text leaves a gap in the numbers instead of
    shifting the later ones, so "page 7" in a citation is page 7 in the viewer. A PDF with no text
    on any page is almost always a scan: it fails with that hint, because Geekbot does no OCR and
    silently indexing nothing would look like success.

    PDFs "encrypted" only to forbid printing or copying have an empty user password and open
    normally; one that needs a real password is a ``LoadError``.
    """
    reader = PdfReader(path)
    if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
        raise LoadError("the PDF is password-protected")
    check_page_limit(len(reader.pages))  # before any page is read (S4.3)
    segments = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text())
        if text:
            segments.append(Segment(text, page=number))
    if not segments:
        raise LoadError("no text found in the PDF (a scan? OCR is not supported)")
    return LoadedDocument(path=path, kind="pdf", title=path.name, segments=tuple(segments))
