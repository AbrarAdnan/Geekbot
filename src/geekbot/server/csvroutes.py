"""The CSV upload route: give a conversation its table (step C5).

::

    POST /v1/csv/upload?thread=<id>&filename=<name>      the file as the raw request body -> 200

A CSV conversation starts with a file. The client reserves a thread id (``POST /v1/threads``), sends
the file here with the id, and then asks questions in that thread. The route does four things, in this
order, and refuses before it does any of them if it can:

1. **Refuses early:** a bad thread name (422); a thread that is already a *chat* conversation (409:
   a thread keeps the mode it started with); a server with no CSV model set up (409); a question or
   another upload being handled right now (409, the same one-at-a-time slot as a question, because
   loading a table is blocking work on the machine the models are using).
2. **Saves the bytes** with the same rules as a document upload (``uploads.save_upload``): a plain
   name, the size cap of the perimeter (100 MB, set in front of every route), all-or-nothing, stored
   as ``<sha16>-<safe name>`` in ``<data>/uploads``. Only ``.csv`` and ``.tsv`` are accepted (415).
3. **Loads the table** in a worker thread and puts it in the registry as the thread's session
   (``csv_mode.sessions``), replacing the thread's earlier one if it had one. A file that is not a
   usable CSV is a 422 that says why; a table that alone does not fit the memory bound is a 413. In
   both cases a file that this upload created is deleted again, so a bad file does not stay on the
   disk (a file that was already there is another conversation's and stays), and the thread keeps the
   session it had.
4. **Answers** with what the user needs to see before asking: the rows, the columns and what the
   loader changed (``CsvUploadOut``). Those come from the file, so they are untrusted text.

Replacing the file of a thread that has been answered is allowed (PLAN.md §6.7: "a new upload replaces
the thread's session"); the earlier turns stay in the thread's memory and the next turn is about the
new file. The UI only offers a file chooser before the first question.
"""

from typing import Annotated

import anyio.to_thread
from fastapi import APIRouter, HTTPException, Query, Request

from geekbot.agent.mode import ModeError, choose_mode
from geekbot.agent.runner import THREADS_DB, ThreadNameError, check_thread_name, read_thread_info
from geekbot.csv_mode.errors import CsvLoadError
from geekbot.csv_mode.sessions import SessionTooLargeError
from geekbot.server.csvmode import (
    CSV_SUFFIXES,
    MAX_COLUMNS_LISTED,
    MAX_NOTES_LISTED,
)
from geekbot.server.routes import (
    AnsweringDep,
    CsvSessionsDep,
    GateDep,
    PerimeterDep,
    SettingsDep,
)
from geekbot.server.schemas import CsvUploadOut
from geekbot.server.security import UPLOADS_DIR
from geekbot.server.uploads import StoredUpload, UploadError, display_name, save_upload

__all__ = ["router"]

router = APIRouter(tags=["csv"])

BUSY = "a question or an upload is being handled; try again when it has finished"


@router.post("/v1/csv/upload")
async def upload_csv(
    request: Request,
    thread: Annotated[str, Query(min_length=1, max_length=64)],
    filename: Annotated[str, Query(min_length=1, max_length=255)],
    settings: SettingsDep,
    answering: AnsweringDep,
    gate: GateDep,
    sessions: CsvSessionsDep,
    perimeter: PerimeterDep,
) -> CsvUploadOut:
    """Receive a CSV as the raw request body and load it as the thread's table (see the module)."""
    try:
        thread_id = check_thread_name(thread)
    except ThreadNameError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if answering.csv is None:
        raise HTTPException(status_code=409, detail="this server has no CSV model set up")
    saved = await read_thread_info(settings.data_dir / THREADS_DB, thread_id)
    try:
        choose_mode(saved.values.get("mode") if saved else None, "csv")
    except ModeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    token = gate.try_acquire(thread_id)
    if token is None:
        raise HTTPException(status_code=409, detail=BUSY)
    try:
        try:
            stored = await save_upload(
                request.stream(),
                filename=filename,
                directory=settings.data_dir / UPLOADS_DIR,
                max_bytes=perimeter.max_upload_bytes,
                suffixes=CSV_SUFFIXES,
            )
        except UploadError as error:
            raise HTTPException(status_code=error.status, detail=str(error)) from error
        try:
            session = await anyio.to_thread.run_sync(sessions.open, thread_id, stored.path)
        except CsvLoadError as error:
            await _discard(stored)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except SessionTooLargeError as error:
            await _discard(stored)
            raise HTTPException(status_code=413, detail=str(error)) from error
        names = session.column_names
        return CsvUploadOut(
            thread_id=thread_id,
            file=display_name(stored.path.name),
            stored_as=stored.path.name,
            sha256=stored.sha256,
            size_bytes=stored.size,
            existed=stored.existed,
            rows=session.rows,
            column_count=len(names),
            columns=names[:MAX_COLUMNS_LISTED],
            notes=session.load_notes[:MAX_NOTES_LISTED],
            memory_bytes=session.memory_bytes,
        )
    finally:
        gate.release(token)


async def _discard(stored: StoredUpload) -> None:
    """Delete a file this upload created and nothing can use. One that was already there stays."""
    if not stored.existed:
        await anyio.to_thread.run_sync(lambda: stored.path.unlink(missing_ok=True))
