"""csv_analyst -- ask questions about a CSV; every answer comes from a real query on the data.

Public API (INTEGRATION.md has the host-app guide):

    from csv_analyst import CsvAnalystSession, to_markdown

    session = CsvAnalystSession.from_path("sales.csv")
    print(to_markdown(session.ask("What is the average revenue by region?")))
"""

from csv_analyst.config import settings
from csv_analyst.contracts import RunRecord, Status
from csv_analyst.ingestion.validators import IngestionError
from csv_analyst.llm.base import LLMProviderError
from csv_analyst.session import CsvAnalystSession, make_provider, to_markdown

__version__ = "1.0.0"
__all__ = [
    "CsvAnalystSession", "IngestionError", "LLMProviderError", "RunRecord", "Status", "make_provider",
    "settings", "to_markdown",
]
