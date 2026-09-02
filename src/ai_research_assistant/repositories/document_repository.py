from sqlalchemy.orm import Session

from ai_research_assistant.models import DocumentModel


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, source: str) -> DocumentModel:
        document = DocumentModel(
            source=source,
        )

        self.session.add(document)
        self.session.flush()

        return document

    def find_by_source(self, source: str) -> DocumentModel | None:
        return (
            self.session.query(DocumentModel)
            .filter(DocumentModel.source == source)
            .first()
        )