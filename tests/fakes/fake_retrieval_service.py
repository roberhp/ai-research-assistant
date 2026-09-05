class FakeRetrievalService:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    def retrieve(self, query: str, limit: int = 5):
        return self.results_by_query.get(query, [])[:limit]