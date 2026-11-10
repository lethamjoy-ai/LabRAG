from src.chunking.common import make_chunks, text_spans, has_table
from src.ingestion.test_metadata import test_regions


def chunk_document_fixed(documents, chunk_size=500, overlap=75):
    documents = list(documents)
    def parts(doc):
        table = has_table(doc.page_content)
        for region_start, region_end in test_regions(doc.page_content):
            for start, end, _ in text_spans(doc.page_content[region_start:region_end], chunk_size, overlap):
                yield region_start + start, region_start + end, doc.metadata.get("section_heading", ""), {
                    "table_review_required": table, "unsplittable": False}
    return make_chunks(documents, "fixed", chunk_size, overlap, parts)
