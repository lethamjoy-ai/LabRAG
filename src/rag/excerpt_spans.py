"""Application-owned excerpt spans; model output never supplies text or offsets."""
import re


def excerpt_spans(source_id, text, limit=1200):
    """Cover each source contiguously, preferring paragraph/line boundaries."""
    if not re.fullmatch(r'S\d+(?:_\d+)?', source_id):
        raise ValueError('Invalid supplied source identifier.')
    spans = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            # Keep labels and their following lines together where practical.
            boundary = text.rfind('\n\n', start + limit // 2, end)
            if boundary < 0:
                boundary = text.rfind('\n', start + limit // 2, end)
            if boundary < 0:
                boundary = text.rfind(' ', start + limit // 2, end)
            if boundary >= 0:
                end = boundary + 1
        spans.append({'excerpt_id': f'{source_id}_E{len(spans)+1}',
                      'start': start, 'end': end, 'text': text[start:end]})
        start = end
    return spans


def verified_menu(evidence):
    """Verify locally supplied boundaries again before resolving model IDs."""
    menu = {}
    for source in evidence:
        sid, text = source['source_id'], source['text']
        expected = excerpt_spans(sid, text)
        if source.get('verified_excerpts') != expected:
            raise ValueError('Invalid supplied excerpt boundaries.')
        for span in expected:
            key = span['excerpt_id']
            if key in menu:
                raise ValueError('Duplicate supplied excerpt identifier.')
            menu[key] = (sid, text[span['start']:span['end']])
    return menu
