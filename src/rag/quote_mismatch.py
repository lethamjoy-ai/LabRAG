"""Bounded, text-free diagnostics. Alignment is explanatory, never acceptance."""
import re
from difflib import SequenceMatcher

LIMIT=2048


def safe_id(value):
    return value if isinstance(value,str) and re.fullmatch(r'S\d+(?:_\d+)?',value) else '[redacted invalid identifier]'


def tokens(text):
    return list(re.finditer(r'\w+|[^\w\s]',text))[:LIMIT]


def values(items):return [m[0] for m in items]


def contains(haystack,needle):
    if not needle:return None
    return next((i for i in range(len(haystack)-len(needle)+1) if haystack[i:i+len(needle)]==needle),None)


def adjacent(a,b):
    x,y=a.metadata,b.metadata
    if not all(x.get(k) and x.get(k)==y.get(k) for k in ('document_id','source_content_hash','test_code')):return False
    try:
        p,q=int(x['page_number']),int(y['page_number'])
        return (p==q and int(x['start_char'])<int(y['start_char'])<=int(x['end_char'])+3 and int(y['end_char'])>int(x['end_char'])) or (q==p+1 and int(y['start_char'])==0)
    except (KeyError,ValueError,TypeError):return False


def diagnose_quote(quote,source_id,sources):
    source=sources.get(source_id)
    result={'cited_source_id':safe_id(source_id),'category':'Mismatch cause cannot be determined',
        'alternate_source_ids':[],'cross_chunk_source_ids':[],
        'token_insertions':None,'token_deletions':None,'token_substitutions':None,
        'offsets':[],'whitespace_restoration_possible':False,'alignment_scope':'Local candidate span; heuristic, not an acceptance rule',
        'whitespace_restoration_succeeded':None,'analysis_truncated':False}
    if source and quote and quote in source.page_content:
        result.update(category='Exact source match',token_insertions=0,token_deletions=0,token_substitutions=0)
        return result
    # Match the SAME whitespace-only pattern used by the existing restorer.
    words=quote.split()
    if source and words and re.search(r'\s+'.join(re.escape(w) for w in words),source.page_content):
        result.update(category='Whitespace-only differences',whitespace_restoration_possible=True,
            token_insertions=0,token_deletions=0,token_substitutions=0)
        return result
    for sid,doc in list(sources.items())[:100]:
        if sid==source_id:continue
        if quote and (quote in doc.page_content or words and re.search(r'\s+'.join(re.escape(w) for w in words),doc.page_content)):
            result['alternate_source_ids'].append(safe_id(sid))
    if result['alternate_source_ids']:
        result['category']='Quote matches a different supplied source'
    if not source:
        if result['alternate_source_ids']:result['category']='Quote matches a different supplied source'
        return result
    qt,st=tokens(quote),tokens(source.page_content);q,s=values(qt),values(st)
    result['analysis_truncated']=bool(len(qt)==LIMIT or len(st)==LIMIT)
    if not q or not s:return result
    # Bound a local source window using identical token blocks.
    blocks=[b for b in SequenceMatcher(None,s,q,autojunk=False).get_matching_blocks() if b.size]
    if not blocks:return result
    start=max(0,blocks[0].a-blocks[0].b)
    last=blocks[-1]
    end=min(len(s),last.a+last.size+len(q)-(last.b+last.size))
    window=s[start:end]
    alignment=SequenceMatcher(None,window,q,autojunk=False)
    ins=delete=sub=0;offsets=[]
    for op,a,b,c,d in alignment.get_opcodes():
        if op=='equal':continue
        if op=='insert':ins+=d-c
        elif op=='delete':delete+=b-a
        else:
            sub+=min(b-a,d-c);delete+=max(0,b-a-(d-c));ins+=max(0,d-c-(b-a))
        offsets.append({'operation':op,'source_token_start':start+a,'source_token_end':start+b,
            'quote_token_start':c,'quote_token_end':d,
            'source_character_start':st[start+a].start() if start+a<len(st) else len(source.page_content),
            'quote_character_start':qt[c].start() if c<len(qt) else len(quote)})
    result.update(token_insertions=ins,token_deletions=delete,token_substitutions=sub,offsets=offsets)
    # A verified crossing needs compatible adjacent source metadata and an exact
    # token sequence spanning their suffix/prefix. Never concatenate for answering.
    if not result['analysis_truncated']:
        for sid,doc in list(sources.items())[:100]:
            if sid==source_id:continue
            for aid,a,bid,b in ((source_id,source,sid,doc),(sid,doc,source_id,source)):
                if not adjacent(a,b):continue
                av,bv=values(tokens(a.page_content)),values(tokens(b.page_content))
                overlap=max((i for i in range(1,min(len(av),len(bv))+1) if av[-i:]==bv[:i]),default=0)
                joined=av+bv[overlap:];position=contains(joined,q)
                if position is not None and position<len(av)<position+len(q):
                    result['cross_chunk_source_ids']=[safe_id(aid),safe_id(bid)]
    qwords=[t for t in q if re.fullmatch(r'\w+',t)]
    swords=[t for t in window if re.fullmatch(r'\w+',t)]
    heading=re.match(r'^\s*([^\n:]{1,60}:)',quote)
    absent_heading=bool(heading and ' '.join(heading[1].split()) not in ' '.join(source.page_content.split()))
    if result['alternate_source_ids']:category='Quote matches a different supplied source'
    elif result['cross_chunk_source_ids']:category='Quote spans multiple chunk boundaries'
    elif alignment.ratio()<.45 or result['analysis_truncated']:category='Mismatch cause cannot be determined'
    elif qwords==swords:category='Quote changes punctuation'
    elif absent_heading:category='Quote includes a heading absent from the cited chunk'
    elif sub:category='Quote substitutes words'
    elif ins and not delete:category='Quote contains inserted words'
    elif delete and not ins:category='Quote omits source words'
    else:category='Mismatch cause cannot be determined'
    result['category']=category
    return result
