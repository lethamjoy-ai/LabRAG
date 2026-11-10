"""Versioned, constrained synthesis from typed, exactly sourced catalog facts.

No free-form model prose is accepted. Clinical values remain source-owned;
templates contribute only grammar. Unrecognized layouts fail closed.
"""
import hashlib
import json
import re
from dataclasses import dataclass,asdict
from src.rag.citation_manager import validate_selection
from src.rag.evidence_validator import source_regions

EXTRACTIVE='extractive_v1'
CONVERSATIONAL='grounded_conversational_v1'
CONVERSATIONAL_V2='grounded_conversational_v2'
MODES=(EXTRACTIVE,CONVERSATIONAL,CONVERSATIONAL_V2)
SYNTHESIS_VERSION='typed_catalog_templates_v1'


@dataclass(frozen=True)
class Fact:
    fact_id:str
    source_id:str
    test_code:str
    test_name:str
    kind:str
    data:dict
    start:int
    end:int


def clean(text):return ' '.join(text.split())


def fact(source_id,doc,kind,data,start,end):
    meta=doc.metadata
    payload=[source_id,meta['test_code'],kind,data,start,end,doc.page_content[start:end]]
    key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:24]
    return Fact(key,source_id,meta['test_code'],meta['test_name'],kind,data,start,end)


LABELS={
    'specimen required':'specimen_type','specimen type':'specimen_type',
    'collection container/tube':'collection_container','container/tube':'container',
    'submission container/tube':'submission_container','specimen volume':'specimen_volume',
    'specimen minimum volume':'minimum_volume','minimum specimen volume':'minimum_volume',
    'preferred':'preferred_tube','acceptable':'acceptable_tube'}
LABEL_PATTERN='|'.join(re.escape(k) for k in sorted(LABELS,key=len,reverse=True))
FIELD=re.compile(r'(?im)^[ \t]*('+LABEL_PATTERN+r'):[ \t]*([^\n]+(?:\n(?![ \t]*(?:[A-Za-z][^\n:]{0,70}:|\d+\.|\n|$))[^\n]+)*)')
UNSAFE=re.compile(r'ignore .*instructions|system prompt|diagnos|prescrib|treat(?:ment)?|dosage|\bmedication\b|\brecommend\b|\btherapy\b',re.I)


def extract_facts(selection,sources,extended_layouts=False):
    if not validate_selection(selection,sources):return []
    result=[]
    for excerpt in selection.excerpts:
        sid=excerpt.source_id;doc=sources[sid];text=excerpt.quote
        base=doc.page_content.find(text)
        if not doc.metadata.get('test_code') or not doc.metadata.get('test_name'):continue
        if not any(start<=base and base+len(text)<=end and code==doc.metadata['test_code']
                   for start,end,_,code in source_regions(doc)):continue
        def add(kind,data,a,b):
            if not UNSAFE.search(text[a:b]):result.append(fact(sid,doc,kind,data,base+a,base+b))
        for m in FIELD.finditer(text):
            kind=LABELS[m[1].lower()];value=clean(m[2])
            if not value:continue
            if kind in {'preferred_tube','acceptable_tube'}:
                # A bare Preferred label is ambiguous outside its source heading.
                before=text[:m.start()]
                headings=list(re.finditer(r'(?im)^\s*([^\n:]+):',before))
                roles=[h[1].strip().lower() for h in headings if h[1].strip().lower() not in {'preferred','acceptable'}]
                if not roles or roles[-1] not in {'collection container/tube','container/tube'}:continue
            if kind in {'specimen_volume','minimum_volume'} and not re.search(r'\d+(?:\.\d+)?\s*(?:mL|uL|µL|μL|L)\b|\bfull tube\b',value,re.I):continue
            specimen_role=None;role_start=None
            if kind=='specimen_type':
                basic=bool(re.fullmatch(r'(?:Serum|Plasma|Urine|Whole blood|Serum SST)(?:\s+or\s+(?:Serum|Plasma|Urine|Whole blood))*',value,re.I))
                respiratory=extended_layouts and value.lower() in {
                    'nasopharyngeal swab',
                    'nasopharyngeal (not throat) aspirate/wash or nasal aspirate/wash'}
                if respiratory:
                    roles=list(re.finditer(r'(?im)^[ \t]*(Preferred|Acceptable):[ \t]*$',text[:m.start()]))
                    if roles:
                        specimen_role=roles[-1][1].lower();role_start=roles[-1].start()
                if not basic and not (respiratory and specimen_role):continue
            data={'value':value}
            if specimen_role:data['specimen_role']=specimen_role
            if kind in {'specimen_volume','minimum_volume'}:
                data['quantities']=[{'number':n[1],'unit':n[2]} for n in re.finditer(r'(\d+(?:\.\d+)?)\s*(mL|µL|μL|uL|L|%)\b',value)]
            add(kind,data,role_start if role_start is not None else m.start(),m.end())
        # Fasting must include both duration and explicit requirement.
        for m in re.finditer(r'(?im)^[ \t]*(?:\d+\.[ \t]*)?Fasting:[ \t]*(\d+(?:\.\d+)?)[ \t]*(hours?)[ \t]*,[ \t]*(required)\.?[ \t]*$',text):
            add('fasting',{'duration':m[1],'unit':m[2],'required':True},m.start(),m.end())
        for block in re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)',text):
            value=clean(block[0]);a,b=block.span()
            prep=re.sub(r'^\d+\.\s*','',value)
            biotin=re.fullmatch(r'For (\d+(?:\.\d+)?) (hours?) before specimen collection, patient should not take multivitamins or dietary supplements(?: \(eg,[^)]*\))? containing biotin(?: \(vitamin B7\))?\.?',prep,re.I)
            if biotin:add('biotin',{'duration':biotin[1],'unit':biotin[2]},a,b)
            if re.search(r'Specimen Type\s+Temperature\s+Time',value,re.I):
                specimen=None
                for line in block[0].splitlines():
                    specimen_pattern='Serum SST|Serum|Plasma|Urine|Whole blood'+('|Fecal|Feces|Stool' if extended_layouts else '')
                    row=re.fullmatch(r'\s*(?:('+specimen_pattern+r')\s+)?((?:Frozen|Refrigerated|Ambient|Room temperature)(?:\s*\([^)]*\))?)\s+(\d+(?:\.\d+)?)\s+(days?|hours?|minutes?)(?:\s+([A-Z ][A-Z ]*))?\s*',line,re.I)
                    if not row:continue
                    specimen=row[1] or specimen
                    if not specimen:continue
                    add('specimen_type',{'value':specimen},a,b)
                    if 'Special Container' in value and not row[5]:continue
                    add('storage',{'specimen':specimen,'temperature':clean(row[2]),'duration':row[3],'unit':row[4],
                        'container':clean(row[5]) if row[5] else ''},a,b)
            instruction=re.sub(r'^\d+\.\s*','',value)
            if instruction.lower().startswith('collection instructions:'):
                instruction=instruction.split(':',1)[1].strip()
            fecal_instruction=extended_layouts and re.fullmatch(
                r'Collect a fresh random fecal specimen, no preservatives\.|'
                r'If specimen is sent refrigerate, send immediately after collection\.|'
                r'If specimen cannot be sent immediately, freeze specimen, and send frozen\.',instruction,re.I)
            if fecal_instruction:
                add('collection_instruction',{'value':instruction},a,b)
            if instruction and not re.search(r'\b(?:Preferred|Acceptable|Specimen Volume|Container/Tube):',instruction,re.I) and re.search(
                r'centrifug|aliquot|anaerob|separate specimens|first tube|remove .*stopper|protect.*light',instruction,re.I):
                add('collection_instruction',{'value':instruction},a,b)
    unique={}
    for item in result:unique.setdefault((item.test_code,item.kind,json.dumps(item.data,sort_keys=True)),item)
    return list(unique.values())


def requested_kinds(query):
    q=query.lower()
    if re.search(r'diagnos|treat|prescrib|reference range|interpret|clinical advice',q):return set()
    kinds=set()
    if re.search(r'stor|stability|frozen|refrigerat|temperature|how long',q):kinds.add('storage')
    if 'fast' in q:kinds.add('fasting')
    if 'biotin' in q:kinds.add('biotin')
    if 'prepar' in q:kinds.update({'fasting','biotin'})
    if re.search(r'process|handling|handle',q):kinds.add('collection_instruction')
    if 'minimum' in q:kinds.add('minimum_volume')
    elif 'volume' in q:kinds.add('specimen_volume')
    if 'submission' in q:kinds.add('submission_container')
    elif re.search(r'collect|container|tube|specimen (?:type|requirements)|what specimen',q):
        kinds.update({'specimen_type','preferred_tube','acceptable_tube','container','collection_container','submission_container','collection_instruction'})
        if re.search(r'requirement|what specimen',q):kinds.add('specimen_volume')
    return kinds


def sentence(item):
    d=item.data;k=item.kind;v=d.get('value','')
    if k=='specimen_type':
        role=d.get('specimen_role')
        suffix=f"{'an' if role=='acceptable' else 'a'} {role} specimen type" if role in {'preferred','acceptable'} else 'the specimen type'
        return f"The catalog lists {v.lower() if v.lower() in {'serum','plasma','urine'} else v} as {suffix}."
    templates={'preferred_tube':'A {v} tube is preferred.','acceptable_tube':'A {v} tube is also acceptable.',
        'collection_container':'The collection container is {v}.','container':'The specified container is {v}.',
        'submission_container':'Submit the specimen in a {v}.','specimen_volume':'The requested specimen volume is {v}.',
        'minimum_volume':'The minimum specimen volume is {v}.','collection_instruction':'The catalog instructs: {v}'}
    if k in templates:return templates[k].format(v=v)
    if k=='fasting':return f"Fasting for {d['duration']} {d['unit']} is required."
    if k=='biotin':return f"Avoid biotin-containing multivitamins or dietary supplements for {d['duration']} {d['unit']} before specimen collection."
    if k=='storage':return f"The catalog lists {d['specimen']} stability of {d['duration']} {d['unit']} when {d['temperature'][0].lower()+d['temperature'][1:]}"+(f" in {d['container']}" if d.get('container') else '')+'.'
    raise ValueError('Unknown fact type')


def render_claims(claims):
    groups={}
    for item in claims:groups.setdefault((item.test_code,item.test_name),[]).append(item)
    paragraphs=[]
    order=['specimen_type','preferred_tube','acceptable_tube','collection_container','container','submission_container','specimen_volume','minimum_volume','fasting','biotin','storage','collection_instruction']
    for (code,name),items in groups.items():
        ordered=sorted(items,key=lambda i:order.index(i.kind))
        title=name if re.search(r'\b'+re.escape(code)+r'\b',name,re.I) else f'{name} ({code})'
        first=sentence(ordered[0])
        direct=f'For {title}, '+first[0].lower()+first[1:]+f' [{ordered[0].source_id}]'
        details=' '.join(sentence(i)+f' [{i.source_id}]' for i in ordered[1:])
        paragraphs.append(direct+('\n\n'+details if details else ''))
    return '\n\n'.join(paragraphs)


def validate_synthesis(answer,claims,selection,sources,extended_layouts=False):
    """Verify typed roles/values, source/test IDs, spans and controlled prose."""
    if not claims or not validate_selection(selection,sources):return False
    bank={i.fact_id:i for i in extract_facts(selection,sources,extended_layouts)}
    if any(bank.get(i.fact_id)!=i for i in claims):return False
    # This is a grammar integrity check, not the evidence-support decision:
    # support above is re-derived independently as typed facts from exact sources.
    return answer==render_claims(claims)


def required_kinds(query):
    essential=set()
    for name,pattern in [('minimum_volume','minimum'),('fasting','fast'),('biotin','biotin'),('storage','stor|stability|frozen|refrigerat|temperature'),('submission_container','submission')]:
        if re.search(pattern,query,re.I):essential.add(name)
    if re.search(r'what specimen|specimen type',query,re.I):essential.add('specimen_type')
    if re.search(r'process|handling|handle',query,re.I):essential.add('collection_instruction')
    return essential


def synthesize(query,selection,sources,extended_layouts=False):
    facts=extract_facts(selection,sources,extended_layouts);kinds=requested_kinds(query)
    claims=[i for i in facts if i.kind in kinds]
    if re.search(r'frozen',query,re.I) and not re.search(r'refrigerat|stor|stability',query,re.I):
        claims=[i for i in claims if i.kind!='storage' or 'frozen' in i.data['temperature'].lower()]
    if re.search(r'refrigerat',query,re.I) and not re.search(r'frozen|stor|stability',query,re.I):
        claims=[i for i in claims if i.kind!='storage' or 'refrigerat' in i.data['temperature'].lower()]
    owners={d.metadata.get('test_code') for d in sources.values() if d.metadata.get('test_code')}
    chosen_owners={i.test_code for i in claims}
    comparison=bool(re.search(r'compare|between|difference|versus|\bvs\b',query,re.I))
    essential=required_kinds(query)
    if extended_layouts and re.search('collect',query,re.I) and re.search('stor',query,re.I):essential.add('collection_instruction')
    supported=bool(claims) and not selection.conflicting_evidence and all(any(i.kind==kind for i in claims) for kind in essential)
    if comparison and (len(chosen_owners)<2 or chosen_owners!=owners):supported=False
    if comparison and any(not any(i.kind==kind and i.test_code==code for i in claims) for code in owners for kind in essential):supported=False
    if re.search(r'exact.*temperature|temperature.*(?:celsius|degrees|°c)|numeric.*temperature',query,re.I):
        if not any(i.kind=='storage' and re.search(r'\d.*(?:°|degrees)',i.data['temperature']) for i in claims):supported=False
    # Competing values for the same field/test cannot be resolved by a template.
    fields={}
    for i in claims:
        if i.kind=='collection_instruction':continue
        key=(i.test_code,i.kind)
        if i.kind=='specimen_type' and i.data.get('specimen_role'):key+=(i.data['specimen_role'],)
        if i.kind=='storage':key+=(i.data['specimen'],re.sub(r'\s*\(preferred\)','',i.data['temperature'],flags=re.I).lower())
        fields.setdefault(key,set()).add(json.dumps(i.data,sort_keys=True))
    if any(len(values)>1 for values in fields.values()):supported=False
    answer=render_claims(claims) if supported else ''
    valid=supported and validate_synthesis(answer,claims,selection,sources,extended_layouts)
    return {'validated':bool(valid),'version':SYNTHESIS_VERSION,'claims':[asdict(i) for i in claims],
        'limitations':['Only verified fields shown are summarized; additional catalog instructions may require reviewing the supporting evidence.'] if re.search(r'requirements|preparation|compare|difference|between',query,re.I) else [],
        'answer':answer if valid else '', 'reason':'Typed facts and source mappings verified' if valid else 'Requested facts could not be verified in the supported source layouts.'}
