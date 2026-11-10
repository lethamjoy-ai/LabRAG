from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

root = Path(__file__).resolve().parents[1]
assets = root / 'tmp/design_spec_qa'
assets.mkdir(parents=True, exist_ok=True)
fontdir = Path('C:/Windows/Fonts')
font = ImageFont.truetype(str(fontdir / 'arial.ttf'), 28)
small = ImageFont.truetype(str(fontdir / 'arial.ttf'), 24)
bold = ImageFont.truetype(str(fontdir / 'arialbd.ttf'), 30)

def canvas(height):
    im = Image.new('RGB', (1600, height), 'white')
    return im, ImageDraw.Draw(im)

def box(d, rect, lines, fill='#EDF3F8', title=False):
    d.rounded_rectangle(rect, radius=18, fill=fill, outline='#5D7385', width=3)
    f = bold if title else font
    x0,y0,x1,y1 = rect
    size = 30 if title else 28
    while max(d.textlength(text, font=f) for text in lines) > x1-x0-32:
        size -= 1
        f = ImageFont.truetype(str(fontdir / ('arialbd.ttf' if title else 'arial.ttf')), size)
    step = 39
    y = (y0+y1-step*len(lines))/2
    for text in lines:
        d.text(((x0+x1)/2,y),text,font=f,anchor='mt',fill='#132D40')
        y += step

def arrow(d, start, end, label=None):
    import math
    d.line([start,end],fill='#405A70',width=4)
    a=math.atan2(end[1]-start[1],end[0]-start[0]); size=15
    points=[end,(end[0]-size*math.cos(a-.5),end[1]-size*math.sin(a-.5)),
            (end[0]-size*math.cos(a+.5),end[1]-size*math.sin(a+.5))]
    d.polygon(points,fill='#405A70')
    if label:
        x,y=(start[0]+end[0])/2,(start[1]+end[1])/2
        bounds=d.textbbox((x,y),label,font=small,anchor='mm')
        d.rectangle((bounds[0]-7,bounds[1]-4,bounds[2]+7,bounds[3]+4),fill='white')
        d.text((x,y),label,font=small,anchor='mm',fill='#132D40')

im,d=canvas(1320)
box(d,(50,40,490,165),['Mayo catalog','PDF HTML TXT'],title=True)
box(d,(570,40,1020,165),['Parse clean and chunk','Fixed or section'])
box(d,(1110,40,1550,165),['Embedding service','MiniLM 384 dimensions'])
arrow(d,(490,102),(570,102));arrow(d,(1020,102),(1110,102))
box(d,(1080,265,1550,440),['Pinecone lab-rag','fixed-chunks','section-chunks'],title=True)
arrow(d,(1330,165),(1330,265),'index')
box(d,(50,300,490,445),['Streamlit chat','User questions and history'],title=True)
box(d,(50,515,490,660),['FastAPI local endpoint','Independent query requests'],title=True)
box(d,(600,340,990,570),['Shared RAG service','LangGraph workflow','Identity and evidence checks'],title=True)
arrow(d,(490,372),(600,410));arrow(d,(490,585),(600,510))
arrow(d,(990,380),(1080,350),'retrieve')
arrow(d,(1080,405),(990,455),'evidence')
box(d,(1080,555,1550,710),['Optional local reranker','Cross encoder top 15 to 5'])
arrow(d,(1080,605),(990,535))
box(d,(600,715,990,890),['OpenAI or Groq','Select verified excerpt IDs','Server configured model'])
arrow(d,(795,570),(795,715))
box(d,(530,990,1060,1155),['Strict quote validation','Typed facts and v2 coverage','Final answer and citations'],title=True)
arrow(d,(795,890),(795,990))
box(d,(45,810,430,990),['Saved evaluation runs','A B C D comparison','Versioned ground truth'])
box(d,(1150,910,1550,1085),['Optional LangSmith','Sanitized workflow spans','Restricted demo capture'])
arrow(d,(600,540),(430,810))
arrow(d,(990,550),(1350,910),'observe')
d.text((800,1250),'Preparation writes vectors only when explicitly requested. Questions read the existing index.',
       font=small,anchor='mm',fill='#405A70')
im.save(assets/'architecture.png')

im,d=canvas(1530)
steps=[(35,['Question from Streamlit or API']),
       (190,['Validate question and resolve follow up']),
       (345,['Retrieve scoped test evidence','Context expansion or optional reranking']),
       (540,['Evidence sufficient and identities resolved']),
       (735,['LLM selects application supplied excerpt IDs']),
       (890,['Resolve source spans and validate exact quotes']),
       (1065,['Conversational v2 when selected','Check fact coverage and supplement verified spans',
               'Render typed facts without another LLM call']),
       (1310,['Final validated answer and structured citations'])]
for y,lines in steps:
    height=140 if len(lines)>2 else 115 if len(lines)>1 else 95
    box(d,(90,y,1110,y+height),lines,title=len(lines)==1)
for i in range(len(steps)-1):
    y,lines=steps[i]; height=140 if len(lines)>2 else 115 if len(lines)>1 else 95
    arrow(d,(600,y+height),(600,steps[i+1][0]),'pass' if i in (3,5) else None)
box(d,(1200,535,1570,790),['Clarify or abstain','Missing evidence','Ambiguous identity','Invalid quotation'],fill='#FFF2E8')
arrow(d,(1110,585),(1200,585),'fail')
arrow(d,(1110,940),(1375,790),'fail')
arrow(d,(1110,1130),(1460,790),'unresolved')
d.text((800,1480),'No intermediate clinical answer is shown. Supporting citations retain source and page identity.',
       font=small,anchor='mm',fill='#405A70')
im.save(assets/'workflow.png')

doc=Document(root/'output/documents/LabRAG_Design_Specification.docx')
def page(title):
    doc.add_page_break();doc.add_heading(title,1)
page('Technical implementation reference')
doc.add_paragraph('Configuration baseline supplied for the presentation, reconciled with the implemented defaults. Historical indexed configurations are identified separately below.')
t=doc.add_table(rows=1,cols=2);t.style='Table Grid'
for c,text in zip(t.rows[0].cells,['Component','Implementation']):c.text=text
rows=[('Knowledge source','Mayo Clinic Laboratories catalog'),
      ('Chunking strategy 1','Fixed-size; default target 500 tokens'),
      ('Fixed chunk overlap','75 tokens by default; configurable'),
      ('Chunking strategy 2','Section-based; preserves detected headings'),
      ('Section overlap','No universal overlap across intact sections. Oversized sections use the configured overlap when split; code default is 75 tokens.'),
      ('Embedding model','sentence-transformers/all-MiniLM-L6-v2'),
      ('Embedding dimensions','384'),('Vector database','Pinecone'),('Index name','lab-rag'),
      ('Similarity metric','Cosine; checked by the Pinecone connection code'),
      ('Namespaces','fixed-chunks and section-chunks'),
      ('Retrieval','Top 5, or initial Top 15 reranked to final Top 5; configurable'),
      ('LLM providers','OpenAI and Groq; answer provider is separate from embedding configuration'),
      ('Observability','LangSmith optional tracing; local node timings and safe API telemetry')]
for row in rows:
    for c,text in zip(t.add_row().cells,row):c.text=text
for i,row in enumerate(t.rows):
    for c in row.cells:
        for p in c.paragraphs:
            p.paragraph_format.space_after=Pt(5)
            for r in p.runs:r.font.size=Pt(10);r.bold=i==0
        if i==0:
            fill=OxmlElement('w:shd');fill.set(qn('w:fill'),'E8EDF2');c._tc.get_or_add_tcPr().append(fill)
doc.add_paragraph('Defaults do not establish the configuration of existing vectors. The verified v3 evaluation labels describe historical size-100, overlap-15 chunks. The 500/75 values above are application defaults, not a claim that the existing corpus was rebuilt. Local token accounting and the embedding-model tokenizer can differ; the embedding service enforces its own input limit.')
doc.add_paragraph('Verification references: config/settings.py; src/chunking/section_chunker.py; src/vectorstore/pinecone_client.py; docs/day5_coverage_v3.md. No live index audit or configuration change was performed for this document.')
page('System architecture diagram')
doc.add_paragraph('The shared service keeps Streamlit and FastAPI on the same RAG implementation. Ingestion, evaluation and tracing remain separate concerns around that workflow.')
doc.add_picture(str(assets/'architecture.png'),width=Inches(6.8))
doc.add_paragraph('Architecture view. Context-aware retrieval, reranking and tracing are conditional features. Evaluation uses its explicit selected protocol rather than assuming the chat settings. The API defaults to independent requests; Streamlit retains conversation history.')
page('Question answering workflow diagram')
doc.add_paragraph('Successful requests proceed through evidence and quote checks before a final answer is displayed. Failed checks produce a visible clarification or abstention rather than an unsupported requirement.')
doc.add_picture(str(assets/'workflow.png'),width=Inches(6.8))
doc.add_paragraph('Workflow view. Conversational synthesis runs only in its selected modes; extractive mode returns validated excerpts. V2 supplements only trusted already retrieved passages. Policy-blocked requests can abstain before retrieval, and provider errors are handled separately from evidence insufficiency.')
doc.core_properties.title='LabRAG Design Specification'
doc.core_properties.version='1.1'
target=root/'output/documents/LabRAG_Design_Specification_v1_1.docx'
doc.save(target)
print(target)
