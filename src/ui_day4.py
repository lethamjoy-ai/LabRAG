"""Native session chat. Services run only on explicit question submission."""
import time
from pathlib import Path
from dataclasses import replace
import streamlit as st
from src.llm.groq_service import GroqService
from src.retrieval.semantic_retriever import SemanticRetriever
from src.utils.conversation_manager import append_turn


def reset_chat():
    for key in ('rag_history','labrag_chat_input','laboratory_question','labrag_draft','labrag_sample',
                '_labrag_pending_question','_labrag_restore_draft','_labrag_empty_submission'):
        st.session_state.pop(key,None)


SAMPLE_PLACEHOLDER='Choose a sample…'


def populate_sample():
    selected=st.session_state.get('labrag_sample',SAMPLE_PLACEHOLDER)
    if selected!=SAMPLE_PLACEHOLDER:st.session_state['labrag_draft']=selected


def queue_question():
    question=st.session_state.get('labrag_draft','').strip()
    if not question:
        st.session_state['_labrag_empty_submission']=True
        return
    st.session_state['_labrag_pending_question']=question
    st.session_state['labrag_draft']=''
    st.session_state['labrag_sample']=SAMPLE_PLACEHOLDER


def final_chat_text(turn):
    """Freeze only the final validated display text, never an intermediate quote."""
    if 'chat_display_answer' in turn:return turn['chat_display_answer']
    conversational=bool((turn.get('synthesis_diagnostics') or {}).get('validated'))
    return turn.get('display_answer',turn['answer']) if conversational else turn['answer']


def show_assistant(turn,tracing):
    if turn.get('llm_provider'):st.caption(f"{turn['llm_provider'].title()} · {turn.get('llm_model','Unknown model')}")
    else:st.caption('Provider/model not recorded for this earlier turn.')
    identities=sorted({(c.get('test_name'),c.get('test_code')) for c in turn.get('citations',[])
        if c.get('test_name') and c.get('test_code') and not str(c['test_name']).startswith('Unresolved')})
    conversational=bool((turn.get('synthesis_diagnostics') or {}).get('validated'))
    if not conversational:
        for name,code in identities:st.caption(f'{name} · {code}')
    st.write(final_chat_text(turn))
    if turn.get('error'):return
    for label in ('missing_information','conflicting_evidence'):
        if turn.get(label):st.caption(label.replace('_',' ').capitalize()+': '+'; '.join(turn[label]))
    with st.expander('Supporting Documents',expanded=False):
        citations=turn.get('citations',[])
        if citations:
            st.markdown('**Supporting source citations**')
            for index,source in enumerate(citations,1):
                st.caption(f"{source.get('test_name','Unresolved test')} · {source.get('test_code','Unresolved code')} · Page {source.get('printed_catalog_page','Unknown')}" if conversational else
                    f"{source.get('source_id','Source')} · {source.get('test_name','Unresolved test')} · Page {source.get('printed_catalog_page','Unknown')}")
                if source.get('source_url'):st.link_button(f"Open reference {index}" if conversational else f"Open source {source['source_id']}",source['source_url'])
        with st.expander('Copy answer',expanded=False):
            st.code(final_chat_text(turn),language=None)
            st.caption('Use the copy icon in the upper-right corner of the answer block.')
        if conversational:
            with st.expander('View supporting evidence',expanded=False):
                st.write('Verified structured claims',turn.get('synthesis_diagnostics',{}).get('claims',[]))
                st.write('Validated source quotations',turn.get('extractive_answer'))
                st.write('Stable source references',citations)
        with st.expander('View Retrieval Details',expanded=False):
            st.write('Answer synthesis validation',turn.get('synthesis_diagnostics') or 'Not attempted in this turn')
            if turn.get('extractive_answer'):st.write('Validated extractive evidence',turn['extractive_answer'])
            st.markdown('**Retrieved evidence**')
            for key,doc in turn.get('sources',{}).items():
                st.caption(key);st.text(doc.page_content)
                with st.expander(f'Source metadata {key}',expanded=False):st.json(doc.metadata)
            st.write('Citation metadata',citations)
            st.write({'resolved_query':turn.get('query'),'namespace':turn.get('namespace'),
                'evidence_check':turn.get('reason'),'status':turn.get('status')})
            st.write(turn.get('retrieval_details',{}))
            st.write([{'source':m.get('source_id',f'S{i}'),'similarity':m['score'],
                'chunk_id':m['document'].metadata.get('chunk_id'),'reranker_score':m.get('reranker_score')}
                for i,m in enumerate(turn.get('matches',[]),1)])
            st.write('Timings',{**turn.get('metrics',{}),'total_seconds':turn.get('total_seconds')})
            st.caption('Cosine similarity is not a confidence score. Missing page metadata is shown as Unknown.')
            if tracing:
                with st.expander('Workflow node timings',expanded=False):st.json(turn.get('trace',[]))


def show_turn(turn,tracing):
    with st.chat_message('user'):st.write(turn['question'])
    with st.chat_message('assistant'):show_assistant(turn,tracing)


def render_history(history,tracing,pending_question=''):
    """Single path for saved answers and the pending assistant status."""
    for turn in history:show_turn(turn,tracing)
    if pending_question:
        with st.chat_message('user'):st.write(pending_question)
        with st.chat_message('assistant'):
            st.status('Searching laboratory documentation and validating the answer...',
                      state='running',expanded=False)
    elif not history:
        st.caption('Ask about specimen requirements, preparation, collection or storage.')


def _render_chat(settings):
    st.subheader('Ask LabRAG')
    st.caption('Evidence-grounded laboratory reference. Verify current official instructions before clinical use; do not enter patient information.')
    new,clear=st.columns(2)
    new.button('New Chat',on_click=reset_chat,use_container_width=True)
    clear.button('Clear Conversation',on_click=reset_chat,use_container_width=True)
    from src.ui_llm import provider_control,model_control
    with st.sidebar:
        with st.expander('Chat Settings',expanded=False):
            provider=provider_control('LLM Provider','chat',default=settings.llm_provider)
            groq_default=settings.llm_model if settings.llm_provider=='groq' else 'openai/gpt-oss-120b'
            previous=st.session_state.get('chat_model_config_default')
            if previous!=groq_default:
                if st.session_state.get('chat_model') in {None,previous,'llama-3.3-70b-versatile'}:
                    st.session_state['chat_model']=groq_default
                st.session_state['chat_model_config_default']=groq_default
            answer_style=st.selectbox('Answer generation mode',['grounded_conversational_v2','grounded_conversational_v1','extractive_v1'],
                format_func=lambda value:{'grounded_conversational_v2':'Grounded conversational v2 (evidence coverage)',
                    'grounded_conversational_v1':'Grounded conversational v1 (historical)',
                    'extractive_v1':'Extractive source quotes'}[value],key='chat_answer_mode')
            model=model_control(settings,provider,'chat','Chat model')
            strategy=st.selectbox('Chat retrieval strategy',['section','fixed'],index=0 if settings.rag_default_strategy=='section' else 1)
            top_k=st.number_input('Chat top-k',min_value=1,max_value=100,value=settings.retrieval_top_k)
            use_reranking=st.checkbox('Enable local reranking',value=False,
                help='Optional CPU cross-encoder. First use may download weights. Errors do not silently fall back.')
            candidate_count=st.number_input('Chat initial candidates',min_value=int(top_k),max_value=100,value=max(15,int(top_k))) if use_reranking else 15
            st.caption('Pinecone: '+st.session_state.get('rag_pinecone_status','not checked in this chat session'))
        st.caption(f'Selected: {provider.title()} · {model}')
    history=st.session_state.get('rag_history',[])
    question=st.session_state.pop('_labrag_pending_question',None)
    question=question.strip() if isinstance(question,str) else ''
    conversation=st.container(height=640,border=False,autoscroll=bool(history) or bool(question))
    with conversation:
        history_slot=st.empty()
    with history_slot.container():
        render_history(history,settings.rag_enable_tracing,question)
    if question:
        started=time.perf_counter()
        try:
            from src.ui_day3 import services
            from src.rag.service import answer_question
            configured=replace(settings,llm_provider=provider,llm_model=model)
            connection,embeddings=services(configured)
            from src.retrieval.reranker import Reranker
            st.session_state.rag_pinecone_status='connected; existing index configuration validated'
            from src.llm.provider_factory import create_llm
            service=GroqService(configured) if provider=='groq' else create_llm(configured)
            result=answer_question(question,history,SemanticRetriever(configured,embeddings,connection),
                service,configured,strategy,int(top_k),
                reranker=Reranker() if use_reranking else None,candidate_count=int(candidate_count),answer_generation_mode=answer_style)
            result['total_seconds']=time.perf_counter()-started
        except Exception as exc:
            from src.llm.groq_service import LLMError
            from src.retrieval.reranker import RerankerError
            detail=getattr(exc,'error_diagnostics',{}) if isinstance(exc,LLMError) else {}
            if detail.get('http_status')==429 and provider=='groq':
                message="Groq's API usage limit has been reached. Please try again after the quota resets or select OpenAI."
            elif isinstance(exc,LLMError):message=detail.get('sanitized_message') or 'The language service is unavailable. Check configuration and try again later.'
            elif isinstance(exc,RerankerError):message='Local reranking is unavailable. Check the model configuration and try again.'
            else:message='Chat failed. Check the Pinecone connection, embedding configuration, and provider availability.'
            result={'question':question,'answer':message,'error':True}
        result.update(llm_provider=provider,llm_model=model)
        result['chat_display_answer']=final_chat_text(result)
        if result.get('error'):st.session_state['_labrag_restore_draft']=question
        history=append_turn(history,result,settings.rag_max_history_turns)
        st.session_state.rag_history=history
        # Replace the loading block directly from committed history. No full-app
        # rerun or second answer renderer sits between generation and display.
        with history_slot.container():
            render_history(history,settings.rag_enable_tracing)
    from src.ui_presentation import sample_questions
    examples=sample_questions(Path(__file__).resolve().parents[1])
    restored=st.session_state.pop('_labrag_restore_draft',None)
    if restored is not None:st.session_state['labrag_draft']=restored
    field,samples,send=st.columns([6,3,1],vertical_alignment='bottom')
    field.text_input('Laboratory question',placeholder='Ask a laboratory question...',key='labrag_draft',max_chars=3000)
    samples.selectbox('Sample Questions',[SAMPLE_PLACEHOLDER]+examples,key='labrag_sample',
        on_change=populate_sample,disabled=not examples,help='Choose a question to fill the input. Edit it, then press Send. No request is made by selecting a sample.')
    send.button('Send',type='primary',on_click=queue_question,use_container_width=True)
    if st.session_state.pop('_labrag_empty_submission',False):st.warning('Enter a laboratory question before sending.')



def render_chat(settings):
    _render_chat(settings)
