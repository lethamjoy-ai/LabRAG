"""Session-only provider/model controls; never initialize network clients."""
import streamlit as st


def provider_control(label,prefix,default='groq'):
    provider=st.selectbox(label,['groq','openai'],index=['groq','openai'].index(default),key=prefix+'_provider',format_func=lambda p:{'groq':'Groq','openai':'OpenAI'}[p])
    if provider=='openai':st.caption('OpenAI API usage may incur charges. API billing is separate from a ChatGPT subscription.')
    return provider


def model_control(settings,provider,prefix,label):
    key=prefix+'_model' if provider=='groq' else prefix+'_openai_model'
    saved=prefix+'_saved_'+provider+'_model'
    active=prefix+'_active_model_provider'
    if st.session_state.get(active)!=provider and saved in st.session_state:
        st.session_state[key]=st.session_state[saved]
    if key not in st.session_state:
        default=settings.llm_model if provider==settings.llm_provider else (
            'openai/gpt-oss-120b' if provider=='groq' else settings.openai_llm_model)
        st.session_state[key]=st.session_state.get(saved,default)
    model=st.text_input(label,key=key)
    st.session_state[saved]=model
    st.session_state[active]=provider
    return model
