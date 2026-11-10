import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from src.llm.groq_service import AnswerSelection
from src.rag.conversational_answer import synthesize,extract_facts,validate_synthesis,render_claims,CONVERSATIONAL


def document(text,code='BUN',name='Blood Urea Nitrogen (BUN), Serum'):
    return Document(page_content=text,metadata={'test_code':code,'test_name':name,'corpus_type':'synthetic','document_id':'fixture'})


def selection(sources):
    return AnswerSelection(excerpts=[{'source_id':sid,'quote':doc.page_content} for sid,doc in sources.items()])


def test_bun_specimen_and_distinct_collection_submission_fields():
    sources={'S1':document('Collection Container/Tube:\nPreferred: Serum gel\nAcceptable: Red top\nSubmission Container/Tube: Plastic vial\nSpecimen Volume: 0.5 mL'),
        'S2':document('Specimen Type        Temperature         Time\nSerum                  Refrigerated (preferred) 7 days\n                       Frozen              365 days')}
    result=synthesize('What specimen is required for BUN?',selection(sources),sources)
    assert result['validated']
    assert '\n\n' in result['answer']  # Direct answer first, supporting details second.
    assert 'serum as the specimen type' in result['answer']
    assert 'Serum gel tube is preferred' in result['answer'] and 'Red top tube is also acceptable' in result['answer']
    assert 'Submit the specimen in a Plastic vial' in result['answer'] and 'requested specimen volume is 0.5 mL' in result['answer']
    assert '365 days' not in result['answer']
    assert {c['source_id'] for c in result['claims']}=={'S1','S2'}


def test_calcium_minimum_does_not_become_requested_volume():
    sources={'S1':document('Specimen Volume: 0.5 mL\nSpecimen Minimum Volume: 0.25 mL','CA','Calcium, Total, Serum')}
    result=synthesize('What is the minimum specimen volume for Calcium, Total, Serum?',selection(sources),sources)
    assert result['validated'] and '0.25 mL' in result['answer'] and '0.5 mL' not in result['answer']
    assert all(c['kind']=='minimum_volume' for c in result['claims'])


def test_c_peptide_fasting_and_biotin_qualifiers():
    text='1. Fasting: 8 hours, required\n\n2. For 12 hours before specimen collection, patient should not take multivitamins or\ndietary supplements (eg, hair, skin, and nail supplements) containing biotin (vitamin B7).'
    sources={'S1':document(text,'CPR','C-Peptide, Serum')}
    result=synthesize('What fasting and biotin preparation is required for C-Peptide?',selection(sources),sources)
    assert result['validated'] and 'fasting for 8 hours is required' in result['answer'].lower()
    assert '12 hours before specimen collection' in result['answer']
    assert {c['kind'] for c in result['claims']}=={'fasting','biotin'}
    missing={'S1':document('Fasting: 8 hours, required','CPR','C-Peptide, Serum')}
    assert not synthesize('What fasting and biotin preparation is required?',selection(missing),missing)['validated']


def test_storage_preserves_specimen_temperature_duration_and_special_container():
    text='Specimen Type        Temperature         Time\nSerum                  Refrigerated (2-8 °C) 7 days\n                       Frozen              365 days'
    sources={'S1':document(text)}
    result=synthesize('How long can BUN serum be refrigerated or frozen?',selection(sources),sources)
    assert result['validated'] and '2-8 °C' in result['answer'] and '7 days' in result['answer'] and '365 days' in result['answer']
    assert all(c['data']['specimen']=='Serum' for c in result['claims'])
    special={'S1':document('Specimen Type  Temperature  Time  Special Container\nSerum SST  Refrigerated  7 days  SERUM GEL TUBE','CAI','Calcium, Ionized, Serum')}
    result=synthesize('What storage conditions apply to CAI?',selection(special),special)
    assert result['validated'] and 'SERUM GEL TUBE' in result['answer']


def test_multi_test_identity_and_serum_plasma_distinctions():
    sources={'S1':document('Specimen Type: Serum\nCollection Container/Tube: Serum gel','CA','Calcium, Total, Serum'),
        'S2':document('Specimen Type: Plasma\nCollection Container/Tube: Heparin tube','OTHER','Other Test, Plasma')}
    result=synthesize('Compare the specimen types and collection requirements between CA and OTHER',selection(sources),sources)
    assert result['validated'] and 'Serum' in result['answer'] and 'plasma' in result['answer']
    assert {c['test_code'] for c in result['claims'] if c['source_id']=='S1'}=={'CA'}
    assert {c['test_code'] for c in result['claims'] if c['source_id']=='S2'}=={'OTHER'}
    bad=AnswerSelection(excerpts=[{'source_id':'S1','quote':sources['S1'].page_content}])
    assert not synthesize('Compare specimen requirements between CA and OTHER',bad,sources)['validated']


@pytest.mark.parametrize('change',[
    lambda f:replace(f,data={'value':'0.5 mL'}),
    lambda f:replace(f,kind='specimen_volume'),
    lambda f:replace(f,source_id='S999'),
    lambda f:replace(f,test_code='CAI'),
    lambda f:replace(f,start=f.start+1),
])
def test_structured_fact_tampering_is_rejected(change):
    sources={'S1':document('Specimen Minimum Volume: 0.25 mL','CA','Calcium, Total, Serum')}
    selected=selection(sources);facts=extract_facts(selected,sources);altered=[change(facts[0])]
    assert not validate_synthesis(render_claims(altered),altered,selected,sources)


def test_unsupported_clinical_claims_and_ambiguity_abstain():
    sources={'S1':document('Specimen Minimum Volume: 0.25 mL','CA','Calcium, Total, Serum')}
    assert not synthesize('Which treatment should I take?',selection(sources),sources)['validated']
    selected=selection(sources);facts=extract_facts(selected,sources)
    assert not validate_synthesis(render_claims(facts)+' This diagnoses cancer.',facts,selected,sources)
    assert not synthesize('What is the submission container?',selected,sources)['validated']
    conflict={'S1':document('Specimen Minimum Volume: 0.25 mL'),'S2':document('Specimen Minimum Volume: 1 mL')}
    assert not synthesize('What is the minimum volume?',selection(conflict),conflict)['validated']
    unresolved={'S1':Document(page_content='Specimen Type: Serum')}
    assert not synthesize('What specimen is required?',selection(unresolved),unresolved)['validated']


def test_quote_validation_still_rejects_missing_words_before_synthesis():
    sources={'S1':document('Specimen Minimum Volume: 0.25 mL')}
    altered=AnswerSelection(excerpts=[{'source_id':'S1','quote':'Minimum Volume: 0.5 mL'}])
    assert not synthesize('What minimum volume?',altered,sources)['validated']


def test_preparation_exceptions_are_not_silently_removed():
    sources={'S1':document('Fasting: 8 hours, required unless otherwise directed\n\nFor 12 hours before specimen collection, patient should not take multivitamins or dietary supplements containing biotin, except in special circumstances.','CPR','C-Peptide, Serum')}
    assert not synthesize('What fasting and biotin instructions apply?',selection(sources),sources)['validated']


def test_selection_conflict_flag_prevents_conversational_claims():
    sources={'S1':document('Specimen Minimum Volume: 0.25 mL')}
    selected=selection(sources);selected.conflicting_evidence=['Conflict noted by selector']
    assert not synthesize('What is the minimum volume?',selected,sources)['validated']


def test_c_peptide_processing_and_storage_are_both_required_when_asked():
    sources={'S1':document('Collection Instructions: Centrifuge and aliquot serum into a plastic vial within 2 hours of collection.\n\nSpecimen Type  Temperature  Time\nSerum  Frozen (preferred)  30 days\n       Refrigerated  7 days','CPR','C-Peptide, Serum')}
    result=synthesize('How should C-Peptide be processed and stored?',selection(sources),sources)
    assert result['validated'] and 'within 2 hours of collection' in result['answer'] and '30 days' in result['answer']
    incomplete={'S1':document('Specimen Type  Temperature  Time\nSerum  Frozen  30 days','CPR','C-Peptide, Serum')}
    assert not synthesize('How should C-Peptide be processed and stored?',selection(incomplete),incomplete)['validated']


def test_unit_specimen_temperature_and_duration_mutations_are_rejected():
    sources={'S1':document('Specimen Minimum Volume: 0.25 mL\n\nSpecimen Type  Temperature  Time\nSerum  Refrigerated (2-8 °C)  7 days')}
    selected=selection(sources);facts=extract_facts(selected,sources)
    volume=next(i for i in facts if i.kind=='minimum_volume')
    assert volume.data['quantities']==[{'number':'0.25','unit':'mL'}]
    altered=replace(volume,data={**volume.data,'value':'0.25 L','quantities':[{'number':'0.25','unit':'L'}]})
    assert not validate_synthesis(render_claims([altered]),[altered],selected,sources)
    storage=next(i for i in facts if i.kind=='storage')
    for key,value in [('specimen','Plasma'),('temperature','Frozen (-20 °C)'),('duration','365'),('unit','hours')]:
        changed=replace(storage,data={**storage.data,key:value})
        assert not validate_synthesis(render_claims([changed]),[changed],selected,sources)


@pytest.mark.parametrize('provider',['groq','openai'])
def test_graph_conversational_mode_and_provider_compatibility(provider):
    from config.settings import Settings
    from src.llm.provider_factory import create_llm
    from src.rag.rag_graph import build_rag_graph
    from src.retrieval.semantic_retriever import SearchResult
    doc=document('Specimen Type: Serum\nSpecimen Volume: 0.5 mL')
    retriever=Mock();retriever.search.return_value=SearchResult('q','fixed','fixed-chunks',[{'document':doc,'score':.8}],0)
    backend=Mock();backend.invoke.return_value={'selected_excerpt_ids':['S1_E1']}
    settings=Settings(llm_provider=provider,llm_model='gpt-4.1-mini' if provider=='openai' else 'openai/gpt-oss-120b')
    graph=build_rag_graph(retriever,create_llm(settings,backend),settings,raw_retrieval=True,answer_generation_mode=CONVERSATIONAL)
    result=graph.invoke({'question':'What specimen is required for BUN?'})
    assert result['status']=='answered' and result['synthesis_diagnostics']['validated']
    assert '[S1]' in result['answer'] and '[S1]' not in result['display_answer']
    assert result['extractive_answer']==doc.page_content+' [S1]'
    assert backend.invoke.call_count==1
    assert result['citations'][0]['source_id']=='S1'
    assert [n['node'] for n in result['trace']][-2:]==['validate_citations','synthesize_answer']


def test_missing_evidence_never_invokes_generation_or_synthesis():
    from config.settings import Settings
    from src.rag.rag_graph import build_rag_graph
    from src.retrieval.semantic_retriever import SearchResult
    retriever=Mock();retriever.search.return_value=SearchResult('q','fixed','fixed-chunks',[],0)
    service=Mock()
    result=build_rag_graph(retriever,service,Settings(),raw_retrieval=True,answer_generation_mode=CONVERSATIONAL).invoke({'question':'What specimen is required for BUN?'})
    assert result['status']=='abstained' and not result.get('synthesis_diagnostics')
    service.select.assert_not_called()


def test_generation_mode_is_signed_and_historical_reports_unchanged(tmp_path):
    from tests.test_evaluation_runner import runner
    from src.evaluation.report_generator import csv_text
    engine=runner(tmp_path)
    run=engine.run(mode='rag',configurations=['A'])
    assert run['parameters']['answer_generation_mode']=='extractive_v1'
    path=tmp_path/'results'/(run['run_id']+'.json');before=path.read_bytes()
    with pytest.raises(ValueError,match='changed'):
        engine.run(mode='rag',configurations=['A'],resume=run['run_id'],answer_generation_mode=CONVERSATIONAL)
    assert path.read_bytes()==before
    fresh=engine.run(mode='rag',configurations=['A'],answer_generation_mode=CONVERSATIONAL)
    assert fresh['answer_generation_mode']==CONVERSATIONAL and 'answer_generation_mode' in csv_text(fresh)
    assert fresh['records'][0]['status']=='Completed'
    historical=Path('evaluation/results/20261106T075303_624a0c1d52.json')
    if historical.exists():
        old=historical.read_bytes();csv_text(json.loads(old));assert historical.read_bytes()==old


def test_chat_displays_friendly_text_and_collapsed_audit_without_changing_composer(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.ui_day3 as services
    import src.ui_presentation as presentation
    import src.rag.rag_graph as graph_module
    sources={'S1':document('Specimen Type: Serum\nSpecimen Volume: 0.5 mL')}
    diagnostic=synthesize('What specimen is required for BUN?',selection(sources),sources)
    assert diagnostic['validated']
    clean=diagnostic['answer'].replace(' [S1]','')
    graph=Mock();graph.invoke.side_effect=lambda p:{'question':p['question'],'answer':diagnostic['answer'],
        'display_answer':clean,'extractive_answer':sources['S1'].page_content+' [S1]',
        'synthesis_diagnostics':diagnostic,'status':'answered','citations':[{'source_id':'S1','test_name':'Blood Urea Nitrogen (BUN), Serum',
        'test_code':'BUN','printed_catalog_page':'1','source_url':'https://example.org/catalog'}]}
    monkeypatch.setattr(services,'services',Mock(return_value=(Mock(),Mock())))
    monkeypatch.setattr(graph_module,'build_rag_graph',Mock(return_value=graph))
    monkeypatch.setattr(presentation,'sample_questions',lambda root:['What specimen is required for BUN?'])
    app=AppTest.from_string('from src.ui_day4 import render_chat\nfrom config.settings import Settings\nrender_chat(Settings())').run(timeout=30)
    next(w for w in app.text_input if w.label=='Laboratory question').set_value('What specimen is required for BUN?')
    next(b for b in app.button if b.label=='Send').click().run(timeout=30)
    assert not app.exception
    assert app.chat_message[1].markdown[0].value==clean and '[S1]' not in clean
    assert any(b.label=='Open reference 1' for b in app.get('link_button'))
    assert any(e.proto.label=='View supporting evidence' and not e.proto.expanded for e in app.get('expander'))
    next(w for w in app.selectbox if w.label=='Sample Questions').select('What specimen is required for BUN?').run(timeout=30)
    assert graph.invoke.call_count==1
    next(b for b in app.button if b.label=='Send').click().run(timeout=30)
    assert len(app.session_state['rag_history'])==2 and graph.invoke.call_count==2
