import streamlit as st
from dotenv import load_dotenv

import phoenix_helpers
import helpers 

load_dotenv()

if "eval_btn_clicked" not in st.session_state:
        st.session_state.eval_btn_clicked = False

def callback2():
    st.session_state.eval_btn_clicked = True

models = helpers.fetch_models()
if models:
    st.session_state.spans_df = phoenix_helpers.get_spans_df()
    st.subheader("Evaluate LLM")

    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None
    if (st.button("Evaluate", on_click=callback2) or st.session_state.eval_btn_clicked):
        if "eval_model" not in st.session_state:
            st.session_state.eval_model = models[0]
        st.selectbox(
            "Choose a model to use for evaluation:", 
            models, key = 'eval_model',
            index=models.index(st.session_state.eval_model) if st.session_state.eval_model in models else 0,
        )
        if st.session_state.eval_model:
            st.session_state.evaluation_result = phoenix_helpers.evaluate_model(st.session_state.spans_df, st.session_state.eval_model)
            st.write(st.session_state.evaluation_result)

                
