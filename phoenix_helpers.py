from phoenix.otel import register
from openinference.instrumentation.groq import GroqInstrumentor
import phoenix as px
import pandas as pd
from dotenv import load_dotenv
from phoenix.trace import SpanEvaluations
from phoenix.evals import HallucinationEvaluator, LiteLLMModel, QAEvaluator, run_evals

load_dotenv()
import os

from groq import Groq

client = Groq(
    api_key=os.environ.get('GROQ_API_KEY'),
)


tracer_provider = register(
  project_name="default",
  endpoint="https://app.phoenix.arize.com/v1/traces",
) 

GroqInstrumentor().instrument(tracer_provider=tracer_provider)

def get_spans_df():
    span_df = px.Client().get_spans_dataframe(project_name="default")
    span_df["attributes.llm.output_messages"] = pd.json_normalize(span_df["attributes.llm.output_messages"])[0].to_list()
    span_df["attributes.llm.input_messages"] = pd.json_normalize(span_df["attributes.llm.input_messages"])[0].to_list()
    return span_df

LLM_EVALUATOR_TEMPLATE = """
You are an evaluator. Your job is to decide if the provided answer is a valid response to the question.

**Your instructions:**
1. Carefully analyze the question to understand its intent.
2. Examine the answer to determine whether it satisfies the intent of the question.
3. Provide your reasoning step-by-step to justify your decision.
4. Output your reasoning in the format strictly provided below:
   - Start with "EXPLANATION:" followed by your reasoning in one or two sentences.
   - End with "LABEL:" followed by either "VALID" or "INVALID" (in uppercase, without quotes or punctuation).

**Important Guidelines:**
- Do not change the output format.
- Do not provide extra information, summaries, or comments outside the specified format.
- The output must consist only of the explanation and label in the specified format.

### Input:
Question:
{question}

Answer:
{answer}

### Expected Output Format:
EXPLANATION: [Your reasoning here.]
LABEL: [VALID or INVALID]

### Example Responses:
**Example 1**:
EXPLANATION: The answer is valid because the question asks for a definition of AI, and the answer provides a clear definition of AI.
LABEL: VALID

**Example 2**:
EXPLANATION: The answer is invalid because the question asks for an explanation about gravity, but the answer discusses photosynthesis instead.
LABEL: INVALID

### Task:
Evaluate the input using the above instructions and respond strictly in the required format.
"""

def evaluate_row(row, model, LLM_EVALUATOR_TEMPLATE):
  question = row['attributes.input.value']
  answer = row['attributes.output.value']
  chat_completion = client.chat.completions.create(
                    messages=[{
                              "role": "user",
                              "content": LLM_EVALUATOR_TEMPLATE.format(question=question, answer=answer),
                              }],
                    model=model,
                  )
  explanation, label = chat_completion.choices[0].message.content.split("LABEL")
  if "INVALID" in label:
    label = "INVALID"
  else:
    label = "VALID"
  return explanation, label


def evaluate_model(df, model, LLM_EVALUATOR_TEMPLATE=LLM_EVALUATOR_TEMPLATE):
  df['explanation'], df['label'] = zip(*df.apply(lambda row: evaluate_row(row, model, LLM_EVALUATOR_TEMPLATE), axis=1))
  df['score' ] = df['label'].apply(lambda x: 1 if x == 'VALID' else 0)
  px.Client().log_evaluations(SpanEvaluations(eval_name="Response Format", dataframe=df))
  return df

def get_dataset(name):
  dataset = px.Client().get_dataset(name=name)
  return dataset

def dataEvalResults(model, df):
  eval_model = LiteLLMModel(model=f"groq/{model}")

  hallucination_evaluator = HallucinationEvaluator(eval_model)
  qa_evaluator = QAEvaluator(eval_model)

  df["reference"] = df["metadata"]
  assert all(column in df.columns for column in ["output", "input", "reference"])

  hallucination_eval_df, qa_eval_df = run_evals(
      dataframe=df, evaluators=[hallucination_evaluator, qa_evaluator], provide_explanation=True
  )

  results_df = df.copy()
  results_df["hallucination_eval"] = hallucination_eval_df["label"]
  results_df["hallucination_explanation"] = hallucination_eval_df["explanation"]
  results_df["qa_eval"] = qa_eval_df["label"]
  results_df["qa_explanation"] = qa_eval_df["explanation"]
  
  return results_df
