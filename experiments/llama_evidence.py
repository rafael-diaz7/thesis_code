import torch
import pandas as pd
from langchain import HuggingFacePipeline, PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

print("device:", torch.cuda.get_device_name())

MODEL_NAME = "TheBloke/Llama-2-13b-Chat-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="cpu"
)

print("downloading model...")

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=generation_config,
)

llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

template = """
<s>[INST] <<SYS>>
    Act as a Clinical Question Answering agent. You will be provided a question and then a context. You must answer the question using text ONLY inside the context. You must not provide any additional information or generate any novel text, only use text from the context. Do not say anything else BUT the answer to the question.
    <</SYS>>

    Context:
    {context}
    Question:
    {question}? [/INST]
    """

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=template,
)

df = pd.read_csv("../data/emrqa_evidence_test.csv")
predictions = []

print("generating predictions...")

for ind, (q, e) in enumerate(df[["question", "evidence"]].values):
    if ind % 100 == 0:
        print(f"processed {ind} questions")
    predictions.append(llm(prompt.format(question=q, context=e)).split("\n")[-1].strip())

predictions_df = pd.DataFrame(
    {
        "true": df["answer"].to_list(),
        "predicted": predictions
    }
)

predictions_df.to_csv("llama_evidence_predictions.csv", index=False)