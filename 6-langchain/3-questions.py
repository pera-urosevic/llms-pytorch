from langchain_chroma import Chroma
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFacePipeline,
)
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import (
    device,
    chroma_path,
    embedding_name,
    model_name,
    questions,
)
from display import display_result


embedding = HuggingFaceEmbeddings(
    model_name=embedding_name,
    model_kwargs={"device": device},
)

db = Chroma(persist_directory=chroma_path, embedding_function=embedding)

tokenizer = AutoTokenizer.from_pretrained(model_name, device=device)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

llm = HuggingFacePipeline(
    pipeline=pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        device=device,
    ),
)

chat = ChatHuggingFace(
    llm=llm,
    device=device,
    temperature=0,
)

system_template = """Your entire response must be derived *only* from the information found in the CONTEXT section.
Do not use any external knowledge, make assumptions, or provide information not explicitly stated in the CONTEXT.
If the information required to answer the question is not present in the CONTEXT, you *must* respond with the exact phrase: "I don't know."

CONTEXT:
{context}
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

qa_chain = RetrievalQA.from_chain_type(
    chat,
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": chat_prompt},
)

for q in questions:
    result = qa_chain.invoke({"query": q})
    display_result(result["result"])
