from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()  # Load from .env file

local_api_key = os.getenv("OPENAI_API_KEY")
if not local_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment.")


client = OpenAI(api_key=local_api_key)

# Replace with your own secure method for managing API keys
#client = OpenAI(api_key="your-api-key-here")

query = "How does the IRS handle audits?"
documents = [
    "The IRS audits are a review of tax returns.",
    "Refunds are processed by the IRS.",
    "The tax code changes every year."
]

# Get embedding for the query
query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)
query_embedding = query_response.data[0].embedding

# Get embeddings for the documents
doc_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=documents
)
doc_embeddings = [d.embedding for d in doc_response.data]

# Calculate cosine similarities
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# Display ranked results
ranked_docs = sorted(zip(documents, similarities), key=lambda x: -x[1])
for doc, score in ranked_docs:
    print(f"Score: {score:.4f} | Doc: {doc}")

# Print most relevant document
print("Most relevant:", ranked_docs[0])

#
#Score: 0.7258 | Doc: The IRS audits are a review of tax returns.
#Score: 0.5551 | Doc: Refunds are processed by the IRS.
#Score: 0.3712 | Doc: The tax code changes every year.
#Most relevant: ('The IRS audits are a review of tax returns.', np.float64(0.7258271303798065))