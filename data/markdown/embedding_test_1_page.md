## **Azure OpenAI Embedding — 1-Page Test** 

Goal: verify that your **text-embedding-3-small** deployment works before processing the full Apple 10-K. This test sends only the short text **“Apple”** . 

## **1. Minimal test code** 

```
from dotenv import load_dotenv
import os
```

```
from langchain_openai import AzureOpenAIEmbeddings
```

```
load_dotenv()
```

```
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv(
"AZURE_OPENAI_API_EMBEDDING_VERSION", "2023-05-15"
```

```
)
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
```

```
if not endpoint or not api_key or not embedding_model:
raise RuntimeError("Missing Azure OpenAI configuration in .env")
```

```
embeddings = AzureOpenAIEmbeddings(
model=embedding_model,
azure_endpoint=endpoint,
api_key=api_key,
api_version=api_version
)
```

```
print("Testing Azure OpenAI embedding deployment...")
```

```
test_embedding = embeddings.embed_query("Apple")
```

```
print("\nEmbedding successful!")
print("Model:", embedding_model)
print("Vector dimensions:", len(test_embedding))
```

## **2. Required .env values** 

```
AZURE_OPENAI_ENDPOINT="https://investiq-uae.services.ai.azure.com"
AZURE_OPENAI_API_KEY="YOUR_API_KEY"
AZURE_OPENAI_API_EMBEDDING_VERSION="2023-05-15"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
```

## **3. Run it** 

Save the Python file, make sure your **.env** is in the project root, activate your virtual environment, then run the file from the terminal: 

```
python your_file_name.py
```

## **4. Expected result** 

You should see something similar to: 

```
Testing Azure OpenAI embedding deployment...
Embedding successful!
Model: text-embedding-3-small
Vector dimensions: 1536
```

## **5. Important** 

Do **not** run the SemanticChunker / full 10-K processing yet. First confirm this small embedding test succeeds. If it works, the next step is to enable the document chunking code and process the Apple 10-K. 

