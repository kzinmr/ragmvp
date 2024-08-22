# `fileloader`

The code is taken and adapted from llama-index and langchain, reducing dependencies for them.

- [Simple Directory Reader](https://docs.llamaindex.ai/en/stable/api_reference/readers/simple_directory_reader/)
- [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)


## Convert files into JSONL dataset.

```bash
docker build -t fileloader -f Dockerfile .
docker run -v /path/to/data:/app/data -v /path/to/output:/app/output fileloader
```
