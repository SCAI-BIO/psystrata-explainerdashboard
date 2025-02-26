# How to run the container

```bash
docker build --no-cache -t ollama-llama3.2 .
docker run -d --rm -p 11434:11434 --name ollama-container ollama-llama3.2
```