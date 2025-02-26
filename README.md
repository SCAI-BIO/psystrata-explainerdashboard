# PsychSTRATA - Explainable ML Dashboard

Run the application by running the following command in the terminal:
```bash
python -m app.main
```

# LLM setup
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama run llama3
```