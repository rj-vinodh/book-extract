# Book Extract Only

This project extracts frames from video files, lets you select book covers, and sends the selected images to a local OpenAI-compatible model to extract metadata such as title, author, publisher, ISBN, and notes.

## Requirements

- Docker
- A local OpenAI-compatible model running on port 9931
- macOS host networking support for Docker if the model runs on your machine

## Build

```bash
docker build -t book-capture:latest .
```

## Run

```bash
docker run --rm -p 7860:7860 book-capture:latest
```

Then open:

```text
http://localhost:7860
```

## Local model connectivity check

If the model is running on your host machine, Docker can usually reach it via `host.docker.internal`.

Example curl check:

```bash
curl -sS http://host.docker.internal:9931/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ggml-org/gemma-4-12B-it-GGUF:Q8_0",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Notes

- The app uses `host.docker.internal` automatically inside Docker when it detects a container environment.
- If you run the app outside Docker, it falls back to `localhost:9931`.
- The app also supports ZIP upload of extracted book images for metadata extraction.
