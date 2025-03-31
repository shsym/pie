#!/bin/bash

API_KEY="YOUR_API_KEY"
URL="http://localhost:8080/v1/completions"

curl -X POST "$URL" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "knock knock."},
      {"role": "assistant", "content": "Who'\''s there?"},
      {"role": "user", "content": "Orange."}
    ]
  }'
