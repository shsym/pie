#!/bin/bash

API_KEY="YOUR_API_KEY"
URL="http://localhost:8080/"

curl -X POST "$URL" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $API_KEY" \
-d '{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What'"'"'s the weather like today?"},
    {"role": "assistant", "content": "I don'"'"'t have access to real-time weather data. Could you please tell me your location so I can better assist you?"},
    {"role": "user", "content": "I'"'"'m in New York City."}
  ],
  "temperature": 0.7
}'
