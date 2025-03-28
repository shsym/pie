## How to run benchamrk

### Install dependencies first if needed
```
pip install -r requirements.txt
```

### Basic usage with defaults (10 concurrent requests for 10 seconds)
```
python benchmark_requests.py
```

### With custom parameters
```
python benchmark_requests.py --concurrent 50 --duration 30 --api-key "YOUR_ACTUAL_API_KEY"
```

If you want to send empty GET requests, use `-e` flag:
```
python benchmark_requests.py --concurrent 50 --duration 30 --api-key "YOUR_ACTUAL_API_KEY" -e
```