# Rate Limiting and HTTP 429 Handling

The base Ingestor class includes comprehensive rate limiting and HTTP 429 (Too Many Requests) handling.

## Features

### Automatic Rate Limiting

All ingestors automatically respect rate limits through:

1. **robots.txt Crawl-Delay**: Respects `Crawl-Delay` directive if present
2. **Configurable delays**: Random delay between `rate_limit_min_ms` and `rate_limit_max_ms`
3. **Per-request timing**: Tracks last fetch time to enforce minimum delays

### HTTP 429 Response Handling

When a server returns HTTP 429, the ingestor automatically:

1. **Parses retry headers** (in order of precedence):
   - `Retry-After` (RFC 7231) - Standard HTTP header
   - `RateLimit-Reset` (IETF draft) - Modern rate limit header
   - `X-RateLimit-Reset` (GitHub, Twitter, etc.) - Common variant

2. **Waits the specified duration**
3. **Retries the request once**
4. **Logs the wait time** (verbosity ≥ 1)

## Supported Headers

### Retry-After (RFC 7231)

Standard HTTP header that can contain:
- **Integer**: Delay in seconds (e.g., `Retry-After: 120`)
- **HTTP-date**: Absolute time (e.g., `Retry-After: Wed, 21 Oct 2026 07:28:00 GMT`)

### RateLimit Headers (IETF Draft)

Modern structured rate limit headers:
- `RateLimit-Limit`: Maximum requests allowed in time window
- `RateLimit-Remaining`: Requests remaining in current window
- `RateLimit-Reset`: Time until rate limit resets

The `RateLimit-Reset` header can be:
- **Unix timestamp**: Seconds since epoch (> 1000000000)
- **Relative seconds**: Seconds until reset (< 1000000000)

### X-RateLimit Headers

Older convention used by many APIs:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Reset time (Unix timestamp or relative)

## Behavior

### On HTTP 429 Response

```
1. Receive 429 response
2. Check Retry-After header
   ├─ If present: Parse and use
   └─ If absent: Check RateLimit-Reset / X-RateLimit-Reset
      ├─ If present: Parse and use
      └─ If absent: Default to 60 seconds
3. Log wait time (if verbosity ≥ 1)
4. Sleep for calculated duration
5. Retry request once
6. Return response (may still be 429 if limit persists)
```

### Default Wait Times

- **With headers**: Uses server-specified delay
- **Without headers**: Defaults to 60 seconds
- **Parse failure**: Falls back to 60 seconds with warning

## Debug Information

At **verbosity level 4**, the ingestor logs all rate limit headers from every response:

```
Rate limit headers: {
    'RateLimit-Limit': '100',
    'RateLimit-Remaining': '42',
    'RateLimit-Reset': '1704067200'
}
```

This helps debug rate limiting issues without triggering 429 responses.

## Example Logs

### With Retry-After Header

```
HTTP 429 (Too Many Requests) - waiting 120s before retry (from Retry-After)
```

### With RateLimit-Reset Header

```
HTTP 429 (Too Many Requests) - waiting 45s before retry (from RateLimit-Reset)
```

### Without Headers

```
Warning: No rate limit headers found, using 60s default
HTTP 429 (Too Many Requests) - waiting 60s before retry (from default)
```

## Configuration

Configure rate limiting per publication in `publications.py`:

```python
'my-journal': {
    'name': 'My Journal',
    'source': 'my-source',
    'ingestor_class': 'MyIngestor',
    'rate_limit_min_ms': 1000,  # Minimum delay: 1 second
    'rate_limit_max_ms': 3000,  # Maximum delay: 3 seconds
},
```

## Implementation

The rate limiting is implemented in the base `Ingestor` class method:
- `_get_with_rate_limit()` - Handles all HTTP requests with rate limiting
- `_apply_rate_limit()` - Enforces minimum delays between requests

All ingestors inherit this behavior automatically.

## Standards References

- **RFC 7231**: Retry-After header
  - https://httpwg.org/specs/rfc7231.html#header.retry-after

- **IETF Draft**: RateLimit Header Fields for HTTP
  - https://www.ietf.org/archive/id/draft-polli-ratelimit-headers-02.html

- **Common Practice**: X-RateLimit-* headers
  - Used by GitHub, Twitter, Stripe, and many others
