# Get Journal DOIs Script

A script to fetch all DOI numbers for articles in a journal using the Crossref API via habanero.

## Installation

First, install the habanero library:

```bash
pip install habanero
```

Or if using conda:

```bash
conda install -c conda-forge habanero
```

## Usage

### Basic Usage

Fetch DOIs for a journal by ISSN:

```bash
./get_journal_dois.py 2309-608X
```

This will print all DOIs to stdout.

### Save to File

Save DOIs to a file (one per line):

```bash
./get_journal_dois.py 2309-608X --output jof_dois.txt
```

### Use Email for Faster Access

Crossref provides faster API access if you include an email (joins the "polite pool"):

```bash
./get_journal_dois.py 2309-608X --email your@email.com --output dois.txt
```

### Quiet Mode

Suppress progress messages (only output DOIs):

```bash
./get_journal_dois.py 2309-608X --quiet > dois.txt
```

### Adjust Rate Limiting

Change the delay between API requests:

```bash
./get_journal_dois.py 2309-608X --rate-limit 0.5  # 0.5 seconds between requests
```

## Examples

### Journal of Fungi (MDPI)
```bash
./get_journal_dois.py 2309-608X --output jof_dois.txt --email scholar@example.com
```

### Mycotaxon
```bash
./get_journal_dois.py 0093-4666 --output mycotaxon_dois.txt
```

### Persoonia (with hyphen or without)
```bash
./get_journal_dois.py 0031-5850 --output persoonia_dois.txt
# or
./get_journal_dois.py 00315850 --output persoonia_dois.txt
```

## Features

- **Cursor-based pagination**: Handles journals with thousands of articles
- **Rate limiting**: Built-in delays to avoid overwhelming the API
- **Polite pool**: Use `--email` for faster API access
- **Progress tracking**: Shows progress for large journals
- **Error handling**: Graceful handling of API errors
- **Flexible ISSN format**: Accepts ISSN with or without hyphen

## Output Format

The script outputs one DOI per line:

```
10.3390/jof12010028
10.3390/jof12010027
10.3390/jof12010026
...
```

## How It Works

1. **Query Crossref**: Uses habanero to query the Crossref API with an ISSN filter
2. **Cursor pagination**: Iterates through all results using cursor-based pagination (1000 results per request)
3. **Extract DOIs**: Pulls the DOI field from each article record
4. **Rate limiting**: Waits between requests to be polite to the API

## API Limits

Crossref's public API has the following characteristics:

- **No authentication required**: Free to use
- **Rate limiting**: Recommended to wait between requests
- **Polite pool**: Faster access if you provide an email via `--email`
- **Cursor pagination**: Efficient for large result sets

## Troubleshooting

### No articles found

If you get "No articles found for ISSN X", check:
- The ISSN is correct (try both print and electronic ISSN)
- The journal is indexed in Crossref
- Try the ISSN with and without hyphen

### Slow performance

- Use `--email` to join the polite pool for faster access
- Increase `--rate-limit` if you're getting rate limited
- Large journals (10,000+ articles) will take several minutes

### habanero not found

Install the library:
```bash
pip install habanero
```

## Integration with Other Scripts

### Use with MDPI ingestor

```bash
# Get all DOIs for Journal of Fungi
./get_journal_dois.py 2309-608X --output jof_dois.txt

# Process each DOI (example - not implemented yet)
cat jof_dois.txt | while read doi; do
    echo "Processing $doi"
    # Add your processing logic here
done
```

### Use with cleanup script

```bash
# Get DOIs
./get_journal_dois.py 2309-608X > dois.txt

# Check which are already in database (example query)
# This would need to be implemented in your ingestion workflow
```

## Technical Details

### Crossref API

- **Endpoint**: https://api.crossref.org/works
- **Filter**: `issn={ISSN}`
- **Pagination**: Cursor-based (more efficient than offset-based)
- **Rate limiting**: ~50 requests/second for anonymous, higher for polite pool

### habanero Library

- Python wrapper for Crossref API
- Documentation: https://habanero.readthedocs.io/
- GitHub: https://github.com/sckott/habanero

## Command-Line Options

```
usage: get_journal_dois.py [-h] [--output OUTPUT] [--email EMAIL] [--quiet]
                           [--rate-limit RATE_LIMIT]
                           issn

positional arguments:
  issn                  ISSN or eISSN of the journal

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output file to save DOIs (one per line)
  --email EMAIL         Email for polite API usage (recommended)
  --quiet, -q           Suppress progress messages
  --rate-limit RATE_LIMIT
                        Delay between API requests in seconds (default: 0.1)
```

## License

This script is part of the skol ingestion system.

## See Also

- [MDPI Ingestor](mdpi.py) - Ingest MDPI journal articles
- [Ingenta Ingestor](ingenta.py) - Ingest Ingenta journals
- [Main Ingestion Script](main.py) - Main ingestion entry point
