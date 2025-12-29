# Session ID Cleanup Script

## Problem

IngentaConnect URLs contain session IDs like `;jsessionid=e7caa4aba81qk.x-ic-live-01` which are random and change between requests. Documents ingested before the session ID cleaning fix have these session IDs in their URLs, leading to:

- Non-deterministic document IDs (based on UUIDs of URLs with random session IDs)
- Duplicate records for the same article
- Incorrect PDF filenames

## Solution

The `cleanup_session_ids.py` script fixes existing records by:

1. **Scanning** the database for documents with session IDs in URLs
2. **Creating** new documents with cleaned URLs (session IDs removed)
3. **Generating** new `_id` fields based on stable URLs (UUID5 of cleaned URL)
4. **Copying** all attachments from old documents to new documents
5. **Optionally deleting** old documents with session IDs

## Usage

### 1. Preview Changes (Dry Run)

Always start with a dry run to see what will be changed:

```bash
cd /data/piggy/src/github.com/piggyatbaqaqi/skol/ingestors
./cleanup_session_ids.py --dry-run
```

This will show:
- How many documents need cleaning
- Examples of URL changes
- New document IDs that will be created

### 2. Execute Cleanup (Keep Old Documents)

Create cleaned copies while keeping the originals:

```bash
./cleanup_session_ids.py
```

This is the safest option - you can manually verify the new documents before deleting old ones.

### 3. Execute Cleanup (Delete Old Documents)

Create cleaned copies and delete the originals:

```bash
./cleanup_session_ids.py --delete-old
```

**Warning**: This permanently deletes old documents after copying. Make sure you have a backup!

### 4. Custom Database

Use with a different database:

```bash
./cleanup_session_ids.py --database skol_prod --dry-run
```

## What Gets Cleaned

The script cleans these URL fields:
- `url` - Article URL
- `pdf_url` - PDF download URL
- `human_url` - Human-readable URL
- `bibtex_link` - BibTeX source URL

### Example Transformation

**Before:**
```json
{
  "_id": "abc123-old-random-id",
  "url": "https://www.ingentaconnect.com/contentone/.../art00011;jsessionid=e7caa4aba81qk.x-ic-live-01",
  "pdf_url": "https://www.ingentaconnect.com/contentone/.../art00011;jsessionid=e7caa4aba81qk.x-ic-live-01?crawler=true",
  "title": "My Article"
}
```

**After:**
```json
{
  "_id": "b070a4c0-7da7-5ab2-998a-5bb7190aec8d",
  "url": "https://www.ingentaconnect.com/contentone/.../art00011",
  "pdf_url": "https://www.ingentaconnect.com/contentone/.../art00011?crawler=true",
  "title": "My Article"
}
```

Note: The new `_id` is deterministic - always the same for the same cleaned URL.

## Safety Features

1. **Dry run mode**: Preview all changes before executing
2. **Confirmation prompt**: Asks before making any changes
3. **Duplicate detection**: Skips if cleaned document already exists
4. **Create before delete**: New documents are fully created before old ones are deleted
5. **Error handling**: Continues processing if individual documents fail
6. **Attachment preservation**: All attachments are copied to new documents

## Output Example

```
Scanning database for documents with session IDs...
Scanned 1523 documents, found 47 needing cleanup

================================================================================
PREVIEW: 47 document(s) would be cleaned
================================================================================

1. Document: abc123-old-random-id
   Title: Phylogenetic relationships of the genus Persoonia
   New ID: b070a4c0-7da7-5ab2-998a-5bb7190aec8d
   URL changes:
     url:
       Old: https://.../art00011;jsessionid=e7caa4aba81qk.x-ic-live-01
       New: https://.../art00011
     pdf_url:
       Old: https://.../art00011;jsessionid=e7caa4aba81qk.x-ic-live-01?crawler=true
       New: https://.../art00011?crawler=true
   Attachments: 1 to copy

...

This will create 47 cleaned document(s)
(old documents will be kept)

Proceed? [y/N]:
```

## Troubleshooting

### "Database not found"
Make sure the database name is correct:
```bash
./cleanup_session_ids.py --database skol_dev
```

### "Unauthorized access"
Set CouchDB credentials:
```bash
export COUCHDB_USER=admin
export COUCHDB_PASSWORD=password
./cleanup_session_ids.py
```

### "Cleaned version already exists"
This is normal - it means the document was already cleaned in a previous run. The script will skip it.

### Error copying attachments
If attachments fail to copy, the document will still be created. Check the error message and manually investigate the attachment.

## After Running

### Verify Results

1. Check document counts:
   ```bash
   # Should see new documents created
   curl http://localhost:5984/skol_dev/_all_docs?limit=10
   ```

2. Verify a cleaned document:
   ```bash
   # Use a new ID from the output
   curl http://localhost:5984/skol_dev/b070a4c0-7da7-5ab2-998a-5bb7190aec8d
   ```

3. Check attachments were copied:
   ```bash
   curl http://localhost:5984/skol_dev/b070a4c0-7da7-5ab2-998a-5bb7190aec8d/article.pdf > test.pdf
   ```

### Clean Up Old Documents

If you ran without `--delete-old` and verified everything works:

1. Make a backup first!
2. Run again with `--delete-old`:
   ```bash
   ./cleanup_session_ids.py --delete-old
   ```

## Prevention

The session ID cleaning is now built into the IngentaConnect ingestor (as of this update), so future ingestions will automatically create documents with cleaned URLs.

**Files modified to prevent future session IDs:**
- `ingestors/ingenta.py`: Added session ID cleaning to `format_pdf_url()` and `_extract_articles_from_issue()`
