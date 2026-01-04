# CouchDB MCP Server Setup

This document explains how to use the CouchDB MCP (Model Context Protocol) server with Claude Code.

## Overview

The CouchDB MCP server provides Claude Code with direct access to your CouchDB databases, enabling:
- Querying documents
- Searching content
- Retrieving attachments
- Executing views
- Listing databases and documents

## Files Created

1. **[couchdb_mcp_server.py](couchdb_mcp_server.py)** - The MCP server implementation
2. **[.mcp.json](.mcp.json)** - MCP server configuration
3. **[.env](.env)** - Environment variables (contains credentials - DO NOT commit!)

## Configuration

### Environment Variables

The `.env` file contains your CouchDB credentials:

```bash
COUCHDB_URL=http://localhost:5984
COUCHDB_USER=admin
COUCHDB_PASSWORD=SU2orange!
COUCHDB_DATABASES=skol_dev,skol_taxa_dev
```

**IMPORTANT:** Add `.env` to your `.gitignore` to avoid committing credentials!

### MCP Configuration

The `.mcp.json` file configures the MCP server for Claude Code:

```json
{
  "mcpServers": {
    "couchdb": {
      "type": "stdio",
      "command": "/usr/bin/python3",
      "args": ["/data/piggy/src/github.com/piggyatbaqaqi/skol/couchdb_mcp_server.py"],
      "env": {
        "COUCHDB_URL": "${COUCHDB_URL:-http://localhost:5984}",
        "COUCHDB_USER": "${COUCHDB_USER}",
        "COUCHDB_PASSWORD": "${COUCHDB_PASSWORD}",
        "COUCHDB_DATABASES": "${COUCHDB_DATABASES:-skol_dev,skol_taxa_dev}"
      }
    }
  }
}
```

## Usage

### Starting the MCP Server

The MCP server is automatically loaded when you start Claude Code in this directory.

To verify it's running, use the `/mcp` command in Claude Code:

```
/mcp
```

This will show all configured MCP servers and their status.

### Available Tools

The CouchDB MCP server provides the following tools:

#### 1. `list_databases`
List all available CouchDB databases.

**Example:**
```
List all CouchDB databases
```

#### 2. `get_document`
Get a specific document by ID.

**Parameters:**
- `database` - Database name (e.g., "skol_dev")
- `doc_id` - Document ID

**Example:**
```
Get document "doc123" from skol_dev database
```

#### 3. `query_documents`
Query documents with pagination.

**Parameters:**
- `database` - Database name
- `limit` - Maximum documents to return (default: 100)
- `skip` - Number of documents to skip (default: 0)
- `include_docs` - Include full document content (default: true)

**Example:**
```
Show me the first 10 documents from skol_taxa_dev
```

#### 4. `search_documents`
Search for documents containing specific text.

**Parameters:**
- `database` - Database name
- `search_text` - Text to search for
- `field` - (Optional) Specific field to search in
- `limit` - Maximum results (default: 50)

**Example:**
```
Search for "Amanita" in skol_dev database
```

#### 5. `get_attachment`
Get an attachment from a document.

**Parameters:**
- `database` - Database name
- `doc_id` - Document ID
- `attachment_name` - Attachment filename

**Example:**
```
Get the "species.txt" attachment from document "doc123" in skol_dev
```

#### 6. `list_attachments`
List all attachments for a document.

**Parameters:**
- `database` - Database name
- `doc_id` - Document ID

**Example:**
```
List all attachments for document "doc123" in skol_dev
```

#### 7. `execute_view`
Execute a CouchDB view.

**Parameters:**
- `database` - Database name
- `design_doc` - Design document name (without "_design/" prefix)
- `view_name` - View name
- `key` - (Optional) Filter by specific key
- `limit` - Maximum results (default: 100)

**Example:**
```
Execute the "by_species" view in the "taxonomy" design doc
```

#### 8. `count_documents`
Count documents in a database.

**Parameters:**
- `database` - Database name
- `doc_type` - (Optional) Filter by document type field

**Example:**
```
How many documents are in skol_taxa_dev?
```

## Example Interactions

### Query Your Data

```
Show me the first 5 documents from skol_dev
```

```
Search for documents containing "mycology" in skol_taxa_dev
```

```
Get document "species_001" from skol_dev
```

### Work with Attachments

```
List all attachments for document "doc_abc123"
```

```
Get the content of "description.txt" attachment from document "doc_abc123"
```

### Database Information

```
How many documents are in each database?
```

```
List all available CouchDB databases
```

## Resources

The MCP server exposes databases as resources with URIs:

- `couchdb://skol_dev` - The skol_dev database
- `couchdb://skol_taxa_dev` - The skol_taxa_dev database

You can reference these URIs in your interactions with Claude Code.

## Troubleshooting

### Server Not Starting

1. **Check Python path:**
   ```bash
   which python3
   ```
   Update the `command` field in `.mcp.json` if needed.

2. **Check MCP SDK installation:**
   ```bash
   pip3 show mcp
   ```

3. **Check CouchDB connection:**
   ```bash
   curl http://localhost:5984
   ```

### Authentication Errors

Verify your credentials in `.env`:
- Ensure `COUCHDB_USER` and `COUCHDB_PASSWORD` are correct
- Test credentials:
  ```bash
  curl -u admin:SU2orange! http://localhost:5984/_all_dbs
  ```

### Database Not Found

Check that the database exists:
```bash
curl -u admin:SU2orange! http://localhost:5984/_all_dbs
```

Update `COUCHDB_DATABASES` in `.env` if needed.

## Security Notes

1. **Never commit `.env` file** - It contains sensitive credentials
2. **Add to .gitignore:**
   ```
   .env
   ```
3. **Use environment variables** for production deployments
4. **Restrict CouchDB access** - Use firewalls and network security
5. **Use HTTPS** for remote CouchDB instances

## Advanced Configuration

### Remote CouchDB

To connect to a remote CouchDB instance, update `.env`:

```bash
COUCHDB_URL=https://your-instance.cloudant.com
COUCHDB_USER=your_username
COUCHDB_PASSWORD=your_password
```

### Multiple Databases

Add more databases to the `COUCHDB_DATABASES` list (comma-separated):

```bash
COUCHDB_DATABASES=skol_dev,skol_taxa_dev,another_db,yet_another_db
```

### Custom Python Environment

If using a conda environment or virtualenv, update the `command` in `.mcp.json`:

```json
"command": "/home/piggy/miniconda3/envs/skol/bin/python3"
```

## Development

To modify the MCP server:

1. Edit [couchdb_mcp_server.py](couchdb_mcp_server.py)
2. Restart Claude Code to reload the server
3. Test with `/mcp` command

## Dependencies

- Python 3.7+
- `mcp` - Model Context Protocol SDK
- `couchdb` - CouchDB Python client

Install dependencies:
```bash
pip3 install mcp couchdb
```

## Support

For issues or questions:
- Check the [MCP documentation](https://modelcontextprotocol.io)
- Review [Claude Code MCP guide](https://code.claude.com/docs/en/mcp)
- Check CouchDB logs for connection issues

## Next Steps

1. Try the example interactions above
2. Explore your databases with natural language queries
3. Create custom views for frequently-used queries
4. Integrate with your existing PySpark workflows

---

**Happy querying!** 🚀
