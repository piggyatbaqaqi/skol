# CouchDB MCP Quick Start Guide

## ✅ Setup Complete!

Your CouchDB MCP server is now fully configured and running!

## 📋 Current Configuration

### Claude CLI Version
- **Version**: 2.0.50 (Claude Code)
- **Node.js**: v20.19.5
- **Location**: `/home/piggy/miniconda3/envs/skol/bin/claude`

### MCP Server Status
- **Name**: couchdb
- **Status**: ✓ Connected
- **Type**: stdio
- **Python**: `/home/piggy/miniconda3/envs/skol/bin/python`
- **Script**: `/data/piggy/src/github.com/piggyatbaqaqi/couchdb_mcp_server.py`

### Connected Databases
- `skol_dev`
- `skol_taxa_dev`

### Server URL
- `http://localhost:5984`

## 🚀 Usage

### Command Line Usage

You can now use the Claude CLI to interact with your CouchDB databases:

```bash
# Start a Claude Code session
claude code

# Within the session, use the /mcp command to see available servers
/mcp
```

### Available MCP Commands

```bash
# List all configured MCP servers
claude mcp list

# Get details about the CouchDB server
claude mcp get couchdb

# Remove the server (if needed)
claude mcp remove couchdb -s local
```

### Natural Language Queries

Once you start a Claude Code session, you can ask natural language questions about your CouchDB databases:

```
How many documents are in skol_dev?
```

```
Search for documents containing "Amanita" in skol_taxa_dev
```

```
Show me the first 10 documents from skol_dev
```

```
Get document "species_001" from skol_taxa_dev
```

```
List all attachments for document "doc123"
```

## 🔧 Available Tools

Your MCP server provides 8 tools:

1. **list_databases** - List all CouchDB databases
2. **get_document** - Get a document by ID
3. **query_documents** - Query with pagination and filters
4. **search_documents** - Full-text search across documents
5. **get_attachment** - Retrieve document attachments
6. **list_attachments** - List all attachments for a document
7. **execute_view** - Execute CouchDB views
8. **count_documents** - Count documents with optional filters

## 🔒 Security

Your credentials are stored in:
- **Environment file**: `.env` (protected by `.gitignore`)
- **Claude config**: `~/.claude.json` (local scope, not shared)

The `.env` file is **not** committed to version control for security.

## 📁 Project Files

- [couchdb_mcp_server.py](couchdb_mcp_server.py) - MCP server implementation
- [.mcp.json](.mcp.json) - Project-scoped MCP configuration
- [.env](.env) - Environment variables (credentials)
- [COUCHDB_MCP_SETUP.md](COUCHDB_MCP_SETUP.md) - Detailed setup documentation
- [MCP_QUICK_START.md](MCP_QUICK_START.md) - This file

## 🎯 Next Steps

1. **Start using it!**
   ```bash
   claude code
   ```

2. **Try a query:**
   ```
   /mcp
   List all databases
   ```

3. **Explore your data:**
   ```
   Search for "mycology" in skol_dev database
   ```

## 🛠️ Troubleshooting

### Server Not Connected

If `claude mcp list` shows the server as disconnected:

1. Check CouchDB is running:
   ```bash
   curl http://localhost:5984
   ```

2. Verify credentials:
   ```bash
   curl -u admin:SU2orange! http://localhost:5984/_all_dbs
   ```

3. Check Python path:
   ```bash
   which python
   ```

4. Test the MCP server directly:
   ```bash
   /home/piggy/miniconda3/envs/skol/bin/python couchdb_mcp_server.py
   ```

### Database Not Found

Check that your databases exist:
```bash
curl -u admin:SU2orange! http://localhost:5984/_all_dbs
```

### Permission Issues

Verify your CouchDB credentials in `.env` are correct.

## 📚 Documentation

- [COUCHDB_MCP_SETUP.md](COUCHDB_MCP_SETUP.md) - Complete setup guide
- [Official MCP Docs](https://modelcontextprotocol.io)
- [Claude Code Docs](https://code.claude.com/docs)

## 🎉 Success!

Your CouchDB MCP server is ready to use! Start exploring your databases with natural language queries.

---

**Need help?** Check [COUCHDB_MCP_SETUP.md](COUCHDB_MCP_SETUP.md) for detailed documentation.
