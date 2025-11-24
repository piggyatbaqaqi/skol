#!/usr/bin/env python3
"""
CouchDB MCP Server
Provides Model Context Protocol access to CouchDB databases.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import couchdb
from mcp.server import Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from mcp.server.stdio import stdio_server


# Initialize MCP server
app = Server("couchdb-server")

# Global CouchDB connection
_server: Optional[couchdb.Server] = None
_databases: Dict[str, Any] = {}


def get_server() -> couchdb.Server:
    """Get or create CouchDB server connection."""
    global _server
    if _server is None:
        url = os.getenv("COUCHDB_URL", "http://localhost:5984")
        username = os.getenv("COUCHDB_USER", "admin")
        password = os.getenv("COUCHDB_PASSWORD", "")

        _server = couchdb.Server(url)
        if username and password:
            _server.resource.credentials = (username, password)

    return _server


def get_database(db_name: str) -> Any:
    """Get or create database connection."""
    global _databases
    if db_name not in _databases:
        server = get_server()
        if db_name in server:
            _databases[db_name] = server[db_name]
        else:
            raise ValueError(f"Database '{db_name}' does not exist")

    return _databases[db_name]


@app.list_resources()
async def list_resources() -> List[Resource]:
    """List available CouchDB databases as resources."""
    server = get_server()
    databases = list(server)

    # Filter to databases of interest
    target_dbs = os.getenv("COUCHDB_DATABASES", "skol_dev,skol_taxa_dev").split(",")

    resources = []
    for db_name in target_dbs:
        if db_name in databases:
            try:
                db = server[db_name]
                doc_count = len(db)
                resources.append(
                    Resource(
                        uri=f"couchdb://{db_name}",
                        name=f"Database: {db_name}",
                        mimeType="application/json",
                        description=f"CouchDB database with {doc_count} documents"
                    )
                )
            except Exception as e:
                print(f"Error accessing database {db_name}: {e}")

    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a CouchDB resource (database or document)."""
    if not uri.startswith("couchdb://"):
        raise ValueError("Invalid URI scheme. Expected 'couchdb://'")

    # Parse URI: couchdb://db_name or couchdb://db_name/doc_id
    parts = uri[11:].split("/", 1)
    db_name = parts[0]

    db = get_database(db_name)

    if len(parts) == 1:
        # List all documents in database
        docs = []
        for doc_id in db:
            try:
                doc = db[doc_id]
                docs.append({
                    "_id": doc.get("_id"),
                    "_rev": doc.get("_rev"),
                    "type": doc.get("type", "unknown")
                })
            except Exception as e:
                print(f"Error reading document {doc_id}: {e}")

        return json.dumps({"database": db_name, "documents": docs}, indent=2)

    else:
        # Get specific document
        doc_id = parts[1]
        if doc_id in db:
            doc = db[doc_id]
            return json.dumps(doc, indent=2)
        else:
            raise ValueError(f"Document '{doc_id}' not found in database '{db_name}'")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available CouchDB tools."""
    return [
        Tool(
            name="list_databases",
            description="List all available CouchDB databases",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        Tool(
            name="get_document",
            description="Get a document by ID from a CouchDB database",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "Database name (e.g., 'skol_dev')"
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID"
                    }
                },
                "required": ["database", "doc_id"]
            }
        ),
        Tool(
            name="query_documents",
            description="Query documents in a database with optional filters",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "Database name"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents to return (default: 100)"
                    },
                    "skip": {
                        "type": "integer",
                        "description": "Number of documents to skip (default: 0)"
                    },
                    "include_docs": {
                        "type": "boolean",
                        "description": "Include full document content (default: true)"
                    }
                },
                "required": ["database"]
            }
        ),
        Tool(
            name="search_documents",
            description="Search for documents containing specific text in any field",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "Database name"
                    },
                    "search_text": {
                        "type": "string",
                        "description": "Text to search for"
                    },
                    "field": {
                        "type": "string",
                        "description": "Optional: specific field to search in"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 50)"
                    }
                },
                "required": ["database", "search_text"]
            }
        ),
        Tool(
            name="get_attachment",
            description="Get an attachment from a document",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "Database name"
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID"
                    },
                    "attachment_name": {
                        "type": "string",
                        "description": "Attachment filename"
                    }
                },
                "required": ["database", "doc_id", "attachment_name"]
            }
        ),
        Tool(
            name="list_attachments",
            description="List all attachments for a document",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "Database name"
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID"
                    }
                },
                "required": ["database", "doc_id"]
            }
        ),
        Tool(
            name="execute_view",
            description="Execute a CouchDB view",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "Database name"
                    },
                    "design_doc": {
                        "type": "string",
                        "description": "Design document name (without '_design/' prefix)"
                    },
                    "view_name": {
                        "type": "string",
                        "description": "View name"
                    },
                    "key": {
                        "type": "string",
                        "description": "Optional: Filter by specific key"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 100)"
                    }
                },
                "required": ["database", "design_doc", "view_name"]
            }
        ),
        Tool(
            name="count_documents",
            description="Count total documents in a database or matching a filter",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "Database name"
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Optional: filter by document type field"
                    }
                },
                "required": ["database"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle tool calls."""

    try:
        if name == "list_databases":
            server = get_server()
            databases = list(server)

            result = []
            for db_name in databases:
                try:
                    db = server[db_name]
                    result.append({
                        "name": db_name,
                        "doc_count": len(db),
                        "uri": f"couchdb://{db_name}"
                    })
                except Exception as e:
                    result.append({
                        "name": db_name,
                        "error": str(e)
                    })

            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        elif name == "get_document":
            db_name = arguments["database"]
            doc_id = arguments["doc_id"]

            db = get_database(db_name)

            if doc_id in db:
                doc = db[doc_id]
                return [TextContent(
                    type="text",
                    text=json.dumps(dict(doc), indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Document '{doc_id}' not found"})
                )]

        elif name == "query_documents":
            db_name = arguments["database"]
            limit = arguments.get("limit", 100)
            skip = arguments.get("skip", 0)
            include_docs = arguments.get("include_docs", True)

            db = get_database(db_name)

            results = []
            for i, doc_id in enumerate(db):
                if i < skip:
                    continue
                if len(results) >= limit:
                    break

                try:
                    if include_docs:
                        doc = db[doc_id]
                        results.append(dict(doc))
                    else:
                        results.append({"_id": doc_id})
                except Exception as e:
                    results.append({"_id": doc_id, "error": str(e)})

            return [TextContent(
                type="text",
                text=json.dumps({
                    "total": len(results),
                    "limit": limit,
                    "skip": skip,
                    "documents": results
                }, indent=2)
            )]

        elif name == "search_documents":
            db_name = arguments["database"]
            search_text = arguments["search_text"].lower()
            field = arguments.get("field")
            limit = arguments.get("limit", 50)

            db = get_database(db_name)

            results = []
            for doc_id in db:
                if len(results) >= limit:
                    break

                try:
                    doc = db[doc_id]

                    # Search in specific field or all fields
                    if field:
                        if field in doc and search_text in str(doc[field]).lower():
                            results.append(dict(doc))
                    else:
                        # Search all fields
                        doc_str = json.dumps(doc).lower()
                        if search_text in doc_str:
                            results.append(dict(doc))
                except Exception as e:
                    pass

            return [TextContent(
                type="text",
                text=json.dumps({
                    "query": search_text,
                    "field": field,
                    "count": len(results),
                    "documents": results
                }, indent=2)
            )]

        elif name == "get_attachment":
            db_name = arguments["database"]
            doc_id = arguments["doc_id"]
            attachment_name = arguments["attachment_name"]

            db = get_database(db_name)

            if doc_id in db:
                doc = db[doc_id]
                if "_attachments" in doc and attachment_name in doc["_attachments"]:
                    attachment = db.get_attachment(doc, attachment_name)
                    if attachment:
                        content = attachment.read()
                        try:
                            # Try to decode as text
                            text_content = content.decode('utf-8')
                            return [TextContent(
                                type="text",
                                text=text_content
                            )]
                        except UnicodeDecodeError:
                            # Binary content
                            return [TextContent(
                                type="text",
                                text=f"Binary attachment ({len(content)} bytes)"
                            )]
                else:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": f"Attachment '{attachment_name}' not found"})
                    )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Document '{doc_id}' not found"})
                )]

        elif name == "list_attachments":
            db_name = arguments["database"]
            doc_id = arguments["doc_id"]

            db = get_database(db_name)

            if doc_id in db:
                doc = db[doc_id]
                attachments = doc.get("_attachments", {})

                result = []
                for name, info in attachments.items():
                    result.append({
                        "name": name,
                        "content_type": info.get("content_type"),
                        "length": info.get("length"),
                        "revpos": info.get("revpos")
                    })

                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "doc_id": doc_id,
                        "attachment_count": len(result),
                        "attachments": result
                    }, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Document '{doc_id}' not found"})
                )]

        elif name == "execute_view":
            db_name = arguments["database"]
            design_doc = arguments["design_doc"]
            view_name = arguments["view_name"]
            key = arguments.get("key")
            limit = arguments.get("limit", 100)

            db = get_database(db_name)

            view_path = f"_design/{design_doc}/_view/{view_name}"

            try:
                if key:
                    results = db.view(view_path, key=key, limit=limit)
                else:
                    results = db.view(view_path, limit=limit)

                rows = [{"key": row.key, "value": row.value, "id": row.id} for row in results]

                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "view": f"{design_doc}/{view_name}",
                        "count": len(rows),
                        "rows": rows
                    }, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"View execution failed: {str(e)}"})
                )]

        elif name == "count_documents":
            db_name = arguments["database"]
            doc_type = arguments.get("doc_type")

            db = get_database(db_name)

            if doc_type:
                count = 0
                for doc_id in db:
                    try:
                        doc = db[doc_id]
                        if doc.get("type") == doc_type:
                            count += 1
                    except Exception:
                        pass

                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "database": db_name,
                        "type": doc_type,
                        "count": count
                    }, indent=2)
                )]
            else:
                count = len(db)
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "database": db_name,
                        "count": count
                    }, indent=2)
                )]

        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
