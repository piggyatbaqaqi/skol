"""
Mock CouchDB database for testing ingestors.

This module provides a MockDatabase class that simulates a CouchDB database
without actually storing data, useful for testing metadata extraction.
"""


class MockDatabase:
    """
    Mock database that doesn't actually store anything.

    This class simulates a CouchDB database interface for testing purposes.
    It collects documents in memory but doesn't persist them.
    """

    def __init__(self):
        """Initialize the mock database with an empty document collection."""
        self.documents = {}
        self.attachments = {}

    def save(self, doc):
        """
        Mock save - store document in memory.

        Args:
            doc: Document dictionary to save

        Returns:
            Tuple of (doc_id, revision)
        """
        doc_id = doc.get('_id', f'mock-id-{len(self.documents)}')
        self.documents[doc_id] = doc.copy()
        return (doc_id, 'mock-rev-1')

    def __contains__(self, doc_id):
        """
        Check if document exists.

        Args:
            doc_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """
        return doc_id in self.documents

    def __getitem__(self, doc_id):
        """
        Get document by ID.

        Args:
            doc_id: Document ID to retrieve

        Returns:
            Document dict if found

        Raises:
            KeyError: If document not found
        """
        return self.documents[doc_id]

    def get(self, doc_id, default=None):
        """
        Get document by ID with default.

        Args:
            doc_id: Document ID to retrieve
            default: Default value if not found

        Returns:
            Document dict if found, default otherwise
        """
        return self.documents.get(doc_id, default)

    def put_attachment(self, doc, file, filename, content_type):
        """
        Mock attachment storage.

        Args:
            doc: Document to attach to
            file: File-like object containing attachment data
            filename: Attachment filename
            content_type: MIME type of attachment

        Returns:
            Mock response
        """
        doc_id = doc.get('_id')
        if doc_id not in self.attachments:
            self.attachments[doc_id] = {}

        # Read file content for size calculation
        file.seek(0)
        content = file.read()

        # Store attachment metadata
        self.attachments[doc_id][filename] = {
            'content_type': content_type,
            'size': len(content)
        }

        # Update document with attachment metadata
        if doc_id in self.documents:
            if '_attachments' not in self.documents[doc_id]:
                self.documents[doc_id]['_attachments'] = {}
            self.documents[doc_id]['_attachments'][filename] = {
                'content_type': content_type,
                'length': len(content)
            }

        return {'ok': True}

    def get_documents(self):
        """
        Get all documents in the database.

        Returns:
            List of all document dictionaries
        """
        return list(self.documents.values())

    def get_attachment_info(self, doc_id):
        """
        Get attachment information for a document.

        Args:
            doc_id: Document ID

        Returns:
            Dictionary of attachment metadata
        """
        return self.attachments.get(doc_id, {})
