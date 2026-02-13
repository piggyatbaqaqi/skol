/**
 * CommentForm - Shared form for new comments, replies, and edits.
 */
import React, { useState } from 'react';

function getCSRFToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]*)/);
  return match ? match[1] : '';
}

const CommentForm = ({
  apiBaseUrl,
  collectionId,
  parentPath = '',
  editMode = false,
  commentId = null,
  initialBody = '',
  initialNomenclature = '',
  onSubmitted,
  onCancel,
  placeholder = 'Write a comment...',
}) => {
  const [body, setBody] = useState(initialBody);
  const [nomenclature, setNomenclature] = useState(initialNomenclature);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!body.trim()) return;
    setSubmitting(true);
    setError(null);

    const url = editMode
      ? `${apiBaseUrl}/collections/${collectionId}/comments/${commentId}/`
      : `${apiBaseUrl}/collections/${collectionId}/comments/`;
    const method = editMode ? 'PUT' : 'POST';

    const payload = editMode
      ? { body, nomenclature }
      : { body, nomenclature, parent_path: parentPath };

    try {
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCSRFToken(),
        },
        credentials: 'same-origin',
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      setBody('');
      setNomenclature('');
      if (onSubmitted) onSubmitted();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="comment-form" onSubmit={handleSubmit}>
      {error && <div className="comment-form-error">{error}</div>}
      <textarea
        className="comment-form-body"
        value={body}
        onChange={(e) => setBody(e.target.value)}
        placeholder={placeholder}
        rows={3}
        required
      />
      <div className="comment-form-nomenclature">
        <input
          type="text"
          value={nomenclature}
          onChange={(e) => setNomenclature(e.target.value)}
          placeholder="Nomenclature (optional)"
        />
      </div>
      <div className="comment-form-actions">
        <button type="submit" disabled={submitting || !body.trim()}>
          {submitting ? 'Saving...' : editMode ? 'Save' : 'Post'}
        </button>
        {onCancel && (
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
};

export default CommentForm;
