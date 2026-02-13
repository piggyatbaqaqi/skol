/**
 * CommentNode - Individual comment display with action buttons.
 */
import React, { useState } from 'react';
import CommentForm from './CommentForm';

function getCSRFToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]*)/);
  return match ? match[1] : '';
}

function formatRelativeTime(isoString) {
  if (!isoString) return '';
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 30) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

function formatDateTime(isoString) {
  if (!isoString) return '';
  return new Date(isoString).toLocaleString();
}

async function apiAction(url, method, onRefresh) {
  try {
    const response = await fetch(url, {
      method,
      headers: { 'X-CSRFToken': getCSRFToken() },
      credentials: 'same-origin',
    });
    if (response.ok && onRefresh) onRefresh();
  } catch (err) {
    console.error('Comment action failed:', err);
  }
}

const CommentNode = ({
  comment,
  currentUserId,
  isAuthor,
  isOwner,
  isAdmin,
  canModerate,
  isCollapsed,
  hasChildren,
  onToggleCollapse,
  apiBaseUrl,
  collectionId,
  onRefresh,
}) => {
  const [editing, setEditing] = useState(false);
  const [replying, setReplying] = useState(false);
  const [selectedVersion, setSelectedVersion] = useState('');

  const isDeleted = comment.deleted;
  const isEdited = comment.updated_at !== comment.created_at;
  const flagCount = comment.flagged_by ? comment.flagged_by.length : 0;
  const isFlaggedByMe = comment.flagged_by && currentUserId
    ? comment.flagged_by.includes(currentUserId)
    : false;

  const handleDelete = () => {
    if (!window.confirm('Delete this comment?')) return;
    apiAction(
      `${apiBaseUrl}/collections/${collectionId}/comments/${comment._id}/`,
      'DELETE',
      onRefresh,
    );
  };

  const handleFlag = () => {
    if (!window.confirm('Flag this comment as inappropriate?\n\nOnly an admin can remove this flag.')) return;
    apiAction(
      `${apiBaseUrl}/collections/${collectionId}/comments/${comment._id}/flag/`,
      'POST',
      onRefresh,
    );
  };

  const handleUnflag = () => {
    apiAction(
      `${apiBaseUrl}/collections/${collectionId}/comments/${comment._id}/flag/`,
      'DELETE',
      onRefresh,
    );
  };

  const handleHide = () => {
    apiAction(
      `${apiBaseUrl}/collections/${collectionId}/comments/${comment._id}/hide/`,
      'POST',
      onRefresh,
    );
  };

  const handleUnhide = () => {
    apiAction(
      `${apiBaseUrl}/collections/${collectionId}/comments/${comment._id}/hide/`,
      'DELETE',
      onRefresh,
    );
  };

  const handleCopyNomenclature = async () => {
    try {
      const response = await fetch(
        `${apiBaseUrl}/collections/${collectionId}/comments/${comment._id}/copy-nomenclature/`,
        {
          method: 'POST',
          headers: { 'X-CSRFToken': getCSRFToken() },
          credentials: 'same-origin',
        },
      );
      if (response.ok) {
        const data = await response.json();
        document.dispatchEvent(
          new CustomEvent('nomenclature-updated', {
            detail: { nomenclature: data.nomenclature },
          }),
        );
        onRefresh();
      }
    } catch (err) {
      console.error('Copy nomenclature failed:', err);
    }
  };

  // Determine which version to display when history dropdown is active
  const historyVersion =
    selectedVersion !== '' ? comment.edit_history[selectedVersion] : null;

  return (
    <div className={`comment-node ${comment.hidden ? 'comment-hidden' : ''}`}>
      <div className="comment-node-header">
        {hasChildren && (
          <button
            className="comment-collapse-btn"
            onClick={onToggleCollapse}
            aria-expanded={!isCollapsed}
          >
            {isCollapsed ? '\u25B6' : '\u25BC'}
          </button>
        )}
        <span className="comment-author">
          {comment.author?.username || '[deleted]'}
        </span>
        <span className="comment-time">
          {formatRelativeTime(comment.created_at)}
        </span>
        {isEdited && !isDeleted && (
          <span className="comment-edited">(edited)</span>
        )}
        {comment.hidden && (
          <span className="comment-hidden-badge">[hidden]</span>
        )}
        {flagCount > 0 && canModerate && (
          <span className="comment-flagged-badge" title={`Flagged by ${flagCount} user${flagCount !== 1 ? 's' : ''}`}>
            flagged ({flagCount})
          </span>
        )}
      </div>

      {!editing ? (
        <div className="comment-body">{comment.body}</div>
      ) : (
        <CommentForm
          apiBaseUrl={apiBaseUrl}
          collectionId={collectionId}
          editMode
          commentId={comment._id}
          initialBody={comment.body}
          initialNomenclature={comment.nomenclature || ''}
          onSubmitted={() => {
            setEditing(false);
            onRefresh();
          }}
          onCancel={() => setEditing(false)}
        />
      )}

      {comment.nomenclature && !isDeleted && (
        <div className="comment-nomenclature">
          <em>Nomenclature: {comment.nomenclature}</em>
          {isOwner && (
            <button
              className="comment-copy-nomenclature-btn"
              onClick={handleCopyNomenclature}
              title="Copy to collection nomenclature"
            >
              Copy to Collection
            </button>
          )}
        </div>
      )}

      {comment.edit_history && comment.edit_history.length > 0 && !isDeleted && (
        <div className="comment-history">
          <select
            value={selectedVersion}
            onChange={(e) => setSelectedVersion(e.target.value)}
          >
            <option value="">Current version</option>
            {comment.edit_history.map((v, i) => (
              <option key={i} value={i}>
                {formatDateTime(v.edited_at)}
              </option>
            ))}
          </select>
          {historyVersion && (
            <div className="comment-history-body">
              <div>{historyVersion.body}</div>
              {historyVersion.nomenclature && (
                <div>
                  <em>Nomenclature: {historyVersion.nomenclature}</em>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {!isDeleted && !editing && (
        <div className="comment-actions">
          <button onClick={() => setReplying(!replying)}>Reply</button>
          {isAuthor && (
            <button onClick={() => setEditing(true)}>Edit</button>
          )}
          {(isAuthor || canModerate) && (
            <button onClick={handleDelete}>Delete</button>
          )}
          <button
            className={isFlaggedByMe ? 'comment-action-active' : ''}
            onClick={handleFlag}
            disabled={isFlaggedByMe}
            title={isFlaggedByMe ? 'You have flagged this comment' : 'Flag for moderator attention'}
          >
            {isFlaggedByMe ? 'Flagged' : 'Flag'}
          </button>
          {canModerate && flagCount > 0 && (
            <button onClick={handleUnflag}>Unflag</button>
          )}
          {canModerate && !comment.hidden && (
            <button onClick={handleHide}>Hide</button>
          )}
          {canModerate && comment.hidden && (
            <button onClick={handleUnhide}>Unhide</button>
          )}
        </div>
      )}

      {replying && (
        <CommentForm
          apiBaseUrl={apiBaseUrl}
          collectionId={collectionId}
          parentPath={comment.path}
          onSubmitted={() => {
            setReplying(false);
            onRefresh();
          }}
          onCancel={() => setReplying(false)}
          placeholder="Write a reply..."
        />
      )}
    </div>
  );
};

export default CommentNode;
