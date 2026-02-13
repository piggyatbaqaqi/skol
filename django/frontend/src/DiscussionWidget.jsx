/**
 * DiscussionWidget - Threaded comment/discussion system for collections.
 *
 * Defaults to collapsed. Shows "Discussion (N)" header. When expanded,
 * fetches comments from the API, renders them as a tree, and provides
 * a form for adding new comments.
 */
import React, { useState, useEffect, useCallback } from 'react';
import CommentThread from './CommentThread';
import CommentForm from './CommentForm';
import './DiscussionWidget.css';

const COOKIE_NAME = 'skol_discussion_collapsed';
const COOKIE_MAX_AGE_DAYS = 30;

function getCSRFToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]*)/);
  return match ? match[1] : '';
}

function getCollapseState() {
  try {
    const match = document.cookie.match(
      new RegExp(`(?:^|;\\s*)${COOKIE_NAME}=([^;]*)`)
    );
    if (match) {
      return new Set(JSON.parse(decodeURIComponent(match[1])));
    }
  } catch (e) {
    // Ignore malformed cookie
  }
  return new Set();
}

function saveCollapseState(collapsedSet) {
  const value = JSON.stringify([...collapsedSet]);
  const expires = new Date(
    Date.now() + COOKIE_MAX_AGE_DAYS * 86400000
  ).toUTCString();
  document.cookie =
    `${COOKIE_NAME}=${encodeURIComponent(value)};` +
    `expires=${expires};path=/;SameSite=Lax`;
}

/**
 * Build a tree from a flat list of comments sorted by sort_key.
 * Each comment gets a `children` array.
 */
function buildTree(comments) {
  const roots = [];
  const byPath = {};

  for (const c of comments) {
    const node = { ...c, children: [] };
    byPath[node.path] = node;

    if (!node.parent_path) {
      roots.push(node);
    } else {
      const parent = byPath[node.parent_path];
      if (parent) {
        parent.children.push(node);
      } else {
        // Orphan (parent hidden/missing) — show at root level
        roots.push(node);
      }
    }
  }

  return roots;
}

const DiscussionWidget = ({ apiBaseUrl, initialCollectionId, container }) => {
  const [collectionId, setCollectionId] = useState(initialCollectionId);
  const [comments, setComments] = useState([]);
  const [count, setCount] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentUserId, setCurrentUserId] = useState(null);
  const [isOwner, setIsOwner] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);
  const [collapsedComments, setCollapsedComments] = useState(getCollapseState);

  // Listen for collection changes via custom DOM event or data attribute
  useEffect(() => {
    const handler = (e) => {
      const newId = e.detail?.collectionId;
      if (newId !== undefined) {
        setCollectionId(newId);
        setExpanded(false);
      }
    };
    document.addEventListener('collection-changed', handler);
    return () => document.removeEventListener('collection-changed', handler);
  }, []);

  // Also watch the container's data attribute for imperative updates
  useEffect(() => {
    if (!container) return;
    const observer = new MutationObserver(() => {
      const newId = container.dataset.collectionId;
      if (newId && newId !== String(collectionId)) {
        setCollectionId(Number(newId) || newId);
        setExpanded(false);
      }
    });
    observer.observe(container, {
      attributes: true,
      attributeFilter: ['data-collection-id'],
    });
    return () => observer.disconnect();
  }, [container, collectionId]);

  // Fetch comment count whenever collectionId changes
  useEffect(() => {
    if (!collectionId) {
      setCount(0);
      return;
    }
    fetch(
      `${apiBaseUrl}/collections/${collectionId}/comments/count/`,
      { credentials: 'same-origin' }
    )
      .then((r) => r.json())
      .then((data) => setCount(data.count || 0))
      .catch(() => setCount(0));
  }, [collectionId, apiBaseUrl]);

  // Fetch full comments when widget is expanded
  const fetchComments = useCallback(async () => {
    if (!collectionId) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${apiBaseUrl}/collections/${collectionId}/comments/`,
        {
          credentials: 'same-origin',
          headers: { 'X-CSRFToken': getCSRFToken() },
        }
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setComments(data.comments || []);
      setCount(data.count || 0);
      setCurrentUserId(data.current_user_id);
      setIsOwner(data.is_owner);
      setIsAdmin(data.is_admin);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [collectionId, apiBaseUrl]);

  useEffect(() => {
    if (expanded && collectionId) fetchComments();
  }, [expanded, collectionId, fetchComments]);

  const commentTree = buildTree(comments);

  const toggleCollapse = (commentId) => {
    setCollapsedComments((prev) => {
      const next = new Set(prev);
      if (next.has(commentId)) {
        next.delete(commentId);
      } else {
        next.add(commentId);
      }
      saveCollapseState(next);
      return next;
    });
  };

  if (!collectionId) return null;

  return (
    <div className="discussion-widget">
      <button
        className="discussion-header"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        <span className="discussion-header-icon">
          {expanded ? '\u25BC' : '\u25B6'}
        </span>
        Discussion ({count})
      </button>

      {expanded && (
        <div className="discussion-body">
          {loading && <div className="discussion-loading">Loading...</div>}
          {error && <div className="discussion-error">{error}</div>}

          <div className="discussion-scroll-area">
            {!loading && comments.length === 0 && (
              <div className="discussion-empty">
                No comments yet. Start the discussion!
              </div>
            )}
            {commentTree.map((node) => (
              <CommentThread
                key={node._id}
                node={node}
                depth={0}
                apiBaseUrl={apiBaseUrl}
                collectionId={collectionId}
                currentUserId={currentUserId}
                isOwner={isOwner}
                isAdmin={isAdmin}
                collapsedComments={collapsedComments}
                onToggleCollapse={toggleCollapse}
                onRefresh={fetchComments}
              />
            ))}
          </div>

          <CommentForm
            apiBaseUrl={apiBaseUrl}
            collectionId={collectionId}
            parentPath=""
            onSubmitted={fetchComments}
            placeholder="Add a comment..."
          />
        </div>
      )}
    </div>
  );
};

export default DiscussionWidget;
