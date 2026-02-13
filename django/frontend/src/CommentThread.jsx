/**
 * CommentThread - Recursive component that renders a comment and its children.
 */
import React from 'react';
import CommentNode from './CommentNode';

const CommentThread = ({
  node,
  depth,
  apiBaseUrl,
  collectionId,
  currentUserId,
  isOwner,
  isAdmin,
  collapsedComments,
  onToggleCollapse,
  onRefresh,
}) => {
  const isCollapsed = collapsedComments.has(node._id);
  const isAuthor = node.author?.user_id === currentUserId;
  const canModerate = isOwner || isAdmin;

  return (
    <div
      className="comment-thread"
      style={depth > 0 ? { marginLeft: 20, borderLeft: '2px solid #e0e0e0', paddingLeft: 12 } : undefined}
    >
      <CommentNode
        comment={node}
        currentUserId={currentUserId}
        isAuthor={isAuthor}
        isOwner={isOwner}
        isAdmin={isAdmin}
        canModerate={canModerate}
        isCollapsed={isCollapsed}
        hasChildren={node.children && node.children.length > 0}
        onToggleCollapse={() => onToggleCollapse(node._id)}
        apiBaseUrl={apiBaseUrl}
        collectionId={collectionId}
        onRefresh={onRefresh}
      />

      {!isCollapsed &&
        node.children &&
        node.children.map((child) => (
          <CommentThread
            key={child._id}
            node={child}
            depth={depth + 1}
            apiBaseUrl={apiBaseUrl}
            collectionId={collectionId}
            currentUserId={currentUserId}
            isOwner={isOwner}
            isAdmin={isAdmin}
            collapsedComments={collapsedComments}
            onToggleCollapse={onToggleCollapse}
            onRefresh={onRefresh}
          />
        ))}
    </div>
  );
};

export default CommentThread;
