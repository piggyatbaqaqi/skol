/**
 * Entry point for ProjectMemberships bundle
 *
 * Initializes the ProjectMemberships panel on elements with
 * [data-project-memberships].
 *
 * Usage in HTML:
 *   <div data-project-memberships
 *        data-api-base-url="/api"
 *        data-collection-id="42"
 *        data-authenticated="true">
 *   </div>
 *
 *   <script src="{% static 'js/project-memberships.bundle.js' %}"></script>
 */
import React from 'react';
import { createRoot } from 'react-dom/client';
import ProjectMemberships from './ProjectMemberships';

function initProjectMemberships() {
  const containers = document.querySelectorAll('[data-project-memberships]');
  containers.forEach((container) => {
    if (container.dataset.projectMembershipsInitialized === 'true') return;

    const apiBaseUrl = container.dataset.apiBaseUrl || '/api';
    const collectionId = parseInt(container.dataset.collectionId, 10);
    const authenticated = container.dataset.authenticated === 'true';

    if (!collectionId) {
      console.warn('ProjectMemberships: missing data-collection-id');
      return;
    }

    const root = createRoot(container);
    root.render(
      <ProjectMemberships
        apiBaseUrl={apiBaseUrl}
        collectionId={collectionId}
        authenticated={authenticated}
      />
    );

    container.dataset.projectMembershipsInitialized = 'true';
  });
}

window.initProjectMemberships = initProjectMemberships;

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initProjectMemberships);
} else {
  initProjectMemberships();
}

export { ProjectMemberships };
export default ProjectMemberships;
