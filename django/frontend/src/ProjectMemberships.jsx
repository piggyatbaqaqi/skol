/**
 * ProjectMemberships - Panel for viewing and managing project membership
 *
 * Shown on the collection "View Details" page.  Lists the projects this
 * collection currently belongs to; lets any authenticated user add it to
 * another project or remove it from an existing one (democratic model).
 *
 * Props (all injected via data-* attributes by projectMembershipsEntry.js):
 *   apiBaseUrl      - e.g. "/api"            (default "/api")
 *   collectionId    - integer collection ID  (required)
 *   authenticated   - boolean                (if false, show read-only view)
 */
import React, { useState, useEffect, useCallback } from 'react';
import CreatableSelect from 'react-select/creatable';
import './ProjectMemberships.css';

const creatableStyles = {
  control: (base, state) => ({
    ...base,
    minHeight: '34px',
    borderColor: state.isFocused ? '#667eea' : '#e0e0e0',
    boxShadow: state.isFocused ? '0 0 0 1px #667eea' : 'none',
    '&:hover': { borderColor: '#667eea' },
  }),
  option: (base, state) => ({
    ...base,
    backgroundColor: state.isSelected ? '#667eea' : state.isFocused ? '#f0f4ff' : 'white',
    color: state.isSelected ? 'white' : '#333',
    cursor: 'pointer',
    padding: '7px 12px',
    '&:active': { backgroundColor: '#667eea' },
  }),
  menu: (base) => ({ ...base, zIndex: 9999 }),
  placeholder: (base) => ({ ...base, color: '#999', fontSize: 13 }),
};

const ProjectMemberships = ({
  apiBaseUrl = '/api',
  collectionId: collectionIdProp,
  authenticated = false,
  listenForCollectionChange = false,
  compact = false,
}) => {
  // In "listen" mode the collection ID comes from the collection-changed event;
  // null means no collection is active (component renders nothing).
  const [collectionId, setCollectionId] = useState(
    listenForCollectionChange ? null : collectionIdProp
  );

  useEffect(() => {
    if (!listenForCollectionChange) return;
    const handler = (e) => setCollectionId(e.detail.collection_id || null);
    document.addEventListener('collection-changed', handler);
    return () => document.removeEventListener('collection-changed', handler);
  }, [listenForCollectionChange]);

  const [memberships, setMemberships] = useState([]);   // [{slug, name}, ...]
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [addOption, setAddOption]     = useState(null);  // react-select option or null
  const [projectOptions, setProjectOptions] = useState([]); // all projects as options
  const [adding, setAdding]           = useState(false);
  const [status, setStatus]           = useState(null); // {type:'success'|'error', msg}
  const [statusVisible, setStatusVisible] = useState(false);
  const [armedSlug, setArmedSlug]     = useState(null); // slug of pill in confirm-remove state

  // In listen mode, reset state whenever the collection switches.
  useEffect(() => {
    if (listenForCollectionChange) {
      setMemberships([]);
      setAddOption(null);
      setStatus(null);
      setError(null);
      setArmedSlug(null);
    }
  }, [collectionId, listenForCollectionChange]);

  // Dismiss armed remove state when clicking outside any membership pill.
  useEffect(() => {
    if (!armedSlug) return;
    const handler = (e) => {
      if (!e.target.closest('.project-membership-item')) {
        setArmedSlug(null);
      }
    };
    document.addEventListener('click', handler);
    return () => document.removeEventListener('click', handler);
  }, [armedSlug]);

  // ---- helpers --------------------------------------------------------

  const clearStatus = () => { setStatus(null); setStatusVisible(false); };

  // In compact mode, show status briefly then fade out.
  useEffect(() => {
    if (!compact || !status) return;
    setStatusVisible(true);
    const t = setTimeout(clearStatus, 3000);
    return () => clearTimeout(t);
  }, [status, compact]); // eslint-disable-line react-hooks/exhaustive-deps

  // Parse username/slug from namespaced slug
  const splitSlug = (namespacedSlug) => {
    const idx = namespacedSlug.indexOf('/');
    if (idx === -1) return null;
    return {
      username: namespacedSlug.slice(0, idx),
      slug: namespacedSlug.slice(idx + 1),
    };
  };

  // Build URL for the project detail page from its namespaced slug.
  const projectPageUrl = (namespacedSlug) => {
    const base = apiBaseUrl.replace(/\/api$/, '');
    return `${base}/projects/${namespacedSlug}/`;
  };

  // ---- fetch current memberships -------------------------------------

  const fetchMemberships = useCallback(async () => {
    if (!collectionId) return;
    setLoading(true);
    setError(null);
    try {
      const [memberResp, allResp] = await Promise.all([
        fetch(`${apiBaseUrl}/projects/?collection_id=${collectionId}`, { credentials: 'same-origin' }),
        fetch(`${apiBaseUrl}/projects/`, { credentials: 'same-origin' }),
      ]);
      if (!memberResp.ok) throw new Error('Failed to load projects');
      const memberData = await memberResp.json();
      setMemberships((memberData.projects || []).map((p) => ({ slug: p.namespaced_slug, name: p.name })));

      if (allResp.ok) {
        const allData = await allResp.json();
        setProjectOptions(
          (allData.projects || []).map((p) => ({
            value: p.namespaced_slug,
            label: p.name,
          }))
        );
      }
    } catch (err) {
      console.error('ProjectMemberships fetch error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, collectionId]);

  useEffect(() => {
    fetchMemberships();
  }, [fetchMemberships]);

  // ---- add to existing project ----------------------------------------

  const addToProject = async (namespacedSlug) => {
    const parts = splitSlug(namespacedSlug);
    if (!parts) return;
    const url = `${apiBaseUrl}/projects/${parts.username}/${parts.slug}/collections/${collectionId}/`;
    const resp = await fetch(url, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'X-CSRFToken': getCsrfToken() },
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${resp.status}`);
    }
  };

  const handleAdd = async () => {
    if (!addOption) return;
    setAdding(true);
    clearStatus();
    try {
      await addToProject(addOption.value);
      setAddOption(null);
      setStatus({ type: 'success', msg: 'Added to project.' });
      await fetchMemberships();
    } catch (err) {
      setStatus({ type: 'error', msg: err.message });
    } finally {
      setAdding(false);
    }
  };

  // ---- create new project then add ------------------------------------

  const handleCreate = async (inputValue) => {
    setAdding(true);
    clearStatus();
    try {
      // Create the project
      const createResp = await fetch(`${apiBaseUrl}/projects/`, {
        method: 'POST',
        credentials: 'same-origin',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfToken(),
        },
        body: JSON.stringify({ name: inputValue }),
      });
      if (!createResp.ok) {
        const body = await createResp.json().catch(() => ({}));
        const msg = body.name ? body.name[0] : (body.detail || `HTTP ${createResp.status}`);
        throw new Error(msg);
      }
      const project = await createResp.json();
      const namespacedSlug = project.namespaced_slug;

      // Add this collection to the new project
      await addToProject(namespacedSlug);

      setAddOption(null);
      setStatus({ type: 'success', msg: `Created project "${project.name}" and added this collection.` });
      await fetchMemberships();
    } catch (err) {
      setStatus({ type: 'error', msg: err.message });
    } finally {
      setAdding(false);
    }
  };

  // ---- remove from project -------------------------------------------

  const handleRemove = async (namespacedSlug) => {
    const parts = splitSlug(namespacedSlug);
    if (!parts) return;
    setArmedSlug(null);
    clearStatus();
    try {
      const url = `${apiBaseUrl}/projects/${parts.username}/${parts.slug}/collections/${collectionId}/`;
      const resp = await fetch(url, {
        method: 'DELETE',
        credentials: 'same-origin',
        headers: { 'X-CSRFToken': getCsrfToken() },
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${resp.status}`);
      }
      setStatus({ type: 'success', msg: 'Removed from project.' });
      await fetchMemberships();
    } catch (err) {
      setStatus({ type: 'error', msg: err.message });
    }
  };

  // ---- render --------------------------------------------------------

  // In listen mode, render nothing until a collection is selected.
  if (listenForCollectionChange && !collectionId) return null;

  if (loading) return <div className="project-memberships"><span style={{ color: '#aaa', fontSize: 13 }}>Loading projects…</span></div>;
  if (error)   return <div className="project-memberships"><span style={{ color: '#c33', fontSize: 13 }}>Could not load projects.</span></div>;

  // ---- compact single-line render ------------------------------------
  if (compact) {
    return (
      <div className="project-memberships-compact">
        <label className="project-memberships-compact-label">Projects:</label>

        {memberships.length === 0 ? (
          <span className="project-memberships-compact-empty">Not in any project.</span>
        ) : (
          <ul className="project-membership-list">
            {memberships.map((m) => (
              <li key={m.slug} className="project-membership-item">
                <a
                  href={projectPageUrl(m.slug)}
                  className="project-membership-link"
                  title={`Open project ${m.name}`}
                >
                  {m.name}
                </a>
                {authenticated && (
                  armedSlug === m.slug ? (
                    <>
                      <button
                        className="project-membership-confirm-remove"
                        onClick={() => handleRemove(m.slug)}
                        aria-label={`Confirm remove from ${m.name}`}
                      >
                        Remove?
                      </button>
                      <button
                        className="project-membership-cancel-remove"
                        onClick={(e) => { e.stopPropagation(); setArmedSlug(null); }}
                        aria-label="Cancel"
                      >
                        ✕
                      </button>
                    </>
                  ) : (
                    <button
                      className="project-membership-remove"
                      title={`Remove from ${m.name}`}
                      onClick={(e) => { e.stopPropagation(); setArmedSlug(m.slug); }}
                      aria-label={`Remove from ${m.name}`}
                    >
                      ×
                    </button>
                  )
                )}
              </li>
            ))}
          </ul>
        )}

        {authenticated && (
          <>
            <div className="project-memberships-compact-select">
              <CreatableSelect
                options={projectOptions}
                value={addOption}
                onChange={setAddOption}
                onCreateOption={handleCreate}
                isDisabled={adding}
                isClearable
                placeholder="Add or create project…"
                formatCreateLabel={(input) => `Create "${input}"`}
                styles={creatableStyles}
                classNamePrefix="project"
                noOptionsMessage={() => 'Type a name to create'}
              />
            </div>
            <button
              className="project-memberships-add-btn"
              onClick={handleAdd}
              disabled={!addOption || adding}
            >
              {adding ? '…' : 'Add'}
            </button>
          </>
        )}

        <span className={`project-memberships-compact-status ${status ? status.type : ''} ${statusVisible ? 'visible' : ''}`}>
          {status ? status.msg : ''}
        </span>
      </div>
    );
  }

  // ---- full stacked render -------------------------------------------
  return (
    <div className="project-memberships">
      <div className="project-memberships-title">Projects</div>

      {memberships.length === 0 ? (
        <div className="project-memberships-empty">Not in any project.</div>
      ) : (
        <ul className="project-membership-list">
          {memberships.map((m) => (
            <li key={m.slug} className="project-membership-item">
              <a
                href={projectPageUrl(m.slug)}
                className="project-membership-link"
                title={`Open project ${m.name}`}
              >
                {m.name}
              </a>
              {authenticated && (
                armedSlug === m.slug ? (
                  <>
                    <button
                      className="project-membership-confirm-remove"
                      onClick={() => handleRemove(m.slug)}
                      aria-label={`Confirm remove from ${m.name}`}
                    >
                      Remove?
                    </button>
                    <button
                      className="project-membership-cancel-remove"
                      onClick={(e) => { e.stopPropagation(); setArmedSlug(null); }}
                      aria-label="Cancel"
                    >
                      ✕
                    </button>
                  </>
                ) : (
                  <button
                    className="project-membership-remove"
                    title={`Remove from ${m.name}`}
                    onClick={(e) => { e.stopPropagation(); setArmedSlug(m.slug); }}
                    aria-label={`Remove from ${m.name}`}
                  >
                    ×
                  </button>
                )
              )}
            </li>
          ))}
        </ul>
      )}

      {authenticated && (
        <div className="project-memberships-add">
          <CreatableSelect
            options={projectOptions}
            value={addOption}
            onChange={setAddOption}
            onCreateOption={handleCreate}
            isDisabled={adding}
            isClearable
            placeholder="Add to project… or type to create"
            formatCreateLabel={(input) => `Create project "${input}"`}
            styles={creatableStyles}
            className="project-select"
            classNamePrefix="project"
            noOptionsMessage={() => 'No projects yet — type a name to create one'}
          />
          <button
            className="project-memberships-add-btn"
            onClick={handleAdd}
            disabled={!addOption || adding}
          >
            {adding ? 'Adding…' : 'Add'}
          </button>
        </div>
      )}

      {status && (
        <div className={`project-memberships-status ${status.type}`}>{status.msg}</div>
      )}
    </div>
  );
};

// ---- CSRF helper -------------------------------------------------------

function getCsrfToken() {
  const name = 'csrftoken';
  const cookies = document.cookie.split(';');
  for (const c of cookies) {
    const [key, val] = c.trim().split('=');
    if (key === name) return decodeURIComponent(val);
  }
  // Fall back to meta tag
  const meta = document.querySelector('meta[name="csrf-token"]');
  return meta ? meta.getAttribute('content') : '';
}

export default ProjectMemberships;
