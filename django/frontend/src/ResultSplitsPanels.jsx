/**
 * ResultSplitsPanels - the Structured Splits and Text Splits tools
 * (docs/plans/result-splitting-features.md, F3).
 *
 * Each panel shows how features divide the current search results, as
 * scored by POST /api/features/structured/ and /api/features/text/
 * (docs/api-reference.md, "Result splits"), and adds the ticked ones to
 * the description.  A score is the expected share of results that differ
 * from the user's answer; it is shown rounded, "separates ~60%".
 */
import React, { useCallback, useState } from 'react';

/** A score as a rounded percentage: 0.598 -> 60. */
function percent(score) {
  return Math.round((score || 0) * 100);
}

/**
 * The parts every panel shares: the loading, error and empty states,
 * the result count, the preview of ticked text and the Add button.
 */
const SplitsFrame = ({
  data, loading, error, isEmpty, selectedTexts, onAdd, label, children,
}) => {
  if (loading) {
    return (
      <div role="status" aria-live="polite" className="feature-loading">
        Loading features...
      </div>
    );
  }
  if (error) {
    return <div role="alert" className="feature-error">{error}</div>;
  }
  if (!data) {
    return (
      <div role="status" aria-live="polite" className="feature-empty">
        Run a search first to see how features split the results.
      </div>
    );
  }
  if (isEmpty) {
    return (
      <div role="status" aria-live="polite" className="feature-empty">
        No features separate these {data.n_results} results.
      </div>
    );
  }
  const preview = selectedTexts.map(t => `${t};`).join(' ');
  return (
    <div className="splits-panel">
      <div className="feature-metadata">{data.n_results} results</div>
      <div className="splits-body" aria-label={label}>{children}</div>
      {preview && (
        <div className="feature-preview">
          <span className="feature-preview-label">Preview:</span>
          <code className="feature-preview-text">{preview}</code>
        </div>
      )}
      <div className="feature-actions">
        <button
          type="button"
          className="feature-add-btn"
          onClick={onAdd}
          disabled={selectedTexts.length === 0}
        >
          Add to Description
        </button>
      </div>
    </div>
  );
};

/**
 * Ticked rows, in the order they were ticked: key -> text to add.
 * ``add`` hands the joined text to ``onAddToDescription`` and clears.
 */
function useSelection(onAddToDescription) {
  const [selected, setSelected] = useState(() => new Map());
  const toggle = useCallback((key, text) => {
    setSelected(prev => {
      const next = new Map(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.set(key, text);
      }
      return next;
    });
  }, []);
  const texts = Array.from(selected.values());
  const add = useCallback(() => {
    if (texts.length === 0) return;
    if (onAddToDescription) {
      onAddToDescription(texts.map(t => `${t};`).join(' ') + ' ');
    }
    setSelected(new Map());
  }, [texts, onAddToDescription]);
  return { selected, toggle, texts, add };
}

/** One attribute: its score, its values as checkboxes, "not mentioned". */
const AttributeGroup = ({ attribute, selected, onToggle }) => {
  const name = attribute.path[attribute.path.length - 1];
  return (
    <fieldset className="splits-attribute">
      <legend>
        <span className="splits-name">{name}</span>{' '}
        <span className="splits-score">
          separates ~{percent(attribute.score)}%
        </span>
      </legend>
      <div className="splits-values">
        {attribute.values.map(v => {
          const key = JSON.stringify([...attribute.path, v.value]);
          return (
            <label
              key={key}
              className="splits-value"
              title={`If yours is ${v.value}, ${percent(v.others)}% of these results differ`}
            >
              <input
                type="checkbox"
                aria-label={v.value}
                checked={selected.has(key)}
                onChange={() => onToggle(key, v.display_text)}
              />
              <span className="splits-name">{v.value}</span>{' '}
              <span className="splits-count">{v.count}</span>
            </label>
          );
        })}
        {attribute.not_mentioned > 0 && (
          <span className="splits-not-mentioned">
            (not mentioned) {attribute.not_mentioned}
          </span>
        )}
      </div>
    </fieldset>
  );
};

/** One part: a heading that collapses it, its attributes, its parts. */
const PartSection = ({ part, nResults, selected, onToggle }) => {
  const [expanded, setExpanded] = useState(true);
  const name = part.path[part.path.length - 1];
  const headingId = `splits-part-${part.path.join('-')}`
    .replace(/[^A-Za-z0-9_-]/g, '_');
  return (
    <section className="splits-part" role="region" aria-labelledby={headingId}>
      <button
        type="button"
        id={headingId}
        className="splits-part-heading"
        aria-expanded={expanded}
        onClick={() => setExpanded(e => !e)}
      >
        <span className="splits-name">{name}</span>{' '}
        <span className="splits-described">
          described in {part.described} of {nResults}
        </span>
      </button>
      {expanded && (
        <div className="splits-part-body">
          {part.attributes.map(a => (
            <AttributeGroup key={a.path.join('/')} attribute={a}
                            selected={selected} onToggle={onToggle} />
          ))}
          {part.parts.map(p => (
            <PartSection key={p.path.join('/')} part={p} nResults={nResults}
                         selected={selected} onToggle={onToggle} />
          ))}
        </div>
      )}
    </section>
  );
};

/**
 * @param {Object} props
 * @param {Object|null} props.data - the /api/features/structured/ response
 * @param {boolean} props.loading
 * @param {string|null} props.error
 * @param {Function} props.onAddToDescription - callback(text)
 */
export const StructuredSplitsPanel = ({
  data, loading, error, onAddToDescription,
}) => {
  const { selected, toggle, texts, add } = useSelection(onAddToDescription);
  return (
    <SplitsFrame data={data} loading={loading} error={error}
                 isEmpty={!data || !data.parts || data.parts.length === 0}
                 selectedTexts={texts} onAdd={add}
                 label="Parts and attributes, by how much they separate the results">
      {data && data.parts && data.parts.map(p => (
        <PartSection key={p.path.join('/')} part={p} nResults={data.n_results}
                     selected={selected} onToggle={toggle} />
      ))}
    </SplitsFrame>
  );
};

/**
 * @param {Object} props
 * @param {Object|null} props.data - the /api/features/text/ response
 * @param {boolean} props.loading
 * @param {string|null} props.error
 * @param {Function} props.onAddToDescription - callback(text)
 */
export const TextSplitsPanel = ({
  data, loading, error, onAddToDescription,
}) => {
  const { selected, toggle, texts, add } = useSelection(onAddToDescription);
  return (
    <SplitsFrame data={data} loading={loading} error={error}
                 isEmpty={!data || !data.features || data.features.length === 0}
                 selectedTexts={texts} onAdd={add}
                 label="Words and word pairs, by how much they separate the results">
      <div className="feature-list">
        {data && data.features && data.features.map(f => (
          <label key={f.text} className="feature-item splits-text">
            <input
              type="checkbox"
              aria-label={f.text}
              checked={selected.has(f.text)}
              onChange={() => toggle(f.text, f.text)}
            />
            <span className="feature-name">{f.text}</span>{' '}
            <span className="splits-count">in {f.count} of {data.n_results}</span>{' '}
            <span className="splits-score">separates ~{percent(f.score)}%</span>
          </label>
        ))}
      </div>
    </SplitsFrame>
  );
};
