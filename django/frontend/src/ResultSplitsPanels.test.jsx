/**
 * Tests for the Structured Splits and Text Splits panels: plan F3 of
 * docs/plans/result-splitting-features.md.
 *
 * test.failing marks tests that must fail until F3 is implemented, as
 * pytest's strict xfail does: a test.failing that passes is an error.
 */
import { fireEvent, render, screen, within } from '@testing-library/react';
import { StructuredSplitsPanel, TextSplitsPanel } from './ResultSplitsPanels';

// The plan's worked example, in the /api/features/structured/ shape.
const STRUCTURED = {
  n_results: 200,
  n_missing: 0,
  n_fallback: 0,
  canonical_version: { synonyms: 'a', spellings: 'b', rules: '1' },
  parts: [
    {
      path: ['Ascomata'],
      described: 180,
      score: 0.598,
      attributes: [
        {
          path: ['Ascomata', 'Color'],
          score: 0.598,
          not_mentioned: 20,
          values: [
            { value: 'Brown', count: 100, others: 0.5,
              display_text: 'Ascomata color brown' },
            { value: 'Yellow', count: 80, others: 0.6,
              display_text: 'Ascomata color yellow' },
            { value: 'Blue', count: 10, others: 0.95,
              display_text: 'Ascomata color blue' },
          ],
        },
      ],
      parts: [
        {
          path: ['Ascomata', 'Peridium'],
          described: 120,
          score: 0.31,
          attributes: [
            {
              path: ['Ascomata', 'Peridium', 'Texture'],
              score: 0.31,
              not_mentioned: 80,
              values: [
                { value: 'smooth', count: 120, others: 0.4,
                  display_text: 'Ascomata peridium texture smooth' },
              ],
            },
          ],
          parts: [],
        },
      ],
    },
    {
      path: ['Asci'],
      described: 20,
      score: 0.18,
      attributes: [
        {
          path: ['Asci', 'Ring'],
          score: 0.18,
          not_mentioned: 180,
          values: [
            { value: 'amyloid', count: 20, others: 0.9,
              display_text: 'Asci ring amyloid' },
          ],
        },
      ],
      parts: [],
    },
  ],
};

const TEXT = {
  n_results: 200,
  n_missing: 0,
  features: [
    { text: 'clamp connections', count: 98, score: 0.4998 },
    { text: 'spores', count: 200, score: 0 },
  ],
};

function structured(props = {}) {
  const onAdd = jest.fn();
  render(<StructuredSplitsPanel data={STRUCTURED} loading={false}
                                error={null} onAddToDescription={onAdd}
                                {...props} />);
  return onAdd;
}

function text(props = {}) {
  const onAdd = jest.fn();
  render(<TextSplitsPanel data={TEXT} loading={false} error={null}
                          onAddToDescription={onAdd} {...props} />);
  return onAdd;
}

describe.each([
  ['StructuredSplitsPanel', structured],
  ['TextSplitsPanel', text],
])('%s states', (_, show) => {
  test.failing('says when it is loading', () => {
    show({ data: null, loading: true });
    expect(screen.getByRole('status')).toHaveTextContent(/loading/i);
  });

  test.failing('shows an error', () => {
    show({ data: null, error: 'couchdb down' });
    expect(screen.getByRole('alert')).toHaveTextContent('couchdb down');
  });

  test.failing('asks for a search before there are results', () => {
    show({ data: null });
    expect(screen.getByRole('status')).toHaveTextContent(/run a search/i);
  });

  test.failing('says when nothing separates the results', () => {
    show({ data: { n_results: 3, parts: [], features: [] } });
    expect(screen.getByRole('status'))
      .toHaveTextContent(/no features separate these 3 results/i);
  });

  test.failing('says how many results it describes', () => {
    show();
    expect(screen.getByText(/200 results/)).toBeInTheDocument();
  });

  test.failing('the Add button waits for a selection', () => {
    show();
    expect(screen.getByRole('button', { name: /add to description/i }))
      .toBeDisabled();
  });
});

describe('StructuredSplitsPanel', () => {
  test.failing('shows each part with how many results describe it', () => {
    structured();
    const part = screen.getByRole('button', { name: /^Ascomata/ });
    expect(part).toHaveTextContent('described in 180 of 200');
    expect(part).toHaveAttribute('aria-expanded', 'true');
  });

  test.failing('parts come in the order the server ranked them', () => {
    structured();
    const parts = screen.getAllByRole('button', { name: /described in/ })
      .map(b => b.textContent);
    expect(parts[0]).toMatch(/^Ascomata/);
    expect(parts[1]).toMatch(/^Peridium/);
    expect(parts[2]).toMatch(/^Asci/);
  });

  test.failing('shows each attribute with how much it separates', () => {
    structured();
    const colour = screen.getByRole('group', { name: /^Color/ });
    expect(colour).toHaveTextContent('separates ~60%');
  });

  test.failing('a value is a checkbox with its count', () => {
    structured();
    const colour = screen.getByRole('group', { name: /^Color/ });
    const brown = within(colour).getByRole('checkbox', { name: /Brown/ });
    expect(brown.closest('label')).toHaveTextContent('100');
    expect(brown.closest('label')).toHaveAttribute(
      'title', 'If yours is Brown, 50% of these results differ');
  });

  test.failing('"not mentioned" is shown with its count, not offered', () => {
    structured();
    const colour = screen.getByRole('group', { name: /^Color/ });
    expect(colour).toHaveTextContent('(not mentioned) 20');
    expect(within(colour).queryByRole('checkbox', { name: /not mentioned/ }))
      .toBeNull();
  });

  test.failing('a nested part sits inside its parent', () => {
    structured();
    const ascomata = screen.getByRole('region', { name: /^Ascomata/ });
    expect(within(ascomata).getByRole('button', { name: /^Peridium/ }))
      .toBeInTheDocument();
  });

  test.failing('a part collapses and expands', () => {
    structured();
    const part = screen.getByRole('button', { name: /^Ascomata/ });
    fireEvent.click(part);
    expect(part).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('checkbox', { name: /Brown/ })).toBeNull();
    fireEvent.click(part);
    expect(screen.getByRole('checkbox', { name: /Brown/ }))
      .toBeInTheDocument();
  });

  test.failing('ticked values are previewed and added', () => {
    const onAdd = structured();
    fireEvent.click(screen.getByRole('checkbox', { name: /Brown/ }));
    fireEvent.click(screen.getByRole('checkbox', { name: /Blue/ }));
    expect(screen.getByText(
      'Ascomata color brown; Ascomata color blue;')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /add to description/i }));
    expect(onAdd).toHaveBeenCalledWith(
      'Ascomata color brown; Ascomata color blue; ');
    expect(screen.getByRole('checkbox', { name: /Brown/ })).not.toBeChecked();
  });
});

describe('TextSplitsPanel', () => {
  test.failing('a feature is a checkbox with its share of results', () => {
    text();
    const row = screen.getByRole('checkbox', { name: 'clamp connections' })
      .closest('label');
    expect(row).toHaveTextContent('in 98 of 200');
    expect(row).toHaveTextContent('separates ~50%');
  });

  test.failing('features come in the order the server ranked them', () => {
    text();
    expect(screen.getAllByRole('checkbox').map(c => c.getAttribute(
      'aria-label'))).toEqual(['clamp connections', 'spores']);
  });

  test.failing('ticked features are previewed and added', () => {
    const onAdd = text();
    fireEvent.click(screen.getByRole('checkbox', { name: 'clamp connections' }));
    expect(screen.getByText('clamp connections;')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /add to description/i }));
    expect(onAdd).toHaveBeenCalledWith('clamp connections; ');
  });
});
