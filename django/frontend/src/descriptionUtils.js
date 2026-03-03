/**
 * Shared utility for inserting text into the description textarea.
 *
 * All "Add to Description" buttons should use this so that semicolon
 * separators are handled consistently.
 */

/**
 * Insert text into a description textarea at the cursor position,
 * automatically prepending "; " if the preceding text doesn't
 * already end with a semicolon.
 *
 * @param {HTMLTextAreaElement} textarea - The target textarea element
 * @param {string} text - The text to insert
 */
export function insertIntoDescription(textarea, text) {
  const start = textarea.selectionStart;
  const end = textarea.selectionEnd;
  const value = textarea.value;

  const before = value.substring(0, start);
  const separator = before.length > 0 && !/;\s*$/.test(before) ? '; ' : '';
  const insert = separator + text;

  textarea.value = value.substring(0, start) + insert + value.substring(end);
  textarea.selectionStart = textarea.selectionEnd = start + insert.length;
  textarea.focus();

  const event = new Event('input', { bubbles: true });
  textarea.dispatchEvent(event);
}
