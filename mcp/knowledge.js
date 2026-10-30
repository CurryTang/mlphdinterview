import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..');
const NOTES_DIR = path.join(PROJECT_ROOT, 'notes');

let notesIndex = null;

/**
 * Scan all markdown files in the notes directory recursively.
 */
function scanDirectory(dir, fileList = []) {
  if (!fs.existsSync(dir)) return fileList;
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== 'node_modules' && entry.name !== '.git' && entry.name !== 'assets' && entry.name !== 'figures') {
        scanDirectory(fullPath, fileList);
      }
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      fileList.push(fullPath);
    }
  }
  return fileList;
}

/**
 * Extract title, headings, and clean excerpt from markdown.
 */
function parseNoteMetadata(filePath) {
  const relativePath = path.relative(PROJECT_ROOT, filePath).replace(/\\/g, '/');
  const content = fs.readFileSync(filePath, 'utf-8');
  const filename = path.basename(filePath);
  const parts = relativePath.split('/');
  const category = parts.length > 1 ? parts[1] : 'Other';
  const isEnglish = filename.endsWith('.en.md');
  const language = isEnglish ? 'en' : 'zh';

  // Extract first H1 heading
  const titleMatch = content.match(/^#\s+(.+)$/m);
  const title = titleMatch ? titleMatch[1].trim() : filename.replace(/\.(en\.)?md$/, '');

  // Extract H2 and H3 headings
  const headings = [];
  const headingRegex = /^(#{2,3})\s+(.+)$/gm;
  let match;
  while ((match = headingRegex.exec(content)) !== null) {
    headings.push({
      level: match[1].length,
      text: match[2].trim(),
    });
  }

  // Create clean snippet/excerpt
  const bodyWithoutHeaders = content
    .replace(/^#+.*$/gm, '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/\[\[.*?\]\]/g, '')
    .trim();
  const excerpt = bodyWithoutHeaders.slice(0, 300).replace(/\s+/g, ' ');

  return {
    relativePath,
    filename,
    category,
    language,
    title,
    headings,
    excerpt,
    size: content.length,
    fullContent: content,
  };
}

/**
 * Initialize or get cached index of all notes.
 */
export function getNotesIndex(forceRefresh = false) {
  if (notesIndex && !forceRefresh) {
    return notesIndex;
  }
  const files = scanDirectory(NOTES_DIR);
  const notes = files.map(parseNoteMetadata);
  
  // Categorize
  const byCategory = {};
  for (const note of notes) {
    if (!byCategory[note.category]) {
      byCategory[note.category] = [];
    }
    byCategory[note.category].push(note);
  }

  notesIndex = {
    all: notes,
    byCategory,
    totalCount: notes.length,
  };
  return notesIndex;
}

/**
 * Search notes by query keyword or phrase.
 */
export function searchNotes(query, options = {}) {
  const { category = 'all', language = 'all', limit = 5 } = options;
  const index = getNotesIndex();
  const q = String(query || '').toLowerCase().trim();
  if (!q) return [];

  const tokens = q.split(/\s+/).filter(Boolean);

  let candidates = index.all;
  if (category && category !== 'all') {
    candidates = candidates.filter((n) => n.category.toLowerCase() === category.toLowerCase());
  }
  if (language && language !== 'all') {
    candidates = candidates.filter((n) => n.language === language);
  }

  const scored = [];
  for (const note of candidates) {
    const titleLower = note.title.toLowerCase();
    const pathLower = note.relativePath.toLowerCase();
    const contentLower = note.fullContent.toLowerCase();

    let score = 0;
    // Exact title match bonus
    if (titleLower.includes(q)) score += 50;
    // Filename / path match bonus
    if (pathLower.includes(q)) score += 30;

    // Heading matches
    for (const h of note.headings) {
      if (h.text.toLowerCase().includes(q)) score += 15;
    }

    // Token frequency in content
    let allTokensPresent = true;
    for (const token of tokens) {
      if (contentLower.includes(token)) {
        score += 5;
      } else {
        allTokensPresent = false;
      }
    }
    if (allTokensPresent && tokens.length > 1) score += 20;

    if (score > 0) {
      // Find best matching snippet
      const pos = contentLower.indexOf(tokens[0]);
      let snippet = note.excerpt;
      if (pos >= 0) {
        const start = Math.max(0, pos - 80);
        const end = Math.min(note.fullContent.length, pos + 220);
        snippet = (start > 0 ? '...' : '') + note.fullContent.slice(start, end).replace(/\s+/g, ' ') + '...';
      }

      scored.push({
        path: note.relativePath,
        title: note.title,
        category: note.category,
        language: note.language,
        score,
        snippet,
        headings: note.headings.slice(0, 8).map((h) => h.text),
      });
    }
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.min(limit, 30));
}

/**
 * Retrieve note content by path or title, with optional section extraction.
 */
export function getNoteContent(notePathOrTitle, options = {}) {
  const { language, section, maxLength = 8000 } = options;
  const index = getNotesIndex();

  let target = index.all.find(
    (n) => n.relativePath.toLowerCase() === notePathOrTitle.toLowerCase() ||
           n.filename.toLowerCase() === notePathOrTitle.toLowerCase()
  );

  if (!target) {
    // Try matching by title or partial path
    const searchVal = notePathOrTitle.toLowerCase().trim();
    target = index.all.find((n) => n.title.toLowerCase().includes(searchVal) || n.relativePath.toLowerCase().includes(searchVal));
  }

  if (!target) {
    return {
      error: `Note not found for identifier: "${notePathOrTitle}". Use search_knowledge_base to find valid note paths.`,
    };
  }

  // If user requested a specific language variant (e.g. en or zh)
  if (language && target.language !== language) {
    let altPath = '';
    if (language === 'en' && !target.filename.endsWith('.en.md')) {
      altPath = target.relativePath.replace(/\.md$/, '.en.md');
    } else if (language === 'zh' && target.filename.endsWith('.en.md')) {
      altPath = target.relativePath.replace(/\.en\.md$/, '.md');
    }
    const altNote = index.all.find((n) => n.relativePath === altPath);
    if (altNote) target = altNote;
  }

  let content = target.fullContent;

  // Extract specific section if requested
  if (section) {
    const secLower = section.toLowerCase().trim();
    const headingLines = content.split('\n');
    let sectionStartIndex = -1;
    let sectionLevel = 2;

    for (let i = 0; i < headingLines.length; i++) {
      const line = headingLines[i];
      const m = line.match(/^(#{1,6})\s+(.+)$/);
      if (m && m[2].toLowerCase().includes(secLower)) {
        sectionStartIndex = i;
        sectionLevel = m[1].length;
        break;
      }
    }

    if (sectionStartIndex >= 0) {
      let sectionEndIndex = headingLines.length;
      for (let j = sectionStartIndex + 1; j < headingLines.length; j++) {
        const line = headingLines[j];
        const m = line.match(/^(#{1,6})\s+(.+)$/);
        if (m && m[1].length <= sectionLevel) {
          sectionEndIndex = j;
          break;
        }
      }
      content = headingLines.slice(sectionStartIndex, sectionEndIndex).join('\n');
    }
  }

  // Handle truncation if too long
  let isTruncated = false;
  if (content.length > maxLength) {
    content = content.slice(0, maxLength);
    isTruncated = true;
  }

  return {
    path: target.relativePath,
    title: target.title,
    category: target.category,
    language: target.language,
    headings: target.headings,
    content: content + (isTruncated ? `\n\n... [Content truncated at ${maxLength} chars. Request specific sections with the "section" argument.]` : ''),
    isTruncated,
  };
}

/**
 * List all knowledge categories and their note counts.
 */
export function listCategoriesAndNotes(categoryFilter = 'all') {
  const index = getNotesIndex();
  const result = {};

  for (const [category, notes] of Object.entries(index.byCategory)) {
    if (categoryFilter !== 'all' && category.toLowerCase() !== categoryFilter.toLowerCase()) {
      continue;
    }
    result[category] = notes
      .filter((n) => n.language === 'zh') // primary list
      .map((n) => ({
        path: n.relativePath,
        title: n.title,
        hasEnglish: index.all.some((en) => en.relativePath === n.relativePath.replace(/\.md$/, '.en.md')),
        headingsCount: n.headings.length,
      }));
  }

  return {
    totalNotes: index.totalCount,
    categories: Object.keys(index.byCategory),
    catalog: result,
  };
}
