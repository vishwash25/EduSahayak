/**
 * EduSahayak - Bloom's Taxonomy Alignment Checker & ShikshakAI Web Client
 */

let currentInputMethod = 'text';
let currentSelectedPdf = null;
let currentSelectedImages = [];
let lastAnalysisResponse = null;
let activeFilter = 'ALL';
let currentSearchQuery = '';

const SAMPLES = {
  cs_exam: `VISVESVARAYA TECHNOLOGICAL UNIVERSITY
DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
B.E. V SEMESTER EXAMINATION - 2025
SUBJECT: OBJECT ORIENTED PROGRAMMING & DESIGN
Time: 3 Hours                          Max Marks: 100

INSTRUCTIONS TO CANDIDATES:
1. Answer all questions from Part A and any four questions from Part B.
2. Assume suitable data wherever necessary.

PART A

Q1. What is Object-Oriented Programming? State four primary OOP principles. (4 Marks) [BL-1]

Q2.
a) Explain how polymorphism enables runtime method dispatch in Java. (6 Marks) [BL-2]
b) Apply binary search algorithm to find key 15 in the sorted array: [2, 5, 8, 12, 15, 23, 38]. (6 Marks) [BL-3]

Q3.
a) Analyze the time complexity of QuickSort in the worst-case and explain why it occurs. (8 Marks) [BL-4]
b) Evaluate whether multiple inheritance is suitable in the given software module and justify your decision. (8 Marks) [BL-5]

Q4. Design a scalable class hierarchy for an automated banking system supporting savings, checking, and loan accounts with ACID compliance. (12 Marks) [BL-6]

Page 1 of 2
*** END OF PART A ***`,

  mismatch_case: `MID SEMESTER EXAMINATION - 2025
DATABASE MANAGEMENT SYSTEMS

Q1. Define what is meant by a Primary Key in relational database design. (2 Marks) [BL-5]

Q2. List three differences between DELETE and TRUNCATE commands. (3 Marks) [BL-4]

Q3. Design a complete normalized relational schema (up to BCNF) for a hospital management system. (15 Marks) [BL-2]

Q4. Apply Dijkstra algorithm to find the shortest path from vertex A to all other vertices in the given weighted graph. (10 Marks) [BL-1]`,

  no_levels: `DEPARTMENT OF INFORMATION SCIENCE
DATA STRUCTURES & ALGORITHMS EXAMINATION

Q1. Explain the difference between linear search and binary search. (5 Marks)

Q2.
a) Define AVL tree and explain its rotation operations. (6 Marks)
b) Implement Dijkstra shortest path algorithm on a graph with 5 vertices. (8 Marks)

Q3. Design a hash table collision resolution mechanism using double hashing. (10 Marks)`
};

// ----------------- TAB SWITCHING -----------------

function switchMainTab(tab) {
  ['bloom', 'ocr', 'enhance', 'pdf'].forEach(t => {
    const view = document.getElementById(`view-${t}`);
    const navBtn = document.getElementById(`nav-${t}`);
    if (view) view.classList.toggle('hidden', t !== tab);
    if (navBtn) {
      if (t === tab) {
        navBtn.className = 'nav-tab px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition flex items-center space-x-2 bg-indigo-600 text-white shadow';
      } else {
        navBtn.className = 'nav-tab px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition flex items-center space-x-2 text-slate-400 hover:text-slate-200';
      }
    }
  });
  if (window.lucide) lucide.createIcons();
}

function switchInputMethod(method) {
  currentInputMethod = method;
  ['text', 'pdf', 'image'].forEach(m => {
    const pane = document.getElementById(`input-pane-${m}`);
    const btn = document.getElementById(`input-tab-${m}`);
    if (pane) pane.classList.toggle('hidden', m !== method);
    if (btn) {
      if (m === method) {
        btn.className = 'input-method-btn px-4 py-2 rounded-lg text-sm font-medium bg-indigo-600 text-white flex items-center space-x-2 transition';
      } else {
        btn.className = 'input-method-btn px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-slate-200 bg-slate-900 hover:bg-slate-850 flex items-center space-x-2 transition';
      }
    }
  });
  if (window.lucide) lucide.createIcons();
}

function loadSample(key) {
  switchInputMethod('text');
  const txtArea = document.getElementById('bloom-text-input');
  if (txtArea && SAMPLES[key]) {
    txtArea.value = SAMPLES[key];
    txtArea.focus();
  }
}

// ----------------- FILE UPLOADS -----------------

function handlePdfSelected(files) {
  if (!files || !files.length) return;
  const file = files[0];
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    alert('Please select a valid PDF file.');
    return;
  }
  currentSelectedPdf = file;
  document.getElementById('pdf-filename').textContent = file.name;
  document.getElementById('pdf-filesize').textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
  document.getElementById('pdf-file-info').classList.remove('hidden');
  document.getElementById('pdf-dropzone').classList.add('hidden');
  if (window.lucide) lucide.createIcons();
}

function clearSelectedPdf() {
  currentSelectedPdf = null;
  document.getElementById('pdf-file-input').value = '';
  document.getElementById('pdf-file-info').classList.add('hidden');
  document.getElementById('pdf-dropzone').classList.remove('hidden');
}

function handleImagesSelected(files) {
  if (!files || !files.length) return;
  for (let i = 0; i < files.length; i++) {
    currentSelectedImages.push(files[i]);
  }
  renderImagesPreview();
}

function renderImagesPreview() {
  const container = document.getElementById('images-preview-list');
  container.innerHTML = '';
  currentSelectedImages.forEach((img, idx) => {
    const card = document.createElement('div');
    card.className = 'bg-slate-900 border border-slate-800 p-2 rounded-xl text-center relative group';
    card.innerHTML = `
      <div class="h-20 bg-slate-950 rounded-lg flex items-center justify-center text-slate-500 mb-1 overflow-hidden">
        <i data-lucide="image" class="w-6 h-6"></i>
      </div>
      <p class="text-[11px] truncate text-slate-300 font-medium">${img.name}</p>
      <p class="text-[10px] text-slate-500">${(img.size / 1024).toFixed(0)} KB</p>
      <button onclick="removeImage(${idx})" class="absolute top-1 right-1 p-1 bg-rose-500/80 hover:bg-rose-600 text-white rounded-md text-xs">
        &times;
      </button>
    `;
    container.appendChild(card);
  });
  if (window.lucide) lucide.createIcons();
}

function removeImage(index) {
  currentSelectedImages.splice(index, 1);
  renderImagesPreview();
}

// ----------------- BLOOM'S ANALYSIS API TRIGGER -----------------

async function triggerBloomAnalysis() {
  const btn = document.getElementById('btn-analyze-bloom');
  const loading = document.getElementById('bloom-loading-state');
  const errorCard = document.getElementById('bloom-error-card');
  const resultsSec = document.getElementById('bloom-results-section');

  errorCard.classList.add('hidden');
  resultsSec.classList.add('hidden');

  let formData = null;
  let jsonBody = null;

  if (currentInputMethod === 'text') {
    const text = document.getElementById('bloom-text-input').value.trim();
    if (!text) {
      showError("Please provide a question paper before analyzing.");
      return;
    }
    jsonBody = { text: text };
  } else if (currentInputMethod === 'pdf') {
    if (!currentSelectedPdf) {
      showError("Please select a question paper PDF to upload.");
      return;
    }
    formData = new FormData();
    formData.append('file', currentSelectedPdf);
  } else if (currentInputMethod === 'image') {
    if (!currentSelectedImages.length) {
      showError("Please select at least one question paper image to upload.");
      return;
    }
    formData = new FormData();
    currentSelectedImages.forEach(img => {
      formData.append('files', img);
    });
  }

  // Show loading
  loading.classList.remove('hidden');
  btn.disabled = true;
  btn.classList.add('opacity-50', 'cursor-not-allowed');

  try {
    let response;
    if (jsonBody) {
      response = await fetch('/api/bloom/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jsonBody)
      });
    } else {
      response = await fetch('/api/bloom/analyze-upload', {
        method: 'POST',
        body: formData
      });
    }

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || data.message || "Failed to analyze question paper.");
    }

    lastAnalysisResponse = data;
    renderResults(data);

  } catch (err) {
    showError(err.message);
  } finally {
    loading.classList.add('hidden');
    btn.disabled = false;
    btn.classList.remove('opacity-50', 'cursor-not-allowed');
  }
}

function showError(msg) {
  const errorCard = document.getElementById('bloom-error-card');
  const errorMsg = document.getElementById('bloom-error-msg');
  errorMsg.textContent = msg;
  errorCard.classList.remove('hidden');
  if (window.lucide) lucide.createIcons();
}

// ----------------- RESULTS RENDERING -----------------

function renderResults(data) {
  const resultsSec = document.getElementById('bloom-results-section');
  resultsSec.classList.remove('hidden');

  // Summary counts
  document.getElementById('stat-total').textContent = data.summary.totalQuestions;
  document.getElementById('stat-matches').textContent = data.summary.matches;
  document.getElementById('stat-mismatches').textContent = data.summary.mismatches;
  document.getElementById('stat-unspecified').textContent = data.summary.notSpecified;
  document.getElementById('stat-ambiguous').textContent = data.summary.ambiguous;

  activeFilter = 'ALL';
  currentSearchQuery = '';
  document.getElementById('bloom-search-input').value = '';
  updateFilterButtons();
  populateTable();

  // Scroll to results
  resultsSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updateFilterButtons() {
  ['all', 'mismatch', 'match', 'unspecified', 'ambiguous'].forEach(f => {
    const btn = document.getElementById(`filter-btn-${f}`);
    if (!btn) return;
    const isTarget = (f === 'all' && activeFilter === 'ALL') ||
                     (f === 'mismatch' && activeFilter === 'MISMATCH') ||
                     (f === 'match' && activeFilter === 'MATCH') ||
                     (f === 'unspecified' && activeFilter === 'NOT_SPECIFIED') ||
                     (f === 'ambiguous' && activeFilter === 'AMBIGUOUS');
    if (isTarget) {
      btn.className = 'filter-btn px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white';
    } else {
      btn.className = 'filter-btn px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-900 hover:bg-slate-800 text-slate-400 border border-slate-700';
    }
  });
}

function setFilter(filter) {
  activeFilter = filter;
  updateFilterButtons();
  populateTable();
}

function handleSearch(val) {
  currentSearchQuery = (val || '').toLowerCase().trim();
  populateTable();
}

function populateTable() {
  if (!lastAnalysisResponse || !lastAnalysisResponse.questions) return;

  const tbody = document.getElementById('bloom-table-body');
  tbody.innerHTML = '';

  let filtered = lastAnalysisResponse.questions.filter(q => {
    if (activeFilter !== 'ALL' && q.status !== activeFilter) return false;
    if (currentSearchQuery) {
      const matchId = (q.fullIdentifier || '').toLowerCase().includes(currentSearchQuery);
      const matchText = (q.questionText || '').toLowerCase().includes(currentSearchQuery);
      const matchReason = (q.reason || '').toLowerCase().includes(currentSearchQuery);
      if (!matchId && !matchText && !matchReason) return false;
    }
    return true;
  });

  if (!filtered.length) {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td colspan="7" class="py-8 text-center text-slate-500 text-sm">
        No questions match the selected filter.
      </td>
    `;
    tbody.appendChild(row);
    return;
  }

  filtered.forEach(q => {
    const tr = document.createElement('tr');
    tr.className = 'table-row-hover border-b border-slate-800/40 cursor-pointer';
    tr.onclick = () => openDetailModal(q);

    // Indentation for subquestions
    const isSub = !!q.subQuestionNumber;
    const indentClass = isSub ? 'pl-8' : 'pl-4';

    // Status Badge
    let statusBadge = '';
    if (q.status === 'MATCH') {
      statusBadge = '<span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold badge-match"><i data-lucide="check" class="w-3 h-3 mr-1"></i> MATCH</span>';
    } else if (q.status === 'MISMATCH') {
      statusBadge = '<span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold badge-mismatch"><i data-lucide="alert-triangle" class="w-3 h-3 mr-1"></i> MISMATCH</span>';
    } else if (q.status === 'NOT_SPECIFIED') {
      statusBadge = '<span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium badge-unspecified">NOT SPECIFIED</span>';
    } else if (q.status === 'AMBIGUOUS') {
      statusBadge = '<span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold badge-ambiguous"><i data-lucide="help-circle" class="w-3 h-3 mr-1"></i> AMBIGUOUS</span>';
    }

    // Assigned BL Tag
    let assignedTag = '<span class="text-xs text-slate-500 italic">None</span>';
    if (q.assignedBloomLevel) {
      assignedTag = `<span class="inline-block px-2 py-0.5 rounded text-xs font-semibold level-tag-${q.assignedBloomLevel}">L${q.assignedBloomLevel} · ${q.assignedBloomName || ''}</span>`;
    }

    // AI BL Tag
    let aiTag = `<span class="inline-block px-2 py-0.5 rounded text-xs font-semibold level-tag-${q.aiBloomLevel}">L${q.aiBloomLevel} · ${q.aiBloomName}</span>`;

    tr.innerHTML = `
      <td class="py-3.5 ${indentClass} whitespace-nowrap">
        <div class="flex items-center space-x-1.5">
          ${isSub ? '<span class="text-slate-600 text-xs">↳</span>' : ''}
          <span class="font-bold ${isSub ? 'text-indigo-300 text-xs' : 'text-white text-sm'}">${q.fullIdentifier}</span>
          ${q.marks ? `<span class="text-[10px] text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded">${q.marks}M</span>` : ''}
        </div>
      </td>
      <td class="py-3.5 px-4 text-slate-200 text-xs sm:text-sm max-w-md">
        <p class="line-clamp-2 leading-relaxed">${escapeHtml(q.questionText)}</p>
      </td>
      <td class="py-3.5 px-4 text-center whitespace-nowrap">
        ${assignedTag}
      </td>
      <td class="py-3.5 px-4 text-center whitespace-nowrap">
        ${aiTag}
      </td>
      <td class="py-3.5 px-4 text-center whitespace-nowrap">
        ${statusBadge}
      </td>
      <td class="py-3.5 px-4 text-slate-300 text-xs leading-relaxed max-w-xs">
        <p class="line-clamp-2">${escapeHtml(q.reason)}</p>
      </td>
      <td class="py-3.5 px-4 text-center whitespace-nowrap">
        <button class="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-indigo-400 hover:text-indigo-300 transition" title="View Detailed Breakdown">
          <i data-lucide="eye" class="w-4 h-4"></i>
        </button>
      </td>
    `;

    tbody.appendChild(tr);
  });

  if (window.lucide) lucide.createIcons();
}

// ----------------- DETAIL MODAL -----------------

function openDetailModal(q) {
  document.getElementById('modal-q-id').textContent = q.fullIdentifier + (q.marks ? ` (${q.marks} Marks)` : '');
  document.getElementById('modal-question-text').textContent = q.questionText;

  // Status Badge
  const statusEl = document.getElementById('modal-status-badge');
  if (q.status === 'MATCH') {
    statusEl.innerHTML = '<span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold badge-match"><i data-lucide="check" class="w-3.5 h-3.5 mr-1"></i> MATCH</span>';
  } else if (q.status === 'MISMATCH') {
    statusEl.innerHTML = '<span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold badge-mismatch"><i data-lucide="alert-triangle" class="w-3.5 h-3.5 mr-1"></i> MISMATCH</span>';
  } else if (q.status === 'NOT_SPECIFIED') {
    statusEl.innerHTML = '<span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium badge-unspecified">NOT SPECIFIED</span>';
  } else {
    statusEl.innerHTML = '<span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold badge-ambiguous"><i data-lucide="help-circle" class="w-3.5 h-3.5 mr-1"></i> AMBIGUOUS</span>';
  }

  // Assigned
  const assignedEl = document.getElementById('modal-assigned-bl');
  const assignedDesc = document.getElementById('modal-assigned-desc');
  if (q.assignedBloomLevel) {
    assignedEl.innerHTML = `<span class="level-tag-${q.assignedBloomLevel} px-2.5 py-1 rounded-lg">Level ${q.assignedBloomLevel} — ${q.assignedBloomName}</span>`;
    assignedDesc.textContent = `Annotated explicitly in the question paper.`;
  } else {
    assignedEl.innerHTML = `<span class="text-slate-400 italic">Not Specified</span>`;
    assignedDesc.textContent = `No Bloom's Taxonomy level tag was present for this question.`;
  }

  // AI
  const aiEl = document.getElementById('modal-ai-bl');
  aiEl.innerHTML = `<span class="level-tag-${q.aiBloomLevel} px-2.5 py-1 rounded-lg">Level ${q.aiBloomLevel} — ${q.aiBloomName}</span>`;
  document.getElementById('modal-confidence').textContent = Math.round((q.confidence || 0.9) * 100) + '%';

  // Reason
  document.getElementById('modal-reason').textContent = q.reason;

  document.getElementById('bloom-detail-modal').classList.remove('hidden');
  if (window.lucide) lucide.createIcons();
}

function closeDetailModal() {
  document.getElementById('bloom-detail-modal').classList.add('hidden');
}

function exportResultsJson() {
  if (!lastAnalysisResponse) return;
  const jsonStr = JSON.stringify(lastAnalysisResponse, null, 2);
  navigator.clipboard.writeText(jsonStr).then(() => {
    alert('JSON analysis report copied to clipboard!');
  }).catch(err => {
    alert('Failed to copy: ' + err);
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

// ----------------- EXISTING FEATURES CLIENT LOGIC -----------------

async function runOcrComparison(file) {
  if (!file) return;
  const loading = document.getElementById('ocr-loading');
  const results = document.getElementById('ocr-results');
  loading.classList.remove('hidden');
  results.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/compare-ocr', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.status === 'success') {
      const p = data.results.paddleocr || {};
      const t = data.results.tesseract || {};
      document.getElementById('paddle-meta').textContent = `${p.time || 0}ms · ${p.confidence || 0}% conf`;
      document.getElementById('paddle-output').textContent = p.text || 'No text recognized.';
      document.getElementById('tesseract-meta').textContent = `${t.time || 0}ms · ${t.confidence || 0}% conf`;
      document.getElementById('tesseract-output').textContent = t.text || 'No text recognized.';
      results.classList.remove('hidden');
    } else {
      alert('OCR Failed: ' + data.message);
    }
  } catch (err) {
    alert('Error running OCR: ' + err.message);
  } finally {
    loading.classList.add('hidden');
  }
}

async function runTextEnhancement() {
  const txt = document.getElementById('enhance-input').value.trim();
  if (!txt) { alert('Please enter noisy text to enhance.'); return; }

  const loading = document.getElementById('enhance-loading');
  const outCard = document.getElementById('enhance-output-card');
  loading.classList.remove('hidden');
  outCard.classList.add('hidden');

  try {
    const res = await fetch('/api/enhance-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ocr_text: txt })
    });
    const data = await res.json();
    if (data.status === 'success') {
      document.getElementById('enhance-output').textContent = data.enhanced_text;
      outCard.classList.remove('hidden');
    } else {
      alert('Enhancement failed: ' + data.message);
    }
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    loading.classList.add('hidden');
  }
}

async function runPdfGeneration() {
  const txt = document.getElementById('pdf-gen-input').value.trim();
  if (!txt) { alert('Please enter text for PDF generation.'); return; }

  const statusEl = document.getElementById('pdf-gen-status');
  statusEl.className = 'p-4 rounded-xl text-sm bg-indigo-950/40 text-indigo-300 border border-indigo-800';
  statusEl.textContent = 'Generating PDF...';
  statusEl.classList.remove('hidden');

  try {
    const res = await fetch('/api/generate-pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: txt })
    });
    const data = await res.json();
    if (data.status === 'success') {
      statusEl.className = 'p-4 rounded-xl text-sm bg-emerald-950/40 text-emerald-300 border border-emerald-800';
      statusEl.textContent = `PDF Generated successfully! File: ${data.file}`;
    } else {
      statusEl.className = 'p-4 rounded-xl text-sm bg-rose-950/40 text-rose-300 border border-rose-800';
      statusEl.textContent = 'PDF generation failed: ' + data.message;
    }
  } catch (err) {
    statusEl.className = 'p-4 rounded-xl text-sm bg-rose-950/40 text-rose-300 border border-rose-800';
    statusEl.textContent = 'Error: ' + err.message;
  }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) lucide.createIcons();
});
