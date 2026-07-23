import { Wllama, LoggerWithoutDebug } from '../wllama/wllama.min.js';

// ————— constants —————

// Each model ships with the system prompt it expects. v4 was TRAINED with its
// prompt verbatim, so it must match exactly. v3 uses the lighter "play along"
// prompt we tuned to tame its identity-blurb reflex.
const MODELS = {
  'parag-v4.1-0.6B': {
    label: 'Parag v4',
    tag: '0.6B',
    ctx: 4096,
    system:
      'You are Parag, a friendly little AI made by Chaitanya (cxaiiii). ' +
      'Chat naturally, play along with fun, keep answers matched to the question, ' +
      'and always finish your sentences.',
    greeting:
      "Namaste! 🙏 I'm Parag v4 — Chaitanya's newest little AI, running entirely in " +
      'your browser. Ask me anything, or just have some fun with me.',
  },
  'parag-v3-0.5B': {
    label: 'Parag v3',
    tag: '0.5B',
    ctx: 4096,
    system:
      'You are Parag, a friendly and playful little AI made by Chaitanya (cxaiiii). ' +
      'Just chat naturally and keep it fun — happily play along with jokes, hypotheticals, ' +
      'and out-of-the-blue questions instead of dodging them. Always give a direct, genuine ' +
      'reply to whatever the user actually says. Only mention who made you or what you are ' +
      'if the user specifically asks about it — never bring your own identity up otherwise. ' +
      'Keep answers short and to the point, and never repeat yourself. If you truly do not ' +
      'know a fact, say so in one short sentence and move on.',
    greeting:
      "Namaste! 🙏 I'm Parag v3 (0.5B), the older model — kept here so you can compare. " +
      'Nothing you say leaves this device.',
  },
};
const DEFAULT_MODEL = 'parag-v4.1-0.6B';
let currentModelId = DEFAULT_MODEL;
const currentSystem = () => MODELS[currentModelId].system;

// The GGUF has no embedded template and its EOS is the base model's
// <|endoftext|>, so we format ChatML ourselves and stop on <|im_end|>.
function buildPrompt(context) {
  let p = '';
  for (const m of context) p += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  return p + '<|im_start|>assistant\n';
}

const MODEL_DIR = './model/';
const MAX_HISTORY = 8; // messages (excl. system) kept in context

// ————— elements —————

const $ = (id) => document.getElementById(id);
const landing = $('landing');
const chatView = $('chat');
const btnLoad = $('btn-load');
const progressWrap = $('progress-wrap');
const progressBar = $('progress-bar');
const progressLabel = $('progress-label');
const progressPct = $('progress-pct');
const loadError = $('load-error');
const messagesEl = $('messages');
const composer = $('composer');
const input = $('input');
const btnSend = $('btn-send');
const btnStop = $('btn-stop');
const statLine = $('stat-line');
const suggestions = $('suggestions');

// ————— state —————

let wllama = null;
let history = [];
let generating = false;

// ————— model download (chunked, same-origin, cached) —————

async function getChunk(url, cache, onBytes) {
  const cached = await cache.match(url);
  if (cached) {
    const blob = await cached.blob();
    onBytes(blob.size);
    return blob;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed (${res.status}) for ${url}`);
  const reader = res.body.getReader();
  const parts = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    onBytes(value.byteLength);
  }
  const blob = new Blob(parts);
  await cache.put(url, new Response(blob)).catch(() => {}); // cache is best-effort
  return blob;
}

const DOWNLOAD_CONCURRENCY = 6;     // parallel chunk fetches

// Download a model's chunks (cached per-model) and assemble one Blob.
// `onProgress(received, total)` is called as bytes stream in.
async function downloadModel(modelId, onProgress) {
  const manifestUrl = MODEL_DIR + 'manifest-' + modelId + '.json';
  const CACHE_NAME = 'parag-model-v2';

  const manifest = await (await fetch(manifestUrl)).json();
  const cache = await caches.open(CACHE_NAME).catch(() => null);
  const total = manifest.totalSize;
  let received = 0;

  const onBytes = (n) => {
    received += n;
    onProgress(received, total);
  };

  // Fetch chunks with a bounded concurrency pool. The old code awaited each
  // chunk before starting the next, so every network round-trip serialized and
  // the download crawled. A pool keeps several fetches in flight at once; the
  // ordered `blobs` array is filled by index so assembly stays correct.
  const chunks = manifest.chunks;
  const blobs = new Array(chunks.length);
  let cursor = 0;

  async function worker() {
    while (true) {
      const i = cursor++;
      if (i >= chunks.length) return;
      const url = MODEL_DIR + chunks[i].file;
      blobs[i] = cache
        ? await getChunk(url, cache, onBytes)
        : await (await fetch(url)).blob().then((b) => (onBytes(b.size), b));
    }
  }

  const pool = Math.min(DOWNLOAD_CONCURRENCY, chunks.length);
  await Promise.all(Array.from({ length: pool }, worker));
  return new Blob(blobs);
}

// ————— model loading (shared by first boot + switching) —————

let switching = false;

// Download + load a model into a fresh Wllama instance. Callbacks let the
// caller drive whatever progress UI it wants (big landing bar vs. the compact
// switcher). Returns the ready Wllama.
async function loadModelInto(modelId, onProgress, onLoadingStage) {
  const blob = await downloadModel(modelId, onProgress);
  onLoadingStage();
  const w = new Wllama(
    { default: './wllama/wllama.wasm' },
    { suppressNativeLog: true, logger: LoggerWithoutDebug }
  );
  w.setCompat('default'); // only kicks in on browsers that need it (Safari)
  await w.loadModel([blob], { n_ctx: MODELS[modelId].ctx });
  return w;
}

function engineLabel() {
  // WebGPU (wllama 3.1+) runs the GGUF on the GPU — 45-69% faster decode.
  const gpu = wllama.isSupportWebGPU();
  console.log('[parag] WebGPU supported:', gpu,
    '| multithread:', wllama.isMultithread(), '| threads:', wllama.getNumThreads());
  return gpu ? 'GPU' : (wllama.isMultithread() ? `${wllama.getNumThreads()} threads` : 'single thread');
}

// ————— boot (first load, from the landing page) —————

btnLoad.addEventListener('click', async () => {
  btnLoad.disabled = true;
  progressWrap.hidden = false;
  loadError.hidden = true;
  progressBar.style.width = '2%';
  progressPct.textContent = '0%';
  progressLabel.textContent = 'Starting download…';
  try {
    wllama = await loadModelInto(
      currentModelId,
      (received, total) => {
        const pct = Math.min(100, (received / total) * 100);
        progressBar.style.width = pct.toFixed(1) + '%';
        progressPct.textContent = pct.toFixed(0) + '%';
        progressLabel.textContent =
          `Downloading ${MODELS[currentModelId].label} — ${(received / 1048576).toFixed(0)} / ${(total / 1048576).toFixed(0)} MB`;
      },
      () => {
        // loading into wasm — no download events for 10-30s, so animate.
        progressLabel.textContent = 'Waking Parag up (loading into memory)…';
        progressPct.textContent = '';
        progressBar.classList.add('indeterminate');
      }
    );
    window.__wllama = wllama; // debugging hook
    statLine.textContent = `local · ${engineLabel()} · ready`;

    progressBar.classList.remove('indeterminate');
    landing.hidden = true;
    chatView.hidden = false;
    renderSwitcher();
    addParagMessage(MODELS[currentModelId].greeting);
    input.focus();
    
    // Optional: Try loading an external system prompt for v4 so it can be edited without code
    try {
      const spRes = await fetch('./system_prompt.txt');
      if (spRes.ok) {
        const text = await spRes.text();
        if (text.trim()) {
          MODELS['parag-v4.1-0.6B'].system = text.trim();
          console.log('[parag] Loaded external system_prompt.txt');
        }
      }
    } catch(e) { /* ignore if not found */ }
    
  } catch (err) {
    console.error(err);
    loadError.textContent =
      'Something went wrong while loading Parag: ' + (err?.message || err) +
      ' — please refresh and try again.';
    loadError.hidden = false;
    btnLoad.disabled = false;
    progressBar.classList.remove('indeterminate');
  }
});

// ————— model switcher —————

async function switchModel(modelId) {
  if (modelId === currentModelId || generating || switching) return;
  switching = true;
  btnSend.disabled = true;
  input.disabled = true;
  renderSwitcher(modelId); // mark target as loading

  try {
    // Free the current model before loading the next (two 380 MB models won't
    // both fit in wasm memory). wllama.exit() releases the worker + weights.
    if (wllama) { await wllama.exit().catch(() => {}); wllama = null; }

    wllama = await loadModelInto(
      modelId,
      (received, total) => {
        statLine.textContent = `loading ${MODELS[modelId].label} · ${Math.round((received / total) * 100)}%`;
      },
      () => { statLine.textContent = `waking ${MODELS[modelId].label}…`; }
    );
    window.__wllama = wllama;
    currentModelId = modelId;
    history = []; // fresh context — the new model shouldn't inherit the old one's turns

    addDivider(`switched to ${MODELS[modelId].label} · ${MODELS[modelId].tag}`);
    addParagMessage(MODELS[modelId].greeting);
    statLine.textContent = `local · ${engineLabel()} · ready`;
  } catch (err) {
    console.error(err);
    statLine.textContent = 'switch failed — please refresh';
  } finally {
    switching = false;
    btnSend.disabled = false;
    input.disabled = false;
    renderSwitcher();
    input.focus();
  }
}

// Paint the switcher buttons: active model highlighted, target shows "loading".
function renderSwitcher(loadingId = null) {
  const wrap = $('model-switch');
  if (!wrap) return;
  wrap.querySelectorAll('.ms-btn').forEach((btn) => {
    const id = btn.dataset.model;
    btn.classList.toggle('active', id === currentModelId && !loadingId);
    btn.classList.toggle('loading', id === loadingId);
    btn.disabled = switching || generating;
  });
}

document.getElementById('model-switch')?.addEventListener('click', (e) => {
  const btn = e.target.closest('.ms-btn');
  if (btn) switchModel(btn.dataset.model);
});

// ————— chat rendering —————

function addUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg msg-user';
  div.textContent = text;
  messagesEl.appendChild(div);
  scrollToEnd();
}

function addParagMessage(text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg-parag';
  wrap.innerHTML = '<div class="avatar">🪷</div><div class="bubble"></div>';
  const bubble = wrap.querySelector('.bubble');
  bubble.textContent = text;
  messagesEl.appendChild(wrap);
  scrollToEnd();
  return bubble;
}

function addDivider(text) {
  const div = document.createElement('div');
  div.className = 'msg-divider';
  div.innerHTML = `<span>${text}</span>`;
  messagesEl.appendChild(div);
  scrollToEnd();
}

function scrollToEnd() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ————— generation —————

// Cuts the reply at a genuine next-turn marker only. Anchored to the start of a
// line and requires the ChatML role tag or a role word immediately followed by a
// newline/colon at line-start — so ordinary prose like "the assistant said" or a
// line ending before "User guide" is NOT chopped mid-sentence.
const TURN_CUT = /<\|im_(?:end|start)\|>|^\s*(?:user|assistant|system)\s*[:\n]/im;

let genId = 0;          // increments per generation; stale callbacks are ignored
let cancelRequested = false; // set by the Stop button (soft-stop)

async function generate(userText) {
  const myGen = ++genId;
  cancelRequested = false;
  generating = true;
  btnSend.disabled = true;
  btnStop.hidden = false;
  suggestions.style.display = 'none';
  renderSwitcher();

  history.push({ role: 'user', content: userText });
  addUserMessage(userText);

  const bubble = addParagMessage('');
  bubble.classList.add('thinking');

  const context = [
    { role: 'system', content: currentSystem() },
    ...history.slice(-MAX_HISTORY),
  ];

  let text = '';       // canonical accumulated reply (never read back from DOM)
  let rawText = '';    // debug only: every delta, ignoring TURN_CUT / frozen
  let tokens = 0;
  let frozen = false;  // once true, we stop appending (cut-marker or user stop)
  const t0 = performance.now();

  const promptStr = buildPrompt(context);
  // Set window.__paragDebug = true in the console to see, per turn, the exact
  // prompt sent and the RAW completion before TURN_CUT chops it. If the raw text
  // contains a "\nUser:" / "\nassistant:" style header, the mid-sentence cutoff
  // is TURN_CUT slicing a hallucinated turn boundary — i.e. the model derailed,
  // not the plumbing.
  if (window.__paragDebug) console.log('[parag] PROMPT >>>\n' + promptStr);

  let result = null;
  try {
    // NOTE: we deliberately do NOT abort. wllama's abort is JS-only — it stops
    // the result loop but leaves the C++ generation session live, and the NEXT
    // createCompletion then resumes that stale session, leaking its tokens into
    // the new reply. Instead we let the session wind down naturally (bounded by
    // max_tokens) while keeping input disabled, so a new generation can never
    // start mid-session. `frozen` just stops us displaying the wind-down tail.
    result = await wllama.createCompletion({
      prompt: promptStr,
      stream: true,
      max_tokens: 2048,
      temp: 0.7,
      // A 0.5B loops hard without penalties (it repeated its identity blurb
      // ~20x live). penalty_repeat curbs verbatim token repeats; the freq +
      // presence penalties are what actually break "same sentence over and
      // over" degeneration without flattening ordinary prose. Slightly higher
      // temp also adds enough variety to keep it from falling into the loop.
      penalty_repeat: 1.1,
      penalty_freq: 0.0,
      penalty_present: 0.0,
      penalty_last_n: 64,
      cache_prompt: false,
      // No `stop` strings: <|im_end|> is a native EOG token so generation stops
      // on it anyway, and string-stops make wllama hold back a lookahead buffer
      // (~9 chars) that isn't flushed at end-of-turn and bleeds into the NEXT
      // reply's opening (the "zone."/"for clarity." leak). TURN_CUT below still
      // guards against the model emitting a literal role marker as text.
      onData: (chunk) => {
        if (myGen !== genId) return;
        const delta = chunk?.choices?.[0]?.text;
        if (window.__paragDebug && typeof delta === 'string') rawText += delta;
        if (frozen) return;
        if (cancelRequested) { frozen = true; return; }
        if (!delta) return;
        text += delta;
        tokens += 1;
        const m = text.match(TURN_CUT);
        if (m) { text = text.slice(0, m.index); frozen = true; return; }
        bubble.textContent = text;
        scrollToEnd();
      },
    });

    // If we never froze, adopt wllama's final full string when it's longer than
    // what streamed (last token can be held in the C++ batch buffer).
    if (!frozen && myGen === genId && result) {
      const full = result?.choices?.[0]?.text;
      if (typeof full === 'string' && full.length > text.length) text = full;
      const m = text.match(TURN_CUT);
      if (m) text = text.slice(0, m.index);
    }
  } catch (err) {
    if (!text) bubble.textContent = '(Sorry, something went wrong: ' + (err?.message || err) + ')';
    console.error(err);
  }

  if (window.__paragDebug) {
    console.log('[parag] RAW completion (pre-TURN_CUT) >>>\n' + rawText);
    console.log('[parag] displayed (post-cut) >>>\n' + text);
    console.log('[parag] frozen=' + frozen + ' tokens=' + tokens);
  }

  bubble.classList.remove('thinking');
  // The 0.5B sometimes emits a stray leading "." (or similar) as its first
  // token — a cosmetic model quirk, not a leak (rawText is per-turn). Strip a
  // run of leading punctuation so ".Love…" renders as "Love…".
  const finalText = text.trim().replace(/^[.,;:!?]+\s*/, '');
  if (finalText) {
    bubble.textContent = finalText;
    history.push({ role: 'assistant', content: finalText });
  } else {
    bubble.textContent = '…';
  }

  const secs = (performance.now() - t0) / 1000;
  if (tokens > 0) {
    const mode = wllama.isMultithread() ? `${wllama.getNumThreads()} threads` : 'single thread';
    statLine.textContent = `local · ${mode} · ${(tokens / secs).toFixed(1)} tok/s`;
  }

  generating = false;
  btnSend.disabled = false;
  btnStop.hidden = true;
  renderSwitcher();
  input.focus();
}

// ————— composer wiring —————

composer.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || generating || !wllama) return;
  input.value = '';
  autosize();
  generate(text);
});

btnStop.addEventListener('click', () => {
  // Soft-stop: freeze the visible reply immediately. The engine winds the
  // session down on its own (bounded by max_tokens); input re-enables once it
  // has fully settled, so the next message can't collide with a live session.
  cancelRequested = true;
  btnStop.hidden = true;
  statLine.textContent = 'stopping…';
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

function autosize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
}
input.addEventListener('input', autosize);

suggestions.addEventListener('click', (e) => {
  const chip = e.target.closest('.chip');
  if (!chip || generating) return;
  input.value = chip.textContent;
  composer.requestSubmit();
});
