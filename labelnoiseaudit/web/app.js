const state = {
  dataset: null,
  auditId: null,
  result: null,
  decisions: {},
  page: 0,
  pageSize: 8,
  filterClass: "all",
  filterStatus: "all",
  query: "",
  selected: null,
  embeddingsInstalled: false,
};

const $ = (id) => document.getElementById(id);

async function api(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function setStatus(message) {
  $("dataset-status").textContent = message || "";
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    document.querySelectorAll(".tab-panel").forEach((panel) => {
      panel.classList.toggle("hidden", panel.dataset.panel !== button.dataset.tab);
    });
  });
});

async function loadMeta() {
  const meta = await api("/api/meta");
  state.embeddingsInstalled = Boolean(meta.embeddings && meta.embeddings.installed);
  $("embed-hint").textContent = meta.embeddings.detail;
  const grid = $("builtin-grid");
  grid.replaceChildren();
  meta.builtins.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "builtin";
    const title = document.createElement("strong");
    title.textContent = item.title;
    const detail = document.createElement("span");
    detail.textContent = item.detail;
    button.append(title, detail);
    button.addEventListener("click", () => openBuiltin(item.id, button));
    grid.append(button);
  });
}

async function openBuiltin(name, button) {
  document.querySelectorAll(".builtin").forEach((item) => item.classList.remove("selected"));
  if (button) button.classList.add("selected");
  setStatus("Loading dataset…");
  try {
    const dataset = await api("/api/datasets/builtin", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    showDataset(dataset);
  } catch (error) {
    setStatus(error.message);
  }
}

function showDataset(dataset) {
  state.dataset = dataset;
  state.result = null;
  state.auditId = null;
  state.decisions = {};
  state.page = 0;
  state.selected = null;
  state.filterClass = "all";
  state.filterStatus = "all";
  state.query = "";
  const classFilter = $("filter-class");
  classFilter.replaceChildren();
  const allClasses = document.createElement("option");
  allClasses.value = "all";
  allClasses.textContent = "All classes";
  classFilter.append(allClasses);
  $("filter-status").value = "all";
  $("filter-query").value = "";
  $("export-btn").disabled = true;
  $("export-links").replaceChildren();
  const labelSelect = $("label-column");
  labelSelect.replaceChildren();
  dataset.columns.forEach((column) => {
    const option = document.createElement("option");
    option.value = column.name;
    option.textContent = column.name;
    if (column.name === dataset.suggested_label) option.selected = true;
    labelSelect.append(option);
  });
  $("column-picker").classList.remove("hidden");
  $("embed-row").classList.toggle("hidden", dataset.kind !== "images");
  $("use-embeddings").disabled = !state.embeddingsInstalled;
  $("allow-download").disabled = !state.embeddingsInstalled;
  renderFeatures();
  renderReview();
  setStatus(`${dataset.n_rows} rows ready.`);
}

function renderFeatures() {
  const dataset = state.dataset;
  const list = $("feature-list");
  list.replaceChildren();
  if (!dataset || dataset.kind === "images") {
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = dataset && dataset.kind === "images"
      ? "Image features default to color histograms, thumbnails, and HOG on CPU."
      : "";
    list.append(note);
    return;
  }
  const label = $("label-column").value;
  const suggested = new Set(dataset.suggested_features || []);
  dataset.columns.forEach((column) => {
    if (column.name === label) return;
    const wrapper = document.createElement("label");
    wrapper.className = "feature";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = column.name;
    box.checked = suggested.size ? suggested.has(column.name) : true;
    box.dataset.kind = column.kind;
    const name = document.createElement("span");
    name.textContent = column.name;
    const kind = document.createElement("span");
    kind.className = "kind";
    kind.textContent = column.kind;
    wrapper.append(box, name, kind);
    list.append(wrapper);
  });
}

$("label-column").addEventListener("change", renderFeatures);

$("csv-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  setStatus("Reading CSV…");
  try {
    showDataset(await api("/api/datasets/csv", { method: "POST", body }));
  } catch (error) {
    setStatus(error.message);
  }
});

$("open-csv-path").addEventListener("click", async () => {
  setStatus("Reading CSV…");
  try {
    showDataset(await api("/api/datasets/csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: $("csv-path").value }),
    }));
  } catch (error) {
    setStatus(error.message);
  }
});

$("zip-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  setStatus("Reading images…");
  try {
    showDataset(await api("/api/datasets/images", { method: "POST", body }));
  } catch (error) {
    setStatus(error.message);
  }
});

$("open-image-path").addEventListener("click", async () => {
  setStatus("Reading images…");
  try {
    showDataset(await api("/api/datasets/images", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: $("image-path").value }),
    }));
  } catch (error) {
    setStatus(error.message);
  }
});

$("run-audit").addEventListener("click", async () => {
  if (!state.dataset) return;
  const features = [...document.querySelectorAll("#feature-list input[type=checkbox]:checked")].map((box) => box.value);
  const kinds = { numeric: [], categorical: [], text: [] };
  document.querySelectorAll("#feature-list input[type=checkbox]:checked").forEach((box) => {
    kinds[box.dataset.kind].push(box.value);
  });
  $("run-audit").disabled = true;
  setStatus("Starting audit…");
  try {
    const started = await api("/api/audits", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: state.dataset.dataset_id,
        label_column: $("label-column").value,
        feature_columns: state.dataset.kind === "images" ? undefined : features,
        column_kinds: state.dataset.kind === "images" ? undefined : kinds,
        n_splits: Number($("folds").value),
        seed: Number($("seed").value),
        embeddings: $("use-embeddings").checked,
        allow_download: $("allow-download").checked,
      }),
    });
    state.auditId = started.audit_id;
    await pollAudit();
  } catch (error) {
    setStatus(error.message);
  } finally {
    $("run-audit").disabled = false;
  }
});

async function pollAudit() {
  while (state.auditId) {
    const payload = await api(`/api/audits/${state.auditId}`);
    if (payload.status === "running") {
      setStatus(payload.progress || "Running…");
      await new Promise((resolve) => setTimeout(resolve, 400));
      continue;
    }
    if (payload.status === "error") {
      setStatus(payload.error || "Audit failed.");
      return;
    }
    state.result = payload.result;
    state.decisions = payload.decisions || {};
    state.page = 0;
    state.selected = state.result.rows[0] ? state.result.rows[0].source_index : null;
    $("export-btn").disabled = false;
    fillFilters();
    renderReview();
    setStatus(`Finished. ${state.result.summary.n_flagged} rows marked Check.`);
    return;
  }
}

function fillFilters() {
  const select = $("filter-class");
  const current = select.value || "all";
  select.replaceChildren();
  const all = document.createElement("option");
  all.value = "all";
  all.textContent = "All classes";
  select.append(all);
  (state.result.class_names || []).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.append(option);
  });
  select.value = [...select.options].some((option) => option.value === current) ? current : "all";
}

function filteredRows() {
  if (!state.result) return [];
  return state.result.rows.filter((row) => {
    if (state.filterClass !== "all" && row.given_label !== state.filterClass) return false;
    if (state.filterStatus !== "all" && row.status !== state.filterStatus) return false;
    if (state.query && !String(row.row_id).toLowerCase().includes(state.query)) return false;
    return true;
  });
}

function renderReview() {
  const body = $("review-body");
  body.replaceChildren();
  const rows = filteredRows();
  const pages = Math.max(1, Math.ceil(rows.length / state.pageSize));
  if (state.page >= pages) state.page = pages - 1;
  const slice = rows.slice(state.page * state.pageSize, (state.page + 1) * state.pageSize);
  if (!state.result) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 9;
    td.className = "empty";
    td.textContent = "Open a dataset and run an audit.";
    tr.append(td);
    body.append(tr);
  } else if (!slice.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 9;
    td.className = "empty";
    td.textContent = "No rows match these filters.";
    tr.append(td);
    body.append(tr);
  } else {
    slice.forEach((row) => body.append(renderRow(row)));
  }
  $("page-label").textContent = state.result ? `Page ${state.page + 1} of ${pages}` : "Page 1";
  renderBars();
  renderNoise();
  renderGrid();
  renderReasons();
}

function renderRow(row) {
  const tr = document.createElement("tr");
  tr.className = row.status === "Check" ? "check" : "clean";
  if (row.source_index === state.selected) tr.classList.add("selected");
  tr.append(cell(String(row.rank)));
  const imageCell = document.createElement("td");
  if (state.result.thumbnail && state.result.thumbnail !== "none") {
    const img = document.createElement("img");
    img.className = "thumb";
    img.alt = "";
    img.src = `/api/audits/${state.auditId}/thumb/${row.source_index}`;
    imageCell.append(img);
  } else {
    imageCell.textContent = "—";
  }
  tr.append(imageCell);
  tr.append(cell(row.row_id));
  tr.append(cell(row.given_label));
  tr.append(cell(row.predicted_label));
  tr.append(cell(row.confidence.toFixed(2)));
  const suspicion = document.createElement("td");
  const bar = document.createElement("div");
  bar.className = "bar";
  const fill = document.createElement("span");
  fill.style.width = `${Math.max(0, Math.min(1, row.ensemble_score)) * 100}%`;
  bar.append(fill);
  const number = document.createElement("div");
  number.textContent = row.ensemble_score.toFixed(2);
  suspicion.append(bar, number);
  tr.append(suspicion);
  const status = document.createElement("td");
  const badge = document.createElement("span");
  badge.className = `badge ${row.status === "Check" ? "check" : "clean"}`;
  badge.textContent = row.status;
  status.append(badge);
  tr.append(status);
  tr.append(decisionCell(row));
  tr.addEventListener("click", (event) => {
    if (event.target.closest("button, select, input")) return;
    state.selected = row.source_index;
    renderReview();
  });
  return tr;
}

function cell(text) {
  const td = document.createElement("td");
  td.textContent = text;
  return td;
}

function decisionCell(row) {
  const td = document.createElement("td");
  const wrap = document.createElement("div");
  wrap.className = "decision";
  const current = state.decisions[String(row.source_index)] || { action: "keep" };
  ["keep", "relabel", "drop"].forEach((action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = action[0].toUpperCase() + action.slice(1);
    button.setAttribute("aria-pressed", String(current.action === action));
    button.addEventListener("click", () => chooseDecision(row, action));
    wrap.append(button);
  });
  if (current.action === "relabel") {
    const select = document.createElement("select");
    state.result.class_names.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.append(option);
    });
    const other = document.createElement("option");
    other.value = "__other__";
    other.textContent = "Other…";
    select.append(other);
    const chosen = current.new_label || row.predicted_label;
    if ([...select.options].some((option) => option.value === chosen)) {
      select.value = chosen;
    } else {
      select.value = "__other__";
    }
    select.addEventListener("change", () => {
      if (select.value === "__other__") {
        const typed = window.prompt("New label");
        if (typed && typed.trim()) chooseDecision(row, "relabel", typed.trim());
        return;
      }
      chooseDecision(row, "relabel", select.value);
    });
    wrap.append(select);
  }
  td.append(wrap);
  return td;
}

async function chooseDecision(row, action, newLabel) {
  const body = { source_index: row.source_index, action };
  if (action === "relabel") body.new_label = newLabel || row.predicted_label;
  try {
    const payload = await api(`/api/audits/${state.auditId}/decisions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.decisions = payload.decisions;
    renderReview();
  } catch (error) {
    setStatus(error.message);
  }
}

function renderBars() {
  const list = $("suspicion-bars");
  list.replaceChildren();
  if (!state.result) {
    const item = document.createElement("li");
    item.className = "empty-bars";
    item.textContent = "Run an audit to rank rows.";
    list.append(item);
    return;
  }
  state.result.rows.slice(0, 5).forEach((row) => {
    const item = document.createElement("li");
    const rank = document.createElement("strong");
    rank.textContent = String(row.rank);
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = row.row_id;
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = row.ensemble_score.toFixed(2);
    const track = document.createElement("div");
    track.className = "bar track";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(4, Math.min(1, row.ensemble_score) * 100)}%`;
    track.append(fill);
    item.append(rank, name, score, track);
    list.append(item);
  });
}

function renderNoise() {
  const list = $("noise-list");
  list.replaceChildren();
  if (!state.result) {
    const item = document.createElement("li");
    item.className = "hint";
    item.textContent = "Estimated from the confident joint after the audit.";
    list.append(item);
    return;
  }
  Object.entries(state.result.noise_rates).forEach(([name, rate]) => {
    const item = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = name;
    const value = document.createElement("strong");
    value.textContent = `${Math.round(rate * 100)}%`;
    item.append(label, value);
    list.append(item);
  });
}

function renderGrid() {
  const card = $("grid-card");
  const grid = $("image-grid");
  if (!state.result || state.result.thumbnail === "none") {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");
  grid.replaceChildren();
  state.result.rows.slice(0, 8).forEach((row) => {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = `tile${row.status === "Check" ? " bad" : ""}`;
    const img = document.createElement("img");
    img.alt = "";
    img.src = `/api/audits/${state.auditId}/thumb/${row.source_index}`;
    const caption = document.createElement("div");
    const mark = document.createElement("span");
    mark.className = `mark ${row.status === "Check" ? "bad" : "ok"}`;
    mark.textContent = row.status === "Check" ? "×" : "✓";
    caption.append(mark, document.createTextNode(` ${row.given_label}`));
    tile.append(img, caption);
    tile.addEventListener("click", () => {
      state.selected = row.source_index;
      renderReview();
    });
    grid.append(tile);
  });
}

function renderReasons() {
  const panel = $("reason-panel");
  if (!state.result || state.selected == null) {
    panel.classList.add("hidden");
    return;
  }
  const row = state.result.rows.find((item) => item.source_index === state.selected);
  if (!row) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  panel.replaceChildren();
  const title = document.createElement("strong");
  title.textContent = `Why row ${row.row_id} is ranked ${row.rank}`;
  const list = document.createElement("ul");
  row.reasons.forEach((reason) => {
    const item = document.createElement("li");
    item.textContent = reason;
    list.append(item);
  });
  panel.append(title, list);
}

$("filter-class").addEventListener("change", (event) => {
  state.filterClass = event.target.value;
  state.page = 0;
  renderReview();
});
$("filter-status").addEventListener("change", (event) => {
  state.filterStatus = event.target.value;
  state.page = 0;
  renderReview();
});
$("filter-query").addEventListener("input", (event) => {
  state.query = event.target.value.trim().toLowerCase();
  state.page = 0;
  renderReview();
});
$("prev-page").addEventListener("click", () => {
  state.page = Math.max(0, state.page - 1);
  renderReview();
});
$("next-page").addEventListener("click", () => {
  state.page += 1;
  renderReview();
});

$("export-btn").addEventListener("click", async () => {
  if (!state.auditId) return;
  setStatus("Writing a new CSV…");
  try {
    const payload = await api(`/api/audits/${state.auditId}/export`, { method: "POST" });
    const box = $("export-links");
    box.replaceChildren();
    const line = document.createElement("p");
    line.textContent = `Exported ${payload.n_out} rows (${payload.n_relabeled} relabeled, ${payload.n_dropped} dropped).`;
    const csv = document.createElement("a");
    csv.href = payload.cleaned_url;
    csv.textContent = "Download cleaned CSV";
    const log = document.createElement("a");
    log.href = payload.log_url;
    log.textContent = "Download audit log";
    box.append(line, csv, document.createTextNode(" · "), log);
    setStatus("Export written. The original file was not changed.");
  } catch (error) {
    setStatus(error.message);
  }
});

loadMeta().catch((error) => setStatus(error.message));
