"use strict";

// Listing text is untrusted, so everything reaches the page through textContent.
const $ = (id) => document.getElementById(id);
const OPTION_KEYS = ["1", "2", "3", "4", "5"];
const KEY_HELP = {
  pick: [["1-5", "take a suggestion"], ["/", "search"], ["↑↓ Enter", "pick a result"], ["s", "can't tell, skip"], ["u", "undo last"]],
  refine: [["Enter", "save and next"], ["↑↓", "browse deeper"], ["→", "go deeper"], ["←", "up a level"], ["a", "attributes"], ["Esc", "back"]],
};

let state = null;
let node = null;
let cursor = -1;
let results = [];
let searchTimer = null;
let valueTimer = null;

async function api(path, body) {
  const options = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

function showError(error) {
  $("error").textContent = error ? String(error.message || error) : "";
}

function pathElement(path) {
  const parts = path.split(" > ");
  const wrapper = document.createElement("span");
  if (parts.length > 1) {
    const parent = document.createElement("span");
    parent.className = "path-parent";
    parent.textContent = parts.slice(0, -1).join(" > ") + " > ";
    wrapper.append(parent);
  }
  wrapper.append(parts[parts.length - 1]);
  return wrapper;
}

function renderOptions(list, paths, keys, onChoose, cursorIndex) {
  list.replaceChildren(...paths.map((path, index) => {
    const item = document.createElement("li");
    if (index === cursorIndex) item.className = "cursor";
    if (keys && keys[index]) {
      const key = document.createElement("span");
      key.className = "key";
      key.textContent = keys[index];
      item.append(key);
    }
    item.append(pathElement(path));
    item.addEventListener("click", () => onChoose(path));
    return item;
  }));
  const active = list.querySelector("li.cursor");
  if (active) active.scrollIntoView({ block: "nearest" });
}

function renderKeys(phase) {
  $("keys").replaceChildren(...KEY_HELP[phase].flatMap(([key, text]) => {
    const kbd = document.createElement("kbd");
    kbd.textContent = key;
    return [kbd, document.createTextNode(` ${text}   `)];
  }));
}

function renderProduct(item) {
  const image = $("image");
  if (item.image_url && item.image_url.startsWith("https://")) image.src = item.image_url;
  else image.removeAttribute("src");
  $("title").textContent = item.title;
  $("price").textContent = typeof item.price === "number" ? `$${item.price.toFixed(2)}` : (item.price || "No price listed");
  $("features").replaceChildren(...(item.features || []).slice(0, 8).map((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    return li;
  }));
  $("description").textContent = item.description || "";
  $("details").replaceChildren(...Object.entries(item.details || {}).flatMap(([key, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
    return [dt, dd];
  }));
}

function render() {
  const { labeled, skipped, target, item } = state;
  $("progress").textContent = `${labeled} of ${target} labeled, ${skipped} skipped`;
  $("bar-fill").style.width = `${Math.min(100, (100 * labeled) / target)}%`;
  $("main").hidden = !item;
  $("done").hidden = Boolean(item);
  if (!item) {
    $("done-text").textContent = `${labeled} products are labeled.`;
    $("keys").replaceChildren();
    return;
  }
  renderProduct(item);
  showPick();
}

function showPick() {
  node = null;
  cursor = -1;
  results = [];
  $("refine").hidden = true;
  $("pick").hidden = false;
  $("search").value = "";
  $("results").replaceChildren();
  renderOptions($("suggestions"), state.item.suggestions.slice(0, 5), OPTION_KEYS, choose);
  renderKeys("pick");
  showError(null);
  document.activeElement.blur();
}

async function choose(path) {
  try {
    const previous = node ? readAttributes() : {};
    node = await api(`/api/node?path=${encodeURIComponent(path)}&asin=${encodeURIComponent(state.item.parent_asin)}`);
    cursor = -1;
    showRefine(previous);
  } catch (error) {
    showError(error);
  }
}

function showRefine(previous) {
  $("pick").hidden = true;
  $("refine").hidden = false;
  $("chosen").replaceChildren(pathElement(node.path));
  $("parent").hidden = !node.parent;
  $("children-title").hidden = node.children.length === 0;
  renderChildren();
  renderAttributes(previous);
  renderKeys("refine");
  showError(null);
  document.activeElement.blur();
}

function renderChildren() {
  renderOptions($("children"), node.children, null, choose, cursor);
  // The chosen category is shown above, so each row only needs its own name.
  for (const row of $("children").querySelectorAll(".path-parent")) row.remove();
}

function renderAttributes(previous) {
  $("attributes").replaceChildren(...node.attributes.map((attribute) => {
    const row = document.createElement("div");
    row.className = "attribute";
    const id = `attr-${attribute.handle}`;
    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = attribute.name;
    const input = document.createElement("input");
    input.id = id;
    input.dataset.handle = attribute.handle;
    input.autocomplete = "off";
    input.setAttribute("list", "value-options");
    input.value = previous[attribute.handle] || attribute.prefill || "";
    input.addEventListener("input", () => suggestValues(input));
    input.addEventListener("focus", () => suggestValues(input));
    row.append(label, input);
    return row;
  }));
}

function suggestValues(input) {
  clearTimeout(valueTimer);
  const path = node.path;
  valueTimer = setTimeout(async () => {
    if (!node || node.path !== path) return;
    const query = new URLSearchParams({ path, handle: input.dataset.handle, q: input.value });
    const values = await api(`/api/values?${query}`);
    $("value-options").replaceChildren(...values.map((value) => {
      const option = document.createElement("option");
      option.value = value;
      return option;
    }));
  }, 80);
}

function readAttributes() {
  const values = {};
  for (const input of $("attributes").querySelectorAll("input")) {
    if (input.value.trim()) values[input.dataset.handle] = input.value.trim();
  }
  return values;
}

async function save() {
  try {
    state = await api("/api/label", { asin: state.item.parent_asin, path: node.path, attributes: readAttributes() });
    render();
  } catch (error) {
    showError(error);
  }
}

async function skip() {
  try {
    state = await api("/api/skip", { asin: state.item.parent_asin });
    render();
  } catch (error) {
    showError(error);
  }
}

async function undo() {
  try {
    state = await api("/api/undo", {});
    render();
  } catch (error) {
    showError(error);
  }
}

function moveCursor(delta, count) {
  if (count === 0) return false;
  cursor = cursor < 0 && delta < 0 ? count - 1 : (cursor + delta + count) % count;
  return true;
}

function renderResults() {
  renderOptions($("results"), results, null, choose, cursor);
}

$("search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    try {
      results = await api(`/api/search?q=${encodeURIComponent($("search").value)}`);
      cursor = results.length ? 0 : -1;
      renderResults();
    } catch (error) {
      showError(error);
    }
  }, 60);
});

$("search").addEventListener("keydown", (event) => {
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    if (moveCursor(event.key === "ArrowDown" ? 1 : -1, results.length)) renderResults();
  } else if (event.key === "Enter") {
    event.preventDefault();
    event.stopPropagation();
    if (results[cursor]) choose(results[cursor]);
  } else if (event.key === "Escape") {
    event.stopPropagation();
    showPick();
  }
});

$("parent").addEventListener("click", () => node.parent && choose(node.parent));
$("save").addEventListener("click", save);
$("done-undo").addEventListener("click", undo);

function isTyping() {
  const tag = document.activeElement && document.activeElement.tagName;
  return tag === "INPUT" || tag === "TEXTAREA";
}

document.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (isTyping()) {
    if (event.key === "Escape") {
      document.activeElement.blur();
      event.preventDefault();
    } else if (event.key === "Enter" && !$("refine").hidden) {
      event.preventDefault();
      save();
    }
    return;
  }
  if (!state || !state.item) {
    if (event.key === "u") undo();
    return;
  }
  if (!$("pick").hidden) {
    const index = OPTION_KEYS.indexOf(event.key);
    if (index >= 0 && state.item.suggestions[index]) choose(state.item.suggestions[index]);
    else if (event.key === "/") { event.preventDefault(); $("search").focus(); }
    else if (event.key === "s") skip();
    else if (event.key === "u") undo();
    else return;
    event.preventDefault();
    return;
  }
  if (event.key === "Enter") {
    if (cursor >= 0 && node.children[cursor]) choose(node.children[cursor]);
    else save();
  } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    if (moveCursor(event.key === "ArrowDown" ? 1 : -1, node.children.length)) renderChildren();
  } else if (event.key === "ArrowRight") {
    const next = node.children[Math.max(cursor, 0)];
    if (next) choose(next);
  } else if (event.key === "ArrowLeft" || event.key === "Backspace") {
    if (node.parent) choose(node.parent);
  } else if (event.key === "a") {
    const first = $("attributes").querySelector("input");
    if (first) first.focus();
  } else if (event.key === "Escape") {
    showPick();
  } else {
    return;
  }
  event.preventDefault();
});

api("/api/state").then((payload) => {
  state = payload;
  render();
}).catch(showError);
