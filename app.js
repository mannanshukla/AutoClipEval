/* --------------------------------------------------
   Autoclip RLHF front-end  – 100 % client-side
   -------------------------------------------------- */

/* ---------- CONFIG ---------- */
const R2_BASE =
  "https://pub-987b418b9ea54a468a9079def11423e1.r2.dev";

/* ---------- STATE ---------- */
let clips  = [];          // loaded from clips.json
let idx    = 0;           // current clip pointer
let scores = [];          // gathered results

/* ---------- BOOT ---------- */
const root = document.getElementById("appRoot");

fetch("clips.json")
  .then(r => r.json())
  .then(data => { clips = data; render(); })
  .catch(err => root.innerHTML =
    `<p class="text-red-600">Failed to load clips.json<br>${err}</p>`);

/* ---------- RENDER ---------- */
function render() {
  root.innerHTML = "";

  /* ---- FINISHED ---- */
  if (idx >= clips.length) {
    const blob = new Blob([JSON.stringify(scores, null, 2)],
                          { type: "application/json" });
    const url  = URL.createObjectURL(blob);

    root.innerHTML = `
      <div class="bg-white rounded-xl shadow p-10 text-center space-y-6">
        <h1 class="text-3xl font-semibold">🎉 Finished!</h1>
        <a href="${url}" download="scores.json"
           class="inline-flex items-center gap-2 rounded-lg px-5 py-3 bg-green-600 text-white hover:bg-green-700 transition">
          Download scores.json
        </a>
      </div>`;
    return;
  }

  const clip  = clips[idx];
  const total = clips.length;
  const count = `${idx + 1} / ${total}`;

  /* URLs */
  const safeId   = encodeURIComponent(clip.id);
  const videoUrl = `${R2_BASE}/${safeId}/${safeId}.mp4`;

  /* template */
  root.innerHTML = `
    <div class="mb-6 flex items-baseline justify-between">
      <h1 class="text-2xl font-bold">Autoclip RLHF Scorer</h1>
      <span class="text-sm text-zinc-500">Clip ${count}</span>
    </div>

    <div class="grid md:grid-cols-2 gap-6">
      <div class="bg-white rounded-xl shadow overflow-hidden">
        <video src="${videoUrl}" controls
               class="w-full aspect-video bg-black"></video>
      </div>

      <div class="space-y-6">
        <article
          class="prose prose-sm max-h-64 overflow-y-auto bg-white rounded-xl shadow p-5"
        >${clip.script}</article>

        <form id="rubricForm" class="space-y-4 bg-white rounded-xl shadow p-5">
          ${rubricBlock("hook",           "Is there a <strong>hook</strong>?")}
          ${rubricBlock("oneClaim",       "Exactly <strong>one</strong> claim?")}
          ${rubricBlock("selfSufficient", "Is it a <strong>self-sufficient</strong> quote?")}

          <button
            id="nextBtn"
            type="submit"
            disabled
            class="w-full mt-4 py-2 rounded-lg bg-blue-600 text-white opacity-50 cursor-not-allowed transition">
            Save & Next
          </button>
        </form>
      </div>
    </div>
  `;

  /* form logic */
  const form = document.getElementById("rubricForm");
  const next = document.getElementById("nextBtn");

  form.addEventListener("input", validate);
  form.addEventListener("change", validate);

  function validate() {
    const ok = ["hook","oneClaim","selfSufficient"].every(name => {
      const chosen = form.querySelector(`input[name='${name}']:checked`);
      const whyVal = form[`${name}Why`].value.trim();
      return chosen && whyVal.length > 0;
    });
    next.disabled = !ok;
    next.classList.toggle("opacity-50", !ok);
    next.classList.toggle("cursor-not-allowed", !ok);
  }

  form.addEventListener("submit", e => {
    e.preventDefault();

    scores.push({
      id: clip.id,
      rubric: {
        hook: form.hook.value === "yes",
        hookWhy: form.hookWhy.value.trim(),
        oneClaim: form.oneClaim.value === "yes",
        oneClaimWhy: form.oneClaimWhy.value.trim(),
        selfSufficient: form.selfSufficient.value === "yes",
        selfSufficientWhy: form.selfSufficientWhy.value.trim()
      }
    });

    idx++;
    render();
  });
}

/* helper: radios + textarea */
function rubricBlock(name, labelHtml) {
  return `
    <fieldset class="space-y-2">
      <legend class="mb-1">${labelHtml}</legend>
      <div class="flex items-center gap-6">
        ${["yes","no"].map(v => `
          <label class="inline-flex items-center gap-1">
            <input type="radio" name="${name}" value="${v}"
                   class="h-4 w-4 text-blue-600 border-zinc-300 focus:ring-blue-500">
            <span class="text-sm">${v[0].toUpperCase() + v.slice(1)}</span>
          </label>`).join("")}
      </div>
      <textarea name="${name}Why" rows="2" required
                placeholder="Why? (required)"
                class="mt-1 w-full rounded-md border-zinc-300 focus:border-blue-500 focus:ring-blue-500"></textarea>
    </fieldset>`;
}
