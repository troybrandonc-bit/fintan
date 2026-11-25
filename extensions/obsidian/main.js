// Teach Fintan (Obsidian plugin): your notes, learnt into your own Fintan model. Notes go only to the Fintan address
// you set (your own machine by default), and only when you run a command.
const { Plugin, PluginSettingTab, Setting, Notice, Modal, requestUrl } = require("obsidian");

const DEFAULTS = { url: "http://127.0.0.1:8900", key: "", kind: "documents", skipFolders: "" };

class AskModal extends Modal {
  constructor(app, plugin) { super(app); this.plugin = plugin; }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h3", { text: "Ask Fintan" });
    const input = contentEl.createEl("textarea", { attr: { rows: 3, style: "width:100%" } });
    const out = contentEl.createEl("div", { attr: { style: "margin-top:10px; white-space:pre-wrap" } });
    const btn = contentEl.createEl("button", { text: "Ask", attr: { style: "margin-top:8px" } });
    btn.onclick = async () => {
      out.setText("Thinking...");
      try {
        const r = await this.plugin.call("/v1/chat/completions", { model: "fintan", max_tokens: 500,
          messages: [{ role: "user", content: input.value }], chat_template_kwargs: { enable_thinking: false } });
        out.setText(((r.choices || [{}])[0].message || {}).content || "(no answer)");
      } catch (e) { out.setText("Could not reach your Fintan: " + e.message); }
    };
  }
  onClose() { this.contentEl.empty(); }
}

module.exports = class TeachFintan extends Plugin {
  async onload() {
    this.settings = Object.assign({}, DEFAULTS, await this.loadData());
    this.addSettingTab(new FintanSettings(this.app, this));
    this.addCommand({ id: "teach-note", name: "Teach Fintan this note", callback: () => this.teachActive() });
    this.addCommand({ id: "teach-vault", name: "Teach Fintan the whole vault", callback: () => this.teachVault() });
    this.addCommand({ id: "learn-now", name: "Start learning now", callback: () => this.learnNow() });
    this.addCommand({ id: "ask", name: "Ask Fintan", callback: () => new AskModal(this.app, this).open() });
    this.addRibbonIcon("graduation-cap", "Teach Fintan this note", () => this.teachActive());
  }

  async call(path, body) {
    const headers = { "Content-Type": "application/json" };
    if (this.settings.key) headers["Authorization"] = "Bearer " + this.settings.key;
    const r = await requestUrl({ url: this.settings.url.replace(/\/+$/, "") + path, method: body ? "POST" : "GET",
      headers, body: body ? JSON.stringify(body) : undefined, throw: false });
    if (r.status === 401) throw new Error("your Fintan needs its API key (plugin settings)");
    return r.json;
  }

  async teachFile(file) {
    const text = (await this.app.vault.cachedRead(file)).trim();
    if (text.length < 40) return { skipped: true };
    return this.call("/fintan/teach", { text: file.basename + ". " + text, source: file.path, kind: this.settings.kind });
  }

  async teachActive() {
    const file = this.app.workspace.getActiveFile();
    if (!file) return new Notice("Open a note first.");
    try {
      const r = await this.teachFile(file);
      if (r.skipped) return new Notice("This note is too short to learn from.");
      new Notice(r.accepted ? (r.kept_as_note ? "Kept as a note in Fintan." : "Fintan will learn this note at its next learning run.")
                            : "Not accepted: " + (r.reason || ""));
    } catch (e) { new Notice("Could not reach your Fintan: " + e.message); }
  }

  async teachVault() {
    const skip = this.settings.skipFolders.split(",").map((s) => s.trim()).filter(Boolean);
    const files = this.app.vault.getMarkdownFiles().filter((f) => !skip.some((p) => f.path.startsWith(p)));
    let ok = 0, failed = 0;
    new Notice("Teaching Fintan " + files.length + " notes...");
    for (const f of files) {
      try { const r = await this.teachFile(f); if (r.accepted) ok++; } catch (e) { failed++; }
    }
    new Notice("Fintan got " + ok + " notes" + (failed ? " (" + failed + " could not be sent)" : "") + ". They are learnt at its next learning run.");
  }

  async learnNow() {
    try {
      const r = await this.call("/fintan/learn", {});
      new Notice(r.started ? "Fintan is learning. It answers again when the run ends." : "Not started: " + (r.reason || ""));
    } catch (e) { new Notice("Could not reach your Fintan: " + e.message); }
  }
};

class FintanSettings extends PluginSettingTab {
  constructor(app, plugin) { super(app, plugin); this.plugin = plugin; }
  display() {
    const el = this.containerEl;
    el.empty();
    const save = () => this.plugin.saveData(this.plugin.settings);
    new Setting(el).setName("Fintan's address").setDesc("Where your Fintan runs (default: this computer).")
      .addText((t) => t.setValue(this.plugin.settings.url).onChange((v) => { this.plugin.settings.url = v; save(); }));
    new Setting(el).setName("API key").setDesc("Only if your Fintan has one.")
      .addText((t) => { t.inputEl.type = "password"; t.setValue(this.plugin.settings.key).onChange((v) => { this.plugin.settings.key = v; save(); }); });
    new Setting(el).setName("Learn notes as").addDropdown((d) => d
      .addOptions({ documents: "Documents", personal: "Facts about me", news: "News" })
      .setValue(this.plugin.settings.kind).onChange((v) => { this.plugin.settings.kind = v; save(); }));
    new Setting(el).setName("Folders to skip").setDesc("Comma separated, e.g. Private, Journal")
      .addText((t) => t.setValue(this.plugin.settings.skipFolders).onChange((v) => { this.plugin.settings.skipFolders = v; save(); }));
  }
}
