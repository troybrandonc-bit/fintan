// Fintan for Pinokio: an open AI model that keeps learning on your own computer.
// Linux with an NVIDIA GPU (24 GB for Fintan-9B, 16 GB or more for Fintan-4B).
module.exports = {
  version: "3.0",
  title: "Fintan",
  description: "An open AI model that keeps learning on your own GPU: teach it your documents, code, topics or facts about you, and it learns them without forgetting what it knew. OpenAI-compatible API on port 8900.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    const installed = info.exists("app/manifest.json") && info.exists("env")
    const running = info.running("start.js")
    if (!installed) {
      return [{ default: true, icon: "fa-solid fa-plug", text: "Install", href: "install.js" }]
    }
    if (running) {
      const local = info.local("start.js")
      return [
        local && local.url ? { default: true, icon: "fa-solid fa-rocket", text: "API: " + local.url, href: local.url + "/fintan/status" } : null,
        { icon: "fa-solid fa-terminal", text: "Terminal", href: "start.js" },
      ].filter(Boolean)
    }
    return [
      { default: true, icon: "fa-solid fa-power-off", text: "Start", href: "start.js" },
      { icon: "fa-solid fa-plug", text: "Reinstall", href: "install.js" },
    ]
  }
}
