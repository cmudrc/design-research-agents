(() => {
  const scriptElement =
    document.currentScript instanceof HTMLScriptElement
      ? document.currentScript
      : Array.from(document.scripts).find((candidate) =>
          candidate.src.includes("color_mode_favicon.js"),
        );

  if (!(scriptElement instanceof HTMLScriptElement) || !scriptElement.src) {
    return;
  }

  const lightHref = new URL("favicon-light.ico", scriptElement.src).toString();
  const darkHref = new URL("favicon-dark.ico", scriptElement.src).toString();
  const colorScheme = window.matchMedia("(prefers-color-scheme: dark)");
  const iconSelector = "link[rel='icon'], link[rel='shortcut icon']";
  const linkId = "drc-color-mode-favicon";

  function resolveMode() {
    const mode = document.documentElement.dataset.theme;
    if (mode === "light" || mode === "dark") {
      return mode;
    }
    return colorScheme.matches ? "dark" : "light";
  }

  function ensureManagedLink() {
    const existing = document.getElementById(linkId);
    if (existing instanceof HTMLLinkElement) {
      return existing;
    }
    const link = document.createElement("link");
    link.id = linkId;
    link.rel = "icon";
    link.type = "image/x-icon";
    document.head.append(link);
    return link;
  }

  function applyFavicon() {
    const href = resolveMode() === "dark" ? darkHref : lightHref;
    const links = new Set(
      Array.from(document.head.querySelectorAll(iconSelector)).filter(
        (element) => element instanceof HTMLLinkElement,
      ),
    );
    links.add(ensureManagedLink());
    for (const link of links) {
      link.href = href;
      link.type = "image/x-icon";
    }
  }

  function handleSystemModeChange() {
    const mode = document.documentElement.dataset.theme;
    if (!mode || mode === "auto") {
      applyFavicon();
    }
  }

  const observer = new MutationObserver(applyFavicon);

  function start() {
    applyFavicon();
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }

  if (typeof colorScheme.addEventListener === "function") {
    colorScheme.addEventListener("change", handleSystemModeChange);
  } else if (typeof colorScheme.addListener === "function") {
    colorScheme.addListener(handleSystemModeChange);
  }
})();
