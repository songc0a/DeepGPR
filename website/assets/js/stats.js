(function () {
  "use strict";

  const formatNumber = (value) => {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return new Intl.NumberFormat("en-US").format(value);
  };

  const formatText = (value) => {
    if (typeof value !== "string" || value.trim() === "") return "—";
    return value;
  };

  const readPath = (data, path) =>
    path.split(".").reduce((current, key) => {
      if (current === null || current === undefined) return undefined;
      return current[key];
    }, data);

  const applyStats = (data) => {
    document.querySelectorAll("[data-stat]").forEach((element) => {
      const value = readPath(data, element.dataset.stat);
      element.textContent = element.dataset.format === "text"
        ? formatText(value)
        : formatNumber(value);
    });

    const generated = document.querySelector("[data-stats-generated]");
    if (generated && typeof data.generated_at === "string") {
      const timestamp = new Date(data.generated_at);
      if (!Number.isNaN(timestamp.getTime())) {
        generated.textContent = `Snapshot generated ${timestamp.toLocaleDateString("en-US", {
          year: "numeric",
          month: "short",
          day: "numeric",
        })}`;
      }
    }
  };

  fetch("data/stats.json", {
    headers: { Accept: "application/json" },
    cache: "no-store",
  })
    .then((response) => {
      if (!response.ok) throw new Error(`Stats request returned ${response.status}`);
      return response.json();
    })
    .then(applyStats)
    .catch(() => {
      const generated = document.querySelector("[data-stats-generated]");
      if (generated) generated.textContent = "Statistics currently unavailable";
    });
})();
