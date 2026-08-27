(function () {
  "use strict";

  const header = document.querySelector(".site-header");
  const toggle = document.querySelector(".nav-toggle");
  const navigation = document.querySelector(".nav-links");

  const updateHeader = () => {
    if (header) {
      header.classList.toggle("is-scrolled", window.scrollY > 12);
    }
  };

  updateHeader();
  window.addEventListener("scroll", updateHeader, { passive: true });

  if (toggle && navigation) {
    toggle.addEventListener("click", () => {
      const open = navigation.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
    });

    navigation.addEventListener("click", (event) => {
      if (event.target.closest("a")) {
        navigation.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      }
    });

    window.addEventListener("resize", () => {
      if (window.innerWidth > 780) {
        navigation.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const value = button.getAttribute("data-copy");
      if (!value) return;

      const original = button.textContent;
      try {
        await navigator.clipboard.writeText(value);
        button.textContent = "Copied";
      } catch (_error) {
        const temporary = document.createElement("textarea");
        temporary.value = value;
        temporary.setAttribute("readonly", "");
        temporary.style.position = "fixed";
        temporary.style.opacity = "0";
        document.body.appendChild(temporary);
        temporary.select();
        document.execCommand("copy");
        temporary.remove();
        button.textContent = "Copied";
      }

      window.setTimeout(() => {
        button.textContent = original;
      }, 1600);
    });
  });

  document.querySelectorAll("[data-year]").forEach((element) => {
    element.textContent = String(new Date().getFullYear());
  });
})();
