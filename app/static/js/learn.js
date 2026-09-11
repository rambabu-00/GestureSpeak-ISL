/**
 * learn.js
 *
 * Renders a placeholder set of ISL signs with search + category
 * filtering, and a detail modal. This data is example/placeholder
 * content only -- it is NOT verified ISL instructional material.
 */

(function () {
    const grid = document.getElementById("signsGrid");
    if (!grid) return;

    // Placeholder data. Emoji stand in for real sign images/video
    // until verified ISL reference media is available.
    const SIGNS = [
        { name: "Hello", category: "greetings", emoji: "👋", desc: "A common greeting sign, placeholder description only." },
        { name: "Thank You", category: "greetings", emoji: "🙏", desc: "A polite expression of gratitude, placeholder description only." },
        { name: "Please", category: "greetings", emoji: "🤲", desc: "Used to make a polite request, placeholder description only." },
        { name: "Sorry", category: "greetings", emoji: "😔", desc: "An apology sign, placeholder description only." },
        { name: "Yes", category: "actions", emoji: "👍", desc: "An affirmative response, placeholder description only." },
        { name: "No", category: "actions", emoji: "👎", desc: "A negative response, placeholder description only." },
        { name: "Help", category: "actions", emoji: "🆘", desc: "Used to ask for assistance, placeholder description only." },
        { name: "Name", category: "people", emoji: "🧑", desc: "Used when introducing yourself, placeholder description only." },
        { name: "Family", category: "people", emoji: "👨‍👩‍👧", desc: "Refers to one's family, placeholder description only." },
        { name: "Friend", category: "people", emoji: "🤝", desc: "Refers to a friend, placeholder description only." },
        { name: "Water", category: "daily", emoji: "💧", desc: "Refers to water/drinking, placeholder description only." },
        { name: "Food", category: "daily", emoji: "🍽️", desc: "Refers to eating/food, placeholder description only." },
        { name: "Emergency", category: "emergency", emoji: "🚨", desc: "Used to signal an emergency, placeholder description only." },
        { name: "Doctor", category: "emergency", emoji: "🩺", desc: "Refers to needing medical help, placeholder description only." },
    ];

    const searchInput = document.getElementById("signSearchInput");
    const categoryFilters = document.getElementById("categoryFilters");

    let activeCategory = "all";
    let query = "";

    function render() {
        const filtered = SIGNS.filter((sign) => {
            const matchesCategory = activeCategory === "all" || sign.category === activeCategory;
            const matchesQuery = sign.name.toLowerCase().includes(query.toLowerCase());
            return matchesCategory && matchesQuery;
        });

        grid.innerHTML = "";

        if (filtered.length === 0) {
            const empty = document.createElement("div");
            empty.className = "no-results";
            empty.textContent = "No signs match your search.";
            grid.appendChild(empty);
            return;
        }

        filtered.forEach((sign) => {
            const card = document.createElement("div");
            card.className = "card sign-card";
            card.innerHTML = `
                <div class="sign-card-visual" aria-hidden="true">${sign.emoji}</div>
                <div class="sign-card-name">${sign.name}</div>
                <button class="btn btn-secondary practice-btn">Practice</button>
            `;
            card.addEventListener("click", (e) => {
                openModal(sign);
            });
            grid.appendChild(card);
        });
    }

    if (categoryFilters) {
        categoryFilters.querySelectorAll(".category-chip").forEach((chip) => {
            chip.addEventListener("click", () => {
                categoryFilters.querySelectorAll(".category-chip").forEach((c) => c.classList.remove("active"));
                chip.classList.add("active");
                activeCategory = chip.dataset.category;
                render();
            });
        });
    }

    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            query = e.target.value;
            render();
        });
    }

    // --- Detail modal ---
    const modal = document.getElementById("signModal");
    const modalClose = document.getElementById("signModalClose");
    const modalTitle = document.getElementById("signModalTitle");
    const modalCategory = document.getElementById("signModalCategory");
    const modalVisual = document.getElementById("signModalVisual");
    const modalDesc = document.getElementById("signModalDesc");

    function openModal(sign) {
        if (!modal) return;
        modalTitle.textContent = sign.name;
        modalCategory.textContent = sign.category;
        modalVisual.textContent = sign.emoji;
        modalDesc.textContent = sign.desc;
        modal.hidden = false;
    }

    function closeModal() {
        if (modal) modal.hidden = true;
    }

    if (modalClose) modalClose.addEventListener("click", closeModal);
    if (modal) {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) closeModal();
        });
    }
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeModal();
    });

    render();
})();
