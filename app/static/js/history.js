/**
 * history.js
 *
 * Renders the recognition history stored in localStorage under the
 * "gesturespeak_history" key (the same key recognition.js writes to
 * when a sentence is spoken). No backend/database is used for this
 * stage -- per the Stage 5 brief, localStorage is the source of truth
 * until a real history backend exists.
 */

(function () {
    const HISTORY_KEY = "gesturespeak_history";
    const listEl = document.getElementById("historyList");
    const clearBtn = document.getElementById("clearHistoryBtn");

    if (!listEl) return;

    function loadHistory() {
        try {
            const raw = localStorage.getItem(HISTORY_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (e) {
            console.warn("Could not read history:", e);
            return [];
        }
    }

    function saveHistory(items) {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
    }

    function formatDateLabel(date) {
        const today = new Date();
        const isToday = date.toDateString() === today.toDateString();
        const yesterday = new Date(today);
        yesterday.setDate(today.getDate() - 1);
        const isYesterday = date.toDateString() === yesterday.toDateString();

        if (isToday) return "Today";
        if (isYesterday) return "Yesterday";
        return date.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
    }

    function formatTime(date) {
        return date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
    }

    function speak(text) {
        if (!("speechSynthesis" in window)) {
            alert("Speech output is not supported in this browser.");
            return;
        }
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    }

    function render() {
        const items = loadHistory();
        listEl.innerHTML = "";

        if (items.length === 0) {
            listEl.innerHTML = `
                <div class="history-empty">
                    <div class="empty-emoji" aria-hidden="true">🕘</div>
                    <p>No recognition history yet. Recognized sentences you speak on the Recognize page will show up here.</p>
                </div>
            `;
            return;
        }

        // Group by date label, preserving newest-first order.
        const groups = new Map();
        items.forEach((item) => {
            const date = new Date(item.time);
            const label = formatDateLabel(date);
            if (!groups.has(label)) groups.set(label, []);
            groups.get(label).push(item);
        });

        groups.forEach((groupItems, label) => {
            const groupEl = document.createElement("div");
            groupEl.className = "history-date-group";

            const labelEl = document.createElement("div");
            labelEl.className = "history-date-label";
            labelEl.textContent = label;
            groupEl.appendChild(labelEl);

            groupItems.forEach((item) => {
                const row = document.createElement("div");
                row.className = "card history-item";
                const date = new Date(item.time);
                row.innerHTML = `
                    <div>
                        <div class="history-item-text">${item.text}</div>
                        <div class="history-item-time">${formatTime(date)}</div>
                    </div>
                    <div class="history-item-actions">
                        <button class="icon-btn" data-action="play" data-id="${item.id}" aria-label="Play">🔊</button>
                        <button class="icon-btn danger" data-action="delete" data-id="${item.id}" aria-label="Delete">🗑</button>
                    </div>
                `;
                groupEl.appendChild(row);
            });

            listEl.appendChild(groupEl);
        });
    }

    listEl.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-action]");
        if (!btn) return;
        const id = btn.dataset.id;
        const items = loadHistory();
        const item = items.find((i) => i.id === id);

        if (btn.dataset.action === "play" && item) {
            speak(item.text);
        } else if (btn.dataset.action === "delete") {
            saveHistory(items.filter((i) => i.id !== id));
            render();
        }
    });

    if (clearBtn) {
        clearBtn.addEventListener("click", () => {
            if (confirm("Clear all recognition history? This cannot be undone.")) {
                saveHistory([]);
                render();
            }
        });
    }

    render();
})();
