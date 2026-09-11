/**
 * emergency.js
 *
 * Emergency Alert page behavior.
 *
 * IMPORTANT: There is no real SMS/call/notification backend
 * connected. The "Send Alert" flow below simulates the confirm ->
 * sending -> success/failure UI states with a short timeout so the
 * interface is ready to be wired to a real notification service
 * later. The demo banner on the page makes this explicit to the user
 * at all times, so nothing here claims (falsely) that a real alert
 * was sent.
 */

(function () {
    const CONTACTS_KEY = "gesturespeak_trusted_contacts";

    // --- Tabs ---
    const tabs = document.querySelectorAll(".emergency-tab");
    const panels = {
        panelAlert: document.getElementById("panelAlert"),
        panelContacts: document.getElementById("panelContacts"),
    };

    tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            tabs.forEach((t) => {
                t.classList.remove("active");
                t.setAttribute("aria-selected", "false");
            });
            tab.classList.add("active");
            tab.setAttribute("aria-selected", "true");

            Object.keys(panels).forEach((key) => {
                if (panels[key]) panels[key].hidden = key !== tab.dataset.panel;
            });
        });
    });

    // --- Trusted contacts (localStorage) ---
    const contactsList = document.getElementById("contactsList");
    const addContactForm = document.getElementById("addContactForm");

    function loadContacts() {
        try {
            const raw = localStorage.getItem(CONTACTS_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (e) {
            return [];
        }
    }

    function saveContacts(contacts) {
        localStorage.setItem(CONTACTS_KEY, JSON.stringify(contacts));
    }

    function renderContacts() {
        if (!contactsList) return;
        const contacts = loadContacts();
        contactsList.innerHTML = "";

        if (contacts.length === 0) {
            contactsList.innerHTML = `
                <div class="contacts-empty">
                    No trusted contacts yet. Add someone below so alerts have
                    somewhere to go.
                </div>
            `;
            return;
        }

        contacts.forEach((contact) => {
            const card = document.createElement("div");
            card.className = "card contact-card";
            card.innerHTML = `
                <div class="contact-info">
                    <span class="contact-avatar" aria-hidden="true">👤</span>
                    <div>
                        <div class="contact-name">${contact.name}</div>
                        <div class="contact-phone">${contact.phone}</div>
                        <div class="contact-check">✓ Receives emergency alerts</div>
                    </div>
                </div>
                <button class="contact-remove" data-id="${contact.id}" aria-label="Remove ${contact.name}">✕</button>
            `;
            contactsList.appendChild(card);
        });
    }

    if (contactsList) {
        contactsList.addEventListener("click", (e) => {
            const btn = e.target.closest(".contact-remove");
            if (!btn) return;
            const contacts = loadContacts().filter((c) => c.id !== btn.dataset.id);
            saveContacts(contacts);
            renderContacts();
        });
    }

    if (addContactForm) {
        addContactForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const name = document.getElementById("contactName").value.trim();
            const phone = document.getElementById("contactPhone").value.trim();
            if (!name || !phone) return;

            const contacts = loadContacts();
            contacts.push({ id: `${Date.now()}`, name, phone });
            saveContacts(contacts);
            renderContacts();
            addContactForm.reset();
        });
    }

    renderContacts();

    // --- Send alert flow (simulated) ---
    const modal = document.getElementById("emergencyModal");
    const modalBody = document.getElementById("emergencyModalBody");
    const modalClose = document.getElementById("emergencyModalClose");

    function openModal(html) {
        if (!modal || !modalBody) return;
        modalBody.innerHTML = html;
        modal.hidden = false;
        wireModalButtons();
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

    function confirmScreen(type, icon) {
        const contacts = loadContacts();
        const contactNote = contacts.length > 0
            ? `${contacts.length} trusted contact(s) will be notified.`
            : `No trusted contacts are set up yet -- add one in the "Trusted Contacts" tab first.`;

        return `
            <div class="emergency-state-icon" aria-hidden="true">${icon}</div>
            <h2 class="modal-title" style="text-align:center;">${type.toUpperCase()}</h2>
            <p style="text-align:center; color:var(--text-secondary); margin-top:0.5rem;">
                Send emergency alert to your trusted contacts?
            </p>
            <ul class="emergency-checklist">
                <li>✓ Trusted contacts selected automatically -- ${contactNote}</li>
                <li>✓ Current location can be included, if your browser shares it</li>
            </ul>
            <div class="btn-row" style="justify-content:center;">
                <button class="btn btn-outline" id="emergencyCancelBtn">Cancel</button>
                <button class="btn btn-danger" id="emergencySendBtn" data-type="${type}">SEND ALERT</button>
            </div>
        `;
    }

    function sendingScreen(type) {
        return `
            <div class="emergency-state-icon" aria-hidden="true">⏳</div>
            <h2 class="modal-title" style="text-align:center;">Sending alert…</h2>
            <p style="text-align:center; color:var(--text-secondary); margin-top:0.5rem;">
                (Simulated -- no real message is being sent yet.)
            </p>
        `;
    }

    function successScreen(type) {
        const time = new Date().toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
        return `
            <div class="emergency-state-icon" aria-hidden="true">✓</div>
            <h2 class="modal-title" style="text-align:center; color:var(--success);">ALERT SENT (DEMO)</h2>
            <p style="text-align:center; color:var(--text-secondary); margin-top:0.5rem;">
                In demo mode: your trusted contacts would have been notified about
                <strong>${type}</strong>.
            </p>
            <div class="emergency-result-grid">
                <div><h4>Location</h4><p>Not connected</p></div>
                <div><h4>Time</h4><p>${time}</p></div>
            </div>
            <button class="btn btn-primary btn-block" id="emergencyDoneBtn">Done</button>
        `;
    }

    function failureScreen(type) {
        return `
            <div class="emergency-state-icon" aria-hidden="true">⚠</div>
            <h2 class="modal-title" style="text-align:center; color:var(--danger);">ALERT COULD NOT BE SENT</h2>
            <p style="text-align:center; color:var(--text-secondary); margin-top:0.5rem;">
                Please check your connection and try again.
            </p>
            <div class="btn-row" style="justify-content:center; margin-top:1.25rem;">
                <button class="btn btn-outline" id="emergencyRetryBtn" data-type="${type}">Try Again</button>
                <a class="btn btn-danger" href="tel:112">Call Emergency Services</a>
            </div>
        `;
    }

    function wireModalButtons() {
        const cancelBtn = document.getElementById("emergencyCancelBtn");
        const sendBtn = document.getElementById("emergencySendBtn");
        const doneBtn = document.getElementById("emergencyDoneBtn");
        const retryBtn = document.getElementById("emergencyRetryBtn");

        if (cancelBtn) cancelBtn.addEventListener("click", closeModal);
        if (doneBtn) doneBtn.addEventListener("click", closeModal);

        if (sendBtn) {
            sendBtn.addEventListener("click", () => {
                const type = sendBtn.dataset.type;
                openModal(sendingScreen(type));
                // Simulated network delay -- this demo always "succeeds".
                // A real implementation would call a backend endpoint here
                // and branch to successScreen()/failureScreen() based on
                // the actual result.
                setTimeout(() => openModal(successScreen(type)), 900);
            });
        }

        if (retryBtn) {
            retryBtn.addEventListener("click", () => {
                const type = retryBtn.dataset.type;
                openModal(confirmScreen(type, "🚨"));
            });
        }
    }

    document.querySelectorAll(".emergency-category-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const type = btn.dataset.type;
            const icon = btn.dataset.icon;
            openModal(confirmScreen(type, icon));
        });
    });
})();
