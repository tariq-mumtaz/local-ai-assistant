const API_URL = "http://localhost:8000";
let conversationId = null;
let uploadedDocs = [];

function getTime() {
    return new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
    });
}

function safeHTML(html) {
    return html
        .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, "")
        .replace(/on\w+="[^"]*"/gi, "")
        .replace(/javascript:/gi, "");
}

function stripIncompleteTag(text) {
    const lastOpen = text.lastIndexOf("<");
    if (lastOpen === -1) return text;
    const lastClose = text.lastIndexOf(">");
    if (lastOpen > lastClose) {
        return text.substring(0, lastOpen);
    }
    return text;
}

function copySVG() {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2"/>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
    </svg>`;
}

function checkSVG() {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="20 6 9 17 4 12"/>
    </svg>`;
}

function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    sidebar.classList.toggle("collapsed");
}

function addMessage(role, text = "", isThinking = false) {
    const emptyState = document.getElementById("empty-state");
    if (emptyState) emptyState.remove();

    const chatBox = document.getElementById("chat-box");

    const wrapper = document.createElement("div");
    // wrapper.className = `message ${role}`;
    wrapper.className = `message ${role} message-wrapper`;
    wrapper.id = isThinking ? "thinking-msg" : "";

    const bubble = document.createElement("div");
    bubble.className = `bubble${isThinking ? " thinking" : ""}`;
    if (role === "ai") {
        bubble.innerHTML = safeHTML(text); // renders HTML
    } else {
        bubble.textContent = text; // keeps user text safe
    }

    const time = document.createElement("div");
    time.className = "timestamp";
    time.textContent = getTime();

    wrapper.appendChild(bubble);
    wrapper.appendChild(time);

    // Add copy button for AI messages only
    if (role === "ai" && !isThinking) {
        const copyBtn = document.createElement("button");
        copyBtn.className = "copy-btn";
        copyBtn.innerHTML = copySVG();
        copyBtn.onclick = () => copyMessage(bubble, copyBtn);
        wrapper.appendChild(copyBtn);
    }

    chatBox.appendChild(wrapper);
    chatBox.scrollTop = chatBox.scrollHeight;

    return bubble;
}

function removeThinking() {
    const thinking = document.getElementById("thinking-msg");
    if (thinking) thinking.remove();
}

function copyMessage(bubble, btn) {
    const text = bubble.innerHTML || bubble.innerText || bubble.textContent;

    navigator.clipboard
        .writeText(text)
        .then(() => {
            btn.innerHTML = checkSVG();
            btn.classList.add("copied");
            setTimeout(() => {
                btn.innerHTML = copySVG();
                btn.classList.remove("copied");
            }, 2000);
        })
        .catch(() => {
            const el = document.createElement("textarea");
            el.value = text;
            document.body.appendChild(el);
            el.select();
            document.execCommand("copy");
            document.body.removeChild(el);
            btn.innerHTML = checkSVG();
            btn.classList.add("copied");
            setTimeout(() => {
                btn.innerHTML = copySVG();
                btn.classList.remove("copied");
            }, 2000);
        });
}

async function sendMessage() {
    const input = document.getElementById("user-input");
    const apiKey = document.getElementById("api-key").value.trim();
    const sendBtn = document.getElementById("send-btn");
    const text = input.value.trim();
    const chatBox = document.getElementById("chat-box");

    if (!text) return;
    if (!apiKey) {
        alert("Please enter an API key");
        return;
    }

    // Show user message
    addMessage("user", text);
    input.value = "";
    input.style.height = "auto";

    // Disable input while waiting
    sendBtn.disabled = true;
    input.disabled = true;

    // Show thinking indicator
    addMessage("ai", "Thinking...", true);

    try {
        const body = { message: text };
        if (conversationId) body.conversation_id = conversationId;

        const res = await fetch(`${API_URL}/chat/stream`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "x-api-key": apiKey,
            },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const data = await res.json();
            removeThinking();
            addMessage("ai", `Error: ${data.detail || "Something went wrong"}`);
            return;
        }

        // Remove thinking and create empty AI bubble for streaming
        removeThinking();
        const aiBubble = addMessage("ai", "");

        // Read stream
        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        let buffer = "";
        let streamBuffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer = decoder.decode(value, { stream: true });

            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;

                const jsonStr = line.replace("data: ", "").trim();
                if (!jsonStr) continue;

                try {
                    const data = JSON.parse(jsonStr);

                    if (data.type === "conv_id" || data.conversation_id) {
                        const cid = data.conversation_id;
                        if (cid) {
                            conversationId = cid;
                            const badge = document.getElementById("conv-label");
                            badge.textContent = `ID: ${cid.slice(0, 8)}...`;
                            badge.style.display = "inline-block";
                        }
                        // conversationId = data.conversation_id;
                        // document.getElementById("conv-label").textContent =
                        //   `ID: ${conversationId.slice(0, 8)}...`;
                    }

                    if (data.type === "token") {
                        streamBuffer += data.text;
                        let displayText = streamBuffer
                            .replace(/<think>[\s\S]*?<\/think>/gi, "")
                            .replace(/<think>[\s\S]*/gi, "") // incomplete think tag
                            .trim();

                        aiBubble.innerHTML = stripIncompleteTag(displayText) || "...";
                        // Auto scroll as tokens arrive
                        const chatBox = document.getElementById("chat-box");
                        chatBox.scrollTop = chatBox.scrollHeight;
                    }

                    if (data.type === "done") {
                        // Remove think tags completely from final response
                        let finalResponse = streamBuffer
                            .replace(/<think>[\s\S]*?<\/think>/gi, "")
                            .trim();

                        // Now safe to render as HTML
                        aiBubble.innerHTML = safeHTML(finalResponse);
                        chatBox.scrollTop = chatBox.scrollHeight;
                        streamBuffer = "";

                        // Add copy button after streaming completes
                        const wrapper = aiBubble.closest(".message-wrapper");
                        if (wrapper && !wrapper.querySelector(".copy-btn")) {
                            const copyBtn = document.createElement("button");
                            copyBtn.className = "copy-btn";
                            copyBtn.innerHTML = copySVG();
                            copyBtn.onclick = () => copyMessage(aiBubble, copyBtn);
                            wrapper.appendChild(copyBtn);
                        }
                        loadConversationList(); // Refresh conversation list to show updated timestamp
                        break;
                    }
                } catch (e) {
                    // Skip malformed chunks
                    console.log(e.message);
                }
            }
        }
    } catch (err) {
        removeThinking();
        addMessage("ai", "Could not reach the API. Is it running?");
    } finally {
        sendBtn.disabled = false;
        input.disabled = false;
        input.focus();
    }
}

async function loadConversationList() {
    const apiKey = document.getElementById("api-key").value.trim();
    if (!apiKey) return;

    try {
        const res = await fetch(`${API_URL}/conversations`, {
            headers: { "x-api-key": apiKey },
        });
        const data = await res.json();
        renderConversationList(data.conversations || []);
    } catch (err) {
        console.log("Could not load conversations");
    }
}

function renderConversationList(convs) {
    const list = document.getElementById("conv-list");

    if (convs.length === 0) {
        list.innerHTML =
            '<p class="no-convs">No conversations yet.<br>Start chatting to see history here.</p>';
        return;
    }

    list.innerHTML = convs
        .map(
            (c) => `
    <div class="conv-item ${c.conv_id === conversationId ? "active" : ""}"
         id="conv-item-${c.conv_id}"
         onclick="loadConversation('${c.conv_id}')">
      <span class="conv-icon">💬</span>
      <div class="conv-info">
        <div class="conv-title">${c.title}</div>
        <div class="conv-date">${c.updated_at}</div>
      </div>
      <button class="conv-del"
        onclick="event.stopPropagation(); deleteConversation('${c.conv_id}')"
        title="Delete">✕</button>
    </div>
  `,
        )
        .join("");
}

async function loadConversation(convId) {
    const apiKey = document.getElementById("api-key").value.trim();
    try {
        const res = await fetch(`${API_URL}/conversations/${convId}/load`, {
            headers: { "x-api-key": apiKey },
        });
        const data = await res.json();

        conversationId = convId;
        const badge = document.getElementById("conv-label");
        badge.textContent = `ID: ${convId.slice(0, 8)}...`;
        badge.style.display = "inline-block";

        const chatBox = document.getElementById("chat-box");
        chatBox.innerHTML = "";

        data.messages.forEach((msg) => {
            if (msg.role === "user") addMessage("user", msg.content);
            else if (msg.role === "assistant") addMessage("ai", msg.content);
        });

        document
            .querySelectorAll(".conv-item")
            .forEach((el) => el.classList.remove("active"));
        const active = document.getElementById(`conv-item-${convId}`);
        if (active) active.classList.add("active");

        await loadDocumentList(convId);
    } catch (e) {
        console.log("Could not load conversation", e);
    }
}

async function deleteConversation(convId) {
    const apiKey = document.getElementById("api-key").value.trim();

    try {
        await fetch(`${API_URL}/conversations/${convId}`, {
            method: "DELETE",
            headers: { "x-api-key": apiKey },
        });

        // If deleting current conversation clear chat
        if (convId === conversationId) {
            newChat();
        }

        loadConversationList();
    } catch (err) {
        console.log("Could not delete conversation");
    }
}

function newChat() {
    conversationId = null;
    uploadedDocs = [];
    renderDocList();
    const badge = document.getElementById("conv-label");
    badge.textContent = "";
    badge.style.display = "none";
    document.getElementById("chat-box").innerHTML = `
    <div class="empty-state" id="empty-state">
      <p>Select a conversation or start typing</p>
    </div>`;
    document
        .querySelectorAll(".conv-item")
        .forEach((el) => el.classList.remove("active"));
}

function renderDocList() {
    const docList = document.getElementById("doc-list");

    if (uploadedDocs.length === 0) {
        docList.innerHTML = '<p class="no-docs">No documents uploaded</p>';
        return;
    }

    docList.innerHTML = uploadedDocs
        .map(
            (filename) => `
        <div class="doc-item" id="doc-${btoa(filename)}">
            <span class="doc-name" title="${filename}">📄 ${filename}</span>
            <button class="doc-remove" onclick="removeDocument('${filename}')" title="Remove">✕</button>
        </div>
    `,
        )
        .join("");
}

async function loadDocumentList(convId) {
    const apiKey = document.getElementById("api-key").value.trim();
    try {
        const res = await fetch(`${API_URL}/documents/${convId}`, {
            headers: { "x-api-key": apiKey },
        });
        const data = await res.json();
        uploadedDocs = data.files || [];
        renderDocList();
    } catch (err) {
        console.log("Could not load documents");
    }
}

async function uploadFile(input) {
    const file = input.files[0];
    if (!file) return;

    const apiKey = document.getElementById("api-key").value.trim();
    const statusEl = document.getElementById("upload-status");

    if (!conversationId) {
        conversationId = crypto.randomUUID();
        const badge = document.getElementById("conv-label");
        badge.textContent = `ID: ${conversationId.slice(0, 8)}...`;
        badge.style.display = "inline-block";
    }

    statusEl.textContent = `Uploading...`;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("conversation_id", conversationId);

    try {
        const res = await fetch(`${API_URL}/upload`, {
            method: "POST",
            headers: { "x-api-key": apiKey },
            body: formData,
        });

        const data = await res.json();

        if (res.ok) {
            statusEl.textContent = `Ready (${data.chunks} chunks)`;
            uploadedDocs.push(file.name);
            renderDocList();
            addMessage(
                "ai",
                `<p>Document <strong>${file.name}</strong> uploaded. You can now ask questions about it.</p>`,
            );
            setTimeout(() => (statusEl.textContent = ""), 3000);
        } else {
            statusEl.textContent = "Upload failed";
            addMessage("ai", `<p>Upload failed: ${data.detail}</p>`);
        }
    } catch (err) {
        statusEl.textContent = "Upload failed";
        addMessage("ai", "<p>Could not upload file. Is the API running?</p>");
    }

    // Reset file input
    input.value = "";
}

async function removeDocument(filename) {
    const apiKey = document.getElementById("api-key").value.trim();

    try {
        const res = await fetch(
            `${API_URL}/documents/file/${encodeURIComponent(filename)}`,
            {
                method: "DELETE",
                headers: { "x-api-key": apiKey, "conv-id": conversationId },
            },
        );

        if (res.ok) {
            uploadedDocs = uploadedDocs.filter((f) => f !== filename);
            renderDocList();
            addMessage("ai", `<p>Document <strong>${filename}</strong> removed.</p>`);
        }
    } catch (err) {
        addMessage("ai", "<p>Could not remove document.</p>");
    }
}

function clearChat() {
    conversationId = null;
    uploadedDocs = [];
    renderDocList();
    document.getElementById("conv-label").textContent = "";
    document.getElementById("chat-box").innerHTML = `
        <div class="empty-state" id="empty-state">
            <p>Upload a document or start chatting</p>
        </div>`;
}


// Load conversations when page opens
window.addEventListener("load", () => {
    loadConversationList();
});
