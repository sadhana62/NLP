// PageEcho Web Application Controller
document.addEventListener("DOMContentLoaded", () => {
    // State Variables
    let activeCharacter = null;
    let charactersList = [];
    let chatHistoryMap = {};
    let ttsEnabled = false;
    let selectedFile = null;
    let selectedEngine = "llm"; // "llm" (Generative) or "classical" (Zero LLM)
    let bookState = {
        hasBook: false,
        title: "",
        hasCharacters: false,
        genre: "",
        message: "",
        indexedChunks: 0
    };

    // DOM Elements - Navigation & Header
    const bookStatusDot = document.getElementById("book-status-dot");
    const activeBookTitle = document.getElementById("active-book-title");
    const indexedChunksVal = document.getElementById("indexed-chunks-val");
    const genreVal = document.getElementById("genre-val");
    const modelSelector = document.getElementById("model-selector");
    const modelSelectWrapper = document.getElementById("model-select-wrapper");
    const btnEngineLlm = document.getElementById("btn-engine-llm");
    const btnEngineClassical = document.getElementById("btn-engine-classical");
    const activeEngineBadge = document.getElementById("active-engine-badge");

    // Sidebar Elements
    const characterGrid = document.getElementById("character-grid");
    const charCountBadge = document.getElementById("char-count");
    const sidebarDesc = document.getElementById("sidebar-desc");
    const sidebarNonFictionAlert = document.getElementById("sidebar-non-fiction-alert");
    const sidebarNonFictionText = document.getElementById("sidebar-non-fiction-text");

    // Chat Header & Controls
    const activeAvatar = document.getElementById("active-avatar");
    const activeCharName = document.getElementById("active-char-name");
    const activeCharRole = document.getElementById("active-char-role");
    const viewDossierBtn = document.getElementById("view-dossier-btn");
    const openComparisonBtn = document.getElementById("open-comparison-btn");
    const toggleVoiceBtn = document.getElementById("toggle-voice-btn");
    const voiceIcon = document.getElementById("voice-icon");
    const toggleContextBtn = document.getElementById("toggle-context-btn");
    const contextDrawer = document.getElementById("context-drawer");
    const closeContextBtn = document.getElementById("close-context-btn");
    const contextContent = document.getElementById("context-content");

    // Comparison Modal Elements
    const comparisonModal = document.getElementById("comparison-modal");
    const closeComparisonBtn = document.getElementById("close-comparison-btn");
    const compareForm = document.getElementById("compare-form");
    const compareInput = document.getElementById("compare-input");
    const runCompareBtn = document.getElementById("run-compare-btn");
    const comparisonResults = document.getElementById("comparison-results");
    const classicalScoreTag = document.getElementById("classical-score-tag");
    const classicalOutputBubble = document.getElementById("classical-output-bubble");
    const classicalDiagnostics = document.getElementById("classical-diagnostics");
    const llmModelTag = document.getElementById("llm-model-tag");
    const llmOutputBubble = document.getElementById("llm-output-bubble");
    const llmDiagnostics = document.getElementById("llm-diagnostics");

    // Chat Stream & Form
    const chatMessages = document.getElementById("chat-messages");
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const liveSentimentBadge = document.getElementById("live-sentiment-badge");

    // Upload Modal Elements
    const uploadModal = document.getElementById("upload-modal");
    const openUploadModalBtn = document.getElementById("open-upload-modal");
    const closeModalBtn = document.getElementById("close-modal-btn");
    const cancelUploadBtn = document.getElementById("cancel-upload-btn");
    const uploadForm = document.getElementById("upload-form");
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const fileInfo = document.getElementById("file-info");
    const fileNameDisplay = document.getElementById("file-name");
    const removeFileBtn = document.getElementById("remove-file-btn");
    const processingLoader = document.getElementById("processing-loader");
    const loaderStatusText = document.getElementById("loader-status-text");
    const uploadProgressFill = document.getElementById("upload-progress-fill");
    const uploadProgressPercent = document.getElementById("upload-progress-percent");
    const loaderStageText = document.getElementById("loader-stage-text");
    const loaderSubtext = document.getElementById("loader-subtext");

    function updateUploadProgress(percent, stageText, subText) {
        if (uploadProgressFill) uploadProgressFill.style.width = `${percent}%`;
        if (uploadProgressPercent) uploadProgressPercent.textContent = `${percent}%`;
        if (loaderStageText) loaderStageText.textContent = stageText || "Processing...";
        if (loaderStatusText) loaderStatusText.textContent = stageText || "Analyzing book...";
        if (loaderSubtext && subText) loaderSubtext.textContent = subText;
    }

    // Learned Dossier Modal Elements
    const dossierModal = document.getElementById("dossier-modal");
    const closeDossierBtn = document.getElementById("close-dossier-btn");
    const dossierAvatar = document.getElementById("dossier-avatar");
    const dossierCharName = document.getElementById("dossier-char-name");
    const dossierCharRole = document.getElementById("dossier-char-role");
    const dossierBackstory = document.getElementById("dossier-backstory");
    const dossierTraits = document.getElementById("dossier-traits");
    const dossierSpeech = document.getElementById("dossier-speech");
    const dossierRelationships = document.getElementById("dossier-relationships");
    const dossierFacts = document.getElementById("dossier-facts");

    // Speech Synthesis
    const synth = window.speechSynthesis;

    // Initialize Application
    initApp();

    async function initApp() {
        setupEventListeners();
        await fetchStatus();
    }

    async function fetchStatus() {
        try {
            const res = await fetch("/api/status");
            if (res.ok) {
                const data = await res.json();
                bookState.hasBook = data.has_book;
                bookState.title = data.active_book;
                bookState.hasCharacters = Boolean(data.has_characters || (data.character_count > 0));
                bookState.genre = data.genre;
                bookState.message = data.classification_message;
                bookState.indexedChunks = data.indexed_chunks;

                // Update models list
                if (data.available_models && data.available_models.length > 0) {
                    modelSelector.innerHTML = "";
                    data.available_models.forEach(model => {
                        const opt = document.createElement("option");
                        opt.value = model;
                        opt.textContent = `${model} ${model.includes('1.5b') ? '(Fast Local)' : ''}`;
                        if (model === data.current_model) opt.selected = true;
                        modelSelector.appendChild(opt);
                    });
                }

                updateUIForBookState();

                if (bookState.hasBook && (bookState.hasCharacters || data.character_count > 0)) {
                    await fetchCharacters();
                }
            }
        } catch (err) {
            console.error("Error fetching app status:", err);
            updateUIForBookState();
        }
    }

    async function fetchCharacters() {
        try {
            const res = await fetch("/api/characters");
            if (res.ok) {
                const data = await res.json();
                charactersList = data.characters || [];
                renderCharacterCards(charactersList);

                if (charactersList.length > 0 && !activeCharacter) {
                    selectCharacter(charactersList[0]);
                }
            }
        } catch (err) {
            console.error("Error fetching characters:", err);
        }
    }

    function updateUIForBookState() {
        // Status dot & Header
        bookStatusDot.className = "status-dot";
        if (!bookState.hasBook) {
            bookStatusDot.classList.add("idle");
            activeBookTitle.textContent = "No PDF Book Uploaded";
            genreVal.textContent = "No Document Loaded";
            indexedChunksVal.textContent = "0 Passages Indexed";
            sidebarDesc.textContent = "Upload a PDF book with characters to extract and chat with them.";
            sidebarNonFictionAlert.classList.add("hidden");
            charCountBadge.textContent = "0 Characters";
            renderEmptyWelcomeState();
            disableChatInput("Upload a PDF book to chat with its characters...");
            viewDossierBtn.classList.add("hidden");
        } else if (!bookState.hasCharacters && charactersList.length === 0) {
            // Book uploaded but NO characters found in document
            bookStatusDot.classList.add("non-fiction");
            activeBookTitle.textContent = bookState.title;
            genreVal.textContent = bookState.genre || "Document";
            indexedChunksVal.textContent = `${bookState.indexedChunks} Passages Indexed`;
            charCountBadge.textContent = "0 Characters Found";
            
            sidebarNonFictionAlert.classList.remove("hidden");
            sidebarNonFictionText.textContent = bookState.message || "No characters or key figures were detected in this document.";
            sidebarDesc.textContent = "No characters detected. Upload a book that contains characters or notable persons.";

            characterGrid.innerHTML = `
                <div style="padding: 16px; background: rgba(255,255,255,0.03); border: 1px dashed var(--border-color); border-radius: var(--radius-md); text-align: center; color: var(--text-muted); font-size: 12px;">
                    No characters or key figures found in this document.
                </div>
            `;

            renderNoCharactersState();
            disableChatInput("Chat disabled — document contains no detected characters.");
            viewDossierBtn.classList.add("hidden");
        } else {
            // Book with characters
            bookStatusDot.classList.add("fiction");
            activeBookTitle.textContent = bookState.title;
            genreVal.textContent = bookState.genre || "Book with Characters";
            indexedChunksVal.textContent = `${bookState.indexedChunks} Passages Indexed`;
            sidebarNonFictionAlert.classList.add("hidden");
            sidebarDesc.textContent = `Learned characters from '${bookState.title}' via NLP character extraction.`;
        }
    }

    function renderEmptyWelcomeState() {
        chatMessages.innerHTML = `
            <div class="welcome-banner">
                <span class="hero-eyebrow">AI BOOK CHARACTER ROLEPLAY &bull; NLP RAG SUITE</span>
                <h1 class="hero-headline">
                    Read the story, <span class="highlight-text">speak with characters.</span><br>
                    The calm art of literary conversation.
                </h1>
                <p class="hero-subtext">
                    Upload any PDF book containing characters (novels, stories, plays, biographies). The NLP engine will automatically identify the characters, learn each persona from the text, and allow you to chat with them in character.
                </p>
                <button class="btn-pill-approach hero-cta-btn" onclick="document.getElementById('open-upload-modal').click()">
                    <span>Select &amp; Upload PDF Book</span>
                </button>
            </div>
        `;
    }

    function renderNoCharactersState() {
        activeAvatar.textContent = "";
        activeCharName.textContent = "No Characters Detected";
        activeCharRole.textContent = bookState.genre || "Document with No Characters Found";

        chatMessages.innerHTML = `
            <div class="non-fiction-notice-card">
                <span class="genre-tag-pill">${escapeHTML(bookState.genre || "Document")}</span>
                <h3 class="notice-title">No Characters Detected in "${escapeHTML(bookState.title)}"</h3>
                <p class="notice-desc">${escapeHTML(bookState.message || "This document does not contain detected characters or persons. Please upload a book that contains characters or notable figures to interact with.")}</p>
                <div style="margin-top: 14px;">
                    <button class="btn-pill-approach" onclick="document.getElementById('open-upload-modal').click()">
                        <span>Upload Another Book</span>
                    </button>
                </div>
            </div>
        `;
    }

    function renderCharacterCards(chars) {
        characterGrid.innerHTML = "";
        charCountBadge.textContent = `${chars.length} Characters`;

        if (chars.length === 0) {
            if (!bookState.hasCharacters) {
                characterGrid.innerHTML = `
                    <div style="padding: 16px; background: rgba(32,31,27,0.03); border: 1px dashed var(--border-color); border-radius: var(--radius-md); text-align: center; color: var(--text-muted); font-size: 12px;">
                        No characters or key figures found in this document.
                    </div>
                `;
            } else {
                characterGrid.innerHTML = `<p class="sidebar-desc">No characters found yet. Upload a PDF book with characters!</p>`;
            }
            return;
        }

        chars.forEach(char => {
            const card = document.createElement("div");
            card.className = `character-card ${activeCharacter && activeCharacter.name === char.name ? 'active' : ''}`;
            
            const traitsHTML = (char.traits || []).map(t => `<span class="trait-tag">${escapeHTML(t)}</span>`).join("");

            card.innerHTML = `
                <div class="card-top">
                    <div class="avatar-bubble"></div>
                    <div class="char-card-info">
                        <h3>${escapeHTML(char.name)}</h3>
                        <span class="char-role">${escapeHTML(char.role || 'Character')}</span>
                    </div>
                </div>
                <p style="font-size: 12px; color: var(--text-secondary); line-height: 1.45; margin-bottom: 8px;">
                    ${escapeHTML(char.speech_style || char.story_backstory || char.backstory || '')}
                </p>
                <div class="traits-list">${traitsHTML}</div>
                <div class="learned-badge">
                    <span>Learned persona</span>
                </div>
            `;

            card.addEventListener("click", () => selectCharacter(char));
            characterGrid.appendChild(card);
        });
    }

    function selectCharacter(char) {
        activeCharacter = char;
        
        // Highlight active card
        document.querySelectorAll(".character-card").forEach(c => c.classList.remove("active"));
        renderCharacterCards(charactersList);

        // Update Chat Header
        activeAvatar.textContent = "";
        activeCharName.textContent = char.name;
        activeCharRole.textContent = `${char.role || 'Character'} — ${char.speech_style || ''}`;

        // Show dossier button
        viewDossierBtn.classList.remove("hidden");

        // Enable input controls
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.placeholder = `Converse with ${char.name}...`;
        userInput.focus();

        // Initialize Chat History for this character
        if (!chatHistoryMap[char.name]) {
            chatHistoryMap[char.name] = [];
            
            // Insert character's initial greeting learned from story context welcoming the human user
            const greeting = char.initial_greeting || `Greetings, my friend. I am ${char.name}. Speak freely with me—what brings you across the pages of '${bookState.title}' to converse today?`;
            chatHistoryMap[char.name].push({
                sender: "character",
                text: greeting,
                sentiment: { label: "Greeting", compound: 0.5, color: "#1c1b18" },
                engine: selectedEngine,
                engineDetails: {
                    method: "Canonical Persona Greeting",
                    intent: "greeting",
                    similarity_score: 1.0,
                    explanation: "Instantiated character's canonical first-person opening greeting welcoming the human reader."
                }
            });
        }
        
        renderChatMessages(char.name);
    }

    function renderChatMessages(charName) {
        const history = chatHistoryMap[charName] || [];
        chatMessages.innerHTML = "";

        if (history.length === 0) {
            chatMessages.innerHTML = `
                <div class="welcome-banner">
                    <span class="hero-eyebrow">IN CONVERSATION &bull; ${escapeHTML(charName).toUpperCase()}</span>
                    <h2 class="hero-headline" style="font-size: 34px;">
                        Speak with <span class="highlight-text">${escapeHTML(charName)}</span>
                    </h2>
                    <p class="hero-subtext">${escapeHTML(activeCharacter ? (activeCharacter.story_backstory || activeCharacter.backstory) : 'Type a message below to chat.')}</p>
                </div>
            `;
            return;
        }

        history.forEach(msg => {
            appendMessageBubble(msg.sender, msg.text, msg.sentiment, msg.avatar, msg.engine, msg.engineDetails);
        });

        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendMessageBubble(sender, text, sentiment, avatar, engine, engineDetails) {
        const banner = chatMessages.querySelector(".welcome-banner");
        if (banner) banner.remove();

        const isUser = sender === "user";
        const wrapper = document.createElement("div");
        wrapper.className = `message-bubble-wrapper ${isUser ? 'user' : 'character'}`;

        const senderName = isUser ? "You" : (activeCharacter ? activeCharacter.name : "Character");
        
        let sentimentHTML = "";
        if (sentiment) {
            sentimentHTML = `
                <span class="sentiment-badge ${isUser ? 'user-badge' : ''}">
                    ${sentiment.label} (${sentiment.compound})
                </span>
            `;
        }

        let engineBadgeHTML = "";
        let diagnosticsHTML = "";

        if (!isUser) {
            const eng = engine || selectedEngine;
            if (eng === "classical" || eng === "classical_nlp") {
                const scoreText = engineDetails && engineDetails.similarity_score !== undefined 
                    ? ` • TF-IDF: ${engineDetails.similarity_score}` 
                    : "";
                engineBadgeHTML = `<span class="engine-bubble-badge classical" title="Generated using Pure Classical NLP (Zero LLM)">🏛️ Classical NLP${scoreText}</span>`;
            } else {
                const modelName = engineDetails?.model_used || modelSelector.value;
                engineBadgeHTML = `<span class="engine-bubble-badge llm" title="Generated using Ollama Neural Transformer">⚡ LLM (${escapeHTML(modelName)})</span>`;
            }

            if (engineDetails) {
                const diagId = "diag-" + Math.random().toString(36).substr(2, 9);
                diagnosticsHTML = `
                    <button type="button" class="diagnostics-toggle-btn" onclick="document.getElementById('${diagId}').classList.toggle('hidden')">
                        Engine Diagnostics &bull;
                    </button>
                    <div class="diagnostics-panel hidden" id="${diagId}">
                        <div><strong>Method:</strong> ${escapeHTML(engineDetails.method || "Dialogue Generation")}</div>
                        ${engineDetails.intent ? `<div><strong>Intent:</strong> ${escapeHTML(engineDetails.intent)}</div>` : ''}
                        ${engineDetails.similarity_score !== undefined ? `<div><strong>Relevance Score:</strong> ${engineDetails.similarity_score}</div>` : ''}
                        ${engineDetails.context_chunks_used ? `<div><strong>Context Chunks Ingested:</strong> ${engineDetails.context_chunks_used}</div>` : ''}
                        ${engineDetails.explanation ? `<div><strong>Explanation:</strong> ${escapeHTML(engineDetails.explanation)}</div>` : ''}
                        ${engineDetails.fallback_note ? `<div style="color: #b45309;"><strong>Note:</strong> ${escapeHTML(engineDetails.fallback_note)}</div>` : ''}
                    </div>
                `;
            }
        }

        let quoteHTML = "";
        if (!isUser && engineDetails && engineDetails.matched_quote) {
            quoteHTML = `<div class="quote-highlight-box">"${escapeHTML(engineDetails.matched_quote)}"</div>`;
        }

        wrapper.innerHTML = `
            <span class="sender-label">${escapeHTML(senderName)}</span>
            <div class="message-bubble">${escapeHTML(text)}${quoteHTML}</div>
            <div class="message-meta-bar">
                ${sentimentHTML}
                ${engineBadgeHTML}
                ${diagnosticsHTML}
            </div>
        `;

        chatMessages.appendChild(wrapper);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function disableChatInput(placeholderText) {
        userInput.disabled = true;
        sendBtn.disabled = true;
        userInput.placeholder = placeholderText;
    }

    // Handle Form Submit (Sending Chat)
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text || !activeCharacter) return;

        userInput.value = "";

        // Push User Message
        const userTurn = { sender: "user", text: text, sentiment: null };
        if (!chatHistoryMap[activeCharacter.name]) chatHistoryMap[activeCharacter.name] = [];
        chatHistoryMap[activeCharacter.name].push(userTurn);

        appendMessageBubble("user", text, null);

        // Fetch User Sentiment preview
        try {
            const sentRes = await fetch("/api/sentiment", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });
            if (sentRes.ok) {
                const userSent = await sentRes.json();
                userTurn.sentiment = userSent;
                updateLiveSentiment(userSent);
            }
        } catch (err) {
            console.error("Error analyzing user sentiment:", err);
        }

        // Show thinking indicator
        const thinkingWrapper = document.createElement("div");
        thinkingWrapper.className = "message-bubble-wrapper character thinking";
        const thinkingLabel = selectedEngine === "classical" 
            ? "Computing TF-IDF cosine similarity & extracting quotes (Classical NLP)..." 
            : "Thinking in character (Generative LLM)...";
        thinkingWrapper.innerHTML = `
            <span class="sender-label">${escapeHTML(activeCharacter.name)}</span>
            <div class="message-bubble" style="font-style: italic; opacity: 0.7;">${thinkingLabel}</div>
        `;
        chatMessages.appendChild(thinkingWrapper);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Call API
        try {
            // Slicing out the turn we just pushed so apiHistory contains only previous turns
            const apiHistory = chatHistoryMap[activeCharacter.name].slice(0, -1).map(m => ({
                role: m.sender === "user" ? "user" : "assistant",
                content: m.text
            }));

            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    character_name: activeCharacter.name,
                    message: text,
                    model: modelSelector.value,
                    chat_history: apiHistory,
                    engine: selectedEngine
                })
            });

            thinkingWrapper.remove();

            if (res.ok) {
                const data = await res.json();
                const charTurn = {
                    sender: "character",
                    text: data.response,
                    sentiment: data.character_sentiment,
                    engine: data.engine || selectedEngine,
                    engineDetails: data.engine_details
                };

                chatHistoryMap[activeCharacter.name].push(charTurn);
                appendMessageBubble("character", data.response, data.character_sentiment, null, charTurn.engine, charTurn.engineDetails);
                updateLiveSentiment(data.character_sentiment);

                // Update Context Drawer Citations
                updateContextDrawer(data.citations);

                // Speech Output if enabled
                if (ttsEnabled) {
                    speakText(data.response);
                }
            } else {
                const errData = await res.json();
                appendMessageBubble("character", `[Error: ${errData.detail || "Failed to generate character response"}]`);
            }
        } catch (err) {
            thinkingWrapper.remove();
            console.error("Chat API error:", err);
            appendMessageBubble("character", "[Error connecting to server]");
        }
    });

    function updateLiveSentiment(sent) {
        if (!sent) return;
        liveSentimentBadge.style.color = sent.color;
        liveSentimentBadge.innerHTML = `${sent.emoji} ${sent.label} (${sent.compound})`;
    }

    function updateContextDrawer(citations) {
        if (!citations || citations.length === 0) {
            contextContent.innerHTML = `<p class="empty-context">No direct book passages retrieved for this response.</p>`;
            return;
        }

        contextContent.innerHTML = citations.map(c => `
            <div class="citation-card">
                <div class="citation-score">Relevance Score: ${c.relevance_score} (Passage #${c.chunk_id})</div>
                <div>"${escapeHTML(c.snippet)}"</div>
            </div>
        `).join("");
    }

    // Voice Output Speech Synthesis
    toggleVoiceBtn.addEventListener("click", () => {
        ttsEnabled = !ttsEnabled;
        toggleVoiceBtn.classList.toggle("active", ttsEnabled);
        voiceIcon.textContent = ttsEnabled ? "Voice on" : "Voice off";
        if (ttsEnabled) {
            speakText(`Voice synthesis enabled for ${activeCharacter ? activeCharacter.name : 'character'}.`);
        } else {
            synth.cancel();
        }
    });

    function speakText(text) {
        if ('speechSynthesis' in window) {
            synth.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            synth.speak(utterance);
        }
    }

    // Toggle Context Drawer
    toggleContextBtn.addEventListener("click", () => {
        contextDrawer.classList.toggle("hidden");
        toggleContextBtn.classList.toggle("active", !contextDrawer.classList.contains("hidden"));
    });
    closeContextBtn.addEventListener("click", () => {
        contextDrawer.classList.add("hidden");
        toggleContextBtn.classList.remove("active");
    });

    // Learned Story Dossier Modal
    viewDossierBtn.addEventListener("click", () => {
        if (!activeCharacter) return;
        dossierAvatar.textContent = "";
        dossierCharName.textContent = activeCharacter.name;
        dossierCharRole.textContent = activeCharacter.role || "Character";

        dossierBackstory.textContent = activeCharacter.story_backstory || activeCharacter.backstory || "Learned from the book narrative.";
        
        // Traits
        dossierTraits.innerHTML = (activeCharacter.traits || []).map(t => `<span class="trait-tag">${escapeHTML(t)}</span>`).join("");
        
        dossierSpeech.textContent = activeCharacter.speech_style || "Speaks in authentic literary tone.";
        dossierRelationships.textContent = activeCharacter.relationships || "Engages with other figures in the story.";
        
        // Facts
        const facts = activeCharacter.learned_facts || [];
        if (facts.length > 0) {
            dossierFacts.innerHTML = facts.map(f => `<li>${escapeHTML(f)}</li>`).join("");
        } else {
            dossierFacts.innerHTML = `<li>Prominent character acting throughout the book.</li>`;
        }

        dossierModal.classList.remove("hidden");
    });
    closeDossierBtn.addEventListener("click", () => dossierModal.classList.add("hidden"));
    dossierModal.addEventListener("click", (e) => {
        if (e.target === dossierModal) dossierModal.classList.add("hidden");
    });

    // Dual Engine Mode Controller
    function setEngineMode(engine) {
        selectedEngine = engine;
        if (btnEngineLlm) btnEngineLlm.classList.toggle("active", engine === "llm");
        if (btnEngineClassical) btnEngineClassical.classList.toggle("active", engine === "classical");

        if (activeEngineBadge) {
            activeEngineBadge.className = `engine-pill-badge ${engine}`;
            activeEngineBadge.innerHTML = engine === "llm" 
                ? "⚡ Generative LLM" 
                : "🏛️ Classical NLP";
        }

        if (modelSelectWrapper) {
            if (engine === "classical") {
                modelSelectWrapper.style.opacity = "0.4";
                modelSelectWrapper.style.pointerEvents = "none";
                modelSelectWrapper.title = "Local LLM model selection is bypassed in Classical NLP Mode (Zero LLM)";
            } else {
                modelSelectWrapper.style.opacity = "1";
                modelSelectWrapper.style.pointerEvents = "auto";
                modelSelectWrapper.title = "";
            }
        }
    }

    if (btnEngineLlm) btnEngineLlm.addEventListener("click", () => setEngineMode("llm"));
    if (btnEngineClassical) btnEngineClassical.addEventListener("click", () => setEngineMode("classical"));

    // Comparison Modal Handling
    if (openComparisonBtn) {
        openComparisonBtn.addEventListener("click", () => {
            comparisonModal.classList.remove("hidden");
            if (activeCharacter) {
                compareInput.placeholder = `Ask ${activeCharacter.name} a question to compare both engines...`;
            }
        });
    }

    if (closeComparisonBtn) {
        closeComparisonBtn.addEventListener("click", () => comparisonModal.classList.add("hidden"));
    }

    if (comparisonModal) {
        comparisonModal.addEventListener("click", (e) => {
            if (e.target === comparisonModal) comparisonModal.classList.add("hidden");
        });
    }

    // Sample prompt chips in comparison modal
    document.querySelectorAll(".sample-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const query = chip.dataset.query;
            if (compareInput) compareInput.value = query;
            runComparison(query);
        });
    });

    if (compareForm) {
        compareForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const q = compareInput.value.trim();
            if (q) runComparison(q);
        });
    }

    async function runComparison(query) {
        const charName = activeCharacter ? activeCharacter.name : (charactersList[0] ? charactersList[0].name : "Character");
        if (!charName) {
            alert("Please upload a PDF book and select a character first.");
            return;
        }

        comparisonResults.classList.remove("hidden");
        runCompareBtn.disabled = true;
        runCompareBtn.innerHTML = "<span>Analyzing &amp; Comparing...</span>";

        classicalOutputBubble.innerHTML = "<em style='color: var(--text-muted);'>Extracting dialogue &amp; calculating TF-IDF cosine similarity...</em>";
        classicalScoreTag.textContent = "Processing...";
        classicalDiagnostics.innerHTML = "";

        llmOutputBubble.innerHTML = "<em style='color: var(--text-muted);'>Querying local Ollama neural model...</em>";
        llmModelTag.textContent = modelSelector.value;
        llmDiagnostics.innerHTML = "";

        try {
            const res = await fetch("/api/compare", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    character_name: charName,
                    message: query,
                    model: modelSelector.value
                })
            });

            if (res.ok) {
                const data = await res.json();
                const classical = data.classical;
                const llm = data.llm;

                // 1. Render Classical Output
                classicalScoreTag.textContent = `Score: ${classical.engine_details?.similarity_score !== undefined ? classical.engine_details.similarity_score : '1.0'}`;
                let classicalHTML = escapeHTML(classical.response);
                if (classical.engine_details?.matched_quote) {
                    classicalHTML += `<div class="quote-highlight-box">"${escapeHTML(classical.engine_details.matched_quote)}"</div>`;
                }
                classicalOutputBubble.innerHTML = classicalHTML;

                classicalDiagnostics.innerHTML = `
                    <div><strong>Intent:</strong> ${escapeHTML(classical.engine_details?.intent || "N/A")}</div>
                    <div><strong>Method:</strong> ${escapeHTML(classical.engine_details?.method || "Classical NLP")}</div>
                    <div><strong>Explanation:</strong> ${escapeHTML(classical.engine_details?.explanation || "")}</div>
                `;

                // 2. Render LLM Output
                llmModelTag.textContent = `Model: ${llm.engine_details?.model_used || modelSelector.value}`;
                llmOutputBubble.innerHTML = escapeHTML(llm.response);

                llmDiagnostics.innerHTML = `
                    <div><strong>Method:</strong> ${escapeHTML(llm.engine_details?.method || "Autoregressive LLM")}</div>
                    <div><strong>RAG Context Chunks:</strong> ${llm.engine_details?.context_chunks_used || 0} excerpts</div>
                    <div><strong>Explanation:</strong> ${escapeHTML(llm.engine_details?.explanation || "")}</div>
                `;
            } else {
                const err = await res.json();
                classicalOutputBubble.textContent = `Error: ${err.detail || "Failed to compare"}`;
                llmOutputBubble.textContent = `Error: ${err.detail || "Failed to compare"}`;
            }
        } catch (err) {
            console.error("Comparison failed:", err);
            classicalOutputBubble.textContent = `Network error connecting to API`;
            llmOutputBubble.textContent = `Network error connecting to API`;
        } finally {
            runCompareBtn.disabled = false;
            runCompareBtn.innerHTML = "<span>Run Side-by-Side Comparison</span>";
        }
    }

    // Modal Handling
    openUploadModalBtn.addEventListener("click", () => uploadModal.classList.remove("hidden"));
    closeModalBtn.addEventListener("click", () => uploadModal.classList.add("hidden"));
    cancelUploadBtn.addEventListener("click", () => uploadModal.classList.add("hidden"));
    uploadModal.addEventListener("click", (e) => {
        if (e.target === uploadModal) uploadModal.classList.add("hidden");
    });

    // File Drag & Drop
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--border-dark)";
        dropZone.style.background = "var(--bg-highlight)";
    });
    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "var(--border-color)";
        dropZone.style.background = "#faf9f5";
    });
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--border-color)";
        dropZone.style.background = "#faf9f5";
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        fileNameDisplay.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        fileInfo.classList.remove("hidden");
    }

    removeFileBtn.addEventListener("click", () => {
        selectedFile = null;
        fileInput.value = "";
        fileInfo.classList.add("hidden");
    });

    // Upload Form Submit
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!selectedFile) {
            alert("Please select a PDF file to upload.");
            return;
        }

        processingLoader.classList.remove("hidden");
        updateUploadProgress(5, "Uploading book file...", "Transferring file to backend storage.");

        const formData = new FormData();
        formData.append("model", modelSelector.value);
        formData.append("file", selectedFile);
        formData.append("stream", "true");

        try {
            const res = await fetch("/api/upload-book", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Upload failed with status ${res.status}`);
            }

            let finalData = null;
            const contentType = res.headers.get("content-type") || "";

            if (contentType.includes("ndjson") && res.body) {
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split("\n");
                    buffer = lines.pop(); // keep trailing incomplete chunk

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed) continue;
                        try {
                            const event = JSON.parse(trimmed);
                            if (event.percent !== undefined) {
                                updateUploadProgress(event.percent, event.stage, event.subtext);
                            }
                            if (event.data) {
                                finalData = event.data;
                            }
                        } catch (pe) {
                            console.error("Error parsing NDJSON chunk:", pe);
                        }
                    }
                }
            } else {
                finalData = await res.json();
                updateUploadProgress(100, "Analysis complete!", "Loaded book data.");
            }

            if (!finalData) {
                throw new Error("No analysis data received from server.");
            }

            setTimeout(() => {
                processingLoader.classList.add("hidden");
                uploadModal.classList.add("hidden");

                // Clear temporary file selection
                selectedFile = null;
                fileInput.value = "";
                fileInfo.classList.add("hidden");

                bookState.hasBook = true;
                bookState.title = finalData.book_title;
                bookState.hasCharacters = Boolean(finalData.has_characters || (finalData.characters && finalData.characters.length > 0));
                bookState.genre = finalData.genre;
                bookState.message = finalData.message;
                bookState.indexedChunks = finalData.indexed_chunks;

                chatHistoryMap = {};
                activeCharacter = null;

                if (finalData.characters && finalData.characters.length > 0) {
                    charactersList = finalData.characters;
                    bookState.hasCharacters = true;
                    updateUIForBookState();
                    renderCharacterCards(charactersList);
                    selectCharacter(charactersList[0]);
                } else {
                    charactersList = [];
                    bookState.hasCharacters = false;
                    updateUIForBookState();
                    renderCharacterCards([]);
                }
            }, 350);
        } catch (err) {
            processingLoader.classList.add("hidden");
            console.error("Upload error:", err);
            alert("Error uploading file: " + err.message);
        }
    });

    function setupEventListeners() {
        // Navigation links
        const navEntrance = document.getElementById("nav-entrance");
        const navSociety = document.getElementById("nav-society");
        const navMembership = document.getElementById("nav-membership");
        const menuToggleBtn = document.getElementById("menu-toggle-btn");
        const characterSidebar = document.getElementById("character-sidebar");

        if (navEntrance) {
            navEntrance.addEventListener("click", (e) => {
                e.preventDefault();
                document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
                navEntrance.classList.add("active");
                if (!userInput.disabled) userInput.focus();
            });
        }

        if (navSociety) {
            navSociety.addEventListener("click", (e) => {
                e.preventDefault();
                document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
                navSociety.classList.add("active");
                if (characterSidebar) {
                    characterSidebar.scrollIntoView({ behavior: "smooth" });
                    characterSidebar.style.outline = "2px solid var(--border-dark)";
                    setTimeout(() => characterSidebar.style.outline = "none", 1200);
                }
            });
        }

        if (navMembership) {
            navMembership.addEventListener("click", (e) => {
                e.preventDefault();
                document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
                navMembership.classList.add("active");
                uploadModal.classList.remove("hidden");
            });
        }

        if (menuToggleBtn && characterSidebar) {
            menuToggleBtn.addEventListener("click", () => {
                characterSidebar.classList.toggle("hidden-mobile");
            });
        }

        // Real-time input sentiment preview
        userInput.addEventListener("input", debounce(async () => {
            const val = userInput.value.trim();
            if (val.length > 3) {
                try {
                    const res = await fetch("/api/sentiment", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ text: val })
                    });
                    if (res.ok) {
                        const s = await res.json();
                        updateLiveSentiment(s);
                    }
                } catch (e) {}
            }
        }, 300));
    }

    function escapeHTML(str) {
        if (!str) return "";
        return String(str).replace(/[&<>'"]/g, 
            tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
        );
    }

    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }
});
