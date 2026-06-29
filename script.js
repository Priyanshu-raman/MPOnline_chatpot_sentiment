const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatContainer = document.getElementById('chat-container');

function scrollToBottom() {
    chatContainer.scrollTo({
        top: chatContainer.scrollHeight,
        behavior: 'smooth'
    });
}

function appendMessage(sender, htmlContent) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message fade-in`;

    let avatarHtml = '';
    if (sender === 'bot') {
        avatarHtml = `<div class="avatar-small">🤖</div>`;
    } else {
        avatarHtml = `<div class="avatar-small">👤</div>`;
    }

    msgDiv.innerHTML = `
        ${avatarHtml}
        <div class="message-bubble">
            ${htmlContent}
        </div>
    `;
    chatContainer.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

function showTyping() {
    const typingMsg = document.createElement('div');
    typingMsg.className = 'message bot-message fade-in';
    typingMsg.id = 'typing-indicator-msg';
    typingMsg.innerHTML = `
        <div class="avatar-small">🤖</div>
        <div class="message-bubble">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatContainer.appendChild(typingMsg);
    scrollToBottom();
}

function hideTyping() {
    const typingIndicator = document.getElementById('typing-indicator-msg');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage('user', `<p>${text}</p>`);
    userInput.value = '';
    
    showTyping();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        const data = await response.json();
        hideTyping();

        if (response.ok) {
            appendMessage('bot', `<p>${data.bot_response}</p>`);
        } else {
            appendMessage('bot', `<p style="color: #ef4444;">Error: ${data.detail || 'Could not connect to model.'}</p>`);
        }
    } catch (error) {
        hideTyping();
        appendMessage('bot', `<p style="color: #ef4444;">Network error. Please make sure the backend is running.</p>`);
    }
});
