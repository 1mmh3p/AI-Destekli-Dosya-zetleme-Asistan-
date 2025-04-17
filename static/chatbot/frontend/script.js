// Socket connection
const socket = io('http://localhost:3000', {
    reconnection: true,
    
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
    timeout: 20000,
    transports: ['websocket']
});

  
// Connection status handlers
socket.on('connection_ack', (data) => {
    console.log('Server connection acknowledgment:', data);
    addMessageToChat('bot', data.message);
});

socket.on('connect', () => {
    console.log('Connected to server');
    document.querySelector('.status-indicator').style.backgroundColor = '#4CAF50';
    document.querySelector('.chat-status span:last-child').textContent = 'Çevrimiçi';
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    document.querySelector('.status-indicator').style.backgroundColor = '#f44336';
    document.querySelector('.chat-status span:last-child').textContent = 'Çevrimdışı';
});

socket.on('connect_error', (err) => {
    console.error('Connection error:', err);
    addMessageToChat('bot', `Bağlantı hatası: ${err.message}`);
});

// DOM elements
const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const faqSuggestions = document.getElementById('faq-suggestions');

// Send message function
function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    addMessageToChat('user', message);
    messageInput.value = '';

    socket.emit('userMessage', message, (ack) => {
        console.log('Server acknowledgment:', ack);
        if (!ack || ack.status !== 'received') {
            addMessageToChat('bot', 'Mesaj gönderilemedi, lütfen daha sonra tekrar deneyin');
        }
    });
    showTypingIndicator();
}

// Add message to chat
function addMessageToChat(sender, content, type = 'text') {
    const messageElement = document.createElement('div');
    
    if (type === 'rich') {
        messageElement.className = 'rich-message';
        messageElement.innerHTML = `
            <h3>${content.title || 'Hukuki Bilgi'}</h3>
            <p>${content.text}</p>
            ${content.links ? `
                <div class="legal-links">
                    ${content.links.map(link => `
                        <a href="${link.url}" target="_blank">
                            <i class="fas fa-external-link-alt"></i>
                            ${link.text}
                        </a>
                    `).join('')}
                </div>
            ` : ''}
        `;
    } else {
        messageElement.className = sender === 'user' ? 'user-message message' : 'bot-message message';
        messageElement.textContent = content;
    }
    
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Typing indicator
function showTypingIndicator() {
    const typingElement = document.createElement('div');
    typingElement.className = 'typing-indicator';
    typingElement.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <span style="margin-left: 10px;">Yazıyor...</span>
    `;
    chatContainer.appendChild(typingElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return typingElement;
}

function hideTypingIndicator(element) {
    if (element && element.parentNode) {
        element.parentNode.removeChild(element);
    }
}

/// تحديث معالجة استجابة البوت
socket.on('botResponse', (response) => {
    // إخفاء مؤشر الكتابة أولاً
    const typingIndicators = document.querySelectorAll('.typing-indicator');
    typingIndicators.forEach(indicator => {
        chatContainer.removeChild(indicator);
    });

    // معالجة الاستجابة بشكل صحيح
    try {
        if (response && typeof response === 'object') {
            if (response.status === 'success' && response.answer) {
                addMessageToChat('bot', response.answer);
            } 
            else if (response.content) {
                addMessageToChat('bot', response.content);
            }
            else if (response.error) {
                addMessageToChat('bot', `Hata: ${response.error}`);
            }
            else {
                console.error('Geçersiz yanıt formatı:', response);
                addMessageToChat('bot', 'Geçersiz yanıt formatı alındı');
            }
        } else {
            addMessageToChat('bot', response || 'Boş yanıt alındı');
        }
    } catch (error) {
        console.error('Yanıt işleme hatası:', error);
        addMessageToChat('bot', 'Yanıt işlenirken hata oluştu');
    }

    // تحديث الاقتراحات القانونية
    updateLegalSuggestions();
});

// Update legal suggestions
function updateLegalSuggestions() {
    const legalItems = [
        { question: "Türk Ceza Kanunu'nda yeni değişiklikler nelerdir?", answer: "Son TCK değişiklikleri şunları içerir..." },
        { question: "İş hukukunda işçi hakları nelerdir?", answer: "İşçilerin temel hakları..." },
        { question: "Türkiye'de boşanma süreci nasıl işler?", answer: "Boşanma süreci Medeni Kanun'da..." },
        { question: "Türk vatandaşlığı nasıl alınır?", answer: "Vatandaşlık başvuru süreci..." }
    ];
    
    const randomLegal = getRandomItems(legalItems, 4);
    faqSuggestions.innerHTML = randomLegal.map(item => `
        <div class="faq-suggestion" onclick="askQuestion('${item.question.replace(/'/g, "\\'")}')">
            ${item.question}
        </div>
    `).join('');
}

function getRandomItems(arr, num) {
    const shuffled = [...arr].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, num);
}

function askQuestion(question) {
    messageInput.value = question;
    
    const userMessage = document.createElement('div');
    userMessage.className = 'user-message message';
    userMessage.textContent = question;
    chatContainer.appendChild(userMessage);
    
    messageInput.value = '';
    
    const typingIndicator = showTypingIndicator();
    
    socket.emit('userMessage', question, (ack) => {
        if (!ack || ack.status !== 'received') {
            hideTypingIndicator(typingIndicator);
            addMessageToChat('bot', 'Soru gönderilemedi, lütfen tekrar deneyin');
        }
    });
    
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Global functions
window.sendMessage = sendMessage;
window.askQuestion = askQuestion;

// Event listeners
sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    updateLegalSuggestions();
});