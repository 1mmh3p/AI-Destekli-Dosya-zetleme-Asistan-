require('dotenv').config();
const express = require('express');
const path = require('path');
const http = require('http');
const { Server } = require('socket.io');
const axios = require('axios');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const morgan = require('morgan');
const legalData = require('./data/legal.json');
const stringSimilarity = require('string-similarity');
const natural = require('natural');
const tokenizer = new natural.WordTokenizer();
const stemmer = natural.PorterStemmer;

const config = {
  port: process.env.PORT || 3000,
  openRouter: {
    url: 'https://openrouter.ai/api/v1/chat/completions',
    apiKey: process.env.OPENROUTER_API_KEY,
    models: {
      gpt3: "openai/gpt-3.5-turbo",
      claude: "anthropic/claude-2"
    }
  },
  security: {
    rateLimit: {
      windowMs: 15 * 60 * 1000,
      max: 100
    }
  }
};

const app = express();

app.use(helmet());
app.use(cors({
  origin: ['http://localhost:5000'], // Frontend kaynağınız
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type'],
  credentials: true
}));

app.use(express.json());
app.use(morgan('combined'));

const limiter = rateLimit(config.security.rateLimit);
app.use(limiter);

app.use(express.static(path.join(__dirname, '../frontend')));

const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "http://localhost:5000",
    methods: ["GET", "POST"]
  }
});

// Metin ön işleme fonksiyonu
function preprocessText(text) {
  const tokens = tokenizer.tokenize(text.toLowerCase());
  return tokens.map(token => stemmer.stem(token));
}

// Yasal eşleşmeyi bulma fonksiyonu
function findBestLegalMatch(userInput) {
  const processedInput = preprocessText(userInput).join(' ');
  const questions = legalData.map(item => ({
    original: item.question,
    processed: preprocessText(item.question).join(' ')
  }));
  
  const matches = stringSimilarity.findBestMatch(processedInput, questions.map(q => q.processed));
  
  if (matches.bestMatch.rating > 0.65) {
    return {
      ...legalData.find(item => item.question === questions[matches.bestMatchIndex].original),
      rating: matches.bestMatch.rating
    };
  }
  return null;
}

// Yasal yanıt alma fonksiyonu
async function getLegalResponse(message, context = []) {
  try {
    const supportedContext = context.map(msg => ({
      ...msg,
      role: msg.role === 'bot' ? 'assistant' : msg.role
    }));

    const response = await axios.post(config.openRouter.url, {
      model: config.openRouter.models.gpt3,
      messages: [
        {
          role: "system",
          content: `Sen Türk hukuk sistemi konusunda uzman bir yapay zeka asistanısın. 
                    Sadece Türkiye'nin yasal mevzuatı hakkında bilgi ver. 
                    Yanıtlarında ilgili kanun maddelerine atıf yap. 
                    Yanıtlarını Türkçe olarak ver. 
                    Belirsiz konularda "Bu konuda kesin bir hüküm bulunmamaktadır" gibi yanıtlar ver.`
        },
        ...supportedContext.filter(msg => ['system', 'user', 'assistant'].includes(msg.role)),
        { role: "user", content: message }
      ],
      temperature: 0.5,
      max_tokens: 800
    }, {
      headers: {
        'Authorization': `Bearer ${config.openRouter.apiKey}`,
        'Content-Type': 'application/json'
      },
      timeout: 10000
    });

    if (!response.data?.choices?.[0]?.message?.content) {
      throw new Error('Geçersiz API yanıtı');
    }

    return {
      text: response.data.choices[0].message.content,
      model: response.data.model,
      usage: response.data.usage
    };
  } catch (error) {
    console.error("API Hatası:", error.response?.data || error.message);
    return {
      text: "Üzgünüm, bir hata oluştu. Lütfen daha sonra tekrar deneyin.",
      error: true
    };
  }
}

const userSessions = new Map();

io.on('connection', (socket) => {
  console.log(`New connection: ${socket.id}`);
  
  userSessions.set(socket.id, {
    id: socket.id,
    ip: socket.handshake.address,
    joinedAt: new Date(),
    conversationHistory: [{
      role: 'bot',
      content: 'Merhaba! Ben Türk hukuk sistemi hakkında bilgi sağlayan bir yapay zeka asistanıyım. Size nasıl yardımcı olabilirim?',
      timestamp: new Date()
    }]
  });

  socket.on('userMessage', async (msg) => {
    if (!msg || typeof msg !== 'string' || msg.length > 500) {
      return socket.emit('error', 'Geçersiz mesaj formatı veya uzunluğu');
    }
    
    const session = userSessions.get(socket.id);
    session.conversationHistory.push({
      role: 'user',
      content: msg,
      timestamp: new Date()
    });
    
    try {
      const legalMatch = findBestLegalMatch(msg);
      if (legalMatch) {
        const response = processLegalAnswer(legalMatch.answer);
        session.conversationHistory.push({
          role: 'bot',
          content: response,
          source: 'legal_db',
          timestamp: new Date()
        });
        return socket.emit('botResponse', {
          type: 'legal',
          content: response,
          metadata: {
            question: legalMatch.question,
            confidence: legalMatch.rating
          }
        });
      }
      
      const legalResponse = await getLegalResponse(msg, session.conversationHistory);
      session.conversationHistory.push({
        role: 'bot',
        content: legalResponse.text,
        source: 'ai',
        model: legalResponse.model,
        timestamp: new Date()
      });
      
      socket.emit('botResponse', {
        type: 'ai',
        content: legalResponse.text,
        metadata: {
          model: legalResponse.model,
          usage: legalResponse.usage
        }
      });
    } catch (error) {
      console.error('Message processing error:', error);
      socket.emit('error', 'Mesaj işlenirken bir hata oluştu');
    }
  });

  socket.on('disconnect', () => {
    userSessions.delete(socket.id);
    console.log(`User disconnected: ${socket.id}`);
  });
});

// Yasal yanıtı işleme fonksiyonu
function processLegalAnswer(answer) {
  return {
    type: 'rich',
    content: answer // Doğrudan bir string ise dönüştürmüyoruz
  };
}

app.get('/api/legal', (req, res) => {
  res.json({
    count: legalData.length,
    items: legalData
  });
});

app.post('/api/ask-legal', async (req, res) => {
  const { question } = req.body;
  if (!question) {
    return res.status(400).json({ error: 'Soru gereklidir' });
  }
  
  const legalMatch = findBestLegalMatch(question);
  if (legalMatch) {
    return res.json({
      source: 'legal_db',
      answer: processLegalAnswer(legalMatch.answer),
      question: legalMatch.question
    });
  }
  
  const legalResponse = await getLegalResponse(question);
  res.json({
    source: 'ai',
    answer: legalResponse.text,
    model: legalResponse.model
  });
});

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

// Sunucuyu başlatma
server.listen(config.port, () => {
  console.log(`Socket.IO server running on port ${config.port}`);
}).on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.log(`Port ${config.port} is in use, trying alternative port...`);
    server.listen(0, () => {
      console.log(`Server running on http://localhost:${server.address().port}`);
    });
  }
});

// Hata yakalama
process.on('unhandledRejection', (reason) => {
  console.error('Unhandled Rejection:', reason);
});