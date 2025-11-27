// main.js
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs').promises; 
const EPub = require("epub"); // Changed from const { EPub } = require("node-epub")
const sqlite3 = require('sqlite3').verbose();

// --- ROBUST PDF-PARSE IMPORT ---
let pdfParseLib = require('pdf-parse');
if (typeof pdfParseLib !== 'function' && pdfParseLib.default) {
    pdfParseLib = pdfParseLib.default;
}
if (typeof pdfParseLib !== 'function') {
    console.error("CRITICAL ERROR: pdf-parse did not load as a function. Type:", typeof pdfParseLib);
}

// --- Database Setup ---
const dbPath = path.join(app.getPath('userData'), 'bookshelf.db');
let db;

function initialiseDatabase() {
    db = new sqlite3.Database(dbPath, (err) => {
        if (err) {
            console.error("Database connection error:", err.message);
        } else {
            console.log("Database connected successfully at:", dbPath);
            db.run(`CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                author TEXT,
                filePath TEXT UNIQUE,
                coverUrl TEXT,
                year TEXT,
                progress REAL DEFAULT 0,
                audioPath TEXT,
                timestamps TEXT,
                currentPage INTEGER DEFAULT 0
            )`, (err) => {
                if (err) console.error("Error creating table:", err.message);
            });
        }
    });
}

const dbRun = (sql, params = []) => {
    return new Promise((resolve, reject) => {
        if (!db) return reject(new Error("Database not initialised."));
        db.run(sql, params, function (err) {
            if (err) return reject(err);
            resolve({ lastID: this.lastID, changes: this.changes });
        });
    });
};

const dbAll = (sql, params = []) => {
    return new Promise((resolve, reject) => {
        if (!db) return reject(new Error("Database not initialised."));
        db.all(sql, params, (err, rows) => {
            if (err) return reject(err);
            resolve(rows);
        });
    });
};

// --- Window Creation ---
function createWindow() {
    const mainWindow = new BrowserWindow({
        width: 1200, 
        height: 800, 
        minWidth: 940,
        minHeight: 600,
        backgroundColor: '#1a1a2e', 
        frame: true, 
        title: 'Narrify',
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'), 
            contextIsolation: true,
            nodeIntegration: false,
        }
    });
    mainWindow.loadFile('index.html');
}

app.whenReady().then(() => {
    initialiseDatabase();
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        if (db) db.close();
        app.quit();
    }
});

// --- IPC Handlers ---

// 1. File Dialogs
ipcMain.handle('dialog:openFile', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
        title: 'Upload Book',
        properties: ['openFile'],
        filters: [
            { name: 'Books', extensions: ['epub', 'pdf', 'docx', 'odt', 'txt'] },
            { name: 'All Files', extensions: ['*'] }
        ]
    });
    return !canceled && filePaths.length > 0 ? filePaths[0] : null;
});

// 2. File Reading Helpers
ipcMain.handle('file:readBuffer', async (event, filePath) => fs.readFile(filePath));
ipcMain.handle('file:readText', async (event, filePath) => fs.readFile(filePath, 'utf-8'));

ipcMain.handle('audio:readFile', async (event, filePath) => {
    try {
        const audioBuffer = await fs.readFile(filePath);
        return `data:audio/mp3;base64,${audioBuffer.toString('base64')}`;
    } catch (error) {
        console.error('Error reading audio file:', error);
        throw error;
    }
});

// 3. Parsers (Used for Preview AND Generation)
const parseEpubText = (filePath) => {
    return new Promise((resolve, reject) => {
        try {
            const epub = new EPub(filePath);
            epub.on('end', () => {
                let fullText = "";
                let processed = 0;
                
                if (!epub.flow || epub.flow.length === 0) {
                    resolve("");
                    return;
                }
                
                epub.flow.forEach(chapter => {
                    epub.getChapter(chapter.id, (err, text) => {
                        if (!err && text) {
                            const cleanText = text.replace(/<[^>]*>/g, ' ');
                            fullText += cleanText + "\n";
                        }
                        processed++;
                        if (processed === epub.flow.length) resolve(fullText);
                    });
                });
            });
            epub.on('error', (err) => {
                console.error('EPUB parsing error:', err);
                reject(err);
            });
            epub.parse();
        } catch (error) {
            reject(error);
        }
    });
};

ipcMain.handle('epub:parse', (event, filePath) => {
    return new Promise((resolve, reject) => {
        try {
            const epub = new EPub(filePath);
            
            epub.on('end', () => {
                const chapters = [];
                const chapterTitles = [];
                let processed = 0;
                
                if (!epub.flow || epub.flow.length === 0) {
                    resolve({
                        title: epub.metadata.title || path.basename(filePath, '.epub'),
                        author: epub.metadata.creator || 'Unknown Author',
                        year: epub.metadata.date ? new Date(epub.metadata.date).getFullYear().toString() : 'N/A',
                        chapters: ['<p>No chapters found in this EPUB file.</p>'],
                        chapterTitles: ['Chapter 1'],
                        coverUrl: null
                    });
                    return;
                }
                
                epub.flow.forEach((chapter, index) => {
                    epub.getChapter(chapter.id, (err, html) => {
                        if (err) {
                            chapters.push({ id: chapter.id, html: `<p>Error loading chapter</p>`, title: `Chapter ${index + 1}` });
                        } else {
                            // Extract chapter title from HTML
                            const titleMatch = html.match(/<h[1-3][^>]*>(.*?)<\/h[1-3]>/i);
                            const title = titleMatch ? titleMatch[1].replace(/<[^>]*>/g, '').trim() : chapter.title || `Chapter ${index + 1}`;
                            chapters.push({ id: chapter.id, html: html, title: title });
                        }
                        
                        processed++;
                        if (processed === epub.flow.length) {
                            // Get cover image
                            let coverUrl = null;
                            if (epub.metadata.cover) {
                                epub.getImage(epub.metadata.cover, (err, data, mimeType) => {
                                    if (!err && data) {
                                        coverUrl = `data:${mimeType};base64,${data.toString('base64')}`;
                                    }
                                    resolveResult();
                                });
                            } else {
                                resolveResult();
                            }
                            
                            function resolveResult() {
                                resolve({
                                    title: epub.metadata.title || path.basename(filePath, '.epub'),
                                    author: epub.metadata.creator || 'Unknown Author',
                                    year: epub.metadata.date ? new Date(epub.metadata.date).getFullYear().toString() : 'N/A',
                                    chapters: chapters.map(c => c.html),
                                    chapterTitles: chapters.map(c => c.title),
                                    coverUrl: coverUrl
                                });
                            }
                        }
                    });
                });
            });
            
            epub.on('error', (err) => {
                console.error('EPUB parsing error:', err);
                reject(err);
            });
            
            epub.parse();
        } catch (error) {
            console.error('EPUB parse catch error:', error);
            reject(error);
        }
    });
});

ipcMain.handle('pdf:parse', async (event, filePath) => {
    try {
        if (typeof pdfParseLib !== 'function') {
            throw new Error('pdf-parse library failed to load correctly. Please restart the app.');
        }

        const dataBuffer = await fs.readFile(filePath);
        const data = await pdfParseLib(dataBuffer); 
        
        const chapters = data.text.split(/\f/g).map(pageText => {
            return pageText.split(/\n\s*\n/g)
                           .filter(p => p.trim().length > 0)
                           .map(p => `<p>${p.trim()}</p>`)
                           .join('');
        });
        
        return {
            title: data.info.Title || path.basename(filePath, '.pdf'),
            author: data.info.Author || 'Unknown Author',
            year: data.info.CreationDate ? new Date(data.info.CreationDate).getFullYear().toString() : 'N/A',
            chapters: chapters.filter(c => c.length > 0),
            chapterTitles: chapters.map((_, i) => `Page ${i + 1}`)
        };
    } catch (error) {
        throw new Error('Failed to parse PDF: ' + error.message);
    }
});

// 4. Database Handlers
ipcMain.handle('db:addBook', async (event, book) => {
    try {
        const sql = `INSERT OR REPLACE INTO books (title, author, filePath, coverUrl, year, progress, audioPath, timestamps, currentPage) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`;
        const result = await dbRun(sql, [
            book.title, book.author, book.filePath, 
            book.coverUrl, book.year, book.progress || 0.0,
            book.audioPath || null,
            book.timestamps || null,
            book.currentPage || 0
        ]);
        return { success: true, id: result.lastID };
    } catch (error) {
        return { success: false, error: error.message };
    }
});

ipcMain.handle('db:getBooks', async () => {
    try {
        return await dbAll(`SELECT * FROM books ORDER BY title`);
    } catch (error) {
        return [];
    }
});

ipcMain.handle('db:deleteBook', async (event, id) => {
    try {
        await dbRun(`DELETE FROM books WHERE id = ?`, [id]);
        return { success: true };
    } catch (error) {
        return { success: false, error: error.message };
    }
});

ipcMain.handle('db:updateProgress', async (event, id, progress, currentPage) => {
    try {
        await dbRun(`UPDATE books SET progress = ?, currentPage = ? WHERE id = ?`, [progress, currentPage || 0, id]);
        return { success: true };
    } catch (error) {
        return { success: false, error: error.message };
    }
});

// 5. Gradio AI Generation
ipcMain.handle('gradio:generate', async (event, config) => {
    try {
        const { Client } = await import("@gradio/client");

        // 1. Get Dynamic URL
        console.log('[Main] Fetching fresh Gradio URL from Render...');
        const AUTH_TOKEN = "Potato"; 
        const urlResponse = await fetch(`${config.spaceUrl}/current`, {
            headers: { 'X-Auth-Token': AUTH_TOKEN }
        });

        if (!urlResponse.ok) throw new Error(`Failed to fetch URL: ${urlResponse.statusText}`);
        const urlData = await urlResponse.json();
        const actualGradioUrl = urlData.url;

        console.log(`[Main] Connecting to: ${actualGradioUrl}`);
        const app = await Client.connect(actualGradioUrl);

        // 2. Extract Text from File
        console.log(`[Main] extracting text from ${config.filePath}...`);
        let extractedText = "";
        const ext = path.extname(config.filePath).toLowerCase();

        if (ext === '.epub') {
            extractedText = await parseEpubText(config.filePath);
        } else if (ext === '.pdf') {
            if (typeof pdfParseLib !== 'function') {
                throw new Error(`pdf-parse library is broken (Type: ${typeof pdfParseLib}). Reinstall via npm.`);
            }
            
            const dataBuffer = await fs.readFile(config.filePath);
            const pdfData = await pdfParseLib(dataBuffer);
            extractedText = pdfData.text;
        } else {
            extractedText = await fs.readFile(config.filePath, 'utf-8');
        }

        if (!extractedText || extractedText.length < 10) {
            throw new Error("Could not extract text from book file.");
        }

        // 3. Call the SINGLE endpoint /predict
        console.log('[Main] Calling /predict...');
        
        const prediction = await app.predict("/predict", { 
            text: extractedText,
            persona: config.persona || "Neutral",
            sfxStyle: config.sfx || "Balanced",    
            voiceAudio: null,
            voiceName: config.voice || "Voice A"   
        });

        console.log('[Main] Prediction received:', prediction);

        // 4. Process Response - Extract both audio and timestamps
        let audioUrl = null;
        let timestamps = null;
        
        const resultData = prediction.data;
        
        // Audio is in data[0] or data[1]
        for (let item of resultData) {
            if (item && typeof item === 'object' && item.url) {
                audioUrl = item.url;
            }
        }
        
        // Timestamps might be in data[2] or another position
        if (resultData[2]) {
            timestamps = resultData[2];
        }

        if (!audioUrl) {
            throw new Error("API returned successfully, but no audio URL was found in the response.");
        }

        // 5. Download and Save Audio
        console.log(`[Main] Downloading audio from ${audioUrl}...`);
        const response = await fetch(audioUrl);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = Buffer.from(arrayBuffer);

        // Determine file extension from URL or default to wav
        let fileExtension = 'wav';
        if (audioUrl.includes('.mp3')) {
            fileExtension = 'mp3';
        } else if (audioUrl.includes('.wav')) {
            fileExtension = 'wav';
        }

        const { app: electronApp } = require('electron');
        const audioDir = path.join(electronApp.getPath('userData'), 'audiobooks');
        await fs.mkdir(audioDir, { recursive: true });
        
        const fileName = `${config.title.replace(/[^a-z0-9]/gi, '_')}_${Date.now()}.${fileExtension}`;
        const savePath = path.join(audioDir, fileName);
        
        await fs.writeFile(savePath, audioBuffer);
        
        return { 
            success: true, 
            audioPath: savePath,
            timestamps: timestamps ? JSON.stringify(timestamps) : null
        };

    } catch (error) {
        console.error('Gradio Generation Error:', error);
        throw new Error(error.message); 
    }
});