// main.js
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs').promises; 
const { EPub } = require("node-epub");
const pdf = require('pdf-parse');
const { promisify } = require('util');
const sqlite3 = require('sqlite3').verbose();

// --- Database Setup ---
const dbPath = path.join(app.getPath('userData'), 'bookshelf.db');
let db;

/**
 * Initialises the SQLite database.
 */
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
                audioPath TEXT
            )`, (err) => {
                if (err) console.error("Error creating table:", err.message);
            });
        }
    });
}

// Promisify DB functions for async/await
const dbAll = (sql, params = []) => {
    return new Promise((resolve, reject) => {
        if (!db) return reject(new Error("Database not initialised."));
        db.all(sql, params, (err, rows) => {
            if (err) return reject(err);
            resolve(rows);
        });
    });
};

const dbRun = (sql, params = []) => {
    return new Promise((resolve, reject) => {
        if (!db) return reject(new Error("Database not initialised."));
        db.run(sql, params, function (err) {
            if (err) return reject(err);
            resolve({ lastID: this.lastID, changes: this.changes });
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
    // mainWindow.webContents.openDevTools();
}

// --- App Lifecycle ---
app.whenReady().then(() => {
    initialiseDatabase();
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        if (db) {
            db.close((err) => {
                if (err) console.error("Error closing database:", err.message);
            });
        }
        app.quit();
    }
});

// --- IPC Handlers ---

// File Dialog
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

// Reads audio file as a Base64 Data URL for playback
ipcMain.handle('audio:readFile', async (event, filePath) => {
    try {
        const audioBuffer = await fs.readFile(filePath);
        const audioBase64 = audioBuffer.toString('base64');
        return `data:audio/mp3;base64,${audioBase64}`;
    } catch (error) {
        console.error('Error reading audio file:', error);
        throw error;
    }
});


ipcMain.handle('audio:save', async (event, audioData, title) => {
    try {
        const audioDir = path.join(app.getPath('userData'), 'audiobooks');
        await fs.mkdir(audioDir, { recursive: true });
        
        const fileName = `${title.replace(/[^a-z0-9]/gi, '_')}_${Date.now()}.mp3`;
        const audioPath = path.join(audioDir, fileName);
        
        await fs.writeFile(audioPath, Buffer.from(audioData));
        
        return audioPath;
    } catch (error) {
        console.error('Error saving audio:', error);
        throw error;
    }
});

// File Readers
ipcMain.handle('file:readBuffer', async (event, filePath) => {
    return fs.readFile(filePath);
});

ipcMain.handle('file:readText', async (event, filePath) => {
    return fs.readFile(filePath, 'utf-8');
});

// --- File Parsers ---

// EPUB Parser
ipcMain.handle('epub:parse', (event, filePath) => {
    return new Promise((resolve, reject) => {
        const epub = new EPub(filePath);
        epub.on('end', () => {
            const chapters = [];
            epub.flow.forEach(chapter => {
                epub.getChapter(chapter.id, (err, html) => {
                    if (err) {
                        chapters.push({ id: chapter.id, html: `<p>Error loading chapter: ${chapter.id}</p>` });
                    } else {
                        chapters.push({ id: chapter.id, html: html });
                    }
                    
                    if (chapters.length === epub.flow.length) {
                        resolve({
                            title: epub.metadata.title || path.basename(filePath, '.epub'),
                            author: epub.metadata.creator || 'Unknown Author',
                            year: epub.metadata.date ? new Date(epub.metadata.date).getFullYear().toString() : 'N/A',
                            chapters: chapters.map(c => c.html)
                        });
                    }
                });
            });
        });
        epub.on('error', reject);
        epub.parse();
    });
});

// PDF Parser
ipcMain.handle('pdf:parse', async (event, filePath) => {
    try {
        const dataBuffer = await fs.readFile(filePath);
        const data = await pdf(dataBuffer);
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
            chapters: chapters.filter(c => c.length > 0)
        };
    } catch (error) {
        console.error('PDF parsing error:', error);
        throw new Error('Failed to parse PDF: ' + error.message);
    }
});

// --- Database Handlers ---

// Add Book
ipcMain.handle('db:addBook', async (event, book) => {
    try {
        // Use INSERT OR REPLACE to prevent UNIQUE constraint errors
        const sql = `INSERT OR REPLACE INTO books (title, author, filePath, coverUrl, year, progress, audioPath) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)`;
        const result = await dbRun(sql, [
            book.title, book.author, book.filePath, 
            book.coverUrl, book.year, book.progress || 0.0,
            book.audioPath || null // Add audioPath
        ]);
        return { success: true, id: result.lastID };
    } catch (error) {
        console.error('Error adding book to DB:', error.message);
        return { success: false, error: error.message };
    }
});

// Get All Books
ipcMain.handle('db:getBooks', async () => {
    try {
        return await dbAll(`SELECT * FROM books ORDER BY title`);
    } catch (error) {
        console.error('Error getting books from DB:', error.message);
        return [];
    }
});

// Delete Book
ipcMain.handle('db:deleteBook', async (event, id) => {
    try {
        await dbRun(`DELETE FROM books WHERE id = ?`, [id]);
        return { success: true };
    } catch (error) {
        console.error('Error deleting book from DB:', error.message);
        return { success: false, error: error.message };
    }
});

// Update Book Progress
ipcMain.handle('db:updateProgress', async (event, id, progress) => {
    try {
        await dbRun(`UPDATE books SET progress = ? WHERE id = ?`, [progress, id]);
        return { success: true };
    } catch (error) {
        console.error('Error updating book progress:', error.message);
        return { success: false, error: error.message };
    }
});