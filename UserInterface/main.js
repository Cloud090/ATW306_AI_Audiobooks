// ============================================================================
// MAIN.JS - Electron Main Process
// ============================================================================
// This is the main process file for the Electron application.
// It handles:
// - Application lifecycle (startup, shutdown)
// - Window creation and management
// - Database operations (SQLite)
// - File system operations (reading books, copying files)
// - IPC (Inter-Process Communication) between main and renderer processes
// - EPUB/PDF parsing
// - AI audiobook generation via Gradio API
// ============================================================================

// Import required Electron modules
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
// app: Controls application lifecycle
// BrowserWindow: Creates and manages application windows
// ipcMain: Handles communication from renderer process
// dialog: Shows native system dialogs (file picker, alerts)

const path = require('path');           // Path manipulation utilities
const fs = require('fs').promises;       // File system operations (async/await version)
const EPub = require("epub");            // Library for parsing EPUB ebook files
const sqlite3 = require('sqlite3').verbose(); // SQLite database for storing book data

// ============================================================================
// PDF-PARSE LIBRARY IMPORT WITH COMPATIBILITY HANDLING
// ============================================================================
// The pdf-parse library can export itself differently depending on how it's built
// This robust import handles both CommonJS and ES Module exports
let pdfParseLib = require('pdf-parse');

// Some versions export as { default: function } instead of direct function
if (typeof pdfParseLib !== 'function' && pdfParseLib.default) {
    pdfParseLib = pdfParseLib.default;
}

// If still not a function, log critical error
// This prevents crashes later when we try to call pdfParseLib()
if (typeof pdfParseLib !== 'function') {
    console.error("CRITICAL ERROR: pdf-parse did not load as a function. Type:", typeof pdfParseLib);
}

// ============================================================================
// DATABASE SETUP
// ============================================================================
// SQLite database stores all book metadata, reading progress, and audio paths
// Located in app's user data directory (persists between app sessions)

// Get platform-specific app data path (e.g., ~/Library/Application Support/Narrify on macOS)
const dbPath = path.join(app.getPath('userData'), 'bookshelf.db');
let db; // Database connection instance

/**
 * Initialises the SQLite database and creates the books table if it doesn't exist
 * Called once when the app starts
 */
function initialiseDatabase() {
    db = new sqlite3.Database(dbPath, (err) => {
        if (err) {
            console.error("Database connection error:", err.message);
        } else {
            console.log("Database connected successfully at:", dbPath);
            
            // Create books table with schema:
            // - id: Auto-incrementing primary key
            // - title, author, year: Book metadata
            // - filePath: Location of book file on disk (UNIQUE to prevent duplicates)
            // - coverUrl: Base64-encoded cover image or URL
            // - progress: Reading progress as decimal (0.0 = unread, 1.0 = finished)
            // - audioPath: Location of generated audiobook file
            // - timestamps: JSON string mapping text positions to audio timestamps
            // - currentPage: Last page user was reading
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

/**
 * Promisified wrapper for database INSERT/UPDATE/DELETE operations
 * Converts callback-based db.run() to async/await style
 * 
 * @param {string} sql - SQL query with ? placeholders
 * @param {Array} params - Values to substitute for placeholders
 * @returns {Promise<{lastID: number, changes: number}>} - Last inserted ID and number of affected rows
 */
const dbRun = (sql, params = []) => {
    return new Promise((resolve, reject) => {
        if (!db) return reject(new Error("Database not initialised."));
        
        db.run(sql, params, function (err) {
            if (err) return reject(err);
            // 'this' context provides lastID (for INSERT) and changes (for UPDATE/DELETE)
            resolve({ lastID: this.lastID, changes: this.changes });
        });
    });
};

/**
 * Promisified wrapper for database SELECT operations
 * Returns all matching rows as an array
 * 
 * @param {string} sql - SQL query with ? placeholders
 * @param {Array} params - Values to substitute for placeholders
 * @returns {Promise<Array>} - Array of result rows
 */
const dbAll = (sql, params = []) => {
    return new Promise((resolve, reject) => {
        if (!db) return reject(new Error("Database not initialised."));
        
        db.all(sql, params, (err, rows) => {
            if (err) return reject(err);
            resolve(rows);
        });
    });
};

// ============================================================================
// FILE MANAGEMENT HELPER
// ============================================================================

/**
 * Copies uploaded book file to app's data directory
 * This ensures the file remains accessible even if the user moves/deletes the original
 * 
 * @param {string} originalPath - Path to the user's original file
 * @returns {Promise<string>} - Path to the copied file in app data
 */
async function copyBookToAppData(originalPath) {
    try {
        // Create books directory in app data if it doesn't exist
        const booksDir = path.join(app.getPath('userData'), 'books');
        await fs.mkdir(booksDir, { recursive: true });
        
        // Generate unique filename by appending timestamp
        // This prevents conflicts if user uploads multiple files with the same name
        const fileName = path.basename(originalPath);
        const timestamp = Date.now();
        const ext = path.extname(fileName);              // e.g., ".epub"
        const baseName = path.basename(fileName, ext);   // e.g., "MyBook"
        const newFileName = `${baseName}_${timestamp}${ext}`; // e.g., "MyBook_1234567890.epub"
        const newPath = path.join(booksDir, newFileName);
        
        // Copy file to app data directory
        await fs.copyFile(originalPath, newPath);
        console.log(`File copied from ${originalPath} to ${newPath}`);
        
        return newPath;
    } catch (error) {
        console.error('Error copying book file:', error);
        throw error;
    }
}

// ============================================================================
// WINDOW CREATION
// ============================================================================

/**
 * Creates the main application window
 * Configures window properties, security settings, and loads the UI
 */
function createWindow() {
    const mainWindow = new BrowserWindow({
        width: 1200,          // Initial window width
        height: 800,          // Initial window height
        minWidth: 940,        // Minimum width to ensure UI remains usable
        minHeight: 600,       // Minimum height to ensure UI remains usable
        backgroundColor: '#1a1a2e',  // Dark background matches app theme
        frame: true,          // Show native window frame (title bar, close button)
        title: 'Narrify',     // Window title shown in taskbar/dock
        autoHideMenuBar: true, // Hide menu bar (File, Edit, etc.) for cleaner look
        webPreferences: {
            // Security configuration
            preload: path.join(__dirname, 'preload.js'),  // Runs before renderer, sets up IPC bridge
            contextIsolation: true,   // Isolates renderer context from Node.js (security best practice)
            nodeIntegration: false,   // Disable direct Node.js access in renderer (security)
            devTools: false,          // Disable developer tools in production
        }
    });
    
    // Load the main HTML file
    mainWindow.loadFile('index.html');
}

// ============================================================================
// APPLICATION LIFECYCLE
// ============================================================================

// When Electron finishes initialising
app.whenReady().then(() => {
    initialiseDatabase();  // Set up database before creating window
    createWindow();        // Create and show main window

    // macOS-specific: Re-create window when dock icon is clicked and no windows exist
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

// When all windows are closed
app.on('window-all-closed', () => {
    // On macOS, apps typically stay open in the dock even with no windows
    // On Windows/Linux, quit the app entirely
    if (process.platform !== 'darwin') {
        if (db) db.close();  // Close database connection cleanly
        app.quit();
    }
});

// ============================================================================
// IPC HANDLERS
// ============================================================================
// These handlers respond to requests from the renderer process (UI)
// All communication between main and renderer uses these secure IPC channels

// ----------------------------------------------------------------------------
// 1. FILE DIALOG HANDLERS
// ----------------------------------------------------------------------------

/**
 * Opens native file picker dialog for book upload
 * Returns selected file path or null if cancelled
 */
ipcMain.handle('dialog:openFile', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
        title: 'Upload Book',
        properties: ['openFile'],  // Single file selection only
        filters: [
            // Show supported file types in dialog
            { name: 'Books', extensions: ['epub', 'pdf', 'docx', 'odt', 'txt'] },
            { name: 'All Files', extensions: ['*'] }  // Fallback to see all files
        ]
    });
    
    // Return first selected file, or null if user cancelled
    return !canceled && filePaths.length > 0 ? filePaths[0] : null;
});

// ----------------------------------------------------------------------------
// 2. FILE COPY HANDLER
// ----------------------------------------------------------------------------

/**
 * Copies book file from user's location to app data directory
 * Ensures file persists even if original is moved/deleted
 */
ipcMain.handle('file:copyBook', async (event, originalPath) => {
    return await copyBookToAppData(originalPath);
});

// ----------------------------------------------------------------------------
// 3. FILE READING HANDLERS
// ----------------------------------------------------------------------------

/**
 * Reads file as raw buffer (for binary files like PDFs)
 * Used during PDF parsing and audio file reading
 */
ipcMain.handle('file:readBuffer', async (event, filePath) => fs.readFile(filePath));

/**
 * Reads file as UTF-8 text string (for plain text books)
 */
ipcMain.handle('file:readText', async (event, filePath) => fs.readFile(filePath, 'utf-8'));

/**
 * Reads audio file and converts to base64 data URL
 * Allows renderer to play audio without direct file access
 */
ipcMain.handle('audio:readFile', async (event, filePath) => {
    try {
        const audioBuffer = await fs.readFile(filePath);
        // Convert to data URL format: data:audio/mp3;base64,XXXXXX
        return `data:audio/mp3;base64,${audioBuffer.toString('base64')}`;
    } catch (error) {
        console.error('Error reading audio file:', error);
        throw error;
    }
});

// ----------------------------------------------------------------------------
// 4. BOOK PARSING HANDLERS
// ----------------------------------------------------------------------------
// These extract text and metadata from EPUB and PDF files
// Used both for preview (configure modal) and text-to-speech generation

/**
 * Helper function: Extracts plain text from EPUB file
 * Used during audiobook generation (we need just the text, not HTML)
 * 
 * @param {string} filePath - Path to EPUB file
 * @returns {Promise<string>} - Concatenated text from all chapters
 */
const parseEpubText = (filePath) => {
    return new Promise((resolve, reject) => {
        try {
            const epub = new EPub(filePath);
            
            // 'end' event fires when EPUB metadata is fully parsed
            epub.on('end', () => {
                let fullText = "";
                let processed = 0;  // Counter to track completion
                
                // epub.flow contains ordered list of all chapters/sections
                if (!epub.flow || epub.flow.length === 0) {
                    resolve("");  // Empty EPUB
                    return;
                }
                
                // Iterate through each chapter
                epub.flow.forEach(chapter => {
                    // Asynchronously get chapter content
                    epub.getChapter(chapter.id, (err, text) => {
                        if (!err && text) {
                            // Strip HTML tags to get plain text
                            const cleanText = text.replace(/<[^>]*>/g, ' ');
                            fullText += cleanText + "\n";
                        }
                        
                        processed++;
                        // When all chapters processed, resolve with full text
                        if (processed === epub.flow.length) resolve(fullText);
                    });
                });
            });
            
            epub.on('error', (err) => {
                console.error('EPUB parsing error:', err);
                reject(err);
            });
            
            // Start parsing the EPUB file
            epub.parse();
        } catch (error) {
            reject(error);
        }
    });
};

/**
 * Full EPUB parser for preview and reader display
 * Returns metadata, chapters as HTML, chapter titles, and cover image
 */
ipcMain.handle('epub:parse', (event, filePath) => {
    return new Promise((resolve, reject) => {
        try {
            const epub = new EPub(filePath);
            
            epub.on('end', () => {
                const chapters = [];        // Array of chapter objects
                const chapterTitles = [];   // Array of chapter title strings
                let processed = 0;
                
                // Handle empty EPUB files
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
                
                // Process each chapter
                epub.flow.forEach((chapter, index) => {
                    epub.getChapter(chapter.id, (err, html) => {
                        if (err) {
                            chapters.push({ 
                                id: chapter.id, 
                                html: `<p>Error loading chapter</p>`, 
                                title: `Chapter ${index + 1}` 
                            });
                        } else {
                            // Extract chapter title from HTML heading tags (h1, h2, or h3)
                            const titleMatch = html.match(/<h[1-3][^>]*>(.*?)<\/h[1-3]>/i);
                            const title = titleMatch 
                                ? titleMatch[1].replace(/<[^>]*>/g, '').trim()  // Remove any nested tags
                                : chapter.title || `Chapter ${index + 1}`;      // Fallback
                            
                            chapters.push({ id: chapter.id, html: html, title: title });
                        }
                        
                        processed++;
                        
                        // When all chapters are processed
                        if (processed === epub.flow.length) {
                            // Attempt to extract cover image
                            let coverUrl = null;
                            if (epub.metadata.cover) {
                                epub.getImage(epub.metadata.cover, (err, data, mimeType) => {
                                    if (!err && data) {
                                        // Convert cover to base64 data URL
                                        coverUrl = `data:${mimeType};base64,${data.toString('base64')}`;
                                    }
                                    resolveResult();
                                });
                            } else {
                                resolveResult();
                            }
                            
                            // Helper function to resolve final result
                            function resolveResult() {
                                resolve({
                                    // Extract metadata from EPUB with fallbacks
                                    title: epub.metadata.title || path.basename(filePath, '.epub'),
                                    author: epub.metadata.creator || 'Unknown Author',
                                    year: epub.metadata.date 
                                        ? new Date(epub.metadata.date).getFullYear().toString() 
                                        : 'N/A',
                                    chapters: chapters.map(c => c.html),      // HTML content
                                    chapterTitles: chapters.map(c => c.title), // Chapter names
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

/**
 * PDF parser for preview and reader display
 * Returns metadata and pages as HTML
 */
ipcMain.handle('pdf:parse', async (event, filePath) => {
    try {
        // Check if pdf-parse loaded correctly
        if (typeof pdfParseLib !== 'function') {
            throw new Error('pdf-parse library failed to load correctly. Please restart the app.');
        }

        // Read PDF file as buffer
        const dataBuffer = await fs.readFile(filePath);
        
        // Parse PDF using pdf-parse library
        const data = await pdfParseLib(dataBuffer);
        
        // Split text by form feed character (\f) which separates pages
        // Then wrap paragraphs in <p> tags for HTML rendering
        const chapters = data.text.split(/\f/g).map(pageText => {
            return pageText.split(/\n\s*\n/g)            // Split by double newlines (paragraphs)
                           .filter(p => p.trim().length > 0)  // Remove empty paragraphs
                           .map(p => `<p>${p.trim()}</p>`)    // Wrap in paragraph tags
                           .join('');
        });
        
        return {
            // Extract metadata from PDF info object
            title: data.info.Title || path.basename(filePath, '.pdf'),
            author: data.info.Author || 'Unknown Author',
            year: data.info.CreationDate 
                ? new Date(data.info.CreationDate).getFullYear().toString() 
                : 'N/A',
            chapters: chapters.filter(c => c.length > 0),     // Remove empty pages
            chapterTitles: chapters.map((_, i) => `Page ${i + 1}`)  // Simple page numbers
        };
    } catch (error) {
        throw new Error('Failed to parse PDF: ' + error.message);
    }
});

// ----------------------------------------------------------------------------
// 5. DATABASE HANDLERS
// ----------------------------------------------------------------------------
// CRUD operations for the books table

/**
 * Adds a new book to the database or updates existing (based on filePath)
 * filePath is unique, so uploading same file again will update, not duplicate
 */
ipcMain.handle('db:addBook', async (event, book) => {
    try {
        // Validate required field
        if (!book.filePath) {
            throw new Error('Cannot save book: file path is missing');
        }
        
        // INSERT OR REPLACE: Updates if filePath already exists, inserts if new
        const sql = `INSERT OR REPLACE INTO books (title, author, filePath, coverUrl, year, progress, audioPath, timestamps, currentPage) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`;
        
        const result = await dbRun(sql, [
            book.title, 
            book.author, 
            book.filePath, 
            book.coverUrl, 
            book.year, 
            book.progress || 0.0,        // Default to 0% progress
            book.audioPath || null,      // Null if no audiobook yet
            book.timestamps || null,     // Null if no timestamps
            book.currentPage || 0        // Start at page 0
        ]);
        
        return { success: true, id: result.lastID };
    } catch (error) {
        return { success: false, error: error.message };
    }
});

/**
 * Retrieves all books from database, ordered alphabetically by title
 */
ipcMain.handle('db:getBooks', async () => {
    try {
        return await dbAll(`SELECT * FROM books ORDER BY title`);
    } catch (error) {
        console.error('Error fetching books:', error);
        return [];  // Return empty array on error (UI handles gracefully)
    }
});

/**
 * Deletes a book from the database by ID
 * Note: Does not delete the actual file on disk (for safety)
 */
ipcMain.handle('db:deleteBook', async (event, id) => {
    try {
        await dbRun(`DELETE FROM books WHERE id = ?`, [id]);
        return { success: true };
    } catch (error) {
        return { success: false, error: error.message };
    }
});

/**
 * Updates reading progress and current page for a book
 * Called when user navigates pages in reader view
 */
ipcMain.handle('db:updateProgress', async (event, id, progress, currentPage) => {
    try {
        await dbRun(
            `UPDATE books SET progress = ?, currentPage = ? WHERE id = ?`, 
            [progress, currentPage || 0, id]
        );
        return { success: true };
    } catch (error) {
        return { success: false, error: error.message };
    }
});

// ----------------------------------------------------------------------------
// 6. GRADIO AI AUDIOBOOK GENERATION
// ----------------------------------------------------------------------------
// Connects to remote Gradio API for text-to-speech generation with streaming

/**
 * Generates audiobook from book text using AI narration
 * 
 * Process:
 * 1. Fetch dynamic Gradio URL from Render deployment
 * 2. Extract text from book file (EPUB/PDF/TXT)
 * 3. Send text + configuration to Gradio API
 * 4. Stream progress updates back to UI
 * 5. Download generated audio file
 * 6. Save audio to app data directory
 * 
 * @param {Object} config - Generation configuration
 * @param {string} config.spaceUrl - Base URL for Gradio space
 * @param {string} config.filePath - Path to book file
 * @param {string} config.title - Book title (for filename)
 * @param {string} config.persona - Narrator style (Neutral/Dramatic/etc)
 * @param {string} config.sfx - Sound effects intensity
 * @param {string} config.voice - Voice ID to use
 * @returns {Promise<{success: boolean, audioPath: string, timestamps: string}>}
 */
ipcMain.handle('gradio:generate', async (event, config) => {
    try {
        // Import Gradio client library (ES module dynamic import)
        const { Client } = await import("@gradio/client");

        // --------------------------------------------------------------------
        // STEP 1: Get Dynamic Gradio URL
        // --------------------------------------------------------------------
        // The Gradio space runs on Render with dynamic URLs that change
        // We fetch the current URL from an authentication endpoint
        console.log('[Main] Fetching fresh Gradio URL from Render...');
        
        const AUTH_TOKEN = "Potato";  // Simple auth token for URL endpoint
        const urlResponse = await fetch(`${config.spaceUrl}/current`, {
            headers: { 'X-Auth-Token': AUTH_TOKEN }
        });

        if (!urlResponse.ok) {
            throw new Error(`Failed to fetch URL: ${urlResponse.statusText}`);
        }
        
        const urlData = await urlResponse.json();
        const actualGradioUrl = urlData.url;  // The actual Gradio space URL

        console.log(`[Main] Connecting to: ${actualGradioUrl}`);
        const app = await Client.connect(actualGradioUrl);

        // --------------------------------------------------------------------
        // STEP 2: Extract Text from Book File
        // --------------------------------------------------------------------
        // Different extraction methods for different file types
        console.log(`[Main] extracting text from ${config.filePath}...`);
        let extractedText = "";
        const ext = path.extname(config.filePath).toLowerCase();

        if (ext === '.epub') {
            // Use EPUB text parser (strips HTML tags)
            extractedText = await parseEpubText(config.filePath);
            
        } else if (ext === '.pdf') {
            // Use PDF parser to extract text
            if (typeof pdfParseLib !== 'function') {
                throw new Error(`pdf-parse library is broken (Type: ${typeof pdfParseLib}). Reinstall via npm.`);
            }
            
            const dataBuffer = await fs.readFile(config.filePath);
            const pdfData = await pdfParseLib(dataBuffer);
            extractedText = pdfData.text;
            
        } else {
            // For TXT/DOCX/ODT, read as plain text
            extractedText = await fs.readFile(config.filePath, 'utf-8');
        }

        // Validate we got meaningful text
        if (!extractedText || extractedText.length < 10) {
            throw new Error("Could not extract text from book file.");
        }

        // --------------------------------------------------------------------
        // STEP 3: Call Gradio API with Streaming
        // --------------------------------------------------------------------
        // Send text and configuration to the /predict endpoint
        // This returns an async iterator that streams progress updates
        console.log('[Main] Calling /predict with streaming...');
        
        const prediction = app.submit("/predict", { 
            text: extractedText,                  // Full book text
            persona: config.persona || "Neutral", // Narration style
            sfxStyle: config.sfx || "Balanced",   // Sound effects level
            voiceAudio: null,                     // Custom voice sample (unused)
            voiceName: config.voice || "Voice A"  // Voice ID
        });

        // --------------------------------------------------------------------
        // STEP 4: Process Streaming Updates
        // --------------------------------------------------------------------
        // The API streams progress updates and eventually the final audio URL
        let audioUrl = null;         // Final audio file URL
        let timestamps = null;       // Text-to-audio timestamp mapping
        let lastStatusMessage = '';  // Prevent duplicate status messages
        let isComplete = false;      // Track completion state
        
        // Iterate through streaming updates
        for await (const update of prediction) {
            console.log('[Main] Streaming update received:', update);
            
            // Updates come as { data: [statusObject, audioObject, ...] }
            if (update.data && Array.isArray(update.data)) {
                const statusData = update.data[0]; // First element is status
                
                if (statusData && typeof statusData === 'object') {
                    const message = statusData.message || statusData.status || '';
                    const status = statusData.status;
                    
                    // Check if generation completed successfully
                    if (status === 'success' || message.includes('Completed audiobook generation')) {
                        isComplete = true;
                    }
                    
                    // Send status update to renderer if message changed
                    if (message && message !== lastStatusMessage) {
                        lastStatusMessage = message;
                        
                        // Use event.sender.send() to push update to renderer
                        event.sender.send('gradio:status-update', {
                            status: status,
                            message: message,
                            isComplete: isComplete
                        });
                    }
                }
                
                // Check for audio output (usually in data[1])
                if (update.data[1] && typeof update.data[1] === 'object' && update.data[1].url) {
                    audioUrl = update.data[1].url;
                    console.log('[Main] Audio URL found:', audioUrl);
                }
                
                // Look for timestamps in any position
                for (let item of update.data) {
                    if (item && typeof item === 'object' && !item.url && item.timestamps) {
                        timestamps = item.timestamps;
                    }
                }
                
                // Break out of loop when we have everything
                if (isComplete && audioUrl) {
                    console.log('[Main] Generation complete, breaking stream loop');
                    break;
                }
            }
        }

        // --------------------------------------------------------------------
        // STEP 5: Validate Result
        // --------------------------------------------------------------------
        if (!audioUrl) {
            throw new Error("API completed but no audio URL was found in the response.");
        }
        
        console.log('[Main] Final audio URL:', audioUrl);

        // --------------------------------------------------------------------
        // STEP 6: Download and Save Audio File
        // --------------------------------------------------------------------
        console.log(`[Main] Downloading audio from ${audioUrl}...`);
        
        // Send download status to renderer
        event.sender.send('gradio:status-update', {
            status: 'running',
            message: 'Downloading generated audio...'
        });
        
        // Fetch audio file from Gradio server
        const response = await fetch(audioUrl);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = Buffer.from(arrayBuffer);

        // Determine file extension from URL
        let fileExtension = 'wav';  // Default to WAV
        if (audioUrl.includes('.mp3')) {
            fileExtension = 'mp3';
        } else if (audioUrl.includes('.wav')) {
            fileExtension = 'wav';
        }

        // Create audiobooks directory in app data
        const { app: electronApp } = require('electron');
        const audioDir = path.join(electronApp.getPath('userData'), 'audiobooks');
        await fs.mkdir(audioDir, { recursive: true });
        
        // Generate unique filename using sanitised title and timestamp
        const fileName = `${config.title.replace(/[^a-z0-9]/gi, '_')}_${Date.now()}.${fileExtension}`;
        const savePath = path.join(audioDir, fileName);
        
        // Write audio file to disk
        await fs.writeFile(savePath, audioBuffer);
        
        // Send final completion status
        event.sender.send('gradio:status-update', {
            status: 'success',
            message: 'Complete!'
        });
        
        // Return success with paths
        return { 
            success: true, 
            audioPath: savePath,                                    // Local path to audio file
            timestamps: timestamps ? JSON.stringify(timestamps) : null  // JSON string of timestamps
        };

    } catch (error) {
        console.error('Gradio Generation Error:', error);
        
        // Send error status to renderer
        event.sender.send('gradio:status-update', {
            status: 'error',
            message: error.message
        });
        
        throw new Error(error.message); 
    }
});