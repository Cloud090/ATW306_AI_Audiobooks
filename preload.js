// preload.js
const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // File operations
    openFile: () => ipcRenderer.invoke('dialog:openFile'),
    copyBook: (filePath) => ipcRenderer.invoke('file:copyBook', filePath),
    readFileBuffer: (filePath) => ipcRenderer.invoke('file:readBuffer', filePath),
    readFileText: (filePath) => ipcRenderer.invoke('file:readText', filePath),
    readAudioFile: (filePath) => ipcRenderer.invoke('audio:readFile', filePath),
    
    // Get file path from File object (for drag & drop)
    getFilePath: (file) => {
        try {
            return webUtils.getPathForFile(file);
        } catch (error) {
            console.error('Error getting file path:', error);
            return null;
        }
    },
    
    // Parsers
    epubParse: (filePath) => ipcRenderer.invoke('epub:parse', filePath),
    pdfParse: (filePath) => ipcRenderer.invoke('pdf:parse', filePath),
    
    // Database
    addBookToDb: (book) => ipcRenderer.invoke('db:addBook', book),
    getBooksFromDb: () => ipcRenderer.invoke('db:getBooks'),
    deleteBook: (id) => ipcRenderer.invoke('db:deleteBook', id),
    updateBookProgress: (id, progress, currentPage) => 
        ipcRenderer.invoke('db:updateProgress', id, progress, currentPage),
    
    // Gradio generation
    generateAudiobook: (config) => ipcRenderer.invoke('gradio:generate', config),
    
    // Status update listener
    onStatusUpdate: (callback) => {
        ipcRenderer.on('gradio:status-update', (event, data) => callback(data));
    },
    
    // Remove status listener
    removeStatusListener: () => {
        ipcRenderer.removeAllListeners('gradio:status-update');
    }
});