// preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // File operations
    openFile: () => ipcRenderer.invoke('dialog:openFile'),
    readFileBuffer: (filePath) => ipcRenderer.invoke('file:readBuffer', filePath),
    readFileText: (filePath) => ipcRenderer.invoke('file:readText', filePath),
    
    // EPUB parsing
    epubParse: (filePath) => ipcRenderer.invoke('epub:parse', filePath),
    
    // Database operations
    addBookToDb: (book) => ipcRenderer.invoke('db:addBook', book),
    getBooksFromDb: () => ipcRenderer.invoke('db:getBooks'),
    deleteBook: (id) => ipcRenderer.invoke('db:deleteBook', id),
    updateBookProgress: (id, progress) => ipcRenderer.invoke('db:updateProgress', id, progress),
    
    // Audio operations
    saveAudio: (audioData, title) => ipcRenderer.invoke('audio:save', audioData, title),
    readAudioFile: (filePath) => ipcRenderer.invoke('audio:readFile', filePath) // *** NEW ***
});