// preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    openFile: () => ipcRenderer.invoke('dialog:openFile'),
    readFileBuffer: (filePath) => ipcRenderer.invoke('file:readBuffer', filePath),
    readFileText: (filePath) => ipcRenderer.invoke('file:readText', filePath),
    epubParse: (filePath) => ipcRenderer.invoke('epub:parse', filePath),
    pdfParse: (filePath) => ipcRenderer.invoke('pdf:parse', filePath),
    addBookToDb: (book) => ipcRenderer.invoke('db:addBook', book),
    getBooksFromDb: () => ipcRenderer.invoke('db:getBooks'),
    deleteBook: (id) => ipcRenderer.invoke('db:deleteBook', id),
    updateBookProgress: (id, progress, currentPage) => ipcRenderer.invoke('db:updateProgress', id, progress, currentPage),
    saveAudio: (audioData, title) => ipcRenderer.invoke('audio:save', audioData, title),
    readAudioFile: (filePath) => ipcRenderer.invoke('audio:readFile', filePath),
    generateAudiobook: (config) => ipcRenderer.invoke('gradio:generate', config)
});
