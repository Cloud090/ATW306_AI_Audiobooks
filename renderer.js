// renderer.js
document.addEventListener('DOMContentLoaded', async () => {
    console.log('App initialised');
    
    // === DOM ELEMENTS ===
    const uploadModalOverlay = document.getElementById('upload-modal-overlay');
    const addBookBtn = document.getElementById('add-book');
    const cancelUploadBtn = document.getElementById('cancel-upload-btn');
    const confirmUploadBtn = document.getElementById('confirm-upload-btn');

    const configureModalOverlay = document.getElementById('configure-modal-overlay');
    const cancelConfigureBtn = document.getElementById('cancel-configure-btn');
    const backConfigureBtn = document.getElementById('back-configure-btn');
    const generateAudiobookBtn = document.getElementById('generate-audiobook-btn');

    const generatingModalOverlay = document.getElementById('generating-modal-overlay');
    const generationProgress = document.getElementById('generation-progress');
    const generationStatus = document.getElementById('generation-status');
    const cancelGenerationBtn = document.getElementById('cancel-generation-btn');

    const completeModalOverlay = document.getElementById('complete-modal-overlay');
    const addToBookshelfBtn = document.getElementById('add-to-bookshelf-btn');
    const listenNowBtn = document.getElementById('listen-now-btn');

    const deleteModalOverlay = document.getElementById('delete-modal-overlay');
    const deleteBookTitle = document.getElementById('delete-book-title');
    const cancelDeleteBtn = document.getElementById('cancel-delete-btn');
    const confirmDeleteBtn = document.getElementById('confirm-delete-btn');

    const browseLink = document.getElementById('browse-link');
    const dropZone = document.getElementById('drop-zone');
    const uploadList = document.querySelector('.upload-list-container');
    const uploadFormFields = document.querySelectorAll('#upload-modal-content .form-group');

    const configCoverPreview = document.getElementById('config-cover-preview');
    const previewPane = document.getElementById('preview-pane');
    const previewPrevBtn = document.getElementById('preview-prev-btn');
    const previewNextBtn = document.getElementById('preview-next-btn');
    const previewPageNumber = document.getElementById('preview-page-number');
    const previewCurrentPage = document.getElementById('preview-current-page');
    const previewTotalPages = document.getElementById('preview-total-pages');

    // Reader elements
    const readerView = document.getElementById('book-reader-view');
    const mainContentArea = document.getElementById('main-content-area');
    const readerBackBtn = document.getElementById('reader-back-btn');
    const readerTitle = document.getElementById('reader-title');
    const readerPageLeftContent = document.getElementById('reader-page-left-content');
    const readerPageRightContent = document.getElementById('reader-page-right-content');
    const prevPageBtn = document.getElementById('prev-page-btn');
    const nextPageBtn = document.getElementById('next-page-btn');
    const pageIndicator = document.getElementById('page-indicator');
    
    // *** UPDATED ***
    const audioControlsBar = document.querySelector('.audio-controls'); // Get the whole bar
    const playPauseBtn = document.getElementById('play-pause-btn');
    const rewindBtn = document.getElementById('rewind-btn');
    const forwardBtn = document.getElementById('forward-btn');
    const speedSlider = document.getElementById('speed-slider');
    const volumeSlider = document.getElementById('volume-slider');
    const speedValue = document.getElementById('speed-value');
    const chapterSelect = document.getElementById('chapter-select');

    // === STATE VARIABLES ===
    let selectedFilePath = null;
    let currentBook = null;
    let generationCancelled = false;
    let currentPreviewPage = 0;
    let totalPreviewPages = 0;
    let bookPages = [];
    let currentBookData = null;
    let generatedCoverUrl = null;
    
    // Reader state
    let currentReaderBook = null;
    let currentChapters = [];
    let currentChapterIndex = 0;
    let totalWordCount = 0;
    let wordsRead = 0;
    let isPlaying = false;
    let bookToDelete = null;
    let currentAudio = null; // *** NEW: For audio playback

    // === DATABASE FUNCTIONS ===
    async function loadBooksFromDatabase() {
        try {
            console.log('Loading books from database...');
            const books = await window.electronAPI.getBooksFromDb();
            console.log('Books retrieved from DB:', books);
            
            const bookshelfGrid = document.getElementById('bookshelf-grid');
            bookshelfGrid.innerHTML = '';
            
            if (books.length === 0) {
                showEmptyState();
            } else {
                books.forEach((book, index) => {
                    console.log(`Adding book ${index + 1}:`, {
                        title: book.title,
                        author: book.author,
                        filePath: book.filePath,
                        coverUrl: book.coverUrl ? 'Has cover' : 'No cover',
                        progress: book.progress,
                        audioPath: book.audioPath ? 'Has audio' : 'No audio' // Log audio path
                    });
                    
                    addBookToBookshelf(
                        book.title,
                        book.author,
                        book.year,
                        book.coverUrl,
                        book.progress || 0,
                        book.id,
                        book.filePath
                        // Note: book object passed to openBookReader will have audioPath
                    );
                });
            }
        } catch (error) {
            console.error('Error loading books from database:', error);
        }
    }

    function showEmptyState() {
        const bookshelfGrid = document.getElementById('bookshelf-grid');
        bookshelfGrid.innerHTML = `
            <div class="empty-state">
                <p class="empty-state-text">No books in bookshelf, click Add Book below to add your first</p>
                <button class="add-book-btn" id="add-book-empty">
                    <span>+</span> ADD BOOK
                </button>
            </div>
        `;
        
        document.getElementById('add-book-empty').addEventListener('click', openUploadModal);
    }

    async function saveBookToDatabase(book) {
        try {
            console.log('Attempting to save book to database:', book);
            
            if (!book.filePath) {
                console.error('Book has no filePath!');
                alert('Cannot save book: file path is missing');
                return;
            }
            
            // *** UPDATED ***: Pass the full book object, which includes audioPath
            const result = await window.electronAPI.addBookToDb(book);
            console.log('Database save result:', result);
            
            if (result.success) {
                console.log('Book saved to database successfully with ID:', result.id);
            } else {
                console.error('Failed to save book:', result.error);
                alert('Failed to save book: ' + result.error);
            }
        } catch (error) {
            console.error('Error saving book to database:', error);
            alert('Error saving book: ' + error.message);
        }
    }

    // === BOOKSHELF FUNCTIONS ===
    function addBookToBookshelf(title, author, year, coverUrl, progress = 0, bookId = null, filePath = null) {
        console.log('Adding book to shelf:', { title, author, bookId, filePath, progress });
        
        const bookshelfGrid = document.getElementById('bookshelf-grid');
        const bookItem = document.createElement('div');
        bookItem.className = 'book-item';
        bookItem.dataset.bookId = bookId;
        bookItem.dataset.filePath = filePath;
        bookItem.dataset.title = title;
        bookItem.dataset.author = author;
        bookItem.dataset.year = year;
        bookItem.dataset.coverUrl = coverUrl || '';
        bookItem.dataset.progress = progress;
        
        // This line ensures the correct format: "Author - year" if year exists
        const authorYearText = (author && year) ? `${author} - ${year}` : (author || 'Unknown Author');
        
        let progressClass = 'yellow';
        let progressText = 'UNREAD';
        let progressWidth = 100;
        
        if (progress > 0 && progress < 100) {
            progressClass = 'blue';
            progressText = `${Math.round(progress)}%`;
            progressWidth = progress;
        } else if (progress === 100) {
            progressClass = 'green';
            progressText = 'READ';
            progressWidth = 100;
        }
        
        const deleteIconSVG = `
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                <line x1="10" y1="11" x2="10" y2="17"></line>
                <line x1="14" y1="11" x2="14" y2="17"></line>
            </svg>`;
        
        bookItem.innerHTML = `
            <div class="book-cover" style="background-color: #555; ${coverUrl ? `background-image: url(${coverUrl});` : ''}">
                <button class="delete-book-btn" title="Delete Book">${deleteIconSVG}</button>
            </div>
            <span class="book-title">${title}</span>
            <span class="book-author">${authorYearText}</span>
            <div class="progress-bar-container">
                <div class="progress-bar ${progressClass}" style="width: ${progressWidth}%;"></div>
                <span class="progress-text">${progressText}</span>
            </div>
        `;
        
        bookItem.addEventListener('click', async (e) => {
            if (e.target.classList.contains('delete-book-btn') || e.target.closest('.delete-book-btn')) {
                e.stopPropagation();
                return;
            }
            
            const bookFilePath = filePath || bookItem.dataset.filePath;
            
            if (bookFilePath && bookFilePath !== 'null' && bookFilePath !== 'undefined' && bookFilePath !== '') {
                
                // *** UPDATED ***: We need the full book object from DB, esp. audioPath
                const books = await window.electronAPI.getBooksFromDb();
                const bookData = books.find(b => b.id === bookId);

                if (bookData) {
                    console.log('Opening book with data:', bookData);
                    openBookReader(bookData);
                } else {
                     alert('Error: Could not find book data.');
                }
            } else {
                alert('Cannot open book: file path is missing. The book may need to be re-uploaded.');
            }
        });
        
        const deleteBtn = bookItem.querySelector('.delete-book-btn');
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            openDeleteModal(bookId, title);
        });
        
        bookshelfGrid.appendChild(bookItem);
        bookshelfGrid.parentElement.scrollLeft = bookshelfGrid.scrollWidth;
    }

    function openDeleteModal(bookId, title) {
        bookToDelete = bookId;
        deleteBookTitle.textContent = title;
        deleteModalOverlay.classList.add('visible');
    }

    function closeDeleteModal() {
        deleteModalOverlay.classList.remove('visible');
        bookToDelete = null;
    }

    async function deleteBook() {
        if (bookToDelete) {
            try {
                const result = await window.electronAPI.deleteBook(bookToDelete);
                if (result.success) {
                    console.log('Book deleted successfully');
                    closeDeleteModal();
                    await loadBooksFromDatabase();
                } else {
                    console.error('Failed to delete book:', result.error);
                    alert('Failed to delete book. Please try again.');
                }
            } catch (error) {
                console.error('Error deleting book:', error);
                alert('Error deleting book. Please try again.');
            }
        }
    }

    function getProgressClass(progress) {
        if (progress === 0) return 'yellow';
        if (progress < 100) return 'blue';
        return 'green';
    }

    function getProgressText(progress) {
        if (progress === 0) return 'UNREAD';
        if (progress < 100) return `${Math.round(progress)}%`;
        return 'READ';
    }

    // === MODAL FUNCTIONS ===
    function openUploadModal() {
        uploadModalOverlay.classList.add('visible');
    }
    
    function closeUploadModal() {
        uploadModalOverlay.classList.remove('visible');
    }

    function openConfigureModal(filePath, title, author, year) {
        console.log('Opening configure modal with filePath:', filePath);
        
        selectedFilePath = filePath;
        
        const filename = filePath.split(/[\\/]/).pop();
        document.getElementById('config-file-name').value = filename;
        document.getElementById('config-book-title').value = title;
        document.getElementById('config-book-author').value = author;
        document.getElementById('config-book-year').value = year || '';
        
        loadBookPreview(filePath);
        configureModalOverlay.classList.add('visible');
    }
    
    function closeConfigureModal() {
        configureModalOverlay.classList.remove('visible');
        if (currentBook) {
            currentBook.destroy();
            currentBook = null;
        }
        configCoverPreview.innerHTML = '';
        configCoverPreview.style.backgroundImage = '';
        previewPane.innerHTML = '<h3>CHAPTER 1</h3><p>Loading...</p>';
        bookPages = [];
        currentPreviewPage = 0;
        totalPreviewPages = 0;
        previewPane.classList.remove('epub-preview', 'pdf-preview', 'fallback-preview');
        
        resetUploadUI();
    }

    function openGeneratingModal() {
        generationCancelled = false;
        generationProgress.style.width = '0%';
        generationStatus.textContent = 'Starting generation...';
        generatingModalOverlay.classList.add('visible');
    }
    
    function closeGeneratingModal() {
        generatingModalOverlay.classList.remove('visible');
    }
    
    function openCompleteModal() {
        completeModalOverlay.classList.add('visible');
    }
    
    function closeCompleteModal() {
        completeModalOverlay.classList.remove('visible');
    }

    // === PREVIEW FUNCTIONS ===
    function updatePreviewNavigation() {
        previewPageNumber.textContent = currentPreviewPage + 1;
        previewCurrentPage.textContent = currentPreviewPage + 1;
        previewTotalPages.textContent = totalPreviewPages;
        
        previewPrevBtn.disabled = currentPreviewPage === 0;
        previewNextBtn.disabled = currentPreviewPage >= totalPreviewPages - 1;
        
        if (bookPages[currentPreviewPage]) {
            previewPane.innerHTML = bookPages[currentPreviewPage];
        }
    }

    async function loadBookPreview(filePath) {
        console.log('Loading preview for:', filePath);
        configCoverPreview.innerHTML = '';
        configCoverPreview.style.backgroundImage = '';
        previewPane.innerHTML = '<p>Loading preview...</p>';
        previewPane.classList.remove('epub-preview', 'pdf-preview', 'fallback-preview');
        bookPages = [];
        currentPreviewPage = 0;
        generatedCoverUrl = null;
        
        const extension = filePath.split('.').pop().toLowerCase();
        console.log('File extension:', extension);
        
        try {
            if (extension === 'epub') {
                await loadEpubPreview(filePath);
            } else if (extension === 'pdf') {
                await loadPdfPreview(filePath);
            } else if (extension === 'docx' || extension === 'doc') {
                await loadDocxPreview(filePath);
            } else if (extension === 'odt') {
                await loadOdtPreview(filePath);
            } else {
                previewPane.classList.add('fallback-preview');
                previewPane.innerHTML = '<h3>Preview Not Available</h3><p>Unsupported file format.</p>';
                configCoverPreview.style.backgroundColor = '#555';
            }
        } catch (error) {
            console.error('Error loading preview:', error);
            previewPane.classList.add('fallback-preview');
            previewPane.innerHTML = `<h3>Error</h3><p>Could not read file: ${error.message}</p>`;
            bookPages = [`<h3>Error</h3><p>Could not read file: ${error.message}</p>`];
            totalPreviewPages = 1;
            updatePreviewNavigation();
        }
    }

    async function loadEpubPreview(filePath) {
        previewPane.classList.add('epub-preview');
        
        try {
            const arrayBuffer = await window.electronAPI.readFileBuffer(filePath);
            currentBook = ePub(arrayBuffer);

            const metadata = await currentBook.loaded.metadata;
            
            if (metadata.cover) {
                const coverUrl = await currentBook.coverUrl(); 
                if (coverUrl) {
                    try {
                        const response = await fetch(coverUrl);
                        const blob = await response.blob();
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            generatedCoverUrl = reader.result;
                            configCoverPreview.style.backgroundImage = `url(${generatedCoverUrl})`;
                            configCoverPreview.style.backgroundSize = 'cover';
                            configCoverPreview.style.backgroundPosition = 'center';
                        };
                        reader.readAsDataURL(blob);
                    } catch (e) {
                        console.error('Error converting cover to base64:', e);
                        configCoverPreview.style.backgroundColor = '#555';
                    }
                } else {
                    configCoverPreview.style.backgroundColor = '#555';
                }
            } else {
                configCoverPreview.style.backgroundColor = '#555';
            }
            
            await currentBook.ready;
            
            const sectionsToLoad = Math.min(5, currentBook.spine.length);
            
            for (let i = 0; i < sectionsToLoad; i++) {
                const section = currentBook.spine.get(i);
                if (section) {
                    const tempDiv = document.createElement('div');
                    await section.render(tempDiv); 
                    
                    const paragraphs = Array.from(tempDiv.querySelectorAll('p'));
                    const chapterTitle = tempDiv.querySelector('h1, h2, h3');
                    
                    let pageHTML = '';
                    if (chapterTitle) {
                        pageHTML += `<h3>${chapterTitle.textContent}</h3>`;
                    }
                    
                    paragraphs.slice(0, 15).forEach(p => {
                        if (p.textContent.trim()) {
                            pageHTML += `<p>${p.textContent}</p>`;
                        }
                    });
                    
                    bookPages.push(pageHTML || '<p>No content available.</p>');
                }
            }
            
            totalPreviewPages = bookPages.length || 1;
            updatePreviewNavigation();

        } catch (error) {
            console.error('Error loading EPUB:', error);
            previewPane.classList.remove('epub-preview');
            previewPane.classList.add('fallback-preview');
            previewPane.innerHTML = `<h3>Error</h3><p>Could not read EPUB file. It may be corrupted or protected.</p>`;
            bookPages = [`<h3>Error</h3><p>Could not read EPUB file.</p>`];
            totalPreviewPages = 1;
            updatePreviewNavigation();
        }
    }

    async function loadPdfPreview(filePath) {
        try {
            previewPane.classList.add('pdf-preview');
            configCoverPreview.style.backgroundColor = '#555';
            previewPane.innerHTML = '<p style="text-align: center; font-family: system-ui; text-indent: 0;">Loading PDF preview...</p>';
            
            const arrayBuffer = await window.electronAPI.readFileBuffer(filePath);
            const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
            const pdf = await loadingTask.promise;
            
            console.log(`PDF loaded: ${pdf.numPages} pages`);
            
            const pagesToLoad = Math.min(5, pdf.numPages);
            bookPages = [];
            
            for (let pageNum = 1; pageNum <= pagesToLoad; pageNum++) {
                const page = await pdf.getPage(pageNum);
                
                const desiredWidth = 800; 
                let viewport = page.getViewport({ scale: 1.0 });
                const scale = desiredWidth / viewport.width;
                viewport = page.getViewport({ scale: scale });

                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;
                
                await page.render({
                    canvasContext: context,
                    viewport: viewport
                }).promise;
                
                const dataUrl = canvas.toDataURL();
                bookPages.push(`<img src="${dataUrl}" alt="Page ${pageNum}">`);
            }

            const firstPage = await pdf.getPage(1);
            const coverViewport = firstPage.getViewport({ scale: 1.0 });
            const coverCanvas = document.createElement('canvas');
            const coverContext = coverCanvas.getContext('2d');
            coverCanvas.height = coverViewport.height;
            coverCanvas.width = coverViewport.width;
            
            await firstPage.render({
                canvasContext: coverContext,
                viewport: coverViewport
            }).promise;
            
            const coverDataUrl = coverCanvas.toDataURL();
            configCoverPreview.style.backgroundImage = `url(${coverDataUrl})`;
            configCoverPreview.style.backgroundSize = 'cover';
            configCoverPreview.style.backgroundPosition = 'center';
            generatedCoverUrl = coverDataUrl;
            
            totalPreviewPages = bookPages.length || 1;
            currentPreviewPage = 0;
            updatePreviewNavigation();
            
        } catch (error) {
            console.error('Error loading PDF:', error);
            previewPane.classList.remove('pdf-preview');
            previewPane.classList.add('fallback-preview');
            previewPane.innerHTML = '<h3>Error Loading PDF</h3><p>Could not read PDF file. It may be corrupted or protected.</p>';
            configCoverPreview.style.backgroundColor = '#555';
            bookPages = ['<h3>Error Loading PDF</h3><p>Could not read PDF file.</p>'];
            totalPreviewPages = 1;
            updatePreviewNavigation();
        }
    }

    async function loadDocxPreview(filePath) {
        previewPane.classList.add('fallback-preview');
        configCoverPreview.style.backgroundColor = '#555';
        previewPane.innerHTML = '<h3>DOCX Preview</h3><p>Word document preview requires additional library support.</p>';
        bookPages = ['<h3>DOCX Preview</h3><p>Word document preview requires additional library support.</p>'];
        totalPreviewPages = 1;
        updatePreviewNavigation();
    }

    async function loadOdtPreview(filePath) {
        previewPane.classList.add('fallback-preview');
        configCoverPreview.style.backgroundColor = '#555';
        previewPane.innerHTML = '<h3>ODT Preview</h3><p>OpenDocument preview requires additional library support.</p>';
        bookPages = ['<h3>ODT Preview</h3><p>OpenDocument preview requires additional library support.</p>'];
        totalPreviewPages = 1;
        updatePreviewNavigation();
    }

    // === BOOK READER FUNCTIONS ===
    async function openBookReader(book) {
        console.log('Opening book reader with book data:', book);
        
        if (!book || !book.filePath) {
            console.error('Invalid book data:', book);
            alert('Cannot open book: invalid book data');
            return;
        }
        
        currentReaderBook = book;
        mainContentArea.classList.add('hidden');
        readerView.classList.remove('hidden');
        
        readerTitle.textContent = book.title || 'Untitled Book';
        readerPageLeftContent.innerHTML = '<p>Loading...</p>';
        readerPageRightContent.innerHTML = '<p>Loading...</p>';
        
        try {
            const extension = book.filePath.split('.').pop().toLowerCase();
            console.log('Book file extension:', extension);
            
            if (extension === 'epub') {
                console.log('Parsing EPUB...');
                const result = await window.electronAPI.epubParse(book.filePath);
                currentChapters = result.chapters || [];
            } else if (extension === 'pdf') {
                console.log('Loading PDF...');
                await loadPdfForReading(book.filePath);
                // PDF function will call loadChapter, so we return here
            } else {
                 currentChapters = ['<h2>Unsupported Format</h2><p>This file format is not supported for reading. Please use EPUB or PDF.</p>'];
            }
            
            if (extension !== 'pdf') {
                if (currentChapters.length === 0) {
                    currentChapters = ['<p>No content could be extracted from this book.</p>'];
                }

                // Calculate total word count
                totalWordCount = currentChapters.reduce((total, chapter) => {
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = chapter;
                    const words = (tempDiv.textContent || '').split(/\s+/).filter(w => w.length > 0).length;
                    return total + words;
                }, 0);
                
                console.log('Total word count:', totalWordCount);
                
                // Populate chapter selector
                chapterSelect.innerHTML = '';
                currentChapters.forEach((_, index) => {
                    const option = document.createElement('option');
                    option.value = index;
                    option.textContent = `Chapter ${index + 1}`;
                    chapterSelect.appendChild(option);
                });
                
                loadChapter(0);
            }

            // Load Audio
            if (book.audioPath) {
                console.log('Loading audio from:', book.audioPath);
                audioControlsBar.style.display = 'flex'; // Show controls
                
                try {
                    const audioDataUrl = await window.electronAPI.readAudioFile(book.audioPath);
                    currentAudio = new Audio(audioDataUrl);
                    
                    // Set initial values
                    currentAudio.playbackRate = parseFloat(speedSlider.value);
                    currentAudio.volume = parseFloat(volumeSlider.value) / 100;
                    
                    // Sync with saved progress
                    currentAudio.addEventListener('loadedmetadata', () => {
                        if (currentReaderBook.progress > 0 && currentAudio.duration) {
                            currentAudio.currentTime = (currentReaderBook.progress / 100) * currentAudio.duration;
                        }
                    });

                    // Add listeners
                    currentAudio.addEventListener('timeupdate', updateAudioProgress);
                    currentAudio.addEventListener('ended', () => {
                        isPlaying = false;
                        playPauseBtn.innerHTML = '▶';
                    });

                } catch (audioError) {
                    console.error('Error loading audio file:', audioError);
                    audioControlsBar.style.display = 'none'; // Hide controls if audio fails
                }

            } else {
                console.log('No audio path for this book.');
                audioControlsBar.style.display = 'none'; // Hide controls
            }
            
        } catch (error) {
            console.error('Error opening book reader:', error);
            readerPageLeftContent.innerHTML = `<p>Error loading book: ${error.message}</p>`;
            readerPageRightContent.innerHTML = `<p>Error loading book: ${error.message}</p>`;
        }
    }

    async function loadPdfForReading(filePath) {
        try {
            console.log('Loading PDF for reading:', filePath);
            
            const availableWidth = (readerPageLeftContent.parentElement.offsetWidth * 0.9);
            console.log(`PDF render width: ${availableWidth}`);

            const arrayBuffer = await window.electronAPI.readFileBuffer(filePath);
            const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
            const pdf = await loadingTask.promise;
            
            console.log(`PDF loaded: ${pdf.numPages} pages`);
            
            currentChapters = [];
            
            for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                const page = await pdf.getPage(pageNum);
                
                let viewport = page.getViewport({ scale: 1.0 });
                const scale = availableWidth / viewport.width;
                viewport = page.getViewport({ scale: scale });

                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;
                
                await page.render({
                    canvasContext: context,
                    viewport: viewport
                }).promise;
                
                const dataUrl = canvas.toDataURL();
                currentChapters.push(`<img src="${dataUrl}" alt="Page ${pageNum}">`);
            }
            
            console.log('PDF pages loaded:', currentChapters.length);
            
            // For PDFs, each page is a "chapter"
            totalWordCount = currentChapters.length * 300; // Estimate 300 words per page
            
            // Populate chapter selector (pages in this case)
            chapterSelect.innerHTML = '';
            currentChapters.forEach((_, index) => {
                if (index % 2 === 0) {
                    const option = document.createElement('option');
                    option.value = index;
                    const endPage = (index + 2 <= currentChapters.length) ? `-${index + 2}` : '';
                    option.textContent = `Pages ${index + 1}${endPage}`;
                    chapterSelect.appendChild(option);
                }
            });
            
            loadChapter(0);
            
        } catch (error) {
            console.error('Error loading PDF:', error);
            readerPageLeftContent.innerHTML = `<p>Error loading PDF: ${error.message}</p>`;
            readerPageRightContent.innerHTML = `<p>Error loading PDF: ${error.message}</p>`;
        }
    }

    function loadChapter(index) {
        console.log('Loading chapter/page:', index);
        
        if (index < 0) index = 0;
        if (index >= currentChapters.length && index > 0) index = currentChapters.length - 1;
        
        currentChapterIndex = index;
        const chapterHtml = currentChapters[index];
        
        const isPdfPage = chapterHtml.includes('<img');
        
        if (isPdfPage) {
            const leftPageHtml = currentChapters[index];
            const rightPageHtml = (index + 1 < currentChapters.length) ? currentChapters[index + 1] : '';
            
            readerPageLeftContent.innerHTML = leftPageHtml;
            readerPageRightContent.innerHTML = rightPageHtml;
            
            readerPageLeftContent.classList.remove('epub-content');
            readerPageRightContent.classList.remove('epub-content');
            
            if (rightPageHtml) {
                pageIndicator.textContent = `Pages ${index + 1}-${index + 2} of ${currentChapters.length}`;
            } else {
                pageIndicator.textContent = `Page ${index + 1} of ${currentChapters.length}`;
            }
            
            prevPageBtn.disabled = index === 0;
            nextPageBtn.disabled = index + 1 >= currentChapters.length - 1; 

        } else {
            // EPUB/Text: Split content of a single chapter.
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = chapterHtml;
            
            const allElements = Array.from(tempDiv.children);
            const midPoint = Math.ceil(allElements.length / 2);
            
            const leftElements = allElements.slice(0, midPoint);
            const rightElements = allElements.slice(midPoint);
            
            readerPageLeftContent.innerHTML = '';
            leftElements.forEach(el => readerPageLeftContent.appendChild(el.cloneNode(true)));
            
            readerPageRightContent.innerHTML = '';
            rightElements.forEach(el => readerPageRightContent.appendChild(el.cloneNode(true)));

            readerPageLeftContent.classList.add('epub-content');
            readerPageRightContent.classList.add('epub-content');
            
            pageIndicator.textContent = `Chapter ${index + 1} of ${currentChapters.length}`;
            
            prevPageBtn.disabled = index === 0;
            nextPageBtn.disabled = index >= currentChapters.length - 1;
        }
        
        chapterSelect.value = index;
        readerPageLeftContent.scrollTop = 0;
        readerPageRightContent.scrollTop = 0;
    }

    // audio progress updates
    function updateAudioProgress() {
        if (!currentAudio || !currentAudio.duration) return;
        
        // Calculate progress percentage
        const progress = (currentAudio.currentTime / currentAudio.duration) * 100;
        
        // Store it on the book object
        currentReaderBook.progress = progress;
        
        // Sync 'wordsRead' for non-audio progress tracking
        wordsRead = Math.floor((progress / 100) * totalWordCount);
    }

    async function updateBookProgress() {
        if (!currentReaderBook) return;
        
        // If no audio, it will be set by page turns or see updateAudioProgress
        const progress = currentReaderBook.progress;
        
        try {
            await window.electronAPI.updateBookProgress(currentReaderBook.id, progress);
            
            const bookItem = document.querySelector(`[data-book-id="${currentReaderBook.id}"]`);
            if (bookItem) {
                const progressBar = bookItem.querySelector('.progress-bar');
                const progressText = bookItem.querySelector('.progress-text');
                
                const progressClass = getProgressClass(progress);
                const progressTextContent = getProgressText(progress);
                
                progressBar.className = `progress-bar ${progressClass}`;
                progressBar.style.width = `${progress}%`;
                progressText.textContent = progressTextContent;
            }
        } catch (error) {
            console.error('Error updating book progress:', error);
        }
    }

    // === UI HELPER FUNCTIONS ===
    function showFileInUI(filePath) {
        if (!filePath) {
            console.error('filePath is null or undefined!');
            alert('Error: No file path provided');
            return;
        }
        
        const filename = filePath.split(/[\\/]/).pop();
        selectedFilePath = filePath;
        
        dropZone.style.display = 'none';
        uploadList.style.display = 'block';
        uploadFormFields.forEach(field => field.style.display = 'flex');
        uploadList.innerHTML = `<div class="upload-item"><span>${filename}</span><span class="remove-file-icon" id="remove-file-btn">×</span></div>`;
        
        const title = filename.replace(/\.[^/.]+$/, "");
        document.getElementById('book-title').value = title;
        document.getElementById('book-author').value = '';
        document.getElementById('book-year').value = '';
        
        const removeBtn = document.getElementById('remove-file-btn');
        if (removeBtn) {
            removeBtn.addEventListener('click', resetUploadUI);
        }
    }
    
    function resetUploadUI() {
        console.log('Resetting upload UI');
        selectedFilePath = null;
        dropZone.style.display = 'block';
        uploadList.style.display = 'none';
        uploadFormFields.forEach(field => field.style.display = 'none');
        uploadList.innerHTML = '';
        document.getElementById('book-title').value = '';
        document.getElementById('book-author').value = '';
        document.getElementById('book-year').value = '';
    }

    // === EVENT LISTENERS ===
    
    addBookBtn.addEventListener('click', openUploadModal);
    cancelUploadBtn.addEventListener('click', () => {
        closeUploadModal();
        resetUploadUI(); 
    });
    
    uploadModalOverlay.addEventListener('click', (event) => {
        if (event.target === uploadModalOverlay) {
            closeUploadModal();
            resetUploadUI();
        }
    });

    browseLink.addEventListener('click', async (event) => {
        event.preventDefault();
        try {
            const filePath = await window.electronAPI.openFile();
            if (filePath) showFileInUI(filePath);
        } catch (error) {
            console.error('Error opening file:', error);
            alert('Error opening file. Please try again.');
        }
    });
    
    dropZone.addEventListener('dragover', (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.add('drag-over');
    });
    
    dropZone.addEventListener('dragleave', (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.remove('drag-over');
    });
    
    dropZone.addEventListener('drop', async (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.remove('drag-over');
        
        const files = event.dataTransfer.files;
        if (files && files.length > 0) {
            const file = files[0];
            if (file.path) {
                showFileInUI(file.path);
            } else {
                console.error('file.path is not available');
                alert('Drag & drop is not fully supported. Please use the Browse button to select your file.');
            }
        }
    });

    confirmUploadBtn.addEventListener('click', () => {
        const title = document.getElementById('book-title').value;
        const author = document.getElementById('book-author').value;
        const year = document.getElementById('book-year').value;
        
        if (!selectedFilePath) {
            alert('Please select a book file');
            return;
        }
        
        if (!title) {
            alert('Please provide a title');
            return;
        }
        
        openConfigureModal(selectedFilePath, title, author, year);
        closeUploadModal();
    });

    cancelConfigureBtn.addEventListener('click', closeConfigureModal);
    configureModalOverlay.addEventListener('click', (event) => {
        if (event.target === configureModalOverlay) closeConfigureModal();
    });
    backConfigureBtn.addEventListener('click', () => {
        closeConfigureModal();
        openUploadModal();
    });
    
    previewPrevBtn.addEventListener('click', () => {
        if (currentPreviewPage > 0) {
            currentPreviewPage--;
            updatePreviewNavigation();
        }
    });

    previewNextBtn.addEventListener('click', () => {
        if (currentPreviewPage < totalPreviewPages - 1) {
            currentPreviewPage++;
            updatePreviewNavigation();
        }
    });
    
    // *** UPDATED ***: Calls the REAL API now
    generateAudiobookBtn.addEventListener('click', async () => {
        const generationData = {
            filePath: selectedFilePath,
            title: document.getElementById('config-book-title').value,
            author: document.getElementById('config-book-author').value,
            year: document.getElementById('config-book-year').value,
            persona: document.getElementById('config-persona').value,
            voice: document.getElementById('config-voice').value,
            sfx: document.getElementById('config-sfx-intensity').value,
            speed: document.getElementById('config-narrator-speed').value,
        };
        
        console.log('Generation data prepared:', generationData);
        
        currentBookData = {
            title: generationData.title,
            author: generationData.author,
            year: generationData.year,
            coverUrl: generatedCoverUrl || null,
            filePath: selectedFilePath,
            progress: 0,
            audioPath: null // Will be filled by generateAudiobook()
        };
        
        console.log('Current book data stored:', currentBookData);
        
        if (!currentBookData.filePath) {
            console.error('WARNING: filePath is missing from currentBookData!');
            alert('Error: File path is missing. Please try uploading again.');
            return;
        }
        
        closeConfigureModal();
        openGeneratingModal();
        
        // *** Call the actual API ***
        try {
            await generateAudiobook(generationData); // This function is defined below
        } catch (error) {
            console.error('Generation failed:', error);
            closeGeneratingModal();
            alert('Error generating audiobook: ' + error.message);
        }
    });

    // === API Integration Functions
    async function generateAudiobook(config) {
        const RENDER_BASE = "https://audioapi-g2ru.onrender.com";
        const AUTH_TOKEN = "Potato";
        
        try {
            // Step 1: Get the current Gradio URL
            updateGenerationProgress('Connecting to API...', 5);
            const gradioUrl = await getGradioUrl(RENDER_BASE, AUTH_TOKEN);
            
            if (!gradioUrl) {
                throw new Error('No Gradio URL configured. Please set the URL first.');
            }
            
            console.log('Using Gradio URL:', gradioUrl);
            
            // Step 2: Read the book file
            updateGenerationProgress('Reading book file...', 15);
            const fileBuffer = await window.electronAPI.readFileBuffer(config.filePath);
            const fileBlob = new Blob([fileBuffer]);
            const fileName = config.filePath.split(/[\\/]/).pop();
            
            // Step 3: Text Tagging Model (TTM)
            updateGenerationProgress('Processing text and adding emotion tags...', 30);
            const taggedText = await callTaggingAPI(gradioUrl, fileBlob, fileName, config);
            
            // Step 4: Tagged Text to Speech (3TS)
            updateGenerationProgress('Generating emotional speech...', 55);
            const audioData = await callTextToSpeechAPI(gradioUrl, taggedText, config);
            
            // Step 5: Sound Effects Model (SEM)
            updateGenerationProgress('Adding sound effects...', 80);
            const finalAudio = await callSoundEffectsAPI(gradioUrl, audioData, config);
            
            // Step 6: Save the audio
            updateGenerationProgress('Saving audiobook...', 95);
            const audioPath = await window.electronAPI.saveAudio(finalAudio, config.title);
            
            // Update book data with audio path
            currentBookData.audioPath = audioPath;
            
            updateGenerationProgress('Complete!', 100);
            
            setTimeout(() => {
                closeGeneratingModal();
                openCompleteModal();
            }, 1000);
            
        } catch (error) {
            console.error('API Error:', error);
            throw error; // This will be caught by the event listener
        }
    }

    async function getGradioUrl(baseUrl, token) {
        try {
            const response = await fetch(`${baseUrl}/current`, {
                headers: { 'X-Auth-Token': token }
            });
            
            if (!response.ok) {
                throw new Error('Failed to get Gradio URL');
            }
            
            const data = await response.json();
            return data.url;
        } catch (error) {
            console.error('Error getting Gradio URL:', error);
            throw error;
        }
    }

    async function callTaggingAPI(gradioUrl, fileBlob, fileName, config) {
        try {
            const formData = new FormData();
            formData.append('file', fileBlob, fileName);
            formData.append('persona', config.persona);
            
            const response = await fetch(`${gradioUrl}/api/ttm`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Text tagging failed: ' + response.statusText);
            }
            
            const result = await response.json();
            return result.tagged_text || result;
        } catch (error) {
            console.error('Tagging API error:', error);
            throw new Error('Failed to process text: ' + error.message);
        }
    }

    async function callTextToSpeechAPI(gradioUrl, taggedText, config) {
        try {
            const response = await fetch(`${gradioUrl}/api/tts`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tagged_text: taggedText,
                    voice: config.voice,
                    speed: parseFloat(config.speed)
                })
            });
            
            if (!response.ok) {
                throw new Error('Text to speech failed: ' + response.statusText);
            }
            
            return await response.arrayBuffer();
        } catch (error) {
            console.error('TTS API error:', error);
            throw new Error('Failed to generate speech: ' + error.message);
        }
    }

    async function callSoundEffectsAPI(gradioUrl, audioData, config) {
        try {
            const formData = new FormData();
            formData.append('audio', new Blob([audioData], { type: 'audio/mpeg' }));
            formData.append('intensity', config.sfx);
            
            const response = await fetch(`${gradioUrl}/api/sfx`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Sound effects failed: ' + response.statusText);
            }
            
            return await response.arrayBuffer();
        } catch (error) {
            console.error('SFX API error:', error);
            throw new Error('Failed to add sound effects: ' + error.message);
        }
    }

    function updateGenerationProgress(message, percent) {
        generationProgress.style.width = `${percent}%`;
        generationStatus.textContent = message;
    }

    cancelGenerationBtn.addEventListener('click', () => {
        generationCancelled = true;
        closeGeneratingModal();
        console.log('Generation cancelled by user.');
        // TODO: Need to add logic to actually cancel the fetch requests if possible
    });

    addToBookshelfBtn.addEventListener('click', async () => {
        console.log('Adding to bookshelf...');
        if (currentBookData) {
            await saveBookToDatabase(currentBookData);
            await loadBooksFromDatabase();
        }
        closeCompleteModal();
    });
    
    listenNowBtn.addEventListener('click', async () => {
        console.log('Listen Now clicked');
        
        if (currentBookData) {
            console.log('Saving book to database...');
            await saveBookToDatabase(currentBookData);
            
            console.log('Loading books from database...');
            await loadBooksFromDatabase();
            
            console.log('Finding newly added book...');
            const books = await window.electronAPI.getBooksFromDb();
            
            const newBook = books.find(b => b.filePath === currentBookData.filePath);
            console.log('Found new book:', newBook);
            
            if (newBook) {
                closeCompleteModal();
                setTimeout(() => {
                    openBookReader(newBook);
                }, 100);
            } else {
                console.error('Could not find newly added book in database');
                alert('Book was added but could not be opened. Please click it from the bookshelf.');
            }
        } else {
            console.error('No current book data');
        }
    });
    
    completeModalOverlay.addEventListener('click', (event) => {
        if (event.target === completeModalOverlay) closeCompleteModal();
    });

    cancelDeleteBtn.addEventListener('click', closeDeleteModal);
    confirmDeleteBtn.addEventListener('click', deleteBook);
    deleteModalOverlay.addEventListener('click', (event) => {
        if (event.target === deleteModalOverlay) closeDeleteModal();
    });

    // Reader controls
    readerBackBtn.addEventListener('click', () => {
        readerView.classList.add('hidden');
        mainContentArea.classList.remove('hidden');
        updateBookProgress(); // Save progress when leaving
        
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.src = ''; // Release memory
            currentAudio = null;
            isPlaying = false;
            playPauseBtn.innerHTML = '▶';
        }
    });

    prevPageBtn.addEventListener('click', () => {
        const isPdfPage = currentChapters[currentChapterIndex].includes('<img');
        if (isPdfPage) {
            if (currentChapterIndex > 0) {
                loadChapter(currentChapterIndex - 2);
            }
        } else {
            if (currentChapterIndex > 0) {
                loadChapter(currentChapterIndex - 1);
            }
        }
    });

    nextPageBtn.addEventListener('click', () => {
        const isPdfPage = currentChapters[currentChapterIndex].includes('<img');
        if (isPdfPage) {
            if (currentChapterIndex + 2 < currentChapters.length) {
                // Update progress based on page turn (if no audio)
                if (!currentAudio) {
                    currentReaderBook.progress = ((currentChapterIndex + 2) / currentChapters.length) * 100;
                    updateBookProgress();
                }
                loadChapter(currentChapterIndex + 2);
            }
        } else {
            if (currentChapterIndex < currentChapters.length - 1) {
                // Update progress based on page turn (if no audio)
                if (!currentAudio) {
                    currentReaderBook.progress = ((currentChapterIndex + 1) / currentChapters.length) * 100;
                    updateBookProgress();
                }
                loadChapter(currentChapterIndex + 1);
            }
        }
    });

    chapterSelect.addEventListener('change', (e) => {
        loadChapter(parseInt(e.target.value));
    });

    // *** UPDATED ***
    playPauseBtn.addEventListener('click', () => {
        if (!currentAudio) return;
        
        if (isPlaying) {
            currentAudio.pause();
        } else {
            currentAudio.play();
        }
        isPlaying = !isPlaying;
        playPauseBtn.innerHTML = isPlaying ? '⏸' : '▶';
        console.log('Audio Play/Pause Toggled:', isPlaying);
    });

    // *** UPDATED ***
    rewindBtn.addEventListener('click', () => {
        if (currentAudio) {
            currentAudio.currentTime = Math.max(0, currentAudio.currentTime - 10);
            console.log('Rewind 10s');
        }
    });

    // *** UPDATED ***
    forwardBtn.addEventListener('click', () => {
        if (currentAudio) {
            currentAudio.currentTime = Math.min(currentAudio.duration, currentAudio.currentTime + 10);
            console.log('Forward 10s');
        }
    });

    // *** UPDATED ***
    speedSlider.addEventListener('input', (e) => {
        const speed = e.target.value;
        speedValue.textContent = `${speed}x`;
        if (currentAudio) {
            currentAudio.playbackRate = parseFloat(speed);
        }
        console.log(`Audio Speed set to: ${speed}x`);
    });

    // *** UPDATED ***
    volumeSlider.addEventListener('input', (e) => {
        const volume = e.target.value;
        if (currentAudio) {
            currentAudio.volume = parseFloat(volume) / 100; // HTML is 0-100, Audio is 0-1
        }
        console.log(`Audio Volume set to: ${volume}`);
    });

    // === INITIALISE APP ===
    resetUploadUI();
    await loadBooksFromDatabase();
    console.log('App fully loaded');
});