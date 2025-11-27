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
    const readerMetadata = document.getElementById('reader-metadata');
    const readerHeader = document.querySelector('.reader-header');
    const readerHeaderTrigger = document.querySelector('.reader-header-trigger'); // The hover zone
    const readerPageLeftContent = document.getElementById('reader-page-left-content');
    const readerPageRightContent = document.getElementById('reader-page-right-content');
    const prevPageBtn = document.getElementById('prev-page-btn');
    const nextPageBtn = document.getElementById('next-page-btn');
    const pageIndicator = document.getElementById('page-indicator');
    
    // Audio Controls
    const audioControlsBar = document.querySelector('.audio-controls'); 
    const playPauseBtn = document.getElementById('play-pause-btn');
    const rewindBtn = document.getElementById('rewind-btn');
    const forwardBtn = document.getElementById('forward-btn');
    const speedSlider = document.getElementById('speed-slider');
    const volumeSlider = document.getElementById('volume-slider');
    const speedValue = document.getElementById('speed-value');
    const chapterSelect = document.getElementById('chapter-select');
    const pageSelect = document.getElementById('page-select');
    const pageSelectGroup = document.getElementById('page-select-group');
    const autoTurnBtn = document.getElementById('auto-turn-btn');

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
    let currentChapterTitles = [];
    let currentChapterIndex = 0;
    let currentPageIndex = 0;
    let totalPages = 0;
    let isPlaying = false;
    let bookToDelete = null;
    let currentAudio = null;
    let timestamps = null;
    let wordsArray = [];
    let currentWordIndex = 0;
    let updateInterval = null;
    let autoTurnEnabled = false;
    let pageTimestamps = []; // Stores which audio time corresponds to which page
    let headerAutoHideTimeout = null;

    // === HEADER VISIBILITY LOGIC ===
    // This handles the behaviour where the header shows on open, hides after 10s,
    // and reappears when hovering the top-left corner.
    
    function showHeader() {
        readerHeader.classList.add('visible');
        
        // Clear existing timeout if any, so it doesn't hide abruptly
        if (headerAutoHideTimeout) {
            clearTimeout(headerAutoHideTimeout);
            headerAutoHideTimeout = null;
        }
    }

    function hideHeader() {
        // Remove the class that gives it opacity
        readerHeader.classList.remove('visible');
    }

    function initialiseHeaderVisibility() {
        // Show immediately upon opening
        showHeader();
        
        // Hide after 10 seconds of inactivity
        headerAutoHideTimeout = setTimeout(() => {
            hideHeader();
        }, 10000); // 10 seconds
    }

    // Set up hover listeners for the trigger zone
    if (readerHeaderTrigger) {
        readerHeaderTrigger.addEventListener('mouseenter', showHeader);
        readerHeaderTrigger.addEventListener('mouseleave', () => {
            // Only hide if we aren't also hovering the header itself
            // We give a small delay to allow moving mouse from trigger to header
            headerAutoHideTimeout = setTimeout(() => {
                // Check if mouse is over header? (CSS hover handles keeping it open slightly, but JS is safer)
                if (!readerHeader.matches(':hover')) {
                    hideHeader();
                }
            }, 500); 
        });
    }

    // Also keep header open if hovering the header itself (e.g. to click Back)
    readerHeader.addEventListener('mouseenter', showHeader);
    readerHeader.addEventListener('mouseleave', () => {
        // Hide shortly after leaving the header
        headerAutoHideTimeout = setTimeout(hideHeader, 1000);
    });

    // === PDF CHAPTER DETECTION ===
    function detectPdfChapters(pdf) {
        return new Promise(async (resolve) => {
            const chapterPatterns = [
                /^chapter\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|[ivxlcdm]+)/i,
                /^part\s+(\d+|one|two|three|four|five|[ivxlcdm]+)/i,
                /^section\s+(\d+|one|two|[ivxlcdm]+)/i,
                /^book\s+(\d+|one|two|three|[ivxlcdm]+)/i,
                /^(\d+)\.\s+[A-Z]/,  // "1. Introduction"
                /^[ivxlcdm]+\.\s+[A-Z]/i  // "I. Introduction"
            ];
            
            const detectedChapters = [];
            
            try {
                for (let pageNum = 1; pageNum <= Math.min(pdf.numPages, 200); pageNum++) {
                    const page = await pdf.getPage(pageNum);
                    const textContent = await page.getTextContent();
                    
                    let pageText = '';
                    for (const item of textContent.items) {
                        pageText += item.str + ' ';
                    }
                    
                    // Check first few lines for chapter indicators
                    const lines = pageText.split('\n').slice(0, 5);
                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (trimmed.length > 0 && trimmed.length < 100) {
                            for (const pattern of chapterPatterns) {
                                if (pattern.test(trimmed)) {
                                    detectedChapters.push({
                                        pageNum: pageNum,
                                        title: trimmed.substring(0, 50) // Limit title length
                                    });
                                    break;
                                }
                            }
                        }
                    }
                }
            } catch (error) {
                console.log('Error detecting chapters:', error);
            }
            
            resolve(detectedChapters);
        });
    }

    // === AUTO-TURN PAGE FUNCTIONALITY ===
    function setupAutoTurn() {
        if (!currentAudio || !autoTurnEnabled) return;
        
        // Calculate page timestamps if not provided by API
        // This maps audio time to specific page pairs
        if (!pageTimestamps || pageTimestamps.length === 0) {
            calculatePageTimestamps();
        }
    }

    function calculatePageTimestamps() {
        if (!currentAudio || !currentChapters.length) return;
        
        const duration = currentAudio.duration;
        // The reader displays 2 pages at a time, so we group them
        const numPagePairs = Math.ceil(currentChapters.length / 2);
        
        pageTimestamps = [];
        
        // Estimate words per page to distribute time proportionally
        const wordsPerPage = currentChapters.map((chapter, index) => {
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = chapter;
            const text = tempDiv.textContent || '';
            const wordCount = text.split(/\s+/).filter(w => w.length > 0).length;
            return { pageIndex: index, wordCount: wordCount };
        });
        
        const totalWords = wordsPerPage.reduce((sum, page) => sum + page.wordCount, 0);
        
        let accumulatedTime = 0;
        for (let i = 0; i < numPagePairs; i++) {
            const leftPageIndex = i * 2;
            const rightPageIndex = i * 2 + 1;
            
            // Get word counts for this pair (left + right page)
            const leftWords = wordsPerPage[leftPageIndex]?.wordCount || 0;
            const rightWords = wordsPerPage[rightPageIndex]?.wordCount || 0;
            const pairWords = leftWords + rightWords;
            
            // Allocate time based on how much text is on these two pages
            const pairDuration = (pairWords / totalWords) * duration;
            
            pageTimestamps.push({
                pageIndex: leftPageIndex, // The index we use to load the pair
                startTime: accumulatedTime,
                endTime: accumulatedTime + pairDuration,
                wordCount: pairWords
            });
            
            accumulatedTime += pairDuration;
        }
        
        console.log('Calculated page timestamps:', pageTimestamps);
    }

    function checkAutoTurnPage() {
        if (!autoTurnEnabled || !currentAudio || !pageTimestamps.length) return;
        
        const currentTime = currentAudio.currentTime;
        
        // Find which page pair corresponds to the current time
        const currentPairTimestamp = pageTimestamps.find(ts => 
            currentTime >= ts.startTime && currentTime < ts.endTime
        );
        
        if (currentPairTimestamp) {
            // We found the correct page pair for this time.
            // If we are NOT currently on this page, turn to it.
            if (currentChapterIndex !== currentPairTimestamp.pageIndex) {
                console.log(`Auto-turning to page pair starting at ${currentPairTimestamp.pageIndex + 1}`);
                loadChapter(currentPairTimestamp.pageIndex);
            }
        }
    }

    // === DATABASE & BOOKSHELF FUNCTIONS ===
    async function loadBooksFromDatabase() {
        try {
            console.log('Loading books from database...');
            const books = await window.electronAPI.getBooksFromDb();
            const bookshelfGrid = document.getElementById('bookshelf-grid');
            bookshelfGrid.innerHTML = '';
            
            if (books.length === 0) {
                showEmptyState();
            } else {
                books.forEach((book) => {
                    addBookToBookshelf(
                        book.title,
                        book.author,
                        book.year,
                        book.coverUrl,
                        book.progress || 0,
                        book.id,
                        book.filePath
                    );
                });
            }
        } catch (error) {
            console.error('Error loading books from database:', error);
        }
    }

    function showEmptyState() {
        const bookshelfGrid = document.getElementById('bookshelf-grid');
        // Ensure CSS centres this correctly (see styles.css .empty-state)
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
            if (!book.filePath) {
                alert('Cannot save book: file path is missing');
                return;
            }
            const result = await window.electronAPI.addBookToDb(book);
            if (!result.success) {
                console.error('Failed to save book:', result.error);
                alert('Failed to save book: ' + result.error);
            }
        } catch (error) {
            console.error('Error saving book to database:', error);
            alert('Error saving book: ' + error.message);
        }
    }

    function addBookToBookshelf(title, author, year, coverUrl, progress = 0, bookId = null, filePath = null) {
        const bookshelfGrid = document.getElementById('bookshelf-grid');
        const bookItem = document.createElement('div');
        bookItem.className = 'book-item';
        bookItem.dataset.bookId = bookId;
        
        const authorYearText = (author && year) ? `${author} - ${year}` : (author || 'Unknown Author');
        
        let progressClass = 'yellow';
        let progressText = 'UNREAD';
        let progressWidth = 0;
        
        if (progress > 0 && progress < 100) {
            progressClass = 'blue';
            progressText = `${Math.round(progress)}%`;
            progressWidth = progress;
        } else if (progress >= 100) {
            progressClass = 'green';
            progressText = 'READ';
            progressWidth = 100;
        } else if (progress === 0) {
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
            
            const books = await window.electronAPI.getBooksFromDb();
            const bookData = books.find(b => b.id === bookId);

            if (bookData) {
                openBookReader(bookData);
            } else {
                 alert('Error: Could not find book data.');
            }
        });
        
        const deleteBtn = bookItem.querySelector('.delete-book-btn');
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            openDeleteModal(bookId, title);
        });
        
        bookshelfGrid.appendChild(bookItem);
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
                    closeDeleteModal();
                    await loadBooksFromDatabase();
                } else {
                    alert('Failed to delete book. Please try again.');
                }
            } catch (error) {
                console.error('Error deleting book:', error);
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
        generationStatus.textContent = 'Connecting to AI Model...';
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
        configCoverPreview.innerHTML = '';
        configCoverPreview.style.backgroundImage = '';
        previewPane.innerHTML = '<p>Loading preview...</p>';
        previewPane.classList.remove('epub-preview', 'pdf-preview', 'fallback-preview');
        bookPages = [];
        currentPreviewPage = 0;
        generatedCoverUrl = null;
        
        const extension = filePath.split('.').pop().toLowerCase();
        
        try {
            if (extension === 'epub') {
                await loadEpubPreview(filePath);
            } else if (extension === 'pdf') {
                await loadPdfPreview(filePath);
            } else {
                previewPane.classList.add('fallback-preview');
                previewPane.innerHTML = '<h3>Preview Not Available</h3><p>Unsupported file format for preview.</p>';
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
            const result = await window.electronAPI.epubParse(filePath);
            
            // Set cover if available
            if (result.coverUrl) {
                generatedCoverUrl = result.coverUrl;
                configCoverPreview.style.backgroundImage = `url(${generatedCoverUrl})`;
                configCoverPreview.style.backgroundSize = 'cover';
                configCoverPreview.style.backgroundPosition = 'center';
            }
            
            // Load preview pages
            const chaptersToPreview = Math.min(5, result.chapters.length);
            for (let i = 0; i < chaptersToPreview; i++) {
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = result.chapters[i];
                
                const chapterTitle = result.chapterTitles ? result.chapterTitles[i] : `Chapter ${i + 1}`;
                const paragraphs = Array.from(tempDiv.querySelectorAll('p'));
                
                let pageHTML = `<h3>${chapterTitle}</h3>`;
                paragraphs.slice(0, 15).forEach(p => {
                    if (p.textContent.trim()) pageHTML += `<p>${p.textContent}</p>`;
                });
                
                bookPages.push(pageHTML || '<p>No content available.</p>');
            }
            
            totalPreviewPages = bookPages.length || 1;
            updatePreviewNavigation();

        } catch (error) {
            throw error;
        }
    }

    async function loadPdfPreview(filePath) {
        try {
            previewPane.classList.add('pdf-preview');
            const arrayBuffer = await window.electronAPI.readFileBuffer(filePath);
            const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
            const pdf = await loadingTask.promise;
            
            const pagesToLoad = Math.min(5, pdf.numPages);
            bookPages = [];
            
            for (let pageNum = 1; pageNum <= pagesToLoad; pageNum++) {
                const page = await pdf.getPage(pageNum);
                const viewport = page.getViewport({ scale: 1.0 });
                
                // High quality rendering - 3x scale
                const scale = (800 / viewport.width) * 3;
                const scaledViewport = page.getViewport({ scale: scale });

                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.height = scaledViewport.height;
                canvas.width = scaledViewport.width;
                
                await page.render({ 
                    canvasContext: context, 
                    viewport: scaledViewport,
                    intent: 'print'
                }).promise;
                
                bookPages.push(`<img src="${canvas.toDataURL('image/png')}" alt="Page ${pageNum}" style="max-width: 100%; height: auto;">`);
            }

            // Render Cover at high quality
            const firstPage = await pdf.getPage(1);
            const coverViewport = firstPage.getViewport({ scale: 3.0 }); // High quality
            const coverCanvas = document.createElement('canvas');
            const coverContext = coverCanvas.getContext('2d');
            coverCanvas.height = coverViewport.height;
            coverCanvas.width = coverViewport.width;
            await firstPage.render({ 
                canvasContext: coverContext, 
                viewport: coverViewport,
                intent: 'print'
            }).promise;
            
            generatedCoverUrl = coverCanvas.toDataURL('image/png');
            configCoverPreview.style.backgroundImage = `url(${generatedCoverUrl})`;
            configCoverPreview.style.backgroundSize = 'cover';
            configCoverPreview.style.backgroundPosition = 'center';
            
            totalPreviewPages = bookPages.length || 1;
            currentPreviewPage = 0;
            updatePreviewNavigation();
            
        } catch (error) {
            throw error;
        }
    }

    // === BOOK READER FUNCTIONS ===
    async function openBookReader(book) {
        if (!book || !book.filePath) {
            alert('Cannot open book: invalid book data');
            return;
        }
        
        currentReaderBook = book;
        mainContentArea.classList.add('hidden');
        readerView.classList.remove('hidden');
        
        readerTitle.textContent = book.title || 'Untitled Book';
        
        // Format metadata: "Author • Year" or handle unknowns
        const author = book.author || 'Unknown Author';
        const year = book.year && book.year !== 'N/A' ? book.year : '';
        readerMetadata.textContent = year ? `${author} • ${year}` : author;
        
        readerPageLeftContent.innerHTML = '<p>Loading...</p>';
        readerPageRightContent.innerHTML = '<p>Loading...</p>';
        
        // Initialise the header visibility logic (shows for 10s then hides)
        initialiseHeaderVisibility();
        
        try {
            const extension = book.filePath.split('.').pop().toLowerCase();
            
            if (extension === 'epub') {
                pageSelectGroup.style.display = 'none'; // EPUBs don't need page dropdown
                const result = await window.electronAPI.epubParse(book.filePath);
                currentChapters = result.chapters || [];
                currentChapterTitles = result.chapterTitles || [];
            } else if (extension === 'pdf') {
                await loadPdfForReading(book.filePath);
                return; // PDF function handles the rest
            } 
            
            if (currentChapters.length === 0) currentChapters = ['<p>No content extracted.</p>'];
            if (currentChapterTitles.length === 0) {
                currentChapterTitles = currentChapters.map((_, i) => `Chapter ${i + 1}`);
            }
            
            // Populate chapter dropdown
            chapterSelect.innerHTML = '';
            currentChapterTitles.forEach((title, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = title;
                chapterSelect.appendChild(option);
            });
            
            // Load saved position or start at beginning
            currentChapterIndex = book.currentPage || 0;
            loadChapter(currentChapterIndex);

            // Load Audio with timestamps
            if (book.audioPath) {
                loadAudioWithTimestamps(book);
            } else {
                audioControlsBar.style.display = 'none';
            }
            
        } catch (error) {
            console.error('Error opening book reader:', error);
            readerPageLeftContent.innerHTML = `<p>Error loading book: ${error.message}</p>`;
        }
    }

    // Export these functions so they can be used in the main renderer.js
    window.detectPdfChapters = detectPdfChapters;
    window.loadPdfForReading = loadPdfForReading;
    window.setupAutoTurn = setupAutoTurn;
    window.calculatePageTimestamps = calculatePageTimestamps;
    window.checkAutoTurnPage = checkAutoTurnPage;
    window.showHeader = showHeader;
    window.hideHeader = hideHeader;
    window.initialiseHeaderVisibility = initialiseHeaderVisibility;
    window.openBookReader = openBookReader;

    async function loadAudioWithTimestamps(book) {
        audioControlsBar.style.display = 'flex';
        try {
            const audioDataUrl = await window.electronAPI.readAudioFile(book.audioPath);
            currentAudio = new Audio(audioDataUrl);
            currentAudio.playbackRate = parseFloat(speedSlider.value);
            currentAudio.volume = parseFloat(volumeSlider.value) / 100;
            
            // Parse timestamps if available
            if (book.timestamps) {
                try {
                    timestamps = JSON.parse(book.timestamps);
                    console.log('Loaded timestamps:', timestamps);
                } catch (e) {
                    console.error('Failed to parse timestamps:', e);
                }
            }
            
            // Load to saved position
            currentAudio.addEventListener('loadedmetadata', () => {
                if (book.progress > 0 && currentAudio.duration) {
                    currentAudio.currentTime = (book.progress / 100) * currentAudio.duration;
                }
                // Once metadata is loaded, we can calculate page durations
                calculatePageTimestamps();
            });

            currentAudio.addEventListener('timeupdate', updateAudioProgress);
            currentAudio.addEventListener('ended', () => {
                isPlaying = false;
                playPauseBtn.innerHTML = '▶';
            });
        } catch (audioError) {
            console.error('Error loading audio file:', audioError);
            audioControlsBar.style.display = 'none';
        }
    }

    async function loadPdfForReading(filePath) {
        try {
            const arrayBuffer = await window.electronAPI.readFileBuffer(filePath);
            const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
            const pdf = await loadingTask.promise;
            
            currentChapters = [];
            currentChapterTitles = [];
            
            // Get the actual viewport dimensions for each page
            const readerContainer = document.querySelector('.reader-content');
            const containerWidth = readerContainer.offsetWidth / 2; // Divided by 2 for two-page view
            const containerHeight = readerContainer.offsetHeight;
            
            for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                const page = await pdf.getPage(pageNum);
                
                // Get original viewport
                let viewport = page.getViewport({ scale: 1.0 });
                
                // Calculate scale to fit container while maintaining aspect ratio
                const scaleX = containerWidth / viewport.width;
                const scaleY = containerHeight / viewport.height;
                const scale = Math.min(scaleX, scaleY);
                
                // Apply HIGH QUALITY scale (3x for crisp rendering)
                const highQualityScale = scale * 3;
                viewport = page.getViewport({ scale: highQualityScale });

                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;
                
                await page.render({ 
                    canvasContext: context, 
                    viewport: viewport,
                    intent: 'print' // Use print quality
                }).promise;
                
                currentChapters.push(`<img src="${canvas.toDataURL('image/png')}" alt="Page ${pageNum}" style="width: 100%; height: 100%; object-fit: contain;">`);
                
                // Every 10 pages is a "chapter" for navigation
                if (pageNum % 10 === 1 || pageNum === 1) {
                    const endPage = Math.min(pageNum + 9, pdf.numPages);
                    currentChapterTitles.push(`Pages ${pageNum}-${endPage}`);
                }
            }
            
            // Populate chapter dropdown (every 10 pages)
            chapterSelect.innerHTML = '';
            for (let i = 0; i < currentChapters.length; i += 10) {
                const option = document.createElement('option');
                option.value = i;
                const endPage = Math.min(i + 10, currentChapters.length);
                option.textContent = `Pages ${i + 1}-${endPage}`;
                chapterSelect.appendChild(option);
            }
            
            currentChapterIndex = currentReaderBook.currentPage || 0;
            loadChapter(currentChapterIndex);
            
            // Load audio
            if (currentReaderBook.audioPath) {
                loadAudioWithTimestamps(currentReaderBook);
            } else {
                audioControlsBar.style.display = 'none';
            }
        } catch (error) {
            throw error;
        }
    }

    function loadChapter(index) {
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
            readerPageLeftContent.classList.add('pdf-content');
            readerPageRightContent.classList.add('pdf-content');
            
            if (rightPageHtml) pageIndicator.textContent = `Pages ${index + 1}-${index + 2} of ${currentChapters.length}`;
            else pageIndicator.textContent = `Page ${index + 1} of ${currentChapters.length}`;
            
            prevPageBtn.disabled = index === 0;
            nextPageBtn.disabled = index + 1 >= currentChapters.length - 1; 
        } else {
            // EPUB - split content across two pages
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = chapterHtml;
            const allElements = Array.from(tempDiv.children);
            const midPoint = Math.ceil(allElements.length / 2);
            
            readerPageLeftContent.innerHTML = '';
            allElements.slice(0, midPoint).forEach(el => readerPageLeftContent.appendChild(el.cloneNode(true)));
            
            readerPageRightContent.innerHTML = '';
            allElements.slice(midPoint).forEach(el => readerPageRightContent.appendChild(el.cloneNode(true)));

            readerPageLeftContent.classList.remove('pdf-content');
            readerPageRightContent.classList.remove('pdf-content');
            readerPageLeftContent.classList.add('epub-content');
            readerPageRightContent.classList.add('epub-content');
            
            const chapterTitle = currentChapterTitles[index] || `Chapter ${index + 1}`;
            pageIndicator.textContent = `${chapterTitle} (${index + 1} of ${currentChapters.length})`;
            prevPageBtn.disabled = index === 0;
            nextPageBtn.disabled = index >= currentChapters.length - 1;
        }
        
        chapterSelect.value = index;
        readerPageLeftContent.scrollTop = 0;
        readerPageRightContent.scrollTop = 0;
        
        // Build words array for highlighting
        if (!isPdfPage) {
            buildWordsArray();
        }
    }

    function buildWordsArray() {
        wordsArray = [];
        const leftText = readerPageLeftContent.innerText || '';
        const rightText = readerPageRightContent.innerText || '';
        const fullText = leftText + ' ' + rightText;
        
        const words = fullText.split(/\s+/).filter(w => w.length > 0);
        wordsArray = words;
        currentWordIndex = 0;
    }

    function updateAudioProgress() {
        if (!currentAudio || !currentAudio.duration) return;
        
        const currentTime = currentAudio.currentTime;
        const progress = (currentTime / currentAudio.duration) * 100;
        currentReaderBook.progress = progress;
        
        // Check auto-turn every update to see if we entered a new page range
        checkAutoTurnPage();
    }

    async function updateBookProgress() {
        if (!currentReaderBook) return;
        try {
            await window.electronAPI.updateBookProgress(
                currentReaderBook.id, 
                currentReaderBook.progress,
                currentChapterIndex
            );
            const bookItem = document.querySelector(`[data-book-id="${currentReaderBook.id}"]`);
            if (bookItem) {
                const progressBar = bookItem.querySelector('.progress-bar');
                const progressText = bookItem.querySelector('.progress-text');
                progressBar.className = `progress-bar ${getProgressClass(currentReaderBook.progress)}`;
                progressBar.style.width = `${currentReaderBook.progress}%`;
                progressText.textContent = getProgressText(currentReaderBook.progress);
            }
        } catch (error) {
            console.error('Error updating book progress:', error);
        }
    }

    // === UI HELPER FUNCTIONS ===
    function showFileInUI(filePath) {
        const filename = filePath.split(/[\\/]/).pop();
        selectedFilePath = filePath;
        dropZone.style.display = 'none';
        uploadList.style.display = 'block';
        uploadFormFields.forEach(field => field.style.display = 'flex');
        uploadList.innerHTML = `<div class="upload-item"><span>${filename}</span><span class="remove-file-icon" id="remove-file-btn">×</span></div>`;
        
        const title = filename.replace(/\.[^/.]+$/, "");
        document.getElementById('book-title').value = title;
        
        const removeBtn = document.getElementById('remove-file-btn');
        if (removeBtn) removeBtn.addEventListener('click', resetUploadUI);
    }
    
    function resetUploadUI() {
        selectedFilePath = null;
        dropZone.style.display = 'block';
        uploadList.style.display = 'none';
        uploadFormFields.forEach(field => field.style.display = 'none');
        uploadList.innerHTML = '';
        document.getElementById('book-title').value = '';
        document.getElementById('book-author').value = '';
        document.getElementById('book-year').value = '';
    }

    function updateGenerationProgress(message, percent) {
        generationProgress.style.width = `${percent}%`;
        generationStatus.textContent = message;
    }

    // === EVENT LISTENERS ===
    addBookBtn.addEventListener('click', openUploadModal);
    cancelUploadBtn.addEventListener('click', () => { closeUploadModal(); resetUploadUI(); });
    uploadModalOverlay.addEventListener('click', (e) => { if (e.target === uploadModalOverlay) { closeUploadModal(); resetUploadUI(); } });

    browseLink.addEventListener('click', async (e) => {
        e.preventDefault();
        const filePath = await window.electronAPI.openFile();
        if (filePath) showFileInUI(filePath);
    });
    
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); });
    dropZone.addEventListener('drop', async (e) => {
        e.preventDefault(); dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0 && e.dataTransfer.files[0].path) showFileInUI(e.dataTransfer.files[0].path);
    });

    confirmUploadBtn.addEventListener('click', () => {
        const title = document.getElementById('book-title').value;
        if (!selectedFilePath) return alert('Please select a book file');
        if (!title) return alert('Please provide a title');
        openConfigureModal(selectedFilePath, title, document.getElementById('book-author').value, document.getElementById('book-year').value);
        closeUploadModal();
    });

    cancelConfigureBtn.addEventListener('click', closeConfigureModal);
    backConfigureBtn.addEventListener('click', () => { closeConfigureModal(); openUploadModal(); });
    
    previewPrevBtn.addEventListener('click', () => { if (currentPreviewPage > 0) { currentPreviewPage--; updatePreviewNavigation(); } });
    previewNextBtn.addEventListener('click', () => { if (currentPreviewPage < totalPreviewPages - 1) { currentPreviewPage++; updatePreviewNavigation(); } });
    
    // Generate Audiobook Button
    generateAudiobookBtn.addEventListener('click', async () => {
        const generationData = {
            spaceUrl: "https://audioapi-g2ru.onrender.com",
            filePath: selectedFilePath,
            title: document.getElementById('config-book-title').value,
            author: document.getElementById('config-book-author').value,
            year: document.getElementById('config-book-year').value,
            persona: document.getElementById('config-persona').value,
            voice: document.getElementById('config-voice').value,
            sfx: document.getElementById('config-sfx-intensity').value,
            speed: document.getElementById('config-narrator-speed').value,
        };
        
        currentBookData = {
            title: generationData.title,
            author: generationData.author,
            year: generationData.year,
            coverUrl: generatedCoverUrl || null,
            filePath: selectedFilePath,
            progress: 0,
            audioPath: null,
            timestamps: null,
            currentPage: 0
        };
        
        closeConfigureModal();
        openGeneratingModal();
        
        try {
            updateGenerationProgress('Processing book content...', 10);
            
            const timer = setInterval(() => {
                const currentWidth = parseFloat(generationProgress.style.width) || 10;
                if (currentWidth < 90) {
                    updateGenerationProgress('Generating audio...', currentWidth + 1);
                }
            }, 1000);

            const result = await window.electronAPI.generateAudiobook(generationData);
            
            clearInterval(timer);
            currentBookData.audioPath = result.audioPath;
            currentBookData.timestamps = result.timestamps;
            
            updateGenerationProgress('Complete!', 100);
            
            setTimeout(() => {
                closeGeneratingModal();
                openCompleteModal();
            }, 1000);
            
        } catch (error) {
            console.error('Generation failed:', error);
            closeGeneratingModal();
            alert('Error generating audiobook: ' + error.message);
        }
    });

    cancelGenerationBtn.addEventListener('click', () => {
        generationCancelled = true;
        closeGeneratingModal();
    });

    addToBookshelfBtn.addEventListener('click', async () => {
        if (currentBookData) {
            await saveBookToDatabase(currentBookData);
            await loadBooksFromDatabase();
        }
        closeCompleteModal();
    });
    
    listenNowBtn.addEventListener('click', async () => {
        if (currentBookData) {
            await saveBookToDatabase(currentBookData);
            await loadBooksFromDatabase();
            const books = await window.electronAPI.getBooksFromDb();
            const newBook = books.find(b => b.filePath === currentBookData.filePath);
            if (newBook) {
                closeCompleteModal();
                setTimeout(() => openBookReader(newBook), 100);
            }
        }
    });
    
    // Delete/Cancel Modals
    completeModalOverlay.addEventListener('click', (e) => { if (e.target === completeModalOverlay) closeCompleteModal(); });
    cancelDeleteBtn.addEventListener('click', closeDeleteModal);
    confirmDeleteBtn.addEventListener('click', deleteBook);
    deleteModalOverlay.addEventListener('click', (e) => { if (e.target === deleteModalOverlay) closeDeleteModal(); });

    // Reader Controls
    readerBackBtn.addEventListener('click', () => {
        readerView.classList.add('hidden');
        mainContentArea.classList.remove('hidden');
        updateBookProgress();
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.src = '';
            currentAudio = null;
            isPlaying = false;
            playPauseBtn.innerHTML = '▶';
        }
    });

    prevPageBtn.addEventListener('click', () => {
        const isPdfPage = currentChapters[currentChapterIndex].includes('<img');
        const step = isPdfPage ? 2 : 1;
        if (currentChapterIndex > 0) {
            loadChapter(currentChapterIndex - step);
            if (!currentAudio) {
                currentReaderBook.progress = ((currentChapterIndex) / currentChapters.length) * 100;
                updateBookProgress();
            }
        }
    });

    nextPageBtn.addEventListener('click', () => {
        const isPdfPage = currentChapters[currentChapterIndex].includes('<img');
        const step = isPdfPage ? 2 : 1;
        if (currentChapterIndex + step < currentChapters.length) {
            if (!currentAudio) {
                currentReaderBook.progress = ((currentChapterIndex + step) / currentChapters.length) * 100;
                updateBookProgress();
            }
            loadChapter(currentChapterIndex + step);
        }
    });

    chapterSelect.addEventListener('change', (e) => loadChapter(parseInt(e.target.value)));

    // Page select dropdown
    pageSelect.addEventListener('change', (e) => loadChapter(parseInt(e.target.value)));

    // Auto-turn toggle
    autoTurnBtn.addEventListener('click', () => {
        autoTurnEnabled = !autoTurnEnabled;
        if (autoTurnEnabled) {
            // Add active class for blue styling
            autoTurnBtn.classList.add('active');
            autoTurnBtn.title = 'Auto-turn pages (ON)';
            // Setup timestamps immediately
            setupAutoTurn();
        } else {
            autoTurnBtn.classList.remove('active');
            autoTurnBtn.title = 'Auto-turn pages (OFF)';
        }
    });

    // Audio Controls Listeners
    playPauseBtn.addEventListener('click', () => {
        if (!currentAudio) return;
        if (isPlaying) currentAudio.pause(); else currentAudio.play();
        isPlaying = !isPlaying;
        playPauseBtn.innerHTML = isPlaying ? '⏸' : '▶';
    });

    rewindBtn.addEventListener('click', () => {
        if (currentAudio) currentAudio.currentTime = Math.max(0, currentAudio.currentTime - 10);
    });

    forwardBtn.addEventListener('click', () => {
        if (currentAudio) currentAudio.currentTime = Math.min(currentAudio.duration, currentAudio.currentTime + 10);
    });

    speedSlider.addEventListener('input', (e) => {
        const speed = e.target.value;
        speedValue.textContent = `${speed}x`;
        if (currentAudio) currentAudio.playbackRate = parseFloat(speed);
    });

    volumeSlider.addEventListener('input', (e) => {
        if (currentAudio) currentAudio.volume = parseFloat(e.target.value) / 100;
    });

    // Initialise
    resetUploadUI();
    await loadBooksFromDatabase();
});