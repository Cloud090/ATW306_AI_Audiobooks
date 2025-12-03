// renderer.js
// This file contains the main JavaScript logic for the Narrify application's user interface (UI).
// It handles DOM manipulation, user interactions, and inter-process communication with the main
// Electron process through the `ipcRenderer` exposed in the preload script.

document.addEventListener('DOMContentLoaded', async () => {
    // Console message to confirm the application's UI script has loaded and started.
    console.log('App initialised');
    
    // === DOM ELEMENTS ===
    // The following constants reference elements in the HTML document.
    // Each reference allows the script to show/hide modals, update progress bars,
    // respond to button clicks, and dynamically change UI elements based on user actions.

    // --- Upload modal (first step where a user selects a file to upload) ---
    const uploadModalOverlay = document.getElementById('upload-modal-overlay');   // Full-screen overlay for the upload modal
    const addBookBtn = document.getElementById('add-book');                       // Button that opens the upload modal from the bookshelf
    const cancelUploadBtn = document.getElementById('cancel-upload-btn');         // Button to close the upload modal without selecting a file
    const confirmUploadBtn = document.getElementById('confirm-upload-btn');       // Button to confirm file upload

    // --- Configuration modal (second step where the user configures metadata) ---
    const configureModalOverlay = document.getElementById('configure-modal-overlay'); // Overlay for the configuration modal
    const cancelConfigureBtn = document.getElementById('cancel-configure-btn');       // Cancels configuration and returns to upload modal
    const backConfigureBtn = document.getElementById('back-configure-btn');           // Goes back one modal step to the upload modal
    const generateAudiobookBtn = document.getElementById('generate-audiobook-btn');   // Starts audiobook generation

    // --- Generation modal (shows real-time progress as the server processes the audiobook request) ---
    const generatingModalOverlay = document.getElementById('generating-modal-overlay'); // Overlay shown during processing
    const generationProgress = document.getElementById('generation-progress');          // Progress bar element whose width changes as progress increases
    const generationStatus = document.getElementById('generation-status');              // Text element showing status messages and timing estimates
    const cancelGenerationBtn = document.getElementById('cancel-generation-btn');       // Allows user to cancel the generation process

    // --- Completion modal (shown once the audiobook generation is finished) ---
    const completeModalOverlay = document.getElementById('complete-modal-overlay'); // Overlay for successful generation
    const addToBookshelfBtn = document.getElementById('add-to-bookshelf-btn');     // Button to add the finished audiobook to the user's library
    const listenNowBtn = document.getElementById('listen-now-btn');                // Button to immediately open the audiobook reader view

    // --- Delete confirmation modal ---
    const deleteModalOverlay = document.getElementById('delete-modal-overlay'); // Overlay shown before removing a book
    const deleteBookTitle = document.getElementById('delete-book-title');       // Displays the title of the book to be deleted
    const cancelDeleteBtn = document.getElementById('cancel-delete-btn');       // Cancels deletion
    const confirmDeleteBtn = document.getElementById('confirm-delete-btn');     // Confirms deletion

    // --- Upload zone elements ---
    const browseLink = document.getElementById('browse-link');                     // "Browse files" link to trigger file selection dialogue
    const dropZone = document.getElementById('drop-zone');                         // Drag-and-drop upload area
    const uploadList = document.querySelector('.upload-list-container');            // Shows selected file info
    const uploadFormFields = document.querySelectorAll('#upload-modal-content .form-group'); // Form inputs inside the upload modal (title, author, year)

    // --- Configuration preview elements (shows book cover, text preview etc.) ---
    const configCoverPreview = document.getElementById('config-cover-preview');
    const previewPane = document.getElementById('preview-pane');                         // Container showing a preview of book pages
    const previewPrevBtn = document.getElementById('preview-prev-btn');                  // Navigation button for previous preview page
    const previewNextBtn = document.getElementById('preview-next-btn');                  // Navigation button for next preview page
    const previewPageNumber = document.getElementById('preview-page-number');            // Numeric page indicator (e.g. 1 of 5)
    const previewCurrentPage = document.getElementById('preview-current-page');          // Shows the current preview page number
    const previewTotalPages = document.getElementById('preview-total-pages');            // Shows the total number of preview pages

    // === READER VIEW ELEMENTS ===
    // These elements belong to the reading/listening interface users see after selecting a book.
    const readerView = document.getElementById('book-reader-view');                  // Entire reader view wrapper
    const mainContentArea = document.getElementById('main-content-area');            // The main content region (bookshelf, UI etc.)
    const readerBackBtn = document.getElementById('reader-back-btn');                // Button that takes the user back to the bookshelf
    const readerTitle = document.getElementById('reader-title');                     // Displays the title of the book in the reader
    const readerMetadata = document.getElementById('reader-metadata');               // Displays metadata such as author, chapter etc.

    // --- Reader header (top bar that hides and reappears based on interaction) ---
    const readerHeader = document.querySelector('.reader-header');
    const readerHeaderTrigger = document.querySelector('.reader-header-trigger');     // Hover area used to reveal the header

    // Actual page content displayed in the left and right “page” panels in the reader
    const readerPageLeftContent = document.getElementById('reader-page-left-content');
    const readerPageRightContent = document.getElementById('reader-page-right-content');

    // --- Page navigation controls within the reader ---
    const prevPageBtn = document.getElementById('prev-page-btn');
    const nextPageBtn = document.getElementById('next-page-btn');
    const pageIndicator = document.getElementById('page-indicator');                  // Shows "Page X of Y" or "Chapter Title"

    // === AUDIO CONTROLS ===
    // These elements affect playback of the generated audiobook.
    const audioControlsBar = document.querySelector('.audio-controls'); // Container for all playback controls (shows/hides with the reader)
    const playPauseBtn = document.getElementById('play-pause-btn');     // Starts or pauses audio playback
    const rewindBtn = document.getElementById('rewind-btn');            // Skips backwards a fixed amount of time (10 seconds)
    const forwardBtn = document.getElementById('forward-btn');          // Skips forwards a fixed amount (10 seconds)
    const speedSlider = document.getElementById('speed-slider');        // Adjusts playback speed
    const volumeSlider = document.getElementById('volume-slider');      // Adjusts audio volume
    const speedValue = document.getElementById('speed-value');          // Displays the current playback speed numerically
    const chapterSelect = document.getElementById('chapter-select');    // Dropdown list of detected chapters
    const pageSelect = document.getElementById('page-select');          // Dropdown list of pages for direct navigation (currently unused, kept for future)
    const pageSelectGroup = document.getElementById('page-select-group'); // Container that can be shown/hidden depending on book type

    // === STATE VARIABLES ===
    // These variables track user selections, reader state, generation progress,
    // and information passed between different components of the UI.
    let selectedFilePath = null;                 // Stores the path of the original file the user has chosen to upload
    let copiedFilePath = null;                   // Stores the new file path for the book in AppData after copying
    let currentBook = null;                      // Holds full data for the book currently being previewed or processed (e.g. a PDF/EPUB instance)
    let generationCancelled = false;             // Indicates whether user has cancelled audiobook generation
    let currentPreviewPage = 0;                  // Tracks which preview page is currently shown in the configuration modal (0-indexed)
    let totalPreviewPages = 0;                   // Total number of preview pages available
    let bookPages = [];                          // Stores extracted pages from the uploaded book (used in preview) as HTML/image data
    let currentBookData = null;                  // Book metadata object sent to/received from the backend during generation
    let generatedCoverUrl = null;                // Temporary URL (Data URL or local path) of generated book cover image

    // Reader-specific state
    let currentReaderBook = null;                // Holds complete data for the currently opened book in the reader (loaded from DB)
    let currentChapters = [];                    // List of chapter content (HTML for EPUB, image data for PDF pages)
    let currentChapterTitles = [];               // Chapter titles for display
    let currentChapterIndex = 0;                 // Index of the currently active chapter/page pair (0-indexed)
    let currentPageIndex = 0;                    // Index of the current page in the reader (currently unused, replaced by currentChapterIndex)
    let totalPages = 0;                          // Total number of pages in the book (currently unused, chapter/page count is in currentChapters.length)
    let isPlaying = false;                       // Tracks if the audio is actively playing
    let bookToDelete = null;                     // Stores reference (ID) to the book selected for deletion
    let currentAudio = null;                     // Reference to the HTMLAudioElement controlling playback
    let timestamps = null;                       // Full list of word/sentence-level timestamps for highlighting
    let wordsArray = [];                         // List of words used for highlighting during playback
    let currentWordIndex = 0;                    // Tracks which word is currently highlighted
    let updateInterval = null;                   // Interval ID for periodic UI updates while audio plays (currently unused/replaced by timeupdate event)
    let headerAutoHideTimeout = null;            // Timer used to hide the reader header after inactivity

    // Converts a "MM:SS" formatted string into total seconds.
    // Used when interpreting timing data provided by the backend.
    function timeToSeconds(timeStr) {
        if (!timeStr) return 0;
        const parts = timeStr.split(':');
        // Expects two parts (minutes and seconds)
        if (parts.length !== 2) return 0;
        return parseInt(parts[0]) * 60 + parseInt(parts[1]);
    }

    // Converts a number of seconds into "MM:SS" format.
    // Used when displaying elapsed and estimated progress times to the user.
    function secondsToTime(seconds) {
        // Handle null or negative input gracefully
        if (!seconds || seconds < 0) return "00:00";
        // Calculate minutes and seconds, rounding down
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        // Format with leading zero padding (e.g., 5 becomes "05")
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    // Updates the generation progress bar and timing display during audiobook creation.
    // This function receives:
    //   - message: descriptive status text from the backend
    //   - elapsedSeconds: how long the generation has been running
    //   - estimatedTotalSeconds: predicted total processing time
    //   - currentChunk / totalChunks: progress information if generation occurs in chunks
    //
    // The function adjusts the progress bar width and updates the text inside the modal.
    function updateGenerationProgressWithTiming(message, elapsedSeconds, estimatedTotalSeconds, currentChunk, totalChunks) {
        let progressPercent = 0;
        
        // If chunk information is available, prefer that because it is more accurate.
        if (currentChunk && totalChunks && totalChunks > 0) {
            // Cap progress at 95% to allow for final processing/download stage
            progressPercent = Math.min(95, Math.floor((currentChunk / totalChunks) * 100));

        // If only timing estimate exists, calculate based on elapsed vs estimated.
        } else if (estimatedTotalSeconds && estimatedTotalSeconds > 0) {
            // Cap progress at 95%
            progressPercent = Math.min(95, Math.floor((elapsedSeconds / estimatedTotalSeconds) * 100));

        // When no information is available, show a baseline progress indicator.
        } else {
            progressPercent = 15;
        }
        
        // Update the progress bar width visually
        generationProgress.style.width = `${progressPercent}%`;
        
        // Format the display times using the helper function
        const elapsedStr = secondsToTime(elapsedSeconds);
        const estimatedStr = secondsToTime(estimatedTotalSeconds);
        
        // Status block with timing information
        if (estimatedTotalSeconds && estimatedTotalSeconds > 0) {
            generationStatus.innerHTML = `
                <div style="margin-bottom: 8px; font-size: 14px; font-weight: 500;">${message}</div>
                <div style="font-size: 12px; color: #aaa; display: flex; gap: 20px; justify-content: center;">
                    <span>Elapsed: <span style="color: #007bff; font-weight: 600;">${elapsedStr}</span></span>
                    <span>Estimated: <span style="color: #28a745; font-weight: 600;">${estimatedStr}</span></span>
                </div>
            `;
        } else {
            // If no estimation yet, show a generic “processing” state
            generationStatus.innerHTML = `
                <div style="margin-bottom: 8px; font-size: 14px; font-weight: 500;">${message}</div>
                <div style="font-size: 12px; color: #aaa;">Processing...</div>
            `;
        }
    }

    // === HEADER VISIBILITY LOGIC ===
    // The reader header automatically hides itself after a few seconds to provide distraction-free reading.
    // It reappears when hovering over a special trigger zone or the header itself.

    // Makes the header visible and clears any auto-hide timers.
    function showHeader() {
        readerHeader.classList.add('visible');
        
        // Clear the existing timeout to prevent immediate hiding
        if (headerAutoHideTimeout) {
            clearTimeout(headerAutoHideTimeout);
            headerAutoHideTimeout = null;
        }
    }

    // Hides the header by removing the class controlling opacity/visibility.
    function hideHeader() {
        readerHeader.classList.remove('visible');
    }

    // Called when the reader opens. Immediately shows the header, then schedules an auto-hide.
    function initialiseHeaderVisibility() {
        showHeader();
        
        // Set a timer to hide the header after 10 seconds of inactivity
        headerAutoHideTimeout = setTimeout(() => {
            hideHeader();
        }, 10000); // 10 seconds of inactivity
    }

    // If the user hovers over the trigger zone, reveal the header.
    if (readerHeaderTrigger) {
        readerHeaderTrigger.addEventListener('mouseenter', showHeader);

        // Hide again when leaving the trigger, after a short delay to avoid flicker.
        readerHeaderTrigger.addEventListener('mouseleave', () => {
            headerAutoHideTimeout = setTimeout(() => {
                // Only hide if the cursor is not now over the header itself
                if (!readerHeader.matches(':hover')) {
                    hideHeader();
                }
            }, 500); // 0.5 second delay
        });
    }

    // Keep header visible when the cursor is over it; hide shortly after leaving.
    readerHeader.addEventListener('mouseenter', showHeader);
    readerHeader.addEventListener('mouseleave', () => {
        // Schedule a hide after 1 second of leaving the header
        headerAutoHideTimeout = setTimeout(hideHeader, 1000);
    });

    // === PDF CHAPTER DETECTION ===
    // Attempts to detect chapters from a PDF by scanning the first 200 pages for patterns
    // typically associated with chapter titles. This helps populate the chapter dropdown
    // and align audio timing with chapter navigation.
    function detectPdfChapters(pdf) {
        return new Promise(async (resolve) => {
            // Common regex patterns for chapter/section titles
            const chapterPatterns = [
                /^chapter\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|[ivxlcdm]+)/i, // e.g. "Chapter 1" or "Chapter Five"
                /^part\s+(\d+|one|two|three|four|five|[ivxlcdm]+)/i,                          // e.g. "Part I"
                /^section\s+(\d+|one|two|[ivxlcdm]+)/i,                                      // e.g. "Section 2"
                /^book\s+(\d+|one|two|three|[ivxlcdm]+)/i,                                   // e.g. "Book One"
                /^(\d+)\.\s+[A-Z]/,  // e.g., "1. Introduction"
                /^[ivxlcdm]+\.\s+[A-Z]/i // e.g., "I. Introduction" (Roman numerals)
            ];
            
            const detectedChapters = [];
            
            try {
                // Loop through early pages only (max 200) - chapter markers usually appear there
                for (let pageNum = 1; pageNum <= Math.min(pdf.numPages, 200); pageNum++) {
                    const page = await pdf.getPage(pageNum);
                    const textContent = await page.getTextContent();
                    
                    // Extract plain text from page items
                    let pageText = '';
                    for (const item of textContent.items) {
                        pageText += item.str + ' ';
                    }
                    
                    // Analyse only the first few lines (max 5) for performance and accuracy
                    const lines = pageText.split('\n').slice(0, 5);

                    for (const line of lines) {
                        const trimmed = line.trim();

                        // Avoid extremely long lines (likely running text) and ignore empty lines
                        if (trimmed.length > 0 && trimmed.length < 100) {
                            for (const pattern of chapterPatterns) {
                                if (pattern.test(trimmed)) {
                                    detectedChapters.push({
                                        pageNum: pageNum,               // The page number the chapter begins on
                                        title: trimmed.substring(0, 50) // Shortened title for UI display
                                    });
                                    break; // Stop checking patterns once one is matched for this line
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

    // Calculates approximate audio start and end times for each page spread in the reader
    // based on the total word count and total audio duration (for EPUBs).
    function calculatePageTimestamps() {
        // If there's no audio or no content loaded, we can't calculate anything
        if (!currentAudio || !currentChapters.length) return;
        
        const duration = currentAudio.duration; // Full audio length in seconds

        // The reader shows two pages at once (left + right)
        // currentChapters is an array of page content (HTML), so we group them in pairs.
        const numPagePairs = Math.ceil(currentChapters.length / 2);
        
        pageTimestamps = []; // Reset global pageTimestamps array
        
        // Count words on every page so we can estimate how much time
        // each page pair should take based on its text density.
        const wordsPerPage = currentChapters.map((chapter, index) => {
            // Convert HTML to plain text for accurate word counting
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = chapter;
            const text = tempDiv.textContent || '';
            
            // Split by whitespace and count non-empty tokens
            const wordCount = text.split(/\s+/).filter(w => w.length > 0).length;
            return { pageIndex: index, wordCount: wordCount };
        });
        
        // Total words across all pages, used for proportional time distribution
        const totalWords = wordsPerPage.reduce((sum, page) => sum + page.wordCount, 0);
        
        let accumulatedTime = 0; // Running marker for start times
        
        for (let i = 0; i < numPagePairs; i++) {
            // Calculate the left and right page indices for this pair
            const leftPageIndex = i * 2;
            const rightPageIndex = i * 2 + 1;
            
            // Fetch their word counts (0 if page missing - e.g., odd number of pages)
            const leftWords = wordsPerPage[leftPageIndex]?.wordCount || 0;
            const rightWords = wordsPerPage[rightPageIndex]?.wordCount || 0;
            const pairWords = leftWords + rightWords;
            
            // Allocate an amount of the total audio duration based on this pair’s word share
            // Formula: (Pair's Words / Total Words) * Total Duration
            const pairDuration = (pairWords / totalWords) * duration;
            
            // Push timestamp entry used during playback to approximate page transition times
            pageTimestamps.push({
                pageIndex: leftPageIndex,       // Index of the pair’s first page
                startTime: accumulatedTime,     // When this pair should begin displaying
                endTime: accumulatedTime + pairDuration, // When it should end
                wordCount: pairWords            // For debugging/tuning
            });
            
            accumulatedTime += pairDuration; // Move to next time segment
        }
        
        console.log('Calculated page timestamps:', pageTimestamps);
    }


    // === DATABASE & BOOKSHELF FUNCTIONS ===
    // Fetches all books from the database and renders the bookshelf grid.
    async function loadBooksFromDatabase() {
        try {
            console.log('Loading books from database...');
            // Call Electron API to get book data
            const books = await window.electronAPI.getBooksFromDb();
            const bookshelfGrid = document.getElementById('bookshelf-grid');
            bookshelfGrid.innerHTML = ''; // Clear old contents
            
            if (books.length === 0) {
                // No books — show centred empty-state UI
                const container = document.querySelector(".bookshelf-container");
                container.classList.add("empty"); // Toggles CSS layout for empty state
                showEmptyState();
            } else {
                // Books exist — remove empty-state layout
                const container = document.querySelector(".bookshelf-container");
                container.classList.remove("empty");
                
                // Add each book tile to the bookshelf grid
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

    // Renders the 'no books in bookshelf' message and the large Add Book button.
    function showEmptyState() {
        // Injects the "no books" screen into the grid
        const bookshelfGrid = document.getElementById('bookshelf-grid');
        
        bookshelfGrid.innerHTML = `
            <div class="empty-state">
                <p class="empty-state-text">No books in bookshelf, click Add Book below to add your first</p>
                <button class="add-book-btn" id="add-book-empty">
                    <span>+</span> ADD BOOK
                </button>
            </div>
        `;
        
        // Button opens standard upload modal
        document.getElementById('add-book-empty').addEventListener('click', openUploadModal);
    }

    // Saves the metadata for a newly generated audiobook into the persistent database.
    async function saveBookToDatabase(book) {
        try {
            if (!book.filePath) {
                alert('Cannot save book: file path is missing');
                return;
            }
            
            // Ask Electron to save book metadata into the DB
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

    // Creates and adds a single book tile element to the bookshelf grid.
    function addBookToBookshelf(title, author, year, coverUrl, progress = 0, bookId = null, filePath = null) {
        const bookshelfGrid = document.getElementById('bookshelf-grid');
        
        // Create the container representing the book's tile
        const bookItem = document.createElement('div');
        bookItem.className = 'book-item';
        bookItem.dataset.bookId = bookId; // Store DB ID for reference
        
        // Author/year formatting (fallbacks included)
        const authorYearText = (author && year) ? `${author} - ${year}` : (author || 'Unknown Author');
        
        // Determine the visual style and text for the progress bar
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
            // UNREAD books show a full bar but yellow
            progressWidth = 100;
        }
        
        // Delete icon for the book tile (SVG markup)
        const deleteIconSVG = `
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                <line x1="10" y1="11" x2="10" y2="17"></line>
                <line x1="14" y1="11" x2="14" y2="17"></line>
            </svg>`;
        
        // Build inner HTML for the visual tile
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
        
        // Clicking a tile opens the reader (unless the click was on delete)
        bookItem.addEventListener('click', async (e) => {
            // Prevent tile click from firing when pressing delete button or its children
            if (e.target.classList.contains('delete-book-btn') || e.target.closest('.delete-book-btn')) {
                e.stopPropagation();
                return;
            }
            
            // Retrieve the latest data for this book from the database
            const books = await window.electronAPI.getBooksFromDb();
            const bookData = books.find(b => b.id === bookId);

            if (bookData) {
                openBookReader(bookData);
            } else {
                alert('Error: Could not find book data.');
            }
        });
        
        // Delete button triggers delete confirmation modal
        const deleteBtn = bookItem.querySelector('.delete-book-btn');
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent tile opening
            openDeleteModal(bookId, title);
        });
        
        // Add tile to the shelf grid
        bookshelfGrid.appendChild(bookItem);
    }

    // Opens the confirmation modal for deleting a book.
    function openDeleteModal(bookId, title) {
        // Populate modal with book info for confirmation
        bookToDelete = bookId;
        deleteBookTitle.textContent = title;
        deleteModalOverlay.classList.add('visible');
    }

    // Closes the delete confirmation modal.
    function closeDeleteModal() {
        deleteModalOverlay.classList.remove('visible');
        bookToDelete = null;
    }

    // Initiates the actual book deletion via the Electron API.
    async function deleteBook() {
        // Performs actual deletion via Electron API
        if (bookToDelete) {
            try {
                const result = await window.electronAPI.deleteBook(bookToDelete);
                if (result.success) {
                    closeDeleteModal();
                    await loadBooksFromDatabase(); // Refresh UI to remove the deleted book
                } else {
                    alert('Failed to delete book. Please try again.');
                }
            } catch (error) {
                console.error('Error deleting book:', error);
            }
        }
    }

    // Helper for styling progress bars consistently based on percentage.
    function getProgressClass(progress) {
        // Helper for styling progress bars consistently
        if (progress === 0) return 'yellow';
        if (progress < 100) return 'blue';
        return 'green';
    }

    // Helper for progress text labels (UNREAD, % complete, READ).
    function getProgressText(progress) {
        // Helper for progress text labels
        if (progress === 0) return 'UNREAD';
        if (progress < 100) return `${Math.round(progress)}%`;
        return 'READ';
    }


    // === MODAL FUNCTIONS ===

    // Shows the upload modal overlay.
    function openUploadModal() {
        uploadModalOverlay.classList.add('visible');
    }

    // Hides the upload modal overlay.
    function closeUploadModal() {
        uploadModalOverlay.classList.remove('visible');
    }

    // Shows the configuration modal, pre-populating fields and starting the file preview.
    function openConfigureModal(filePath, title, author, year) {
        // Store selected file path so we know what to process on save
        selectedFilePath = filePath;
        
        // Auto-fill form inputs
        const filename = filePath.split(/[\\/]/).pop();
        document.getElementById('config-file-name').value = filename;
        document.getElementById('config-book-title').value = title;
        document.getElementById('config-book-author').value = author;
        document.getElementById('config-book-year').value = year || '';
        
        // Load the visual preview (PDF/EPUB)
        loadBookPreview(filePath);
        
        configureModalOverlay.classList.add('visible');
    }

    // Hides the configuration modal and resets related state/UI elements.
    function closeConfigureModal() {
        configureModalOverlay.classList.remove('visible');
        
        // Destroy previous preview instance if needed (e.g. PDF.js document)
        if (currentBook) {
            // Note: currentBook is used by PDF.js for the document instance, and this destroy
            // method is typically available in libraries like pdf.js.
            // If the library does not expose a destroy method, this line would need amending.
            // Assuming it's a generic document/parser instance here.
            currentBook.destroy(); 
            currentBook = null;
        }

        // Reset preview UI
        configCoverPreview.innerHTML = '';
        configCoverPreview.style.backgroundImage = '';
        previewPane.innerHTML = '<h3>CHAPTER 1</h3><p>Loading...</p>';
        
        bookPages = [];
        currentPreviewPage = 0;
        totalPreviewPages = 0;
        previewPane.classList.remove('epub-preview', 'pdf-preview', 'fallback-preview');
        
        resetUploadUI(); // Also resets the state of the first modal
    }

    // Shows the generation progress modal.
    function openGeneratingModal() {
        generationCancelled = false;
        generationProgress.style.width = '0%'; // Progress bar reset
        generationStatus.textContent = 'Connecting to AI Model...';
        generatingModalOverlay.classList.add('visible');
    }

    // Hides the generation progress modal.
    function closeGeneratingModal() {
        generatingModalOverlay.classList.remove('visible');
    }

    // Shows the completion modal after successful generation.
    function openCompleteModal() {
        completeModalOverlay.classList.add('visible');
    }

    // Hides the completion modal.
    function closeCompleteModal() {
        completeModalOverlay.classList.remove('visible');
    }


    // === PREVIEW FUNCTIONS ===
    // Updates the page numbers and displays the correct page content in the configuration preview.
    function updatePreviewNavigation() {
        // Update page number indicators
        previewPageNumber.textContent = currentPreviewPage + 1;
        previewCurrentPage.textContent = currentPreviewPage + 1;
        previewTotalPages.textContent = totalPreviewPages;
        
        // Enable/disable navigation arrows based on current page limits
        previewPrevBtn.disabled = currentPreviewPage === 0;
        previewNextBtn.disabled = currentPreviewPage >= totalPreviewPages - 1;
        
        // Display the corresponding parsed HTML page from the preview array
        if (bookPages[currentPreviewPage]) {
            previewPane.innerHTML = bookPages[currentPreviewPage];
        }
    }

    // Export functions needed by other parts (e.g., event handlers or other modules)
    window.openConfigureModal = openConfigureModal;
    window.closeConfigureModal = closeConfigureModal;
    window.updatePreviewNavigation = updatePreviewNavigation;
    window.showFileInUI = showFileInUI;
    window.resetUploadUI = resetUploadUI;

    // Determines the file type and calls the appropriate preview loader (EPUB, PDF, or fallback).
    async function loadBookPreview(filePath) {
        // Reset UI elements for the new file preview
        configCoverPreview.innerHTML = '';
        configCoverPreview.style.backgroundImage = '';
        previewPane.innerHTML = '<p>Loading preview...</p>';
        previewPane.classList.remove('epub-preview', 'pdf-preview', 'fallback-preview');
        
        bookPages = [];
        currentPreviewPage = 0;
        generatedCoverUrl = null;
        
        const extension = filePath.split('.').pop().toLowerCase(); // Detect file type
        
        try {
            // Choose preview loader based on file extension
            if (extension === 'epub') {
                await loadEpubPreview(filePath);
            } else if (extension === 'pdf') {
                // PDF loading requires the `pdfjsLib` global object, which must be loaded separately
                await loadPdfPreview(filePath);
            } else {
                // Unsupported formats fall back to a generic message
                previewPane.classList.add('fallback-preview');
                previewPane.innerHTML = '<h3>Preview Not Available</h3><p>Unsupported file format for preview.</p>';
                // Set minimal page info for navigation to work without crashing
                totalPreviewPages = 1;
                updatePreviewNavigation();
            }
        } catch (error) {
            // If preview fails, show readable error instead of crashing
            console.error('Error loading preview:', error);
            previewPane.classList.add('fallback-preview');
            previewPane.innerHTML = `<h3>Error</h3><p>Could not read file: ${error.message}</p>`;
            
            // Display the error as the single preview page
            bookPages = [`<h3>Error</h3><p>Could not read file: ${error.message}</p>`];
            totalPreviewPages = 1;
            updatePreviewNavigation();
        }
    }

    // Load a preview of an EPUB file by parsing chapters and extracting content.
    async function loadEpubPreview(filePath) {
        // Add a class to indicate EPUB preview styling
        previewPane.classList.add('epub-preview');

        try {
            // Parse the EPUB using the Electron API (which likely uses a node module like `epub-parser`)
            const result = await window.electronAPI.epubParse(filePath);

            // Set the cover image if available
            if (result.coverUrl) {
                generatedCoverUrl = result.coverUrl;
                configCoverPreview.style.backgroundImage = `url(${generatedCoverUrl})`;
                configCoverPreview.style.backgroundSize = 'cover';
                configCoverPreview.style.backgroundPosition = 'center';
            }

            // Determine how many chapters to preview (max 5)
            const chaptersToPreview = Math.min(5, result.chapters.length);

            for (let i = 0; i < chaptersToPreview; i++) {
                // Create a temporary container to parse chapter HTML
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = result.chapters[i];

                // Use chapter title if available, otherwise fallback
                const chapterTitle = result.chapterTitles ? result.chapterTitles[i] : `Chapter ${i + 1}`;

                // Get all paragraph elements from chapter
                const paragraphs = Array.from(tempDiv.querySelectorAll('p'));

                // Build page HTML starting with the chapter title
                let pageHTML = `<h3>${chapterTitle}</h3>`;

                // Include up to 15 paragraphs for preview to avoid loading too much content
                paragraphs.slice(0, 15).forEach(p => {
                    // Only include non-empty paragraphs
                    if (p.textContent.trim()) pageHTML += `<p>${p.textContent}</p>`;
                });

                // Add the page HTML to the global preview pages array
                bookPages.push(pageHTML || '<p>No content available.</p>');
            }

            // Set total preview pages and update navigation controls
            totalPreviewPages = bookPages.length || 1;
            updatePreviewNavigation();

        } catch (error) {
            // Propagate any parsing errors to the main error handler
            throw error;
        }
    }

    // Load a preview of a PDF file by rendering pages to an image.
    async function loadPdfPreview(filePath) {
        try {
            // Add a class to indicate PDF preview styling
            previewPane.classList.add('pdf-preview');

            // Read the PDF file as an ArrayBuffer using the Electron API
            const arrayBuffer = await window.electronAPI.readFileBuffer(filePath);

            // Load PDF using pdf.js (requires pdfjsLib to be available in global scope)
            const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
            const pdf = await loadingTask.promise;
            currentBook = pdf; // Store PDF instance for potential cleanup later

            // Determine pages to preview (max 5)
            const pagesToLoad = Math.min(5, pdf.numPages);
            bookPages = [];

            // Render each page to a canvas and store as an image
            for (let pageNum = 1; pageNum <= pagesToLoad; pageNum++) {
                const page = await pdf.getPage(pageNum);
                const viewport = page.getViewport({ scale: 1.0 });

                // Scale page for high-quality rendering (3x)
                // Scaling based on a target width of 800px ensures consistent quality/size
                const scale = (800 / viewport.width) * 3;
                const scaledViewport = page.getViewport({ scale: scale });

                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.height = scaledViewport.height;
                canvas.width = scaledViewport.width;

                await page.render({ 
                    canvasContext: context, 
                    viewport: scaledViewport,
                    intent: 'print' // Use print intent for better rendering quality
                }).promise;

                // Convert canvas to a Data URL (image) and add to preview pages
                bookPages.push(`<img src="${canvas.toDataURL('image/png')}" alt="Page ${pageNum}" style="max-width: 100%; height: auto;">`);
            }

            // Render the first page as the cover at high quality
            const firstPage = await pdf.getPage(1);
            const coverViewport = firstPage.getViewport({ scale: 3.0 });
            const coverCanvas = document.createElement('canvas');
            const coverContext = coverCanvas.getContext('2d');
            coverCanvas.height = coverViewport.height;
            coverCanvas.width = coverViewport.width;

            await firstPage.render({ 
                canvasContext: coverContext, 
                viewport: coverViewport,
                intent: 'print'
            }).promise;

            // Set the generated cover URL (Data URL)
            generatedCoverUrl = coverCanvas.toDataURL('image/png');
            configCoverPreview.style.backgroundImage = `url(${generatedCoverUrl})`;
            configCoverPreview.style.backgroundSize = 'cover';
            configCoverPreview.style.backgroundPosition = 'center';

            // Update preview navigation
            totalPreviewPages = bookPages.length || 1;
            currentPreviewPage = 0;
            updatePreviewNavigation();

        } catch (error) {
            // Propagate any PDF loading/rendering errors
            throw error;
        }
    }

    // === BOOK READER FUNCTIONS ===
    // Opens the main reader view for a selected book.
    async function openBookReader(book) {
        // Validate book object
        if (!book || !book.filePath) {
            alert('Cannot open book: invalid book data');
            return;
        }

        // Set the current book and switch to reader view
        currentReaderBook = book;
        mainContentArea.classList.add('hidden'); // Hide bookshelf
        readerView.classList.remove('hidden');   // Show reader

        // Display title
        readerTitle.textContent = book.title || 'Untitled Book';

        // Format metadata: "Author • Year" or just author if year unknown
        const author = book.author || 'Unknown Author';
        const year = book.year && book.year !== 'N/A' ? book.year : '';
        readerMetadata.textContent = year ? `${author} • ${year}` : author;

        // Show loading placeholders for left and right pages
        readerPageLeftContent.innerHTML = '<p>Loading...</p>';
        readerPageRightContent.innerHTML = '<p>Loading...</p>';

        // Initialise header visibility logic (auto-hide after 10s)
        initialiseHeaderVisibility();

        try {
            // Determine file extension
            const extension = book.filePath.split('.').pop().toLowerCase();

            if (extension === 'epub') {
                // EPUB-specific setup
                pageSelectGroup.style.display = 'none'; // EPUB uses chapters, not page numbers
                const result = await window.electronAPI.epubParse(book.filePath);
                currentChapters = result.chapters || []; // Chapter content (HTML strings)
                currentChapterTitles = result.chapterTitles || [];
            } else if (extension === 'pdf') {
                // PDF-specific setup
                await loadPdfForReading(book.filePath);
                // PDF function handles rendering/navigation setup, so we return here
                return; 
            }

            // Fallback content if chapters or titles are missing
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

            // Load saved page or start at the first chapter
            currentChapterIndex = book.currentPage || 0;
            loadChapter(currentChapterIndex);

            // Load audio if an audio file path exists for this book
            if (book.audioPath) {
                loadAudioWithTimestamps(book);
            } else {
                audioControlsBar.style.display = 'none'; // Hide controls if no audio
            }

        } catch (error) {
            console.error('Error opening book reader:', error);
            readerPageLeftContent.innerHTML = `<p>Error loading book: ${error.message}</p>`;
        }
    }

    // Export key functions for main renderer.js (mostly for debug or external module calls)
    window.detectPdfChapters = detectPdfChapters;
    window.loadPdfForReading = loadPdfForReading;
    window.showHeader = showHeader;
    window.hideHeader = hideHeader;
    window.initialiseHeaderVisibility = initialiseHeaderVisibility;
    window.openBookReader = openBookReader;

    // Loads the audio file (as a Data URL) and parses any word/sentence timestamps.
    async function loadAudioWithTimestamps(book) {
        audioControlsBar.style.display = 'flex'; // Show controls
        try {
            // Read audio file as data URL via Electron API to bypass file access issues
            const audioDataUrl = await window.electronAPI.readAudioFile(book.audioPath);
            currentAudio = new Audio(audioDataUrl);

            // Apply speed and volume settings from UI controls
            currentAudio.playbackRate = parseFloat(speedSlider.value);
            currentAudio.volume = parseFloat(volumeSlider.value) / 100;

            // Parse timestamps if provided in the book data
            if (book.timestamps) {
                try {
                    timestamps = JSON.parse(book.timestamps);
                    console.log('Loaded timestamps:', timestamps);
                } catch (e) {
                    console.error('Failed to parse timestamps:', e);
                }
            }

            // Set saved position and calculate page durations once metadata is loaded
            currentAudio.addEventListener('loadedmetadata', () => {
                if (book.progress > 0 && currentAudio.duration) {
                    // Set audio position based on saved progress percentage
                    currentAudio.currentTime = (book.progress / 100) * currentAudio.duration;
                }
                // Calculate page transition times based on word count/duration (for EPUB)
                calculatePageTimestamps();
            });

            // Update progress bar as audio plays
            currentAudio.addEventListener('timeupdate', updateAudioProgress);

            // Handle audio end
            currentAudio.addEventListener('ended', () => {
                isPlaying = false;
                playPauseBtn.innerHTML = '▶';
            });

        } catch (audioError) {
            console.error('Error loading audio file:', audioError);
            audioControlsBar.style.display = 'none';
        }
    }

    // Loads and renders all pages of a PDF for reading, preparing the content array for navigation.
    async function loadPdfForReading(filePath) {
        try {
            // Read the PDF file as an ArrayBuffer using Electron API
            const arrayBuffer = await window.electronAPI.readFileBuffer(filePath);

            // Load the PDF document using pdf.js
            const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
            const pdf = await loadingTask.promise;

            // Reset content arrays
            currentChapters = [];
            currentChapterTitles = [];

            // Get reader container dimensions for scaling pages
            const readerContainer = document.querySelector('.reader-content');
            const containerWidth = readerContainer.offsetWidth / 2; // Half width for two-page spread
            const containerHeight = readerContainer.offsetHeight;

            // Loop through each page in the PDF
            for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                const page = await pdf.getPage(pageNum);

                // Get original viewport (1:1 scale)
                let viewport = page.getViewport({ scale: 1.0 });

                // Calculate scale to fit container while keeping aspect ratio
                const scaleX = containerWidth / viewport.width;
                const scaleY = containerHeight / viewport.height;
                const scale = Math.min(scaleX, scaleY);

                // Increase scale for high-quality rendering (3x)
                const highQualityScale = scale * 3;
                viewport = page.getViewport({ scale: highQualityScale });

                // Create a canvas for rendering this page
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;

                // Render the page to canvas with print-quality intent
                await page.render({ 
                    canvasContext: context, 
                    viewport: viewport,
                    intent: 'print'
                }).promise;

                // Add page as an image to the chapters array (each PDF page is a "chapter" here)
                currentChapters.push(`<img src="${canvas.toDataURL('image/png')}" alt="Page ${pageNum}" style="width: 100%; height: 100%; object-fit: contain;">`);

                // Create a chapter title for navigation every 10 pages for easier browsing
                if (pageNum % 10 === 1 || pageNum === 1) {
                    const endPage = Math.min(pageNum + 9, pdf.numPages);
                    currentChapterTitles.push(`Pages ${pageNum}-${endPage}`);
                }
            }

            // Populate chapter dropdown (navigation menu)
            chapterSelect.innerHTML = '';
            // Loop in steps of 10 pages to create the dropdown options
            for (let i = 0; i < currentChapters.length; i += 10) {
                const option = document.createElement('option');
                option.value = i;
                const endPage = Math.min(i + 10, currentChapters.length);
                option.textContent = `Pages ${i + 1}-${endPage}`;
                chapterSelect.appendChild(option);
            }

            // Load saved page or start from beginning
            currentChapterIndex = currentReaderBook.currentPage || 0;
            loadChapter(currentChapterIndex);

            // Load audio with timestamps if available
            if (currentReaderBook.audioPath) {
                loadAudioWithTimestamps(currentReaderBook);
            } else {
                audioControlsBar.style.display = 'none';
            }
        } catch (error) {
            // Propagate any errors
            throw error;
        }
    }

    // Loads a specific chapter/page spread into the reader's left and right content panels.
    function loadChapter(index) {
        // Clamp index to valid range
        if (index < 0) index = 0;
        if (index >= currentChapters.length && index > 0) index = currentChapters.length - 1;
        currentChapterIndex = index;

        const chapterHtml = currentChapters[index];
        const isPdfPage = chapterHtml.includes('<img'); // Check if content is a PDF page image

        if (isPdfPage) {
            // --- PDF DISPLAY LOGIC (Two pages side-by-side) ---
            const leftPageHtml = currentChapters[index];
            // Check for the existence of the next page
            const rightPageHtml = (index + 1 < currentChapters.length) ? currentChapters[index + 1] : '';

            // Render content
            readerPageLeftContent.innerHTML = leftPageHtml;
            readerPageRightContent.innerHTML = rightPageHtml;

            // Apply PDF-specific styles
            readerPageLeftContent.classList.remove('epub-content');
            readerPageRightContent.classList.remove('epub-content');
            readerPageLeftContent.classList.add('pdf-content');
            readerPageRightContent.classList.add('pdf-content');

            // Update page indicator text
            if (rightPageHtml) pageIndicator.textContent = `Pages ${index + 1}-${index + 2} of ${currentChapters.length}`;
            else pageIndicator.textContent = `Page ${index + 1} of ${currentChapters.length}`;

            // Enable/disable navigation buttons (PDFs navigate two pages at a time)
            prevPageBtn.disabled = index === 0;
            // Next button is disabled if the next page is the last one, or if it doesn't exist.
            nextPageBtn.disabled = index + 1 >= currentChapters.length - 1; 
        } else {
            // --- EPUB DISPLAY LOGIC (Split content across two pages) ---
            // Create a temporary element to parse the HTML content
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = chapterHtml;
            const allElements = Array.from(tempDiv.children); // Get all top-level elements (p, h1, etc.)
            const midPoint = Math.ceil(allElements.length / 2); // Split content by element count

            // Populate left page with the first half of elements
            readerPageLeftContent.innerHTML = '';
            allElements.slice(0, midPoint).forEach(el => readerPageLeftContent.appendChild(el.cloneNode(true)));

            // Populate right page with the second half of elements
            readerPageRightContent.innerHTML = '';
            allElements.slice(midPoint).forEach(el => readerPageRightContent.appendChild(el.cloneNode(true)));

            // Update classes for EPUB-specific styles
            readerPageLeftContent.classList.remove('pdf-content');
            readerPageRightContent.classList.remove('pdf-content');
            readerPageLeftContent.classList.add('epub-content');
            readerPageRightContent.classList.add('epub-content');

            // Update page indicator with chapter title
            const chapterTitle = currentChapterTitles[index] || `Chapter ${index + 1}`;
            pageIndicator.textContent = `${chapterTitle} (${index + 1} of ${currentChapters.length})`;
            // EPUBs navigate one chapter/page at a time
            prevPageBtn.disabled = index === 0;
            nextPageBtn.disabled = index >= currentChapters.length - 1;
        }

        // Set chapter dropdown value to match the current index
        chapterSelect.value = index;

        // Scroll pages to top
        readerPageLeftContent.scrollTop = 0;
        readerPageRightContent.scrollTop = 0;

        // Build words array for EPUB highlighting
        if (!isPdfPage) {
            buildWordsArray();
        }
    }

    // Extracts all words from the visible two-page spread for word-level highlighting (EPUB only).
    function buildWordsArray() {
        wordsArray = [];
        // Get text content from the rendered pages
        const leftText = readerPageLeftContent.innerText || '';
        const rightText = readerPageRightContent.innerText || '';
        const fullText = leftText + ' ' + rightText;

        // Split by any whitespace and filter out empty strings
        const words = fullText.split(/\s+/).filter(w => w.length > 0);
        wordsArray = words;
        currentWordIndex = 0;
    }

    // Updates the current book's playback progress percentage based on audio time.
    function updateAudioProgress() {
        if (!currentAudio || !currentAudio.duration) return;

        const currentTime = currentAudio.currentTime;
        const progress = (currentTime / currentAudio.duration) * 100;
        currentReaderBook.progress = progress;
    }

    // Persists the current book's playback progress and chapter index to the database.
    async function updateBookProgress() {
        if (!currentReaderBook) return;
        try {
            await window.electronAPI.updateBookProgress(
                currentReaderBook.id, 
                currentReaderBook.progress,
                currentChapterIndex
            );

            // Update progress bar in the book list UI dynamically
            const bookItem = document.querySelector(`[data-book-id="${currentReaderBook.id}"]`);
            if (bookItem) {
                const progressBar = bookItem.querySelector('.progress-bar');
                const progressText = bookItem.querySelector('.progress-text');
                // Use helpers to get correct class and text
                progressBar.className = `progress-bar ${getProgressClass(currentReaderBook.progress)}`;
                progressBar.style.width = `${currentReaderBook.progress}%`;
                progressText.textContent = getProgressText(currentReaderBook.progress);
            }
        } catch (error) {
            console.error('Error updating book progress:', error);
        }
    }

    // === UI HELPER FUNCTIONS ===

    // Displays the selected file's name in the upload list and initiates file copying.
    async function showFileInUI(filePath) {
        try {
            const filename = filePath.split(/[\\/]/).pop();
            selectedFilePath = filePath;
            
            // Show loading state
            dropZone.style.display = 'none';
            uploadList.style.display = 'block';
            uploadList.innerHTML = '<div class="upload-item"><span>Copying file...</span></div>';
            
            // Copy file to app data directory via Electron API
            copiedFilePath = await window.electronAPI.copyBook(filePath);
            console.log('File copied to:', copiedFilePath);
            
            // Update UI with success and show remove button
            uploadList.innerHTML = `<div class="upload-item"><span>${filename}</span><span class="remove-file-icon" id="remove-file-btn">×</span></div>`;
            // Show metadata form fields
            uploadFormFields.forEach(field => field.style.display = 'flex');

            // Pre-fill title by removing the file extension
            const title = filename.replace(/\.[^/.]+$/, "");
            document.getElementById('book-title').value = title;

            // Attach event listener to the remove button
            const removeBtn = document.getElementById('remove-file-btn');
            if (removeBtn) removeBtn.addEventListener('click', resetUploadUI);
        } catch (error) {
            console.error('Error copying file:', error);
            alert('Error copying file: ' + error.message);
            resetUploadUI(); // Revert to initial state on failure
        }
    }

    // Resets the upload UI back to the initial state (showing the drop zone).
    function resetUploadUI() {
        selectedFilePath = null;
        copiedFilePath = null;
        dropZone.style.display = 'block';
        uploadList.style.display = 'none';
        uploadFormFields.forEach(field => field.style.display = 'none');
        uploadList.innerHTML = '';
        // Clear input values
        document.getElementById('book-title').value = '';
        document.getElementById('book-author').value = '';
        document.getElementById('book-year').value = '';
    }

    // Updates the generation progress bar and simple message (less detailed than the one with timing).
    function updateGenerationProgress(message, percent) {
        generationProgress.style.width = `${percent}%`;
        generationStatus.textContent = message;
    }

    // Updates the generation progress with elapsed time and estimated total time.
    function updateGenerationProgressWithTime(message, percent, elapsedSeconds, estimatedTotalSeconds) {
        // Adjust displayed percentage based on time (prefers time if available)
        let displayPercent = percent;
        if (estimatedTotalSeconds && elapsedSeconds) {
            displayPercent = Math.min(95, Math.floor((elapsedSeconds / estimatedTotalSeconds) * 100));
        }
        generationProgress.style.width = `${displayPercent}%`;

        // Local helper to format time in MM:SS
        const formatTime = (seconds) => {
            if (!seconds) return '--:--';
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        };

        const elapsedStr = formatTime(elapsedSeconds);
        const estimatedStr = formatTime(estimatedTotalSeconds);

        // Update status element with message and time info
        generationStatus.innerHTML = `
            <div style="margin-bottom: 8px;">${message}</div>
            <div style="font-size: 12px; color: #aaa;">
                Elapsed: ${elapsedStr} ${estimatedTotalSeconds ? `/ Estimated: ${estimatedStr}` : ''}
            </div>
        `;
    }

    // === EVENT LISTENERS ===
    
    // --- Upload Modal Events ---
    
    // Open the "Add Book" modal when the Add Book button is clicked
    addBookBtn.addEventListener('click', openUploadModal);

    // Cancel upload modal and reset UI when cancel button is clicked
    cancelUploadBtn.addEventListener('click', () => { 
        closeUploadModal(); 
        resetUploadUI(); 
    });

    // Prevent default behaviour for drag events on the whole modal overlay
   uploadModalOverlay.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
    });

    uploadModalOverlay.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
    });

    uploadModalOverlay.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
    });

    // Drop zone specific handlers (visual feedback on drag/drop)
    dropZone.addEventListener('dragenter', (e) => { 
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragover', (e) => { 
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => { 
        e.preventDefault();
        e.stopPropagation();
        // Only remove class if we're actually leaving the drop zone element itself
        if (e.target === dropZone) {
            dropZone.classList.remove('drag-over');
        }
    });

    // Handle the actual file drop event
    dropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
        
        console.log('Drop event triggered');
        console.log('Files:', e.dataTransfer.files);
        
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            const file = files[0];
            console.log('File object:', file);
            
            // Get the file path using the Electron preload script API
            // Note: `getFilePath` likely extracts the file path from the file object, which is needed for Electron.
            const filePath = window.electronAPI.getFilePath(file);
            console.log('File path:', filePath);
            
            if (filePath) {
                // Check if file extension is supported
                const ext = filePath.split('.').pop().toLowerCase();
                const supportedFormats = ['epub', 'pdf', 'txt', 'docx', 'odt'];
                
                console.log('File extension:', ext);
                
                if (supportedFormats.includes(ext)) {
                    await showFileInUI(filePath);
                } else {
                    alert('Unsupported file format. Please upload EPUB, PDF, TXT, DOCX, or ODT files.');
                }
            } else {
                console.error('No file path found on dropped file');
                alert('Error: Could not read file path. Please try using the Browse button instead.');
            }
        } else {
            console.error('No files in drop event');
        }
    });

     // Browse link handler - opens native file dialogue
    browseLink.addEventListener('click', async (e) => {
        e.preventDefault();
        // `openFile` returns the selected path
        const filePath = await window.electronAPI.openFile();
        if (filePath) await showFileInUI(filePath);
    });

    // Confirm upload: validate inputs and open configuration modal
    confirmUploadBtn.addEventListener('click', () => {
        const title = document.getElementById('book-title').value;
        if (!copiedFilePath) return alert('Please select a book file');
        if (!title) return alert('Please provide a title');
        
        // Use the copied file path instead of the original
        openConfigureModal(
            copiedFilePath, 
            title, 
            document.getElementById('book-author').value, 
            document.getElementById('book-year').value
        );
        closeUploadModal();
    });
    
    // --- Configuration Modal Events ---

    // Cancel configuration modal
    cancelConfigureBtn.addEventListener('click', closeConfigureModal);

    // Go back from configuration modal to upload modal
    backConfigureBtn.addEventListener('click', () => { 
        closeConfigureModal(); 
        openUploadModal(); 
    });

    // Preview book navigation (previous page)
    previewPrevBtn.addEventListener('click', () => { 
        if (currentPreviewPage > 0) { 
            currentPreviewPage--; 
            updatePreviewNavigation(); 
        } 
    });
    // Preview book navigation (next page)
    previewNextBtn.addEventListener('click', () => { 
        if (currentPreviewPage < totalPreviewPages - 1) { 
            currentPreviewPage++; 
            updatePreviewNavigation(); 
        } 
    });

    // === Generate Audiobook Button ===
    generateAudiobookBtn.addEventListener('click', async () => {
        // Gather generation settings from configuration inputs
        const generationData = {
            // Note: spaceUrl is hardcoded here, likely pointing to a Hugging Face Space or similar AI endpoint
            spaceUrl: "https://audioapi-g2ru.onrender.com", 
            filePath: selectedFilePath, // This is the copied path in AppData
            title: document.getElementById('config-book-title').value,
            author: document.getElementById('config-book-author').value,
            year: document.getElementById('config-book-year').value,
            persona: document.getElementById('config-persona').value,
            voice: document.getElementById('config-voice').value,
            sfx: document.getElementById('config-sfx-intensity').value,
            speed: document.getElementById('config-narrator-speed').value,
        };

        // Initialise current book data object for eventual saving to DB
        currentBookData = {
            title: generationData.title,
            author: generationData.author,
            year: generationData.year,
            coverUrl: generatedCoverUrl || null, // Use the generated preview cover
            filePath: selectedFilePath,
            progress: 0,
            audioPath: null, // To be filled by the generation result
            timestamps: null, // To be filled by the generation result
            currentPage: 0
        };

        closeConfigureModal();
        openGeneratingModal();

        // Variables for tracking progress and elapsed time
        let generationStartTime = Date.now();
        let currentElapsed = 0;
        let currentEstimated = 0;
        let currentChunk = 0;
        let totalChunks = 0;
        let elapsedTimerInterval = null;

        // Start live elapsed timer updating UI every second
        elapsedTimerInterval = setInterval(() => {
            currentElapsed = Math.floor((Date.now() - generationStartTime) / 1000);
            updateGenerationProgressWithTiming(
                // Safely get the current status message from the status element
                generationStatus.querySelector('div')?.textContent || 'Processing...', 
                currentElapsed, 
                currentEstimated, 
                currentChunk, 
                totalChunks
            );
        }, 1000); // Update every 1 second

        // Listen for streaming status updates from Electron API (e.g. "Processing chunk 5 of 20")
        window.electronAPI.onStatusUpdate((statusData) => {
            console.log('Status update received:', statusData);

            const message = statusData.message || 'Processing...';
            const isComplete = statusData.isComplete || false;

            // Try to extract progress/chunk info from the message string
            const progressInfo = parseProgressFromMessage(message);
            if (progressInfo) {
                currentChunk = progressInfo.current;
                totalChunks = progressInfo.total;

                if (progressInfo.estimated) {
                    // Convert estimated time string to seconds
                    currentEstimated = timeToSeconds(progressInfo.estimated);
                }
            }

            // Update UI with progress/timing info
            updateGenerationProgressWithTiming(message, currentElapsed, currentEstimated, currentChunk, totalChunks);

            // Note: Final completion is handled by the `await window.electronAPI.generateAudiobook(generationData)` call resolving,
            // but this internal flag is kept for potential future use.
            if (isComplete) {
                console.log('Generation marked as complete by API');
            }
        });

        try {
            // Initial status update before the API call starts
            updateGenerationProgressWithTiming('Connecting to AI Model...', 0, 0, 0, 0);

            // Call API to generate audiobook (this blocks until generation is finished or failed)
            const result = await window.electronAPI.generateAudiobook(generationData);

            // Stop elapsed timer
            if (elapsedTimerInterval) {
                clearInterval(elapsedTimerInterval);
                elapsedTimerInterval = null;
            }

            // Remove status listener (to prevent further updates)
            window.electronAPI.removeStatusListener();

            console.log('Generation result received:', result);

            // Final status update for completion
            currentElapsed = Math.floor((Date.now() - generationStartTime) / 1000);
            updateGenerationProgressWithTiming('Complete!', currentElapsed, currentEstimated, totalChunks, totalChunks);
            generationProgress.style.width = '100%';

            // Store results in the book data object
            currentBookData.audioPath = result.audioPath;
            currentBookData.timestamps = result.timestamps;

            // Brief delay before opening completion modal
            setTimeout(() => {
                closeGeneratingModal();
                openCompleteModal();
            }, 1000);

        } catch (error) {
            console.error('Generation failed:', error);

            // Clean up timers and listeners on failure
            if (elapsedTimerInterval) {
                clearInterval(elapsedTimerInterval);
                elapsedTimerInterval = null;
            }

            window.electronAPI.removeStatusListener();
            closeGeneratingModal();
            alert('Error generating audiobook: ' + error.message);
        }
    });

    // Utility function to extract chunk progress and estimated time from a status message string.
    function parseProgressFromMessage(message) {
        // Matches patterns like "(1 of 10, 00:30 / 25:00)"
        // Captures: 1: current chunk, 2: total chunks, 3: elapsed time (MM:SS, optional), 4: estimated time (MM:SS, optional)
        const progressPattern = /\((\d+)\s+of\s+(\d+)(?:,\s*(\d+:\d+)\s*\/\s*(\d+:\d+))?\)/;
        const match = message.match(progressPattern);

        if (match) {
            const current = parseInt(match[1]);
            const total = parseInt(match[2]);
            const elapsed = match[3] || null;
            const estimated = match[4] || null;

            return { current, total, elapsed, estimated };
        }
        return null;
    }

    // Cancel audiobook generation
    cancelGenerationBtn.addEventListener('click', () => {
        generationCancelled = true;
        closeGeneratingModal();
        // Note: Actual cancellation logic in the backend (e.g. stopping the AI request) is assumed
        // to be handled by the Electron process if a flag/API call is made. This only updates the UI state.
    });

    // --- Completion Modal Events ---
    
    // Add generated book to bookshelf (saves to DB and refreshes list)
    addToBookshelfBtn.addEventListener('click', async () => {
        if (currentBookData) {
            await saveBookToDatabase(currentBookData);
            await loadBooksFromDatabase(); // Refresh bookshelf
        }
        closeCompleteModal();
    });

    // Listen to audiobook immediately (saves to DB, refreshes list, then opens reader)
    listenNowBtn.addEventListener('click', async () => {
        if (currentBookData) {
            // Save the book first
            await saveBookToDatabase(currentBookData);
            await loadBooksFromDatabase();
            
            // Fetch the newly added book (with its assigned ID)
            const books = await window.electronAPI.getBooksFromDb();
            const newBook = books.find(b => b.filePath === currentBookData.filePath);
            
            if (newBook) {
                closeCompleteModal();
                // Short delay to allow modal to close smoothly before transition
                setTimeout(() => openBookReader(newBook), 100);
            }
        }
    });

    // --- Delete/Cancel Modals ---
    
    // Close completion modal if clicking outside the content area
    completeModalOverlay.addEventListener('click', (e) => { 
        if (e.target === completeModalOverlay) closeCompleteModal(); 
    });
    // Delete confirmation modal actions
    cancelDeleteBtn.addEventListener('click', closeDeleteModal);
    confirmDeleteBtn.addEventListener('click', deleteBook);
    // Close delete modal if clicking outside the content area
    deleteModalOverlay.addEventListener('click', (e) => { 
        if (e.target === deleteModalOverlay) closeDeleteModal(); 
    });

    // === Reader Controls ===

    // Exit reader and return to main content (bookshelf)
    readerBackBtn.addEventListener('click', () => {
        readerView.classList.add('hidden');
        mainContentArea.classList.remove('hidden');
        updateBookProgress(); // Save current progress before closing

        // Stop and reset audio playback
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.src = '';
            currentAudio = null;
            isPlaying = false;
            playPauseBtn.innerHTML = '▶';
        }
    });

    // Previous page/chapter button
    prevPageBtn.addEventListener('click', () => {
        // PDF navigation steps by 2 pages, EPUB by 1 chapter
        const isPdfPage = currentChapters[currentChapterIndex]?.includes('<img');
        const step = isPdfPage ? 2 : 1;
        if (currentChapterIndex > 0) {
            // Save progress only if there's no audio (manual reading)
            if (!currentAudio) {
                currentReaderBook.progress = ((currentChapterIndex) / currentChapters.length) * 100;
                updateBookProgress();
            }
            loadChapter(currentChapterIndex - step);
        }
    });

    // Next page/chapter button
    nextPageBtn.addEventListener('click', () => {
        // PDF navigation steps by 2 pages, EPUB by 1 chapter
        const isPdfPage = currentChapters[currentChapterIndex]?.includes('<img');
        const step = isPdfPage ? 2 : 1;
        if (currentChapterIndex + step < currentChapters.length) {
            // Save progress only if there's no audio (manual reading)
            if (!currentAudio) {
                currentReaderBook.progress = ((currentChapterIndex + step) / currentChapters.length) * 100;
                updateBookProgress();
            }
            loadChapter(currentChapterIndex + step);
        }
    });

    // Chapter dropdown selection
    chapterSelect.addEventListener('change', (e) => loadChapter(parseInt(e.target.value)));

    // Page select dropdown (currently only populated for PDFs in 10-page groups)
    pageSelect.addEventListener('change', (e) => loadChapter(parseInt(e.target.value)));

    // === Audio Controls ===

    // Play/pause toggle
    playPauseBtn.addEventListener('click', () => {
        if (!currentAudio) return;
        if (isPlaying) currentAudio.pause(); else currentAudio.play();
        isPlaying = !isPlaying;
        playPauseBtn.innerHTML = isPlaying ? '⏸' : '▶';
    });

    // Rewind 10 seconds
    rewindBtn.addEventListener('click', () => {
        if (currentAudio) currentAudio.currentTime = Math.max(0, currentAudio.currentTime - 10);
    });

    // Forward 10 seconds
    forwardBtn.addEventListener('click', () => {
        if (currentAudio) currentAudio.currentTime = Math.min(currentAudio.duration, currentAudio.currentTime + 10);
    });

    // Change playback speed
    speedSlider.addEventListener('input', (e) => {
        const speed = e.target.value;
        speedValue.textContent = `${speed}x`;
        if (currentAudio) currentAudio.playbackRate = parseFloat(speed);
    });

    // Change volume
    volumeSlider.addEventListener('input', (e) => {
        // Volume is a value between 0 and 1. Slider is 0-100.
        if (currentAudio) currentAudio.volume = parseFloat(e.target.value) / 100;
    });

    // Initialise upload UI and load books on startup
    resetUploadUI();
    await loadBooksFromDatabase();
});