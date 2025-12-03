# ATW306 - Emotional Audiobook Project

The following repository contains the codebase for an emotional audiobook generator. It comprises of two sections, which will need to be set up and run separately - the Frontend, and the Gradio Interface.

Multiple working branches have been left for historical purposes, but everything needed to run the audiobook generator is in the main branch.

## Frontend

## Gradio Interface

To set up the Gradio Interface, unzip all files locally, then run "DownloadTagModels.bat" to download the text tagging models (these were too large to fit on free Github). To start it, first visit "https://audioapi-g2ru.onrender.com/" in your browser, to initialise the Render static endpoint (when the Gradio API is not running, this times out). Then run "InitialiseGradio.bat", and wait for Gradio to initialise. No need to do anything else from this point - all requests are managed by the Frontend, and this just runs in the background.

To Summarise:

### Setup
1. Unzip folder locally
2. Run "DownloadTagModels.bat"

### Run
1. Visit "https://audioapi-g2ru.onrender.com/" in your browser
2. Run "InitialiseGradio.bat"
3. Wait, and leave running while Frontend is in use
