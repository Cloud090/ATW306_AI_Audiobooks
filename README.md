# ATW306 - Emotional Audiobook Project
#Github: https://github.com/Cloud090/ATW306_AI_Audiobooks
The following repository contains the codebase for an emotional audiobook generator. It comprises of two sections, which will need to be set up and run separately - the Frontend, and the Gradio Interface.

Multiple working branches have been left for historical purposes, but everything needed to run the audiobook generator is in the main branch.

## Frontend

The frontend is located in the "Frontend" folder within the main branch when cloned and can be run in either of the following two ways:

### Option 1:
1. Run Start.bat contained within the Fronend folder. If NodeJS isn't installed it will be installed & nodejs will install all required dependencies for the fronend.
2. Enjoy :)

### Option 2:
1. Ensure NodeJS is installed if not either:
   a. Install via CMD using: ```winget install OpenJS.NodeJS```
   b. Install using the installer from nodejs's website: https://nodejs.org/en/download
2. open to the directory the frontend files are contained within in CMD
3. Run ```npm start```
4. Enjoy :)

## Gradio Interface

The Gradio Interface is located in the "GradioInterface" folder. 

To set up the Gradio Interface, unzip all files locally, then run "DownloadTagModels.bat" to download the text tagging models (these were too large to fit on free Github). To start it, first visit "https://audioapi-g2ru.onrender.com/" in your browser, to initialise the Render static endpoint (when the Gradio API is not running, this times out). Then run "InitialiseGradio.bat", and wait for Gradio to initialise. No need to do anything else from this point - all requests are managed by the Frontend, and this just runs in the background.

To Summarise:

### Setup
1. Install requirements via ``pip install -r requirements.txt``
2. Unzip folder locally
3. Run "DownloadTagModels.bat"

### Run
1. Visit "https://audioapi-g2ru.onrender.com/" in your browser
2. Run "InitialiseGradio.bat"
3. Wait, and leave running while Frontend is in use
