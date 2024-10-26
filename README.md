# Live Audio Transcription and Translation

This project provides real-time transcription and translation of audio sent from a client (web) to a server (Python). The client records audio, sends it to the server for processing, and displays subtitles in selected languages. The application currently supports English, Persian, and Arabic, with potential for future language expansion.

## Features
- Real-time audio transcription and translation
- Multiple language support with selectable subtitles
- Continuous subtitle display with language choice

## Components
- **Server**: `stream_server.py` - Python WebSocket server for processing audio and returning transcription and translation.
- **Client**: `stream_client.html` - Web client that records audio and displays subtitles in selected languages.
- **Shell Scripts**:
  - `install-requirements.sh`: Installs the required Python packages.
  - `start.sh`: Sets the necessary CUDA libraries and starts the server.

## Setup and Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Run the installation script to install dependencies:
    ```bash
    chmod +x install-requirements.sh
    ./install-requirements.sh
    ```
   > **Note**: The `install-requirements.sh` script installs dependencies from `requirements.txt` and upgrades `faster-whisper` to the latest version, which may cause an error with WhisperX. Ignore this error as it does not impact the functionality of the project.

3. Before running the server, ensure necessary CUDA libraries are available by setting `LD_LIBRARY_PATH`:
    ```bash
    chmod +x start.sh
    ./start.sh
    ```
   > **Note**: Running `start.sh` will export the `LD_LIBRARY_PATH` using `nvidia.cublas.lib` and `nvidia.cudnn.lib` paths for CUDA compatibility.

## Usage
You have two options for running the project:

1. **Using `start.sh`**:
Run the `start.sh` script, which sets the `LD_LIBRARY_PATH`, starts the server, and opens the client in your default web browser.

    ```bash
    ./start.sh
    ```

2. **Manual Start**:
    - **Start the Server**:
    Alternatively, you can run the server directly:
        ```bash
        python3 stream_server.py
        ```
        You can customize the behavior of the server by using the following command-line arguments when running stream_server.py:

        - `--interface`:
        Specify the server interface to bind to. Default is `0.0.0.0`.

        - `--port`:
        Specify the port number for the server. Default is `2800`.

        - `--open_browser`:
        Set to True to open the browser automatically if the server starts successfully. Default is `True`.

    - **Open the Client**:
        - Open `stream_client.html` in a browser (using `http://localhost:2800`).
        - Select up to three languages for subtitles from the dropdown menu. The client will display subtitles as the server processes incoming audio.

## Project Structure
```plaintext
.
├── stream_server.py          # Python WebSocket server
├── stream_client.html        # Web client for recording and displaying subtitles
├── requirements.txt          # List of required Python packages
├── install-requirements.sh   # Shell script for installing dependencies
└── start.sh                  # Shell script for setting LD_LIBRARY_PATH and running the server
```
## Troubleshooting

- **Permissions**:
If `install-requirements.sh` or `start.sh` scripts cannot execute, ensure they have proper permissions:

    ```bash

    chmod +x install-requirements.sh start.sh
    ```

- **WebSocket Connection**:
Ensure that `stream_server.py` is running on the same network and accessible via `ws://localhost:2800`.

- **Loading CUDA libraries**:
If you encounter the following error:

```arduino
Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}
```

Ensure that you have set the `LD_LIBRARY_PATH` correctly, either by running the `start.sh` script or manually exporting the path by executing the following command in your terminal:
```bash
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```