# tongflow-modal-ffmpeg

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Media transcoding, muxing, and frame/track extraction with **FFmpeg**, running on [Modal](https://modal.com). No model weights — pure media processing.

## Capabilities

- **Concatenate clips** (`concat-videos`) — join multiple videos end to end.
- **Mux audio + video** (`merge-video-audio`) — merge an audio track and a video into one file.
- **Split video & audio** (`remove-video-audio`) — demux a video into separate video and audio tracks.
- **Extract audio track** (`extract-audio`) — pull the audio out as its own asset.
- **Extract first frame** (`get-first-frame`) — grab the first frame as an image.
- **Extract last frame** (`get-last-frame`) — grab the last frame as an image.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin deploys to your Modal account automatically and caches the build. No Hugging Face token required.
