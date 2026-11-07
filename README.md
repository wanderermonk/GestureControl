# GestureControl

Control your Mac with hand gestures through the webcam: move the cursor, click, drag and scroll, switch desktops, snap windows, and change the volume or media playback.

It uses MediaPipe hand tracking and runs entirely on your Mac. Nothing is uploaded.

## Setup (once)

```bash
cd ~/Documents/Projects_Folder/GestureControl
bash setup.sh
```

This creates a `.venv`, installs the dependencies, downloads the hand model (~7.5 MB) and runs the self-tests. It needs Python 3.10–3.13. If you don't have one, run `brew install python@3.12`.

## Run

```bash
./run.sh              # normal
./run.sh --dry-run    # see detected gestures; nothing is clicked or pressed
./run.sh --no-preview # start hidden (menu bar icon only)
```

Or double-click **Start GestureControl.command** in Finder.

### macOS permissions (first run)

Grant both of these to the app you launch it from (Terminal, iTerm, VS Code…), then **quit and reopen that app**:

1. **Camera**: System Settings → Privacy & Security → Camera
2. **Accessibility**: System Settings → Privacy & Security → Accessibility. This lets it move the mouse and press keys.

The first time you use window snapping, macOS also asks to let the terminal control **System Events**. Allow it.

## Gestures

| Pose | How | Does |
|---|---|---|
| ☝️ **Point** | index finger up | moves the cursor (follows your index knuckle) |
| 🤏 **Pinch** | thumb tip → index tip, quick | left click (two quick pinches = double click) |
| 🤏 **Pinch & hold** | hold the pinch > 0.35 s, then move | drag; release the pinch to drop |
| **Middle pinch** | thumb tip → middle tip | right click |
| ✌️ **Two fingers** | index + middle up, move hand up/down | scroll |
| 🖐 **Open palm swipe** | ← / → | previous / next desktop (Space) |
| 🖐 **Open palm swipe** | ↑ / ↓ | Mission Control / App Exposé |
| **Three fingers swipe** | ← / → / ↑ / ↓ | snap window left / right / maximise / minimise |
| **Three fingers hold** | hold still 1 s | screenshot |
| 👍 / 👎 **hold** | | volume up / down (repeats while held) |
| 🤙 **Shaka hold** | thumb + pinky | play / pause |
| 👌 **OK hold** | 1.2 s | close window (cmd+W) |
| 🤘 **Rock hold** | index + pinky, 1 s | **pause / resume gesture control** |
| 🖐 **Open palm hold** | hold still ~1 s | start listening for a voice command |
| ✊ **Fist** | | nothing; rest your hand here |

**Tips**

- Sit about 50–80 cm from the camera with your hand roughly at chest height. The grey box in the preview is the part of the image that maps to your whole screen.
- Make a fist to "lift the mouse" while you reposition your hand.
- Swipes need a quick, deliberate movement. A hold needs your hand to stay still, and a ring fills up around your palm while you hold.
- Pause with 🤘 when you want to type or talk with your hands.

## Train it on your hand

The built-in rules work for most hands. Training makes recognition fit *your* hand, camera and lighting, and lets you invent new poses.

```bash
./run.sh --train               # record all 9 built-in poses (~2 min)
./run.sh --train point palm    # re-record just a couple
./run.sh --train vulcan        # add a new pose called "vulcan" (any name)
./run.sh --list-trained        # what's in your model + estimated accuracy
./run.sh --forget vulcan       # remove a pose
./run.sh --reset-training      # delete the model, back to built-in rules
```

For each pose: press **SPACE**, wait for the countdown, and hold the pose for 2 seconds. Then do a second round while slowly moving and tilting your hand. **S** skips a pose, **R** redoes it and **Q** stops early; whatever you've recorded so far is kept. At the end you get an accuracy estimate for each pose and a hint for any that get confused.

- Only the 21 hand-point coordinates are stored (in `models/my_gestures.npz`). No camera images are ever saved.
- Your trained poses take priority when the model is confident (`tracking.trained_min_confidence`). Otherwise the built-in rules decide. `./run.sh --no-model` ignores the model for one session.
- Pinch, middle pinch and OK are always detected by measuring finger distance, because that gives the most reliable clicks, so they can't be trained.
- To use a new pose, map it in `config.yaml`, e.g. `vulcan_hold: "app:Spotify"` or `vulcan_swipe_left: "media:previous"`.

## Voice commands

Speech recognition runs offline with [Vosk](https://alphacephei.com/vosk/). No audio is recorded, saved or uploaded.

**Two ways to give a command** (`voice.activation: both`):

- Say the **wake word** first: *"Computer, next desktop"*. You can also say *"Computer"*, pause, and then give the command.
- **Hold an open palm still** for about a second (the mic indicator turns yellow), then just say *"next desktop"*.

After a command it keeps listening for 5 more seconds, so you can say *"scroll down… scroll down"* without the wake word each time. Anything said without activation, like normal conversation or videos, is ignored.

**Some default commands** (`./run.sh --voice-commands` lists them all):

| Say | Does |
|---|---|
| next / previous desktop, mission control, show windows, show desktop | desktops |
| snap left / right, maximize / center / minimize / close window, full screen | windows |
| click, double click, right click, scroll up / down | mouse at the cursor |
| copy that, paste, undo, go back, new tab | editing |
| volume up / down, mute, play / pause music, next / previous song | media |
| take screenshot, lock screen, open safari / finder / terminal | system |
| pause / resume gestures, stop / start listening | control |

Add your own under `voice.commands` in `config.yaml`, e.g. `"open mail": "app:Mail"` or `"zoom in": "hotkey:cmd+="`. Use plain English words; startup warns you about any word the speech model doesn't know.

### Train it on your voice

```bash
./run.sh --train-voice                    # wake word + every command, 3x each (~4 min)
./run.sh --train-voice "snap left" mute   # just some phrases
./run.sh --voice-test                     # live: what it hears, your voice score, what it would do
./run.sh --reset-voice                    # forget voice training
```

Training does three things:

1. **Learns your pronunciation.** It records how the recogniser hears *you* say each phrase. If it consistently hears "necks desktop", that becomes an alias for "next desktop".
2. **Enrols your voice for "only respond to me".** It averages a voice fingerprint (a speaker embedding) from all your samples and calibrates a threshold. Afterwards, commands spoken in other voices are rejected. Turn this off with `voice.only_my_voice: false`.
3. **Checks itself.** It replays your recordings through the real command recogniser and shows a score for each phrase, flagging any that need re-training or a more distinct wording.

The profile (`models/voice_profile.json`) contains only text aliases and numbers, no audio. For Indian English, set `voice.model: vosk-model-small-en-in-0.4`, re-run `bash setup.sh`, then re-run `--train-voice`.

The first time, macOS asks to let your terminal use the **Microphone**. Allow it.

## Customise

Edit `config.yaml` (menu bar ✋ → *Edit gestures*), then choose *Reload config*. Any `<pose>_hold` or `<pose>_swipe_<dir>` can be mapped to any of these:

```yaml
gestures:
  shaka_hold: "app:Spotify"
  palm_swipe_up: "hotkey:cmd+space"        # Spotlight
  two_hold: { action: "media:mute", hold: 1.5 }
  rock_swipe_right: "media:next"
```

The action types are `hotkey:`, `media:`, `window:`, `system:`, `app:` and `builtin:toggle_pause`. The full list is at the top of `config.yaml`. You can also tune cursor smoothing, scroll speed, the active region and all timings there.

## Menu bar

The ✋ icon shows ⏸ when paused. The menu shows the current pose and lets you enable or disable control, show or hide the preview, edit and reload the config, and quit.

In the preview window: **P** pauses, **H** hides the preview, **Q** quits.

## Troubleshooting

- **Cursor doesn't move / keys do nothing**: check the Accessibility permission. Toggle it off and on, then restart the terminal.
- **"Could not open camera"**: check the Camera permission. If you have several cameras, try `camera: index: 1`.
- **Cursor jittery**: lower `cursor.min_cutoff` (e.g. 0.6). **Cursor laggy**: raise `cursor.beta`.
- **Clicks firing by accident**: lower `tracking.pinch_on` (e.g. 0.22).
- **Window snapping does nothing**: allow System Settings → Privacy & Security → Automation → your terminal → System Events.
- **Voice does nothing**: run `./run.sh --voice-test`. If it hears nothing, allow your terminal under Privacy & Security → Microphone. If it hears you but says "not your voice", run `--train-voice` again or set `voice.speaker_threshold` lower (e.g. 0.4).
- **Space switching does nothing**: make sure System Settings → Keyboard → Keyboard Shortcuts → Mission Control has "Move left/right a space" enabled.

## Project layout

```
gesture_control/
  __main__.py     main loop: camera → tracker → classifier → engine → preview/menu bar
  tracker.py      webcam + MediaPipe HandLandmarker
  gestures.py     pose classification from 21 landmarks
  engine.py       state machine: cursor, click/drag, scroll, swipes, holds
  filters.py      One Euro smoothing filter
  learned.py      personal pose model (k-nearest-neighbour on landmarks)
  trainer.py      guided recording UI (--train)
  voice.py        offline speech: wake word, command matching, voice ID, mic thread
  voice_trainer.py  --train-voice / --voice-test
  actions.py      "type:value" action dispatcher (+ dry-run backend)
  backend_macos.py Quartz mouse/keyboard, media keys, window placement
  overlay.py      preview drawing
  menubar.py      rumps menu bar helper (child process)
tests/            synthetic-hand tests for the classifier and engine
```
