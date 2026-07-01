# ReSpeaker Lite AEC Investigation — Technical Report

**Branch:** `respeaker-hw-aec`
**Date:** 2026-07-01
**Hardware:** Raspberry Pi 4 Model B Rev 1.5, ReSpeaker Lite (Seeed, XMOS XU316), WaveShare/JMTek "USB PnP Audio Device"

## 1. Background

The original working architecture used PipeWire's `libpipewire-module-echo-cancel`
(WebRTC AEC) with capture from the ReSpeaker Lite's mic and playback relayed to a
separate WaveShare/JMTek USB audio stick wired to the speaker. This worked, but the
WaveShare stick had a history of USB enumeration/power failures on the Pi's shared
internal hub.

Goal for this session: move the speaker wire to the ReSpeaker Lite's onboard 2-pin
"hardware AEC" terminal, which claims to cancel echo in the DSP itself, eliminating
the need for a second USB device and PipeWire's software AEC layer.

## 2. Experiment 1 — ReSpeaker Lite onboard "hardware AEC"

**Change:** Removed the PipeWire echo-cancel module entirely. Set the ReSpeaker Lite
as the default PipeWire sink and source directly (`wpctl set-default`), relying on
the device's own DSP to cancel echo using its own speaker output as reference.

**Test:** Ran the full app (`agent.service`), had live conversations, monitored
`journalctl -u agent.service`.

**Result:** Severe echo leak. Bot's own TTS output was transcribed back nearly
verbatim by STT in the large majority of turns. The app's own three-layer text-based
Echo Guard (playback gate, Jaccard similarity filter, cooldown) failed to catch most
of it, because Jaccard similarity is computed against the bot's *entire* last
sentence — a short echoed fragment (5 overlapping words out of a 20+ word sentence)
scores ~0.22, well under the 0.55 rejection threshold.

**Controlled follow-up:** Tested digital output volume at 100%, 120%, and 200% via
`wpctl set-volume`. Leak severity was **identical at all three levels**, ruling out
volume/clipping as the cause.

**External validation:** Seeed Studio's own product forum
(`respeaker-lite-aec-support/283114`, `respeaker-lite-microphone-and-speaker-configurations/284874`)
confirms this is a known, unresolved firmware bug in the onboard AEC over USB — other
users report the same failure. Not a configuration gap on our end.

**Conclusion:** Hardware AEC does not work on this device. Ruled out.

## 3. Digression — USB port topology

User question: since the WaveShare stick's original power/enumeration issues were the
motivation for this whole change, would moving it to a different physical stack of
USB ports (the Pi 4B has 2 stacks of 2) avoid the problem?

**Investigation:** `cat /proc/device-tree/model` confirmed Raspberry Pi 4 Model B Rev
1.5. `lsusb -t` showed:

```
Bus 001.Port 001: Dev 001, Class=root_hub, Driver=xhci_hcd/1p, 480M
    |__ Port 001: Dev 002, If 0, Class=Hub, Driver=hub/4p, 480M   <- VIA Labs 2109:3431
        |__ ReSpeaker Lite (480M)
Bus 002.Port 001: Dev 001, Class=root_hub, Driver=xhci_hcd/4p, 5000M   <- empty
```

**Finding:** All 4 physical USB-A ports on the Pi 4B funnel USB 2.0/Full-Speed traffic
(which is what USB Audio Class devices use — audio doesn't need SuperSpeed) through a
**single internal VIA Labs 4-port hub**, regardless of which physical stack is used.
The "two stacks of two" is a visual/physical grouping, not an electrically independent
one. This hub also has one shared power budget across all 4 ports.

**Conclusion:** moving devices between stacks would not have fixed the original power
starvation — both devices would still land on the same hub, same power budget. The
actual fix for that problem is a self-powered external USB hub, independent of which
Pi port is used.

## 4. Experiment 2 — Software AEC retargeted at the ReSpeaker Lite (both directions)

Since onboard hardware AEC was ruled out, and the speaker wire had already been
physically moved to the ReSpeaker's 2-pin terminal, the plan pivoted to: bring back
PipeWire's software AEC, but point *both* capture and playback at the ReSpeaker
Lite's own ALSA nodes (rather than splitting across two physical devices like the
original setup did).

**Exact node names (from `wpctl inspect`):**
- Capture: `alsa_input.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo`
- Playback: `alsa_output.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo`

**Config applied** (`setup.sh`, `setup_drew.sh`, and live on the running system):
`libpipewire-module-echo-cancel` with `aec.args` (webrtc gain control, extended
filter, delay agnostic, high pass filter, noise suppression) and both
`capture.props.target.object` and `playback.props.target.object` pointing at the
ReSpeaker Lite. WirePlumber routing rules added to enforce these targets (WirePlumber
0.5 does not honor `target.object` from module args alone). `echo-cancel-source` /
`echo-cancel-sink` confirmed as PipeWire's default source/sink after restart.

**Sanity tests:** `tests/test_speaker.py` and `tests/test_microphone.py` both passed
functionally (tones audible, recording/playback round-tripped), but note both scripts
resolve the **raw ALSA hw device** via `device_finder.py`, not the `pulse`/AEC-routed
path that `main.py` actually uses (`_get_pulse_device_index()` looks for a device
literally named `pulse`/`pipewire`). The mic test also flagged very low raw capture
amplitude (peak 35/32767, ~0.1%) — this reflects the raw hardware capture level before
any AGC, not necessarily the real app path (which goes through
`webrtc.gain_control = true` in the AEC module).

## 5. Objective AEC measurement (this is the decisive test)

Rather than relying on subjective live-conversation impressions, ran a controlled,
objective test using PipeWire's native CLI tools (`pw-record` / `pw-play`, no
`pactl`/PulseAudio-utils needed): record from `echo-cancel-source` while
simultaneously playing a known signal through `echo-cancel-sink`, and measure how much
of the played signal leaks into the capture.

### 5a. Baseline (noise floor)

3s recording from `echo-cancel-source`, no playback: **RMS 1.3** (near-silent room).

### 5b. Synthetic 1 kHz tone test

Played a 5s, 16 kHz mono, 1 kHz sine tone through `echo-cancel-sink` while recording
`echo-cancel-source` for 7s (tone starts at ~t=1s).

| Window | RMS | 1kHz band energy |
|---|---|---|
| t=0.0–0.5s (pre-tone) | 0.8–1.1 | ~1,600 |
| t=1.0–2.5s (tone playing, filter converging) | 124–428 | 0.9M–4.1M |
| t=3.0–4.5s (tone playing, sustained) | 84–8,861 | 30K–212M |
| t=6.0s (tone playing) | 3,050 | 29.8M |

Highly erratic, non-converging leak — RMS swings between ~40 and ~8,800 (out of a
32,767 full-scale) with no stabilization. (Note: pure sine tones are a known
harder/less-representative case for WebRTC-style adaptive filters than natural
speech, since they lack the broadband spectral content the filter uses to converge —
so this was treated as a preliminary signal, not the final verdict.)

### 5c. Real TTS speech test (representative of actual app usage)

Synthesized real TTS audio via the app's own `TextToSpeechClient.synthesize_to_audio()`
(same Azure voice/format the app uses live) — a 9.7s sentence, source RMS 2,636.
Played through `echo-cancel-sink` while recording `echo-cancel-source` for 13s
(playback starts at ~t=1.5s, ends at ~t=11.2s).

| Window | Captured RMS | dB over noise floor |
|---|---|---|
| t=0.0–1.0s (pre-playback) | 28–54 | +26 to +32 dB |
| t=1.5–10.0s (playback active) | 1,265–9,778 | **+60 to +77 dB** |
| t=10.5–12.0s (after playback ends) | 2.6–7.6 | +6 to +15 dB (back to floor) |

**Captured echo RMS during playback (avg ~5,000) exceeds the original TTS source RMS
(2,636)** — the mic is hearing its own speaker at essentially full, uncancelled
volume, tracking the playback window exactly. This is not "some leak" — it is
functionally **zero effective echo cancellation**.

## 6. Root cause

WebRTC-style AEC (and the `libspa-aec-webrtc` backend PipeWire uses) depends on
estimating a stable delay between the reference (playback) signal and the captured
echo, typically tens of milliseconds on independent hardware. Here, capture and
playback are both routed through the **same USB audio interface, same driver, same
clock domain**. This breaks the delay-estimation assumptions the algorithm is built
on (`webrtc.delay_agnostic = true` does not compensate for this — it's designed for
variable-but-*present* delay, not a same-device degenerate case). The erratic,
non-converging leak pattern observed in both tests is consistent with this failure
mode, not with a tunable parameter problem.

## 7. Status of the two approaches tried this session

| Approach | Result | Root cause |
|---|---|---|
| ReSpeaker onboard "hardware AEC" | Failed — severe leak, identical across volume levels | Broken/unreliable firmware feature (confirmed via Seeed's own forum) |
| Software AEC, capture+playback both on ReSpeaker | Failed — near-zero cancellation (objectively measured) | Same-device clock domain breaks AEC delay estimation |

Both failure modes share the same underlying cause: using **one physical USB device**
for both mic and speaker undermines echo cancellation, whether done in firmware or in
software. The only architecture validated to actually work (the original setup, prior
to this investigation) uses **two separate physical devices** — this is also the
configuration WebRTC-style AEC is designed and normally deployed for.

## 8. Files touched this session

- `setup.sh` — PipeWire echo-cancel module + WirePlumber routing, retargeted at
  ReSpeaker Lite for both capture and playback (superseded pending the decision below)
- `/home/likhitjuttada/setup_drew.sh` (outside git repo) — same config, adapted to its
  `sudo -u likhitjuttada` invocation pattern; also fixed its stale index-based `.env`
  heredoc to match the repo's name-based device resolution
- `src/audio/device_finder.py` — added `respeaker`/`seeed` to `DEFAULT_OUTPUT_NAME_HINTS`
- `.env` — `AUDIO_OUTPUT_DEVICE_NAME=respeaker`
- Live system config: `~/.config/pipewire/pipewire.conf.d/99-echo-cancel.conf`,
  `~/.config/wireplumber/wireplumber.conf.d/99-echo-cancel-routing.conf`

## 9. Decision point

Given both same-device approaches are ruled out, the options are:

1. **Revert to two-device AEC** — restore ReSpeaker (mic) + separate USB audio device
   (speaker), software AEC bridging them, as originally worked. Address the earlier
   power-starvation issue with a self-powered external USB hub for the speaker device
   (not a different Pi port — see §3, all ports share one hub anyway).
2. **Keep ReSpeaker-only wiring, drop AEC entirely** — rely solely on `main.py`'s
   text-similarity Echo Guard, and fix its Layer 2 similarity-dilution bug (Jaccard
   score computed against the full last sentence dilutes short echoed fragments below
   the 0.55 rejection threshold).
3. **Further tuning** — try explicit fixed delay instead of `delay_agnostic`, other
   `aec.args`, before committing to an architecture change. Unlikely to overcome the
   fundamental same-device clock issue documented in §6.
