# ReSpeaker 2-Mic HAT Troubleshooting Guide for Raspberry Pi

## Hardware
- **Device**: Keyestudio ReSpeaker 2-Mic Voice Card (WM8960 codec)
- **Tested on**: Raspberry Pi 4 Model B
- **OS**: Debian Trixie (arm64)
- **Kernel**: 6.12.47+rpt-rpi-v8

## Issue Summary

The ReSpeaker microphone shows up as a detected sound card but cannot be used for recording due to a kernel API incompatibility. The WM8960 codec fails to initialize with the error:

```
wm8960 1-001a: No MCLK configured
wm8960 1-001a: ASoC: error at snd_soc_dai_hw_params on wm8960-hifi: -22
```

## Root Cause

**Kernel 6.12+ API Change**: The `snd_soc_pcm_runtime` structure changed from using `rtd->id` to `rtd->num`. The seeed-voicecard driver was not updated for this change, causing compilation failures.

---

## Solution: Install Patched Driver

### Step 1: Install Prerequisites

```bash
sudo apt update
sudo apt install -y dkms git i2c-tools libasound2-plugins gcc-aarch64-linux-gnu
```

### Step 2: Clone the Driver Repository

Use the HinTak fork for better Raspberry Pi 4 support:

```bash
cd ~
git clone https://github.com/HinTak/seeed-voicecard.git seeed-voicecard-hintak
cd seeed-voicecard-hintak
```

### Step 3: Apply Kernel 6.12 Compatibility Patch

Fix the API incompatibility:

```bash
sed -i 's/rtd->id/rtd->num/g' seeed-voicecard.c
```

### Step 4: Install the Driver

```bash
sudo ./install.sh
```

> **Note**: You may see warnings during compilation - these are normal. The critical fix is the `rtd->id` to `rtd->num` change.

### Step 5: Configure Boot Options

Edit the boot configuration:

```bash
sudo nano /boot/firmware/config.txt
```

Add these lines at the end under the `[all]` section:

```
dtparam=i2s=on
dtoverlay=seeed-2mic-voicecard
```

Save and exit (Ctrl+X, Y, Enter).

### Step 6: Reboot

```bash
sudo reboot
```

---

## Verification

After reboot, verify the setup:

### 1. Check Driver Loaded Successfully

```bash
dmesg | grep -i wm8960 | head -20
```

**Expected**: No "No MCLK configured" errors. You should see successful initialization messages.

### 2. Check Device Tree Overlay

```bash
dtoverlay -l
```

**Expected**: Should show `seeed-2mic-voicecard` in the list (though this may vary).

### 3. List Audio Devices

```bash
arecord -l
```

**Expected output**:
```
card 3: seeed2micvoicec [seeed-2mic-voicecard], device 0: bcm2835-i2s-wm8960-hifi wm8960-hifi-0
```

### 4. Test Recording and Playback

Assuming card 3 is ReSpeaker and card 4 is USB speaker:

```bash
arecord -D plughw:3,0 -f S16_LE -r 16000 -c 2 -d 5 test.wav && aplay -D plughw:4,0 test.wav
```

This records 5 seconds of audio and plays it back.

---

## Troubleshooting

### Issue: Device Tree Overlay Fails to Load

**Check manually**:
```bash
sudo dtoverlay seeed-2mic-voicecard
dmesg | tail -20
```

If you see "Failed to apply overlay", the overlay may be incompatible with your kernel.

### Issue: Compilation Fails with "has no member named 'id'"

**Solution**: Apply the patch from Step 3 above.

### Issue: "No MCLK configured" Errors Persist

Check boot config:
```bash
cat /boot/firmware/config.txt | grep -i i2s
cat /boot/firmware/config.txt | grep -i seeed
```

Ensure both lines are present and not commented out.

### Issue: Can't Hear Playback

Check mixer settings:
```bash
alsamixer -c 3
```

Use arrow keys to navigate, `M` to unmute channels. Ensure "Capture" and "Playback" are unmuted and volumes are up.

### Issue: Recording Format Not Supported

Try different sample rates and channels:
```bash
# Mono, 16kHz
arecord -D plughw:3,0 -f S16_LE -r 16000 -c 1 -d 5 test.wav

# Stereo, 48kHz
arecord -D plughw:3,0 -f S16_LE -r 48000 -c 2 -d 5 test.wav
```

---

## Alternative: Use Older Kernel (6.6 LTS)

If patching doesn't work, downgrade to Raspberry Pi OS **Bookworm** which uses the well-tested kernel 6.6 LTS. The seeed-voicecard driver fully supports this kernel without modifications.

---

## Configuration Files Modified

1. **`/boot/firmware/config.txt`** - Added I2S and device tree overlay
2. **`~/seeed-voicecard-hintak/seeed-voicecard.c`** - Patched for kernel 6.12 compatibility
3. **`/boot/firmware/overlays/seeed-2mic-voicecard.dtbo`** - Installed by driver

## Key Commands Reference

| Action | Command |
|--------|---------|
| List capture devices | `arecord -l` |
| List playback devices | `aplay -l` |
| Check kernel version | `uname -r` |
| View kernel messages | `dmesg \| grep -i wm8960` |
| Check loaded overlays | `dtoverlay -l` |
| Test recording | `arecord -D plughw:3,0 -f S16_LE -r 16000 -c 2 -d 5 test.wav` |
| Mixer settings | `alsamixer -c 3` |

---

## Credits

- **Driver**: HinTak fork of seeed-voicecard - https://github.com/HinTak/seeed-voicecard
- **Original**: Seeed Studio - https://github.com/respeaker/seeed-voicecard

## Notes

- The patch changes `rtd->id` to `rtd->num` in 4 locations (lines 103, 133, 153, 418)
- This is required for kernel 6.12+ due to ALSA/ASoC API changes
- Once applied, the driver compiles successfully and the MCLK is properly configured
