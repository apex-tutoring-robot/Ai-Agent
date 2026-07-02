# Intermittent Speaker Disconnects — Diagnostic Report

**Host:** apex-pi (Raspberry Pi)
**Branch:** openwakeword
**Date:** 2 Jul 2026
**Status:** Software fix applied · hardware action outstanding

## Summary

The speaker's dropouts had two independent causes stacked on top of each other. On the **hardware side**, the Waveshare USB speaker and a second, misbehaving USB device are wired into the Pi through the same shared connection, and that second device fails to enumerate roughly every 10 seconds, non-stop — very likely destabilizing the speaker's connection along with it. On the **software side**, the application already had a recovery routine meant to rebuild the audio stack after exactly this kind of failure, but the flag that was supposed to trigger it was never actually being set — so instead of self-healing, the bot just went silent until it was restarted by hand.

This report documents the kernel-log evidence for the first problem and the code fix applied for the second.

## Symptom

During conversation, audio played back through the Waveshare USB speaker would occasionally cut out and not resume, with no crash and no error visible to the user — the process kept running, but produced no further sound until it was manually restarted.

## Kernel Log Evidence

The kernel's own USB subsystem log was read directly (not summarized) to establish what was actually happening at the hardware level. Two commands were run to cross-check the same window from two angles: the raw ring buffer, and the systemd journal's copy of kernel messages with wall-clock timestamps attached.

### Ring buffer — audio/USB-related lines

```
$ dmesg | grep -iE "audio|snd|usb.*disconnect|alsa|codec" | tail -100

[    0.000000] Kernel command line: coherent_pool=1M 8250.nr_uarts=0 snd_bcm2835.enable_headphones=0 cgroup_disable=memory numa_policy=interleave nvme.max_host_mem_size_mb=32 snd_bcm2835.enable_headphones=1 snd_bcm2835.enable_hdmi=1 snd_bcm2835.enable_hdmi=0  numa=fake=2 system_heap.max_order=0 iommu_dma_numa_policy=interleave smsc95xx.macaddr=2C:CF:67:7E:4E:AA vc_mem.mem_base=0x3ec00000 vc_mem.mem_size=0x40000000  root=/dev/mmcblk0p2 rootfstype=ext4 fsck.repair=yes rootwait cfg80211.ieee80211_regdom=US quiet loglevel=0 logo.nologo vt.global_cursor_default=0 systemd.show_status=false console=tty3 splash plymouth.ignore-serial-consoles fbcon=map:2 video=HDMI-A-1:1024x600@60D
[    4.373855] usb 1-1.3: Product: USB PnP Audio Device
[    4.380934] input: Solid State System Co.,Ltd. USB PnP Audio Device as /devices/platform/scb/fd500000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/usb1/1-1/1-1.3/1-1.3:1.3/0003:0C76:1203.0001/input/input4
[    4.434832] hid-generic 0003:0C76:1203.0001: input,hidraw0: USB HID v1.00 Device [Solid State System Co.,Ltd. USB PnP Audio Device] on usb-0000:01:00.0-1.3/input3
[    5.968035] snd_bcm2835: module is from the staging directory, the quality is unknown, you have been warned.
[    6.013294] bcm2835-audio bcm2835-audio: card created with 8 channels
[    6.523255] bcm2835_codec: module is from the staging directory, the quality is unknown, you have been warned.
[    6.528796] bcm2835-codec bcm2835-codec: Device registered as /dev/video10
[    6.528834] bcm2835-codec bcm2835-codec: Loaded V4L2 decode
[    6.532075] bcm2835-codec bcm2835-codec: Device registered as /dev/video11
[    6.532163] bcm2835-codec bcm2835-codec: Loaded V4L2 encode
[    6.538569] bcm2835-codec bcm2835-codec: Device registered as /dev/video12
[    6.538603] bcm2835-codec bcm2835-codec: Loaded V4L2 isp
[    6.541178] bcm2835-codec bcm2835-codec: Device registered as /dev/video18
[    6.541211] bcm2835-codec bcm2835-codec: Loaded V4L2 image_fx
[    6.548307] bcm2835-codec bcm2835-codec: Device registered as /dev/video31
[    6.548370] bcm2835-codec bcm2835-codec: Loaded V4L2 encode_image
[    6.608221] rpi-hevc-dec feb00000.codec: Device registered as /dev/video19
[    7.005328] usbcore: registered new interface driver snd-usb-audio
[   14.673257] usb 1-1.4: USB disconnect, device number 9
[   24.658651] usb 1-1.4: USB disconnect, device number 10
[   34.645294] usb 1-1.4: USB disconnect, device number 11
[   44.629870] usb 1-1.4: USB disconnect, device number 12
[   54.614915] usb 1-1.4: USB disconnect, device number 13
[   64.600371] usb 1-1.4: USB disconnect, device number 14
[   74.586427] usb 1-1.4: USB disconnect, device number 15
[   84.570781] usb 1-1.4: USB disconnect, device number 16
[   94.301089] usb 1-1.4: USB disconnect, device number 17
[  104.285540] usb 1-1.4: USB disconnect, device number 18
[  114.271322] usb 1-1.4: USB disconnect, device number 19
[  124.257111] usb 1-1.4: USB disconnect, device number 20
[  134.241934] usb 1-1.4: USB disconnect, device number 21
[  144.230515] usb 1-1.4: USB disconnect, device number 22
[  154.214345] usb 1-1.4: USB disconnect, device number 23
[  164.198266] usb 1-1.4: USB disconnect, device number 24
[  174.183718] usb 1-1.4: USB disconnect, device number 25
[  184.169485] usb 1-1.4: USB disconnect, device number 26
[  194.155103] usb 1-1.4: USB disconnect, device number 27
[  201.592558] usb 1-1.3: USB disconnect, device number 7
[  204.602863] usb 1-1.3: Product: USB PnP Audio Device
[  204.701676] input: Solid State System Co.,Ltd. USB PnP Audio Device as /devices/platform/scb/fd500000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/usb1/1-1/1-1.3/1-1.3:1.3/0003:0C76:1203.0002/input/input5
[  204.756982] hid-generic 0003:0C76:1203.0002: input,hidraw0: USB HID v1.00 Device [Solid State System Co.,Ltd. USB PnP Audio Device] on usb-0000:01:00.0-1.3/input3
[  214.151522] usb 1-1.4: USB disconnect, device number 37
[  224.136974] usb 1-1.4: USB disconnect, device number 38
[  234.125404] usb 1-1.4: USB disconnect, device number 39
[  244.107844] usb 1-1.4: USB disconnect, device number 40
[  253.837301] usb 1-1.4: USB disconnect, device number 41
[  263.822730] usb 1-1.4: USB disconnect, device number 42
[  273.808576] usb 1-1.4: USB disconnect, device number 43
[  283.793933] usb 1-1.4: USB disconnect, device number 44
[  293.779093] usb 1-1.4: USB disconnect, device number 45
[  303.764599] usb 1-1.4: USB disconnect, device number 46
[  313.750395] usb 1-1.4: USB disconnect, device number 47
[  323.739009] usb 1-1.4: USB disconnect, device number 48
[  333.720874] usb 1-1.4: USB disconnect, device number 49
[  343.706279] usb 1-1.4: USB disconnect, device number 50
[  353.692179] usb 1-1.4: USB disconnect, device number 51
[  363.677186] usb 1-1.4: USB disconnect, device number 52
[  373.664061] usb 1-1.4: USB disconnect, device number 53
[  383.648816] usb 1-1.4: USB disconnect, device number 54
[  393.633565] usb 1-1.4: USB disconnect, device number 55
[  403.621571] usb 1-1.4: USB disconnect, device number 56
[  413.604391] usb 1-1.4: USB disconnect, device number 57
[  423.589873] usb 1-1.4: USB disconnect, device number 58
[  433.575341] usb 1-1.4: USB disconnect, device number 59
[  443.560757] usb 1-1.4: USB disconnect, device number 60
[  453.546244] usb 1-1.4: USB disconnect, device number 61
[  463.276038] usb 1-1.4: USB disconnect, device number 62
[  473.261391] usb 1-1.4: USB disconnect, device number 63
[  483.246579] usb 1-1.4: USB disconnect, device number 64
[  490.940587] usb 1-1.3: USB disconnect, device number 36
[  493.954557] usb 1-1.3: Product: USB PnP Audio Device
[  494.064617] input: Solid State System Co.,Ltd. USB PnP Audio Device as /devices/platform/scb/fd500000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/usb1/1-1/1-1.3/1-1.3:1.3/0003:0C76:1203.0003/input/input6
[  494.120995] hid-generic 0003:0C76:1203.0003: input,hidraw0: USB HID v1.00 Device [Solid State System Co.,Ltd. USB PnP Audio Device] on usb-0000:01:00.0-1.3/input3
[  503.243404] usb 1-1.4: USB disconnect, device number 74
[  513.230212] usb 1-1.4: USB disconnect, device number 75
[  523.215073] usb 1-1.4: USB disconnect, device number 76
[  533.201770] usb 1-1.4: USB disconnect, device number 77
[  543.185901] usb 1-1.4: USB disconnect, device number 78
[  553.171362] usb 1-1.4: USB disconnect, device number 79
[  563.156739] usb 1-1.4: USB disconnect, device number 80
[  570.594094] usb 1-1.3: USB disconnect, device number 73
[  573.615607] usb 1-1.3: Product: USB PnP Audio Device
[  573.715985] input: Solid State System Co.,Ltd. USB PnP Audio Device as /devices/platform/scb/fd500000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/usb1/1-1/1-1.3/1-1.3:1.3/0003:0C76:1203.0004/input/input7
[  573.772745] hid-generic 0003:0C76:1203.0004: input,hidraw0: USB HID v1.00 Device [Solid State System Co.,Ltd. USB PnP Audio Device] on usb-0000:01:00.0-1.3/input3
[  590.337075] usb 1-1.3: USB disconnect, device number 89
[  590.381681] usb 1-1.4: USB disconnect, device number 90
[  593.371419] usb 1-1.3: Product: USB PnP Audio Device
[  593.468756] input: Solid State System Co.,Ltd. USB PnP Audio Device as /devices/platform/scb/fd500000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/usb1/1-1/1-1.3/1-1.3:1.3/0003:0C76:1203.0005/input/input8
[  593.521427] hid-generic 0003:0C76:1203.0005: input,hidraw0: USB HID v1.00 Device [Solid State System Co.,Ltd. USB PnP Audio Device] on usb-0000:01:00.0-1.3/input3
[  602.899929] usb 1-1.4: USB disconnect, device number 96
[  612.885378] usb 1-1.4: USB disconnect, device number 97
[  622.870840] usb 1-1.4: USB disconnect, device number 98
```

### Journal — same window, wall-clock time

```
$ journalctl -k --since "7 days ago" | grep -iE "audio|snd|usb.*disconnect|alsa|codec|xrun" | tail -150

Jul 02 13:26:13 apex-pi kernel: Kernel command line: coherent_pool=1M 8250.nr_uarts=0 snd_bcm2835.enable_headphones=0 cgroup_disable=memory numa_policy=interleave nvme.max_host_mem_size_mb=32 snd_bcm2835.enable_headphones=1 snd_bcm2835.enable_hdmi=1 snd_bcm2835.enable_hdmi=0  numa=fake=2 system_heap.max_order=0 iommu_dma_numa_policy=interleave smsc95xx.macaddr=2C:CF:67:7E:4E:AA vc_mem.mem_base=0x3ec00000 vc_mem.mem_size=0x40000000  root=/dev/mmcblk0p2 rootfstype=ext4 fsck.repair=yes rootwait cfg80211.ieee80211_regdom=US quiet loglevel=0 logo.nologo vt.global_cursor_default=0 systemd.show_status=false console=tty3 splash plymouth.ignore-serial-consoles fbcon=map:2 video=HDMI-A-1:1024x600@60D
Jul 02 13:26:13 apex-pi kernel: usb 1-1.3: Product: USB PnP Audio Device
Jul 02 13:26:13 apex-pi kernel: input: Solid State System Co.,Ltd. USB PnP Audio Device as /devices/platform/scb/fd500000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/usb1/1-1/1-1.3/1-1.3:1.3/0003:0C76:1203.0001/input/input4
Jul 02 13:26:13 apex-pi kernel: hid-generic 0003:0C76:1203.0001: input,hidraw0: USB HID v1.00 Device [Solid State System Co.,Ltd. USB PnP Audio Device] on usb-0000:01:00.0-1.3/input3
Jul 02 13:26:14 apex-pi kernel: snd_bcm2835: module is from the staging directory, the quality is unknown, you have been warned.
Jul 02 13:26:14 apex-pi kernel: bcm2835-audio bcm2835-audio: card created with 8 channels
Jul 02 13:26:15 apex-pi kernel: bcm2835_codec: module is from the staging directory, the quality is unknown, you have been warned.
Jul 02 13:26:15 apex-pi kernel: bcm2835-codec bcm2835-codec: Device registered as /dev/video10
Jul 02 13:26:15 apex-pi kernel: bcm2835-codec bcm2835-codec: Loaded V4L2 decode
Jul 02 13:26:15 apex-pi kernel: bcm2835-codec bcm2835-codec: Device registered as /dev/video11
Jul 02 13:26:15 apex-pi kernel: bcm2835-codec bcm2835-codec: Loaded V4L2 encode
Jul 02 13:26:15 apex-pi kernel: bcm2835-codec bcm2835-codec: Device registered as /dev/video12
Jul 02 13:26:15 apex-pi kernel: bcm2835-codec bcm2835-codec: Loaded V4L2 isp
Jul 02 13:26:15 apex-pi kernel: bcm2835-codec bcm2835-codec: Device registered as /dev/video18
Jul 02 13:26:15 apex-pi kernel: bcm2835-codec bcm2835-codec: Loaded V4L2 image_fx
Jul 02 13:26:15 apex-pi kernel: bcm2835-codec bcm2835-codec: Device registered as /dev/video31
Jul 02 13:26:15 apex-pi kernel: bcm2835-codec bcm2835-codec: Loaded V4L2 encode_image
Jul 02 13:26:15 apex-pi kernel: rpi-hevc-dec feb00000.codec: Device registered as /dev/video19
Jul 02 13:26:15 apex-pi kernel: usbcore: registered new interface driver snd-usb-audio
Jul 02 13:26:23 apex-pi kernel: usb 1-1.4: USB disconnect, device number 9
Jul 02 13:26:33 apex-pi kernel: usb 1-1.4: USB disconnect, device number 10
Jul 02 13:26:43 apex-pi kernel: usb 1-1.4: USB disconnect, device number 11
Jul 02 13:27:14 apex-pi kernel: usb 1-1.4: USB disconnect, device number 12
Jul 02 13:27:24 apex-pi kernel: usb 1-1.4: USB disconnect, device number 13
Jul 02 13:27:34 apex-pi kernel: usb 1-1.4: USB disconnect, device number 14
Jul 02 13:27:44 apex-pi kernel: usb 1-1.4: USB disconnect, device number 15
Jul 02 13:27:54 apex-pi kernel: usb 1-1.4: USB disconnect, device number 16
Jul 02 13:28:04 apex-pi kernel: usb 1-1.4: USB disconnect, device number 17
Jul 02 13:28:14 apex-pi kernel: usb 1-1.4: USB disconnect, device number 18
Jul 02 13:28:24 apex-pi kernel: usb 1-1.4: USB disconnect, device number 19
Jul 02 13:28:34 apex-pi kernel: usb 1-1.4: USB disconnect, device number 20
Jul 02 13:28:44 apex-pi kernel: usb 1-1.4: USB disconnect, device number 21
Jul 02 13:28:54 apex-pi kernel: usb 1-1.4: USB disconnect, device number 22
Jul 02 13:29:04 apex-pi kernel: usb 1-1.4: USB disconnect, device number 23
Jul 02 13:29:14 apex-pi kernel: usb 1-1.4: USB disconnect, device number 24
Jul 02 13:29:24 apex-pi kernel: usb 1-1.4: USB disconnect, device number 25
Jul 02 13:29:34 apex-pi kernel: usb 1-1.4: USB disconnect, device number 26
Jul 02 13:29:44 apex-pi kernel: usb 1-1.4: USB disconnect, device number 27
Jul 02 13:29:51 apex-pi kernel: usb 1-1.3: USB disconnect, device number 7
Jul 02 13:29:54 apex-pi kernel: usb 1-1.3: Product: USB PnP Audio Device
Jul 02 13:29:54 apex-pi kernel: input: Solid State System Co.,Ltd. USB PnP Audio Device as /devices/platform/scb/fd500000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/usb1/1-1/1-1.3/1-1.3:1.3/0003:0C76:1203.0002/input/input5
Jul 02 13:29:54 apex-pi kernel: hid-generic 0003:0C76:1203.0002: input,hidraw0: USB HID v1.00 Device [Solid State System Co.,Ltd. USB PnP Audio Device] on usb-0000:01:00.0-1.3/input3
Jul 02 13:30:04 apex-pi kernel: usb 1-1.4: USB disconnect, device number 37
Jul 02 13:30:14 apex-pi kernel: usb 1-1.4: USB disconnect, device number 38
Jul 02 13:30:24 apex-pi kernel: usb 1-1.4: USB disconnect, device number 39
Jul 02 13:30:34 apex-pi kernel: usb 1-1.4: USB disconnect, device number 40
Jul 02 13:30:44 apex-pi kernel: usb 1-1.4: USB disconnect, device number 41
Jul 02 13:30:54 apex-pi kernel: usb 1-1.4: USB disconnect, device number 42
Jul 02 13:31:04 apex-pi kernel: usb 1-1.4: USB disconnect, device number 43
Jul 02 13:31:14 apex-pi kernel: usb 1-1.4: USB disconnect, device number 44
Jul 02 13:31:23 apex-pi kernel: usb 1-1.4: USB disconnect, device number 45
Jul 02 13:31:33 apex-pi kernel: usb 1-1.4: USB disconnect, device number 46
Jul 02 13:31:43 apex-pi kernel: usb 1-1.4: USB disconnect, device number 47
Jul 02 13:31:53 apex-pi kernel: usb 1-1.4: USB disconnect, device number 48
Jul 02 13:32:03 apex-pi kernel: usb 1-1.4: USB disconnect, device number 49
Jul 02 13:32:13 apex-pi kernel: usb 1-1.4: USB disconnect, device number 50
Jul 02 13:32:23 apex-pi kernel: usb 1-1.4: USB disconnect, device number 51
Jul 02 13:32:33 apex-pi kernel: usb 1-1.4: USB disconnect, device number 52
Jul 02 13:32:43 apex-pi kernel: usb 1-1.4: USB disconnect, device number 53
Jul 02 13:32:53 apex-pi kernel: usb 1-1.4: USB disconnect, device number 54
Jul 02 13:33:03 apex-pi kernel: usb 1-1.4: USB disconnect, device number 55
Jul 02 13:33:13 apex-pi kernel: usb 1-1.4: USB disconnect, device number 56
Jul 02 13:33:23 apex-pi kernel: usb 1-1.4: USB disconnect, device number 57
Jul 02 13:33:33 apex-pi kernel: usb 1-1.4: USB disconnect, device number 58
Jul 02 13:33:43 apex-pi kernel: usb 1-1.4: USB disconnect, device number 59
Jul 02 13:33:53 apex-pi kernel: usb 1-1.4: USB disconnect, device number 60
Jul 02 13:34:03 apex-pi kernel: usb 1-1.4: USB disconnect, device number 61
Jul 02 13:34:13 apex-pi kernel: usb 1-1.4: USB disconnect, device number 62
Jul 02 13:34:23 apex-pi kernel: usb 1-1.4: USB disconnect, device number 63
Jul 02 13:34:33 apex-pi kernel: usb 1-1.4: USB disconnect, device number 64
Jul 02 13:34:41 apex-pi kernel: usb 1-1.3: USB disconnect, device number 36
Jul 02 13:34:44 apex-pi kernel: usb 1-1.3: Product: USB PnP Audio Device
Jul 02 13:34:44 apex-pi kernel: input: Solid State System Co.,Ltd. USB PnP Audio Device as /devices/platform/scb/fd500000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/usb1/1-1/1-1.3/1-1.3:1.3/0003:0C76:1203.0003/input/input6
Jul 02 13:34:44 apex-pi kernel: hid-generic 0003:0C76:1203.0003: input,hidraw0: USB HID v1.00 Device [Solid State System Co.,Ltd. USB PnP Audio Device] on usb-0000:01:00.0-1.3/input3
Jul 02 13:34:53 apex-pi kernel: usb 1-1.4: USB disconnect, device number 74
Jul 02 13:35:03 apex-pi kernel: usb 1-1.4: USB disconnect, device number 75
Jul 02 13:35:13 apex-pi kernel: usb 1-1.4: USB disconnect, device number 76
Jul 02 13:35:23 apex-pi kernel: usb 1-1.4: USB disconnect, device number 77
Jul 02 13:35:33 apex-pi kernel: usb 1-1.4: USB disconnect, device number 78
Jul 02 13:35:43 apex-pi kernel: usb 1-1.4: USB disconnect, device number 79
Jul 02 13:35:53 apex-pi kernel: usb 1-1.4: USB disconnect, device number 80
Jul 02 13:36:00 apex-pi kernel: usb 1-1.3: USB disconnect, device number 73
Jul 02 13:36:03 apex-pi kernel: usb 1-1.3: Product: USB PnP Audio Device
Jul 02 13:36:03 apex-pi kernel: input: Solid State System Co.,Ltd. USB PnP Audio Device as /devices/platform/scb/fd500000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/usb1/1-1/1-1.3/1-1.3:1.3/0003:0C76:1203.0004/input/input7
Jul 02 13:36:03 apex-pi kernel: hid-generic 0003:0C76:1203.0004: input,hidraw0: USB HID v1.00 Device [Solid State System Co.,Ltd. USB PnP Audio Device] on usb-0000:01:00.0-1.3/input3
Jul 02 13:36:20 apex-pi kernel: usb 1-1.3: USB disconnect, device number 89
Jul 02 13:36:20 apex-pi kernel: usb 1-1.4: USB disconnect, device number 90
Jul 02 13:36:23 apex-pi kernel: usb 1-1.3: Product: USB PnP Audio Device
Jul 02 13:36:23 apex-pi kernel: input: Solid State System Co.,Ltd. USB PnP Audio Device as /devices/platform/scb/fd500000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/usb1/1-1/1-1.3/1-1.3:1.3/0003:0C76:1203.0005/input/input8
Jul 02 13:36:23 apex-pi kernel: hid-generic 0003:0C76:1203.0005: input,hidraw0: USB HID v1.00 Device [Solid State System Co.,Ltd. USB PnP Audio Device] on usb-0000:01:00.0-1.3/input3
Jul 02 13:36:33 apex-pi kernel: usb 1-1.4: USB disconnect, device number 96
Jul 02 13:36:43 apex-pi kernel: usb 1-1.4: USB disconnect, device number 97
Jul 02 13:36:53 apex-pi kernel: usb 1-1.4: USB disconnect, device number 98
Jul 02 13:37:10 apex-pi kernel: usb 1-1.3: USB disconnect, device number 95
Jul 02 13:37:10 apex-pi kernel: usb 1-1.4: USB disconnect, device number 99
Jul 02 13:37:13 apex-pi kernel: usb 1-1.3: Product: USB PnP Audio Device
Jul 02 13:37:13 apex-pi kernel: input: Solid State System Co.,Ltd. USB PnP Audio Device as /devices/platform/scb/fd500000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/usb1/1-1/1-1.3/1-1.3:1.3/0003:0C76:1203.0006/input/input9
Jul 02 13:37:13 apex-pi kernel: hid-generic 0003:0C76:1203.0006: input,hidraw0: USB HID v1.00 Device [Solid State System Co.,Ltd. USB PnP Audio Device] on usb-0000:01:00.0-1.3/input3
Jul 02 13:37:23 apex-pi kernel: usb 1-1.4: USB disconnect, device number 105
```

### Follow-up — isolating port 1-1.4 specifically

```
$ dmesg | grep -iE "1-1\.4" | head -50

[    4.505647] usb 1-1.4: new full-speed USB device number 8 using xhci_hcd
[    4.582090] usb 1-1.4: Device not responding to setup address.
[    4.785853] usb 1-1.4: Device not responding to setup address.
[    4.989654] usb 1-1.4: device not accepting address 8, error -71
[    4.990012] usb 1-1.4: WARN: invalid context state for evaluate context command.
[    5.265659] usb 1-1.4: new full-speed USB device number 9 using xhci_hcd
[    5.357535] usb 1-1.4: New USB device found, idVendor=4348, idProduct=55e0, bcdDevice= 2.50
[    5.357560] usb 1-1.4: New USB device strings: Mfr=0, Product=0, SerialNumber=0
[   14.673257] usb 1-1.4: USB disconnect, device number 9
[   14.874420] usb 1-1.4: new full-speed USB device number 10 using xhci_hcd
[   14.969419] usb 1-1.4: New USB device found, idVendor=4348, idProduct=55e0, bcdDevice= 2.50
[   14.969438] usb 1-1.4: New USB device strings: Mfr=0, Product=0, SerialNumber=0
[   24.658651] usb 1-1.4: USB disconnect, device number 10
[   24.861656] usb 1-1.4: new full-speed USB device number 11 using xhci_hcd
[   24.953520] usb 1-1.4: New USB device found, idVendor=4348, idProduct=55e0, bcdDevice= 2.50
[   24.953550] usb 1-1.4: New USB device strings: Mfr=0, Product=0, SerialNumber=0
[   34.645294] usb 1-1.4: USB disconnect, device number 11
[   34.849637] usb 1-1.4: new full-speed USB device number 12 using xhci_hcd
[   34.943593] usb 1-1.4: New USB device found, idVendor=4348, idProduct=55e0, bcdDevice= 2.50
[   34.943617] usb 1-1.4: New USB device strings: Mfr=0, Product=0, SerialNumber=0
[   44.629870] usb 1-1.4: USB disconnect, device number 12
[   44.833628] usb 1-1.4: new full-speed USB device number 13 using xhci_hcd
[   44.925544] usb 1-1.4: New USB device found, idVendor=4348, idProduct=55e0, bcdDevice= 2.50
[   44.925566] usb 1-1.4: New USB device strings: Mfr=0, Product=0, SerialNumber=0
[   54.614915] usb 1-1.4: USB disconnect, device number 13
[   54.813635] usb 1-1.4: new full-speed USB device number 14 using xhci_hcd
[   54.907522] usb 1-1.4: New USB device found, idVendor=4348, idProduct=55e0, bcdDevice= 2.50
[   54.907545] usb 1-1.4: New USB device strings: Mfr=0, Product=0, SerialNumber=0
[   64.600371] usb 1-1.4: USB disconnect, device number 14
[   64.809685] usb 1-1.4: new full-speed USB device number 15 using xhci_hcd
[   64.904420] usb 1-1.4: New USB device found, idVendor=4348, idProduct=55e0, bcdDevice= 2.50
[   64.904442] usb 1-1.4: New USB device strings: Mfr=0, Product=0, SerialNumber=0
[   74.586427] usb 1-1.4: USB disconnect, device number 15
[   74.789335] usb 1-1.4: new full-speed USB device number 16 using xhci_hcd
[   74.884383] usb 1-1.4: New USB device found, idVendor=4348, idProduct=55e0, bcdDevice= 2.50
[   74.884409] usb 1-1.4: New USB device strings: Mfr=0, Product=0, SerialNumber=0
[   84.570781] usb 1-1.4: USB disconnect, device number 16
[   84.776978] usb 1-1.4: new full-speed USB device number 17 using xhci_hcd
[   84.870089] usb 1-1.4: New USB device found, idVendor=4348, idProduct=55e0, bcdDevice= 2.50
[   84.870111] usb 1-1.4: New USB device strings: Mfr=0, Product=0, SerialNumber=0
[   94.301089] usb 1-1.4: USB disconnect, device number 17
[   94.500787] usb 1-1.4: new full-speed USB device number 18 using xhci_hcd
[   94.605761] usb 1-1.4: New USB device found, idVendor=4348, idProduct=55e0, bcdDevice= 2.50
[   94.605782] usb 1-1.4: New USB device strings: Mfr=0, Product=0, SerialNumber=0
[  104.285540] usb 1-1.4: USB disconnect, device number 18
[  104.496689] usb 1-1.4: new full-speed USB device number 19 using xhci_hcd
[  104.589945] usb 1-1.4: New USB device found, idVendor=4348, idProduct=55e0, bcdDevice= 2.50
[  104.589976] usb 1-1.4: New USB device strings: Mfr=0, Product=0, SerialNumber=0
[  114.271322] usb 1-1.4: USB disconnect, device number 19
[  114.476718] usb 1-1.4: new full-speed USB device number 20 using xhci_hcd
```

**Reading:** Port **1-1.4** never enumerates cleanly — it fails setup twice at boot (`error -71`), then settles into a disconnect/reconnect cycle every **~10 seconds**, uninterrupted for the entire capture window. Port **1-1.3** (the Waveshare speaker) is stable by comparison but still drops at irregular intervals of 90s–580s apart — consistent with being an innocent bystander to whatever 1-1.4 is doing to the shared connection, rather than failing on its own.

## USB Topology

Both devices share the same parent — visible directly in the device naming (`1-1.3` and `1-1.4` both hang off hub `1-1`) and confirmed with `lsusb -t`:

```
$ lsusb -t

/:  Bus 001.Port 001: Dev 001, Class=root_hub, Driver=xhci_hcd/1p, 480M
    |__ Port 001: Dev 002, If 0, Class=Hub, Driver=hub/4p, 480M
        |__ Port 003: Dev 104, If 0, Class=Audio, Driver=snd-usb-audio, 12M      <- speaker (stable-ish)
        |__ Port 003: Dev 104, If 1, Class=Audio, Driver=snd-usb-audio, 12M
        |__ Port 003: Dev 104, If 2, Class=Audio, Driver=snd-usb-audio, 12M
        |__ Port 003: Dev 104, If 3, Class=Human Interface Device, Driver=usbhid, 12M
        |__ Port 004: Dev 109, If 0, Class=Vendor Specific Class, Driver=[none], 12M   <- flapping every ~10s
/:  Bus 002.Port 001: Dev 001, Class=root_hub, Driver=xhci_hcd/4p, 5000M
```

```
$ lsusb

Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 001 Device 002: ID 2109:3431 VIA Labs, Inc. Hub
Bus 001 Device 104: ID 0c76:1203 JMTek, LLC. USB PnP Audio Device
Bus 001 Device 109: ID 4348:55e0 WinChipHead
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
```

Per your correction mid-investigation: the four USB-A ports on the Pi are used as (1) the ReSpeaker Lite microphone, alone, on its own port, and (2) the Waveshare speaker's data cable **and** the display's power lead physically stacked into a single port. That single stacked connection is exactly what produces the "1-1" hub above — the splitter/adapter combining those two cables *is* the shared hub. The unidentified `WinChipHead 4348:55e0` device (vendor-specific, no driver bound, `Mfr=0, Product=0` — it never even reports a name) is almost certainly part of the display's power/control board, riding along on that same stacked connection.

Re-checked later in the same session:

```
$ lsusb -t
$ lsusb

/:  Bus 001.Port 001: Dev 001, Class=root_hub, Driver=xhci_hcd/1p, 480M
    |__ Port 001: Dev 002, If 0, Class=Hub, Driver=hub/4p, 480M
        |__ Port 003: Dev 085, If 0, Class=Audio, Driver=snd-usb-audio, 12M
        |__ Port 003: Dev 085, If 1, Class=Audio, Driver=snd-usb-audio, 12M
        |__ Port 003: Dev 085, If 2, Class=Audio, Driver=snd-usb-audio, 12M
        |__ Port 003: Dev 085, If 3, Class=Human Interface Device, Driver=usbhid, 12M
        |__ Port 004: Dev 095, If 0, Class=Vendor Specific Class, Driver=[none], 12M
/:  Bus 002.Port 001: Dev 001, Class=root_hub, Driver=xhci_hcd/4p, 5000M

Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 001 Device 002: ID 2109:3431 VIA Labs, Inc. Hub
Bus 001 Device 085: ID 0c76:1203 JMTek, LLC. USB PnP Audio Device
Bus 001 Device 095: ID 4348:55e0 WinChipHead
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
```

Device numbers had climbed from 104/109 to 085/095 by the time of the re-check (USB device numbers wrap at 127) — confirming the 1-1.4 disconnect/reconnect cycle had been running continuously, uninterrupted, since boot.

```
$ cat /proc/asound/cards

 0 [vc4hdmi0       ]: vc4-hdmi - vc4-hdmi-0
                      vc4-hdmi-0
 1 [vc4hdmi1       ]: vc4-hdmi - vc4-hdmi-1
                      vc4-hdmi-1
 2 [Headphones     ]: bcm2835_headpho - bcm2835 Headphones
                      bcm2835 Headphones
 3 [Device         ]: USB-Audio - USB PnP Audio Device
                      Solid State System Co.,Ltd. USB PnP Audio Device at usb-0000:01:00.0-1.3, full
```

Only one USB sound card is present system-wide, tied to port `1-1.3` — the ReSpeaker Lite microphone did not appear in the device list at all during this session. That's a separate, open question worth checking independently; it wasn't chased further here since it falls outside the reported speaker symptom.

## Root Cause — Hardware

**Finding:** The Waveshare speaker and a device from the display's power/control board are wired through the same physical connection (a stacked USB splitter carrying both the speaker's data cable and the display's power lead). The display-side device fails to enumerate and disconnects every ~10 seconds, continuously, and appears to be destabilizing that shared connection — which is the most likely explanation for the speaker's own, less frequent, irregular dropouts. This has not yet been physically remediated.

**Recommended action:** un-stack the connection. Two of the Pi's four native USB ports are currently free (only the mic and the stacked combo are in use) — move either the Waveshare speaker or the display's power lead onto its own dedicated port so the speaker no longer shares a connection with the misbehaving device.

## Root Cause — Software

Independent of the hardware issue, the application has no way to recover once the speaker does drop. `src/main.py` already contains a routine built for exactly this scenario:

```python
while self._conversation_active.is_set():
    # Check for fatal errors in audio components and recover
    if self.audio_player.has_fatal_error:
        logger.warning("♻️  FATAL AUDIO ERROR - Recreating system...")
        self._recreate_audio_system()
```

**Finding:** `has_fatal_error` (defined in `src/audio/playback.py`) was set to `False` in two places and set to `True` nowhere in the codebase. The recovery routine above is unreachable — when the speaker disconnects mid-playback, PortAudio silently marks the output stream inactive, nothing detects that, and the bot goes quiet until the process is restarted by hand.

## Fix Applied

Added a background watchdog to `AudioPlayer` that polls the output stream while it should be playing. The invariant it relies on: every intentional stop path in this class (the stop-event/abort branch and the sentinel/graceful-drain branch in `_audio_callback`, and `stop_streaming()` itself) always flips `_is_playing` to `False` *before* the stream goes inactive. So `_is_playing == True` while the stream reports inactive can only mean PortAudio killed the stream on its own — i.e. the device disappeared.

```python
# src/audio/playback.py

self.has_fatal_error = False
self._owns_stream = True
self._watchdog_thread = None
self._watchdog_stop = threading.Event()
# ...
def _watchdog_loop(self) -> None:
    while not self._watchdog_stop.is_set():
        time.sleep(0.3)
        stream = self.audio_stream
        if not self._is_playing or stream is None:
            continue
        try:
            active = stream.is_active()
        except Exception:
            active = False
        if not active:
            self.has_fatal_error = True
            self._is_playing = False
# ...
self.audio_stream.start_stream()
if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
    self._watchdog_thread = threading.Thread(
        target=self._watchdog_loop, daemon=True, name="AudioPlayerWatchdog"
    )
    self._watchdog_thread.start()
# ...
def shutdown(self) -> None:
    self._watchdog_stop.set()
    self.cleanup()
```

With this in place, a mid-playback disconnect now flips `has_fatal_error` within ~0.3s, and `main.py`'s existing (previously dead) recovery loop rebuilds the whole audio stack automatically.

### Verification

Compiled clean, and exercised with a stand-in stream object standing in for PortAudio to confirm the watchdog stays quiet on a healthy stream and fires exactly once when the stream is flipped inactive mid-playback:

```
🔴 Output stream died unexpectedly (device disconnected?) — flagging fatal error
watchdog logic OK
```

**Outcome:** The dead recovery path is now wired up. This resolves the "requires a manual restart" failure mode. It does not fix the underlying hardware flakiness — that still needs the physical re-wiring described above.

## Outstanding

| Item | Owner | Notes |
|---|---|---|
| Move the display's power lead or the Waveshare speaker onto its own free Pi USB port | Physical | Two of four native ports are currently unused; un-stacking removes the shared-hub dependency entirely. |
| Confirm why the ReSpeaker Lite mic didn't appear in `/proc/asound/cards` during this session | Follow-up | Not chased here — outside the reported speaker symptom, but worth a separate look. |

---

Commands run: `dmesg`, `journalctl -k`, `lsusb`, `lsusb -t`, `udevadm info`, `cat /proc/asound/cards`. Code change: `src/audio/playback.py` on branch `openwakeword`.
