#!/usr/bin/env python3
import os
import time
import gi
import signal
import sys
import subprocess
import threading

REPO_PATH = '/home/tutor/Ai-Agent'
PID_FILE = "splash.pid"

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

def signal_handler(sig, frame):
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    Gtk.main_quit()
    sys.exit(0)

class SplashScreen(Gtk.Window):
    def __init__(self):
        with open(PID_FILE, "w") as f:
             f.write(str(os.getpid()))

        super().__init__(title="Loading")
        self.fullscreen()
        self.set_decorated(False)
        
        # 1. CSS for Background, Text, AND the new Wi-Fi Warning
        screen = Gdk.Screen.get_default()
        provider = Gtk.CssProvider()
        style_context = Gtk.StyleContext()
        style_context.add_provider_for_screen(screen, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        
        css = b"""
        window { background-color: black; }
        label { color: white; font-size: 48px; font-weight: bold; }
        #wifi-warning { 
            color: red; 
            background-color: rgba(0, 0, 0, 0.8); 
            font-size: 36px;
            padding: 10px;
        }
        """
        provider.load_from_data(css)

        # 2. Overlay Container
        overlay = Gtk.Overlay()
        self.add(overlay)

        # 3. Load Background Image Safely
        img_path = "/home/tutor/Downloads/JARVIS.png" 
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                filename=img_path,
                width=1024,
                height=600,
                preserve_aspect_ratio=True
            )
            background_image = Gtk.Image.new_from_pixbuf(pixbuf)
            overlay.add(background_image)
        except Exception as e:
            print(f"Image load failed: {e}")

        # 4. Foreground UI (Spinner and Init Text)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=30)
        vbox.set_valign(Gtk.Align.CENTER)
        vbox.set_halign(Gtk.Align.CENTER)

        self.spinner = Gtk.Spinner()
        self.spinner.set_size_request(100, 100)
        self.spinner.start()

        self.label = Gtk.Label(label="Initializing System...")
        
        vbox.pack_start(self.spinner, False, False, 0)
        vbox.pack_start(self.label, False, False, 0)
        overlay.add_overlay(vbox)

        # ---> NEW: WI-FI WARNING LABEL <---
        self.wifi_warning = Gtk.Label(label="NO WI-FI CONNECTION")
        self.wifi_warning.set_name("wifi-warning") # Hooks into the CSS above
        # Position it near the top center of the screen
        self.wifi_warning.set_valign(Gtk.Align.START)
        self.wifi_warning.set_halign(Gtk.Align.CENTER)
        self.wifi_warning.set_margin_top(20)
        
        # Add to the overlay and hide it by default
        overlay.add_overlay(self.wifi_warning)
        self.wifi_warning.hide()
        
        self.show_all()
        # Keep it hidden immediately after show_all() forces everything visible
        self.wifi_warning.hide() 

        # 5. Start the background system tasks
        update_thread = threading.Thread(target=self.run_background_tasks)
        update_thread.daemon = True
        update_thread.start()

        # ---> NEW: START GTK WI-FI MONITOR <---
        # This tells GTK to run this function every 1000ms natively
        GLib.timeout_add_seconds(1, self.check_wifi_state)


    # ---> NEW: WI-FI CHECK FUNCTION <---
    def check_wifi_state(self):
        """ Checks the hardware state file and updates the warning label."""
        state_file = '/sys/class/net/wlan0/operstate'
        is_connected = False
        
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    if f.read().strip() == "up":
                        is_connected = True
        except Exception:
            pass

        # Toggle UI visibility
        if is_connected:
            self.wifi_warning.hide()
        else:
            self.wifi_warning.show()
            
        return True # Returning True tells GLib to keep running this timer


    # Safely updates the label from the background thread
    def update_label(self, text):
        self.label.set_text(text)
        return False

    # The background worker
    def run_background_tasks(self):
        env = os.environ.copy()
        env["DEBIAN_FRONTEND"] = "noninteractive"
    
        try:
            GLib.idle_add(self.update_label, "Checking for system updates...")
            subprocess.run(["sudo", "apt-get", "update", "-qq"], check=True, env=env)
            
            sim_result = subprocess.run(["sudo", "apt-get", "-s", "upgrade"], capture_output=True, text=True, env=env)
            needs_update = any(line.startswith("Inst") for line in sim_result.stdout.splitlines())
            
            if needs_update:
                GLib.idle_add(self.update_label, "Installing system updates...")
                subprocess.run(["sudo", "apt-get", "full-upgrade", "-y"], check=True, env=env)
                subprocess.run(["sudo", "apt-get", "dist-upgrade", "-y"], check=True, env=env)
                subprocess.run(["sudo", "apt-get", "auto-remove", "-y"], check=True, env=env)
                GLib.idle_add(self.update_label, "Rebooting to apply system updates...")
                time.sleep(1)
                subprocess.run(["sudo", "reboot"], check=True)
                return 
            else:
                GLib.idle_add(self.update_label, "No system updates found. Checking AI...")
            
            subprocess.run(['git', 'fetch'], cwd=REPO_PATH, check=True)
            status = subprocess.run(
                   ['git', 'status', '-uno'], 
                   cwd=REPO_PATH, 
                   capture_output=True, 
                   text=True
            )
            
            if "Your branch is behind" in status.stdout:
                GLib.idle_add(self.update_label, "Installing AI updates...")
                subprocess.run(['git', 'pull'], cwd=REPO_PATH, check=True)
                GLib.idle_add(self.update_label, "Updates complete. Booting Jarvis...")
            else:
                GLib.idle_add(self.update_label, "No AI updates found. Booting Jarvis...")
                
        except subprocess.CalledProcessError as e:
            GLib.idle_add(self.update_label, "An error occurred during boot.")
            print(f"Subprocess failed: {e}")

# Register the signal
signal.signal(signal.SIGTERM, signal_handler)

win = SplashScreen()
win.connect("destroy", Gtk.main_quit)
Gtk.main()
