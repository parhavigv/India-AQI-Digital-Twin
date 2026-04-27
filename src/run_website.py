#!/usr/bin/env python3
"""
India AQI Digital Twin v6.0 — Local Server Launcher
Run this file to start the website on your computer.
"""
import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8080
HTML_FILE = "india_aqi_digital_twin_v6.html"

# Make sure server runs from the folder containing the HTML file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if not os.path.exists(HTML_FILE):
    print(f"ERROR: '{HTML_FILE}' not found in {script_dir}")
    print("Make sure both files are in the same folder.")
    input("Press Enter to exit...")
    sys.exit(1)

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress request logs for clean output

print("=" * 50)
print("  India AQI Digital Twin v6.0")
print("=" * 50)
print(f"  Starting server on http://localhost:{PORT}")
print(f"  Opening browser automatically...")
print(f"  Press Ctrl+C to stop the server.")
print("=" * 50)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    url = f"http://localhost:{PORT}/{HTML_FILE}"
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped. Goodbye!")
