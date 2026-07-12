#!/usr/bin/env python3
"""
Test script for interacting directly with Hera's native IPC server.
Bypasses HTTP/SwarmUI and talks directly to the candle-based FluxEngine via SOL payload.
"""
import socket
import json
import base64
import argparse

def test_draw(prompt, output_file, width=512, height=512):
    payload = {
        "action": "generate_image",
        "payload": {
            "prompt": prompt,
            "width": width,
            "height": height
        }
    }
    
    print(f"Connecting to Hera IPC socket at /tmp/hera-core.sock...")
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect("/tmp/hera-core.sock")
        
        print(f"Sending native SOL payload for prompt: '{prompt}'")
        client.sendall(json.dumps(payload).encode('utf-8') + b'\n')

        print("Waiting for Hera FluxEngine to complete Generation...")
        print("(This may take time depending on GPU and model quantization)")
        
        response_data = b""
        while True:
            chunk = client.recv(8192)
            if not chunk:
                break
            response_data += chunk
            if b'\n' in chunk:
                break

        resp = json.loads(response_data.decode('utf-8'))
        print(f"\nHera responded with status: {resp.get('status')}")
        
        if resp.get("status") == "success" and "data" in resp:
            print(f"Success! Image data received natively.")
            b64str = resp["data"].get("url", "")
            if "," in b64str:
                b64str = b64str.split(",", 1)[1]
            with open(output_file, "wb") as f:
                f.write(base64.b64decode(b64str))
            print(f"Saved generated image to {output_file}")
        else:
            print("Hera Engine Error:", json.dumps(resp, indent=2))
            
    except Exception as e:
        print("Socket/IPC error:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hera Native IPC Flux Test")
    parser.add_argument("--prompt", type=str, default="A highly detailed rendering of a futuristic geometric object, vivid colors, neon lights", help="Prompt for image generation")
    parser.add_argument("--output", type=str, default="hera_native_output.png", help="Path to save the output image")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    
    args = parser.parse_args()
    test_draw(args.prompt, args.output, args.width, args.height)
