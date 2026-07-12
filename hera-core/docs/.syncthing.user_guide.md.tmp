# Hera Core: User Guide

Welcome to the **Hera Core** component. Hera is the Sovereign Execution Engine of the Vilaros ecosystem.

## How it works for Users
As a Vilaros user, Hera handles all the artificial intelligence and heavy lifting natively on your hardware. Unlike legacy setups that connect to OpenAI or Anthropic across the internet, Hera uses **Open Source Models** (like Llama, Qwen, Flux, and Whisper) completely privately on your device.

When you type a prompt in the Vilaros interface, or when the Agent (Ava/Imaginclaw) decides to generate an image, it sends a pure binary protocol payload straight to Hera over a blazingly fast Unix socket (`/tmp/hera-core.sock`). Hera uses its native Rust processing power (`candle-core`) to run those models on your GPU in milliseconds.

## Headless Execution
In previous architectures, Hera ran as a sluggish Web Server that processed standard HTTP packets. In this latest evolution, Hera has shed the Web Server entirely. 

Hera now runs as a **Headless PURE IPC Daemon**. It opens zero network ports, which makes it totally immune to traditional network breaches or port-scanning. Try looking for it on `localhost:3305` — it's not there! It is perfectly invisible, and only internal system actors (like Vilaros OS itself) can speak to it.

## Troubleshooting
If you ever experience hanging generations or failed image requests:
1. Make sure Hera is running via `pm2 ls`.
2. Ensure the daemon has file permissions in `/tmp` by running `ls -la /tmp/hera-core.sock`. 
3. If the socket file is missing or corrupted by a dirty shutdown, restart the server gracefully: `pm2 restart hera-core`. Hera automatically cleans up and recreates its socket on boot.
