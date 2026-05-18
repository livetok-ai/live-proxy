import argparse
import asyncio
import os

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def run_client(port: int):
    server_url = f"http://localhost:{port}/connection"
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        print("Error: GOOGLE_API_KEY not found in environment or .env file.")
        return

    print("Initializing RTCPeerConnection...")
    pc = RTCPeerConnection()

    # Load the fake audio and video from the downloaded MP4 file
    mp4_path = os.path.join(os.path.dirname(__file__), "sample.mp4")
    if not os.path.exists(mp4_path):
        print(f"Error: Sample MP4 file not found at: {mp4_path}")
        return

    print(f"Loading media from {mp4_path}...")
    player = MediaPlayer(mp4_path)

    # Add audio and video tracks to the peer connection
    if player.audio:
        print("Adding audio track...")
        pc.addTrack(player.audio)
    else:
        print("Warning: No audio track found in MP4.")

    if player.video:
        print("Adding video track...")
        pc.addTrack(player.video)
    else:
        print("Warning: No video track found in MP4.")

    # Create the WebRTC offer
    print("Creating local offer...")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Payload for the proxy connection API
    payload = {
        "sdp": pc.localDescription.sdp,
        "model": "gemini-2.5-flash-native-audio-latest;yolo",
        "system_instructions": "You are a helpful assistant. Please transcribe or answer any speech you hear.",
        "voice": "Puck",
        "language": "en-US",
        "api_key": google_api_key,
    }

    print(f"Sending POST request to {server_url}...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(server_url, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    print(f"Error starting connection: HTTP {response.status} - {text}")
                    await pc.close()
                    return

                res_data = await response.json()
                answer_sdp = res_data["sdp"]
                connection_id = res_data["id"]
                print(f"Connection created successfully! ID: {connection_id}")

                # Set the remote answer
                answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
                await pc.setRemoteDescription(answer)
                print("Remote description set successfully. Streaming...")

                # Wait and stream for 30 seconds to let the model hear and respond
                print("Streaming for 30 seconds... Please check proxy logs for transcriptions.")
                for i in range(30):
                    await asyncio.sleep(1)
                    if i % 5 == 0:
                        print(f"Streaming... {30 - i} seconds remaining.")
    except aiohttp.ClientConnectorError as e:
        print(
            f"Connection failed: Could not connect to proxy server at {server_url}. Is the server running? Details: {e}"
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Close connections
        print("Closing WebRTC peer connection...")
        await pc.close()
        print("Client stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC test client for live-proxy")
    parser.add_argument("--port", type=int, default=8085, help="Port of the live-proxy server (default: 8085)")
    args = parser.parse_args()
    asyncio.run(run_client(args.port))
