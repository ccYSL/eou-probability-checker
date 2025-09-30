import fetch from 'node-fetch';
import { writeFile } from 'node:fs/promises';
import path from 'path';

const url = 'https://huggingface.co/livekit/turn-detector/resolve/main/model_quantized.onnx';
const dest = path.join('src', 'model_quantized.onnx');

async function downloadModel() {
  console.log(`Downloading model from ${url} ...`);
  
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch model: ${res.status} ${res.statusText}`);
  }

  const buffer = await res.arrayBuffer();
  await writeFile(dest, Buffer.from(buffer));
  console.log(`Model saved to ${dest}`);
}

downloadModel().catch(err => {
  console.error(err);
  process.exit(1);
});
