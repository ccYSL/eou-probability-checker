import { AutoTokenizer } from "@huggingface/transformers";
import ort from "onnxruntime-node";
import path from "node:path"
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const modelPath = path.join(__dirname, "model_quantized.onnx");
const session = await ort.InferenceSession.create(modelPath);
const tokenizer = await AutoTokenizer.from_pretrained("livekit/turn-detector");

type ChatItem = {role: "user" | "assistant", content: string}

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map((x) => Math.exp(x - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map((x) => x / sumExp);
}

function getUnicodeCategory(char: string): string {
  const code = char.codePointAt(0);
  if (!code) return '';

  // Basic punctuation ranges (simplified version to match Python unicodedata.category)
  if (
    (code >= 0x21 && code <= 0x2f) || // !"#$%&'()*+,-./
    (code >= 0x3a && code <= 0x40) || // :;<=>?@
    (code >= 0x5b && code <= 0x60) || // [\]^_`
    (code >= 0x7b && code <= 0x7e) || // {|}~
    (code >= 0xa0 && code <= 0xbf) || // Latin-1 punctuation
    (code >= 0x2000 && code <= 0x206f) || // General punctuation
    (code >= 0x3000 && code <= 0x303f)
  ) {
    // CJK symbols and punctuation
    return 'P';
  }
  return '';
}

function normalizeText(text: string): string {
  if (!text) return '';

  let normalized = text.toLowerCase().normalize('NFKC');

  // Remove punctuation except apostrophes and hyphens
  // Using character-by-character approach to match Python logic
  normalized = Array.from(normalized)
    .filter((ch) => {
      const category = getUnicodeCategory(ch);
      return !(category.startsWith('P') && ch !== "'" && ch !== '-');
    })
    .join('');

  // Collapse whitespace and trim
  normalized = normalized.replace(/\s+/g, ' ').trim();

  return normalized;
}

function normalizeChat(items: ChatItem[]): ChatItem[] {
  const normalized: ChatItem[] = [];
  let lastMsg: ChatItem | undefined;

  for (const msg of items) {
    const content = normalizeText(msg.content);
    if (!content) continue;

    if (lastMsg && lastMsg.role === msg.role) {
      lastMsg.content += ` ${content}`;
    } else {
      lastMsg = { role: msg.role, content };
      normalized.push(lastMsg);
    }
  }

  return normalized;
}

export default async function getEOU(chat: ChatItem[]): Promise<number> {
  let properInput = tokenizer.apply_chat_template(normalizeChat(chat), {tokenize: false}) as string;
  properInput = properInput.slice(0, properInput.lastIndexOf('<|im_end|>'))

  const inputIds = tokenizer.encode(properInput)

  const inputTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from(inputIds.map((x) => BigInt(x))),
    [1, inputIds.length]
  );
  
  // Run inference
  const result = await session.run({ input_ids: inputTensor });
  const logits = result.logits;
  
  const seqLen = logits.dims[1];
  const vocabSize = logits.dims[2];
  
  // Calculate the starting index for the last token's logits
  const lastTokenStart = (seqLen - 1) * vocabSize;
  const lastTokenLogits: number[] = [];
  
  for (let i = 0; i < vocabSize; i++) {
    lastTokenLogits.push(logits.data[lastTokenStart + i] as number);
  }

  // Apply softmax
  const probs = softmax(lastTokenLogits);

  // Get the EOU token ID (should be the same as <|im_end|>)
  const eouTokenIds = tokenizer.encode('<|im_end|>');
  const eouTokenId = eouTokenIds[eouTokenIds.length - 1];
  
  const eouProb = parseFloat(probs[eouTokenId].toFixed(4))
  return eouProb
}