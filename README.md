# EOU (End of utterance) Probability Checker
![npm](https://img.shields.io/npm/v/eou-probability-checker)
![downloads](https://img.shields.io/npm/dm/eou-probability-checker)
![license](https://img.shields.io/npm/l/eou-probability-checker)
![styled with](https://img.shields.io/badge/styled_with-prettier-ff69b4.svg)
### Simple End of Utterance (EOU) detection using LiveKit's ML model. Tells you the probability that someone is done speaking.
## Install

```bash
npm install eou-probability-checker
```

## Downloading the ML Model
#### If the model isnt in the src directory then you can download it using:
```bash
npm run download-model
```
## Usage
```typescript
import getEOU from 'eou-probability-checker';

const chat = [
  { role: "user", content: "Hello, how are you?" },
  { role: "assistant", content: "Hi! I'm good, thanks. How can I help you today?" },
  { role: "user", content: "I was wondering if you could tell me a joke" },
];

// Gets EOU probability for the last message, uses prior messages for accurate results.
const probability = await getEOU(chat);
console.log(probability); // 0-1
```

Higher numbers = more likely they're done speaking.

## Credits

Uses [LiveKit's turn detector model](https://huggingface.co/livekit/turn-detector)