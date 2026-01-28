import { GoogleGenAI, Type } from "@google/genai";
import { ModelId, ImageGenSize, SlideContent, PresentationStructure, Attachment } from "../types";

// Helper to ensure API key selection for paid features
const ensureApiKey = async () => {
  if (window.aistudio && window.aistudio.hasSelectedApiKey && window.aistudio.openSelectKey) {
    const hasKey = await window.aistudio.hasSelectedApiKey();
    if (!hasKey) {
      await window.aistudio.openSelectKey();
      // Re-check after dialog
      const hasKeyNow = await window.aistudio.hasSelectedApiKey();
      if (!hasKeyNow) {
        throw new Error("API Key selection is required for this feature.");
      }
    }
  }
};

const getAI = () => new GoogleGenAI({ apiKey: process.env.API_KEY });

export const streamChat = async (
  modelId: string,
  history: { role: string; parts: any[] }[],
  message: string,
  attachments: Attachment[],
  grounding: { search: boolean }
) => {
  const ai = getAI();
  
  const tools: any[] = [];
  if (grounding.search) {
    tools.push({ googleSearch: {} });
  }

  // Construct current turn parts
  const currentParts: any[] = [];
  
  // Handle attachments
  for (const att of attachments) {
    if (att.mimeType === 'application/pdf') {
      // Native PDF support
      currentParts.push({
        inlineData: {
          mimeType: 'application/pdf',
          data: att.data
        }
      });
    } else if (att.mimeType.startsWith('image/')) {
      // Native Image support
      currentParts.push({
        inlineData: {
          mimeType: att.mimeType,
          data: att.data
        }
      });
    } else {
      // Text-based files (Docx, Excel converted to text)
      currentParts.push({
        text: `\n[Context from file "${att.name}":]\n${att.data}\n`
      });
    }
  }

  // Add the text prompt
  if (message.trim()) {
    currentParts.push({ text: message });
  }

  // Sanitize history to match API expected format
  const historyParts = history.map(h => ({
    role: h.role,
    parts: h.parts
  }));

  const chat = ai.chats.create({
    model: modelId,
    history: historyParts,
    config: {
      tools: tools.length > 0 ? tools : undefined,
    }
  });

  return chat.sendMessageStream({
    message: currentParts.length === 1 && currentParts[0].text ? currentParts[0].text : currentParts
  });
};

export const generateImage = async (prompt: string, size: ImageGenSize) => {
  await ensureApiKey(); // Required for Pro Image model
  const ai = getAI();
  
  const response = await ai.models.generateContent({
    model: ModelId.GEMINI_3_PRO_IMAGE,
    contents: {
      parts: [{ text: prompt }]
    },
    config: {
      imageConfig: {
        imageSize: size,
        aspectRatio: '1:1', 
      }
    }
  });

  const images: string[] = [];
  if (response.candidates?.[0]?.content?.parts) {
    for (const part of response.candidates[0].content.parts) {
      if (part.inlineData && part.inlineData.data) {
        images.push(`data:image/png;base64,${part.inlineData.data}`);
      }
    }
  }
  return images;
};

export const editImage = async (base64Image: string, prompt: string) => {
  const ai = getAI();
  
  const response = await ai.models.generateContent({
    model: ModelId.GEMINI_2_5_FLASH_IMAGE,
    contents: {
      parts: [
        {
          inlineData: {
            mimeType: 'image/jpeg',
            data: base64Image.split(',')[1],
          },
        },
        { text: prompt },
      ],
    },
  });

  const images: string[] = [];
  if (response.candidates?.[0]?.content?.parts) {
    for (const part of response.candidates[0].content.parts) {
      if (part.inlineData && part.inlineData.data) {
        images.push(`data:image/png;base64,${part.inlineData.data}`);
      }
    }
  }
  return images;
};

export const generateSlideContent = async (topic: string, count: number): Promise<PresentationStructure | null> => {
  const ai = getAI();
  
  const response = await ai.models.generateContent({
    model: ModelId.GEMINI_3_FLASH,
    contents: `Create a McKinsey-style management consulting presentation outline about: ${topic}.
    I need exactly ${count} slides.
    
    Style Guidelines:
    1. Titles must be "Action Titles" (complete sentences that summarize the slide's main insight).
    2. Content should be MECE (Mutually Exclusive, Collectively Exhaustive).
    3. Determine the 'sentiment' of the topic (positive, neutral, negative, urgent).
    4. Suggest a professional 'themeColor' hex code based on the sentiment (e.g., Navy for neutral, Red for urgent, Green for growth).
    
    Output JSON.`,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          slides: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                title: { type: Type.STRING, description: "Full sentence action title" },
                content: { 
                  type: Type.ARRAY,
                  items: { type: Type.STRING }
                },
                speakerNotes: { type: Type.STRING }
              },
              required: ["title", "content"]
            }
          },
          sentiment: {
            type: Type.STRING,
            enum: ["positive", "neutral", "negative", "urgent"]
          },
          themeColor: {
            type: Type.STRING,
            description: "Hex color code for the theme"
          }
        },
        required: ["slides", "sentiment", "themeColor"]
      }
    }
  });

  if (response.text) {
    try {
      return JSON.parse(response.text) as PresentationStructure;
    } catch (e) {
      console.error("Failed to parse JSON", e);
      throw new Error("Failed to generate valid slide structure.");
    }
  }
  return null;
};

export const analyzeStock = async (ticker: string) => {
  const ai = getAI();
  const response = await ai.models.generateContent({
    model: ModelId.GEMINI_3_PRO,
    contents: `Perform a deep-dive investment analysis on ${ticker}.
    
    Persona: You are a Senior Equity Research Analyst at a top-tier Wall Street hedge fund. 
    Tone: Professional, Objective, Data-Driven, Critical.
    
    Requirements:
    1. Use Google Search to fetch real-time price, recent news, and financial data.
    2. Analyze Valuation (P/E, Market Cap, EV/EBITDA vs Peers).
    3. Evaluate the Competitive Moat & Growth Drivers.
    4. Assess Risks (Macro, Regulatory, Execution).
    5. Provide a Technical Analysis overview (Trends, Support/Resistance).
    6. Conclude with an Institutional Verdict: BUY, SELL, or HOLD, with a clear thesis.
    
    Format using Markdown with clear headers.`,
    config: {
      tools: [{ googleSearch: {} }],
      systemInstruction: "You are a world-class financial analyst. Your analysis must be rigorous, citing numbers and specific events. Do not give generic advice."
    }
  });
  return response;
};