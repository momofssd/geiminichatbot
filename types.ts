export enum ModelId {
  GEMINI_3_FLASH = 'gemini-3-flash-preview',
  GEMINI_3_PRO = 'gemini-3-pro-preview',
  GEMINI_2_5_FLASH_IMAGE = 'gemini-2.5-flash-image', // For Image Editing
  GEMINI_3_PRO_IMAGE = 'gemini-3-pro-image-preview', // For Image Gen
}

export enum AppMode {
  CHAT = 'chat',
  IMAGE_STUDIO = 'image_studio',
  PPT_BUILDER = 'ppt_builder',
  STOCK_ANALYSIS = 'stock_analysis',
}

export enum ImageGenSize {
  SIZE_1K = '1K',
  SIZE_2K = '2K',
  SIZE_4K = '4K',
}

export interface UsageMetadata {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  totalTokenCount?: number;
}

export interface Attachment {
  id: string;
  type: 'image' | 'pdf' | 'text' | 'file';
  mimeType: string;
  data: string; // Base64 for binary, or raw text for parsed files
  name: string;
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  attachments?: Attachment[];
  groundingMetadata?: {
    search?: { title: string; uri: string }[];
  };
  usage?: UsageMetadata;
  isLoading?: boolean;
}

export interface GroundingConfig {
  search: boolean;
}

export interface SlideContent {
  title: string;
  content: string[];
  speakerNotes?: string;
}

export interface PresentationStructure {
  slides: SlideContent[];
  sentiment: 'positive' | 'neutral' | 'negative' | 'urgent';
  themeColor: string;
}

declare global {
  interface AIStudio {
    hasSelectedApiKey: () => Promise<boolean>;
    openSelectKey: () => Promise<void>;
  }

  interface Window {
    aistudio?: AIStudio;
  }
}