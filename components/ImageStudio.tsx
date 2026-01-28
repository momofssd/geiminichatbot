import React, { useState, useRef } from 'react';
import { Wand2, Download, Image as ImageIcon, Sparkles, AlertCircle, Loader2 } from 'lucide-react';
import { generateImage, editImage } from '../services/geminiService';
import { ImageGenSize } from '../types';

enum StudioMode {
  GENERATE = 'generate',
  EDIT = 'edit'
}

export const ImageStudio: React.FC = () => {
  const [mode, setMode] = useState<StudioMode>(StudioMode.GENERATE);
  const [prompt, setPrompt] = useState('');
  const [size, setSize] = useState<ImageGenSize>(ImageGenSize.SIZE_1K);
  const [generatedImages, setGeneratedImages] = useState<string[]>([]);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAction = async () => {
    if (!prompt.trim()) return;
    if (mode === StudioMode.EDIT && !uploadedImage) return;

    setIsProcessing(true);
    setError(null);
    setGeneratedImages([]);

    try {
      if (mode === StudioMode.GENERATE) {
        // Nano Banana Pro Generation
        const images = await generateImage(prompt, size);
        setGeneratedImages(images);
      } else {
        // Nano Banana Image Editing
        const images = await editImage(uploadedImage!, prompt);
        setGeneratedImages(images);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to process image. Make sure you have selected an API Key.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-col h-full w-full max-w-5xl mx-auto overflow-y-auto">
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6">Image Studio</h2>
        
        {/* Streamlit-style Radio Toggle */}
        <div className="flex gap-6 mb-8 border-b border-[#3b3d45]">
             <button
                onClick={() => setMode(StudioMode.GENERATE)}
                className={`pb-2 px-1 font-semibold text-sm transition-all border-b-2 ${
                    mode === StudioMode.GENERATE
                    ? 'border-[#FF4B4B] text-white'
                    : 'border-transparent text-gray-400 hover:text-gray-200'
                }`}
            >
                Generation (Pro)
            </button>
            <button
                onClick={() => setMode(StudioMode.EDIT)}
                className={`pb-2 px-1 font-semibold text-sm transition-all border-b-2 ${
                    mode === StudioMode.EDIT
                    ? 'border-[#FF4B4B] text-white'
                    : 'border-transparent text-gray-400 hover:text-gray-200'
                }`}
            >
                Editing (Flash)
            </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            
            {/* Input Column */}
            <div className="space-y-6">
                <div className="bg-[#262730] p-6 rounded-md border border-[#3b3d45]">
                    {mode === StudioMode.EDIT && (
                        <div className="mb-6">
                            <label className="block text-sm font-semibold text-white mb-2">Source Image</label>
                            <div 
                                onClick={() => fileInputRef.current?.click()}
                                className={`border-2 border-dashed rounded-md h-48 flex flex-col items-center justify-center cursor-pointer transition-colors ${
                                    uploadedImage ? 'border-[#FF4B4B] bg-[#0E1117]' : 'border-gray-600 hover:border-gray-400 bg-[#0E1117]'
                                }`}
                            >
                                {uploadedImage ? (
                                    <img src={uploadedImage} alt="Source" className="h-full object-contain p-2" />
                                ) : (
                                    <>
                                        <ImageIcon className="w-8 h-8 text-gray-400 mb-2" />
                                        <span className="text-gray-400 text-sm">Drag and drop file here</span>
                                        <span className="text-gray-600 text-xs mt-1">Limit 200MB per file</span>
                                    </>
                                )}
                            </div>
                            <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleFileUpload} />
                        </div>
                    )}

                    <div className="mb-6">
                        <label className="block text-sm font-semibold text-white mb-2">
                            {mode === StudioMode.GENERATE ? 'Prompt' : 'Instructions'}
                        </label>
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder={mode === StudioMode.GENERATE ? "A cyberpunk cat eating ramen..." : "Make it look like a painting..."}
                            className="w-full bg-[#0E1117] border border-[#4B4B4B] rounded-md p-3 text-white placeholder-gray-500 focus:border-[#FF4B4B] outline-none resize-none h-32 text-sm font-mono"
                        />
                    </div>

                    {mode === StudioMode.GENERATE && (
                        <div className="mb-6">
                            <label className="block text-sm font-semibold text-white mb-2">Resolution</label>
                            <div className="flex gap-2">
                                {[ImageGenSize.SIZE_1K, ImageGenSize.SIZE_2K, ImageGenSize.SIZE_4K].map((s) => (
                                    <button
                                        key={s}
                                        onClick={() => setSize(s)}
                                        className={`px-3 py-1 rounded-sm border text-xs font-mono transition-colors ${
                                            size === s
                                            ? 'bg-[#FF4B4B] border-[#FF4B4B] text-white'
                                            : 'bg-[#0E1117] border-[#4B4B4B] text-gray-300 hover:border-gray-300'
                                        }`}
                                    >
                                        {s}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    <button
                        onClick={handleAction}
                        disabled={isProcessing || !prompt.trim() || (mode === StudioMode.EDIT && !uploadedImage)}
                        className="w-full py-2 bg-[#FF4B4B] hover:bg-[#FF3333] disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-md font-semibold transition-colors flex items-center justify-center gap-2"
                    >
                        {isProcessing ? <Loader2 className="animate-spin" size={16} /> : <Sparkles size={16} />}
                        {mode === StudioMode.GENERATE ? 'Generate' : 'Apply Edit'}
                    </button>
                    
                    {error && (
                        <div className="mt-4 p-3 bg-red-900/20 border border-red-800 rounded text-red-200 text-xs flex gap-2 items-start">
                            <AlertCircle size={14} className="mt-0.5 shrink-0" />
                            <span>{error}</span>
                        </div>
                    )}
                </div>
            </div>

            {/* Output Column */}
            <div className="bg-[#262730] border border-[#3b3d45] p-6 rounded-md flex flex-col items-center justify-center min-h-[400px]">
                {generatedImages.length > 0 ? (
                    <div className="space-y-4 w-full">
                        {generatedImages.map((img, idx) => (
                            <div key={idx} className="relative group">
                                <img src={img} alt="Generated" className="w-full rounded-md shadow-sm" />
                                <a 
                                    href={img} 
                                    download={`gemini-generated-${Date.now()}.png`}
                                    className="absolute top-2 right-2 p-2 bg-black/60 backdrop-blur text-white rounded-md opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black/80"
                                >
                                    <Download size={16} />
                                </a>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="text-center text-gray-500">
                        <ImageIcon size={40} className="mx-auto mb-3 opacity-30" />
                        <p className="text-sm">Output will appear here</p>
                    </div>
                )}
            </div>
        </div>
      </div>
    </div>
  );
};