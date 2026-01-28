import React, { useState } from 'react';
import { TrendingUp, Search, Loader2, AlertTriangle, DollarSign, Activity, FileText } from 'lucide-react';
import { analyzeStock } from '../services/geminiService';
import ReactMarkdown from 'react-markdown';
import { GenerateContentResponse } from '@google/genai';

export const StockAnalysis: React.FC = () => {
  const [ticker, setTicker] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [groundingSources, setGroundingSources] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!ticker.trim()) return;

    setIsAnalyzing(true);
    setReport(null);
    setGroundingSources([]);
    setError(null);

    try {
      const response: GenerateContentResponse = await analyzeStock(ticker);
      
      if (response.text) {
        setReport(response.text);
      } else {
        setError("No analysis generated. Please try again.");
      }

      const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
      if (chunks) {
        const sources = chunks
            .map((c: any) => c.web)
            .filter((w: any) => w);
        setGroundingSources(sources);
      }

    } catch (err) {
      console.error(err);
      setError("Failed to generate analysis. Ensure your API key has search permissions.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="flex flex-col h-full w-full max-w-5xl mx-auto overflow-y-auto">
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 flex items-center gap-3 text-white">
            <TrendingUp className="text-[#FF4B4B]" size={32} />
            Institutional Stock Analysis
            <span className="text-xs font-normal text-gray-400 bg-[#3b3d45] px-2 py-1 rounded ml-2">Wall St. Persona</span>
        </h2>

        {/* Input Section */}
        <div className="bg-[#262730] p-8 rounded-lg border border-[#3b3d45] mb-8">
            <form onSubmit={handleAnalyze} className="max-w-2xl mx-auto text-center">
                <label className="block text-lg font-medium text-gray-200 mb-4">
                    Enter Company Ticker or Name
                </label>
                <div className="relative flex items-center">
                    <div className="absolute left-4 text-gray-400">
                        <DollarSign size={20} />
                    </div>
                    <input
                        type="text"
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value.toUpperCase())}
                        placeholder="e.g. NVDA, AAPL, TSLA"
                        className="w-full bg-[#0E1117] border border-[#4B4B4B] rounded-full py-4 pl-12 pr-32 text-xl font-bold text-white tracking-wide focus:border-[#FF4B4B] focus:ring-1 focus:ring-[#FF4B4B] outline-none placeholder-gray-600 uppercase"
                    />
                    <button
                        type="submit"
                        disabled={!ticker.trim() || isAnalyzing}
                        className="absolute right-2 top-2 bottom-2 bg-[#FF4B4B] hover:bg-[#FF3333] text-white px-6 rounded-full font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                        {isAnalyzing ? <Loader2 className="animate-spin" /> : 'Analyze'}
                    </button>
                </div>
                <p className="text-xs text-gray-500 mt-3">
                    Powered by Gemini 3.0 Pro with Real-time Google Search Grounding
                </p>
            </form>
        </div>

        {/* Report Section */}
        {error && (
            <div className="bg-red-900/20 border border-red-800 p-4 rounded-lg flex items-center gap-3 text-red-300">
                <AlertTriangle size={20} />
                {error}
            </div>
        )}

        {report && (
            <div className="animate-fade-in grid grid-cols-1 lg:grid-cols-4 gap-8">
                {/* Main Report */}
                <div className="lg:col-span-3">
                    <div className="bg-[#262730]/50 p-8 rounded-lg border border-[#3b3d45]">
                        <div className="flex items-center gap-3 mb-6 pb-4 border-b border-[#3b3d45]">
                            <Activity className="text-blue-400" size={24} />
                            <h3 className="text-2xl font-bold text-white">Investment Memo: {ticker}</h3>
                        </div>
                        <div className="markdown-content text-gray-200">
                            <ReactMarkdown>{report}</ReactMarkdown>
                        </div>
                    </div>
                </div>

                {/* Sources Sidebar */}
                <div className="lg:col-span-1 space-y-6">
                    <div className="bg-[#1E1E1E] p-4 rounded-lg border border-[#3b3d45] sticky top-8">
                        <h4 className="text-sm font-bold text-gray-300 mb-4 flex items-center gap-2">
                            <Search size={14} /> Data Sources
                        </h4>
                        {groundingSources.length > 0 ? (
                            <ul className="space-y-3">
                                {groundingSources.map((source: any, idx: number) => (
                                    <li key={idx} className="text-xs">
                                        <a 
                                            href={source.uri} 
                                            target="_blank" 
                                            rel="noreferrer"
                                            className="block p-2 rounded bg-[#262730] hover:bg-[#3b3d45] transition-colors border border-transparent hover:border-gray-600"
                                        >
                                            <div className="font-semibold text-blue-300 truncate mb-1">
                                                {source.title || new URL(source.uri).hostname}
                                            </div>
                                            <div className="text-gray-500 truncate text-[10px]">
                                                {source.uri}
                                            </div>
                                        </a>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p className="text-xs text-gray-500 italic">No direct web sources returned.</p>
                        )}
                    </div>

                    <div className="bg-blue-900/10 p-4 rounded-lg border border-blue-800/30">
                        <div className="flex items-start gap-2">
                            <FileText size={16} className="text-blue-400 mt-0.5" />
                            <div>
                                <h5 className="text-xs font-bold text-blue-300 mb-1">Disclaimer</h5>
                                <p className="text-[10px] text-gray-400 leading-relaxed">
                                    This report is generated by AI for informational purposes only. It does not constitute financial advice. Always do your own due diligence.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        )}
      </div>
    </div>
  );
};