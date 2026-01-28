import React from 'react';
import { MessageSquare, Image as ImageIcon, Sparkles, PlusCircle, Presentation, TrendingUp } from 'lucide-react';
import { AppMode } from '../types';

interface SidebarProps {
  currentMode: AppMode;
  setMode: (mode: AppMode) => void;
  onNewChat: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentMode, setMode, onNewChat }) => {
  return (
    <div className="w-20 md:w-72 bg-[#262730] flex flex-col h-full transition-all duration-300">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
            <h1 className="text-xl font-bold text-white tracking-tight hidden md:block">
              OmniGem Assistant
            </h1>
            <Sparkles className="w-5 h-5 text-white md:hidden" />
        </div>

        {/* New Chat Button */}
        <button
          onClick={() => {
            setMode(AppMode.CHAT);
            onNewChat();
          }}
          className="w-full flex items-center justify-center gap-2 bg-white text-[#262730] hover:bg-gray-200 px-4 py-2 rounded-md transition-colors mb-6 font-semibold text-sm"
        >
          <PlusCircle size={16} />
          <span className="hidden md:block">New Conversation</span>
        </button>

        <div className="space-y-1">
            <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 hidden md:block px-2">Apps</div>
            <button
            onClick={() => setMode(AppMode.CHAT)}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-md transition-all text-sm ${
                currentMode === AppMode.CHAT
                ? 'bg-[#3b3d45] text-white font-semibold'
                : 'text-gray-300 hover:bg-[#31333F]'
            }`}
            >
            <MessageSquare className="w-4 h-4" />
            <span className="hidden md:block">Chat</span>
            </button>

            <button
            onClick={() => setMode(AppMode.IMAGE_STUDIO)}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-md transition-all text-sm ${
                currentMode === AppMode.IMAGE_STUDIO
                ? 'bg-[#3b3d45] text-white font-semibold'
                : 'text-gray-300 hover:bg-[#31333F]'
            }`}
            >
            <ImageIcon className="w-4 h-4" />
            <span className="hidden md:block">Image Studio</span>
            </button>

            <button
            onClick={() => setMode(AppMode.PPT_BUILDER)}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-md transition-all text-sm ${
                currentMode === AppMode.PPT_BUILDER
                ? 'bg-[#3b3d45] text-white font-semibold'
                : 'text-gray-300 hover:bg-[#31333F]'
            }`}
            >
            <Presentation className="w-4 h-4" />
            <span className="hidden md:block">Presentation</span>
            </button>

            <button
            onClick={() => setMode(AppMode.STOCK_ANALYSIS)}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-md transition-all text-sm ${
                currentMode === AppMode.STOCK_ANALYSIS
                ? 'bg-[#3b3d45] text-white font-semibold'
                : 'text-gray-300 hover:bg-[#31333F]'
            }`}
            >
            <TrendingUp className="w-4 h-4" />
            <span className="hidden md:block">Stock Analysis</span>
            </button>
        </div>
      </div>

      <div className="mt-auto p-6 border-t border-[#3b3d45]">
        <div className="hidden md:block">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">About</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
                Powered by Gemini 3. Flash for speed, Pro for reasoning.
            </p>
        </div>
      </div>
    </div>
  );
};