import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatInterface } from './components/ChatInterface';
import { ImageStudio } from './components/ImageStudio';
import { PPTBuilder } from './components/PPTBuilder';
import { AppMode } from './types';

function App() {
  const [currentMode, setMode] = useState<AppMode>(AppMode.CHAT);
  // Chat session key to force remount of ChatInterface
  const [chatSessionKey, setChatSessionKey] = useState<number>(0);

  const handleNewChat = () => {
    setChatSessionKey(prev => prev + 1);
  };

  return (
    <div className="flex h-screen w-screen bg-[#0E1117] text-white overflow-hidden">
      <Sidebar 
        currentMode={currentMode} 
        setMode={setMode} 
        onNewChat={handleNewChat}
      />
      <main className="flex-1 h-full relative">
        {currentMode === AppMode.CHAT && (
          <ChatInterface key={chatSessionKey} />
        )}
        {currentMode === AppMode.IMAGE_STUDIO && (
          <ImageStudio />
        )}
        {currentMode === AppMode.PPT_BUILDER && (
          <PPTBuilder />
        )}
      </main>
    </div>
  );
}

export default App;