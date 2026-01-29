import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Bot, User, Search, ExternalLink, Cpu, Copy, Check, Terminal, FileText, FileSpreadsheet } from 'lucide-react';
import { ChatMessage } from '../types';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface MessageBubbleProps {
  message: ChatMessage;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`group flex gap-4 ${isUser ? 'bg-transparent' : 'bg-[#262730]/30'} p-6 rounded-lg mb-4 border border-transparent hover:border-[#3b3d45] transition-colors relative`}>
      <div className={`w-8 h-8 rounded-md flex items-center justify-center shrink-0 ${
        isUser ? 'bg-[#FF4B4B]' : 'bg-gray-600'
      }`}>
        {isUser ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-white" />}
      </div>

      <div className="flex-1 overflow-hidden">
        <div className="flex justify-between items-center mb-2">
            <div className="font-semibold text-sm text-gray-300">
                {isUser ? "You" : "Assistant"}
            </div>
            {!isUser && !message.isLoading && (
                <button 
                    onClick={handleCopy} 
                    className="text-gray-500 hover:text-white transition-colors p-1 rounded hover:bg-[#3b3d45]"
                    title="Copy response"
                >
                    {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
                </button>
            )}
        </div>
        
        {/* Attachments Display */}
        {message.attachments && message.attachments.length > 0 && (
        <div className="flex gap-2 mb-4 flex-wrap">
            {message.attachments.map((att, i) => (
                <div key={i} className="border border-[#3b3d45] rounded overflow-hidden bg-[#1E1E1E]">
                    {att.type === 'image' ? (
                         <img src={`data:${att.mimeType};base64,${att.data}`} alt={att.name} className="max-h-[200px] max-w-full" />
                    ) : (
                        <div className="flex items-center gap-3 p-3">
                             {att.name.endsWith('.xlsx') ? (
                                <FileSpreadsheet className="w-8 h-8 text-green-400" />
                             ) : att.name.endsWith('.docx') ? (
                                <FileText className="w-8 h-8 text-blue-400" />
                             ) : (
                                <FileText className="w-8 h-8 text-red-400" />
                             )}
                             <div>
                                 <p className="text-sm font-medium text-white">{att.name}</p>
                                 <p className="text-[10px] text-gray-500 uppercase">{att.type === 'pdf' ? 'PDF' : 'PARSED TEXT'}</p>
                             </div>
                        </div>
                    )}
                </div>
            ))}
        </div>
        )}

        {/* Text Content */}
        <div className="markdown-content text-gray-200 text-sm md:text-base">
        {message.isLoading ? (
            <div className="flex gap-1 items-center h-6">
                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
            </div>
        ) : isUser ? (
             // User messages: Preserve whitespace and layout exactly as typed
             <div className="whitespace-pre-wrap font-sans leading-relaxed">{message.text}</div>
        ) : (
             // Model messages: Use Markdown
            <ReactMarkdown
                components={{
                    code({ node, inline, className, children, ...props }: any) {
                        const match = /language-(\w+)/.exec(className || '');
                        const [codeCopied, setCodeCopied] = useState(false);

                        const handleCodeCopy = () => {
                            navigator.clipboard.writeText(String(children).replace(/\n$/, ''));
                            setCodeCopied(true);
                            setTimeout(() => setCodeCopied(false), 2000);
                        };

                        return !inline && match ? (
                            <div className="relative group/code my-4 rounded-md overflow-hidden border border-[#3b3d45]">
                                <div className="flex justify-between items-center bg-[#1E1E1E] px-4 py-1.5 border-b border-[#3b3d45]">
                                    <span className="text-xs text-gray-400 font-mono flex items-center gap-1">
                                        <Terminal size={12} />
                                        {match[1]}
                                    </span>
                                    <button
                                        onClick={handleCodeCopy}
                                        className="text-gray-400 hover:text-white text-xs flex items-center gap-1 transition-colors"
                                    >
                                        {codeCopied ? (
                                            <>
                                                <Check size={12} className="text-green-400" />
                                                <span>Copied</span>
                                            </>
                                        ) : (
                                            <>
                                                <Copy size={12} />
                                                <span>Copy Code</span>
                                            </>
                                        )}
                                    </button>
                                </div>
                                <SyntaxHighlighter
                                    style={vscDarkPlus}
                                    language={match[1]}
                                    PreTag="div"
                                    customStyle={{ margin: 0, borderRadius: 0, padding: '1rem', fontSize: '0.875rem' }}
                                    {...props}
                                >
                                    {String(children).replace(/\n$/, '')}
                                </SyntaxHighlighter>
                            </div>
                        ) : (
                            <code className="bg-[#3b3d45] text-[#FF4B4B] px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
                                {children}
                            </code>
                        );
                    }
                }}
            >
                {message.text}
            </ReactMarkdown>
        )}
        </div>

        {/* Grounding Sources */}
        {!isUser && message.groundingMetadata && (
          <div className="mt-4 flex flex-wrap gap-2 pt-2 border-t border-[#3b3d45]/50">
            {message.groundingMetadata.search?.map((source, idx) => (
               <a 
                 key={`search-${idx}`}
                 href={source.uri}
                 target="_blank"
                 rel="noopener noreferrer"
                 className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#3b3d45] hover:bg-[#4b4d55] transition-colors text-xs text-blue-300 no-underline"
               >
                 <Search className="w-3 h-3" />
                 <span className="truncate max-w-[200px]">{source.title || source.uri}</span>
                 <ExternalLink className="w-3 h-3 opacity-50" />
               </a>
            ))}
          </div>
        )}

        {/* Token Usage Stats */}
        {!isUser && message.usage && (
            <div className="mt-3 flex items-center gap-4 text-[10px] text-gray-500 font-mono opacity-60 hover:opacity-100 transition-opacity">
                <div className="flex items-center gap-1">
                    <Cpu size={10} />
                    <span>Usage:</span>
                </div>
                <div>In: {message.usage.promptTokenCount || 0}</div>
                <div>Out: {message.usage.candidatesTokenCount || 0}</div>
                <div>Total: {message.usage.totalTokenCount || 0}</div>
            </div>
        )}
      </div>
    </div>
  );
};