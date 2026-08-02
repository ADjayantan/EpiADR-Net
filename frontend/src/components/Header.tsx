import React from 'react';
import { Activity, Cpu, Database, GitBranch, Layers, Sparkles, Terminal, ShieldCheck, FileText } from 'lucide-react';

interface HeaderProps {
  activeTab: 'predict' | 'train' | 'organs' | 'api' | 'repo';
  setActiveTab: (tab: 'predict' | 'train' | 'organs' | 'api' | 'repo') => void;
  useTissueConditioning: boolean;
  setUseTissueConditioning: (val: boolean) => void;
  onOpenAiReport: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  useTissueConditioning,
  setUseTissueConditioning,
  onOpenAiReport
}) => {
  return (
    <header className="bg-slate-900 border-b border-slate-800 text-slate-100 sticky top-0 z-40 shadow-xl">
      {/* Top Banner Status Bar */}
      <div className="bg-slate-950 px-4 py-1.5 border-b border-slate-800/80 text-xs flex flex-wrap items-center justify-between text-slate-400 gap-2">
        <div className="flex items-center space-x-3">
          <span className="flex items-center space-x-1 font-mono text-emerald-400 font-bold">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping mr-1"></span>
            EpiADR-Net v2.1.0-Release
          </span>
          <span className="hidden sm:inline-block text-slate-600">|</span>
          <span className="hidden sm:flex items-center space-x-1 text-slate-300">
            <GitBranch className="w-3.5 h-3.5 text-indigo-400" />
            <span className="font-mono text-[11px]">GitHub: 643c0b7 (main)</span>
          </span>
          <span className="hidden md:inline-block text-slate-600">|</span>
          <span className="hidden md:flex items-center space-x-1 text-slate-300">
            <Database className="w-3.5 h-3.5 text-cyan-400" />
            <span>SIDER 4.1 (7,325 ADRs) & GTEx V8 Transcriptomics</span>
          </span>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={() => setUseTissueConditioning(!useTissueConditioning)}
            className={`flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold border transition-all ${
              useTissueConditioning
                ? 'bg-indigo-600/30 border-indigo-500 text-indigo-300'
                : 'bg-amber-600/20 border-amber-500 text-amber-300'
            }`}
            title="Toggle Tissue-Conditioned Gene Cross-Attention Mode vs Baseline Molecule-Only Mode"
          >
            <Layers className="w-3 h-3" />
            <span>{useTissueConditioning ? 'Tissue-Conditioned (GTEx V8)' : 'Molecule-Only Baseline'}</span>
          </button>
        </div>
      </div>

      {/* Main Header Container */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand Title */}
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-600 rounded-xl text-white shadow-lg shadow-indigo-500/20">
            <Cpu className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-extrabold tracking-tight bg-gradient-to-r from-white via-indigo-100 to-indigo-300 bg-clip-text text-transparent">
                EpiADR-Net
              </h1>
              <span className="bg-gradient-to-r from-emerald-500 to-teal-500 text-slate-950 font-black text-[10px] px-1.5 py-0.5 rounded uppercase tracking-wider">
                Zero-Shot Organ AI
              </span>
            </div>
            <p className="text-xs text-slate-400 font-medium hidden sm:block">
              Tissue-Conditioned Adverse Drug Reaction & Organ Toxicity Platform
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1 bg-slate-950 p-1 rounded-xl border border-slate-800/80">
          <button
            onClick={() => setActiveTab('predict')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              activeTab === 'predict'
                ? 'bg-indigo-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Predict & XAI</span>
          </button>

          <button
            onClick={() => setActiveTab('train')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              activeTab === 'train'
                ? 'bg-indigo-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>Train & Benchmark</span>
          </button>

          <button
            onClick={() => setActiveTab('organs')}
            className={`hidden md:flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              activeTab === 'organs'
                ? 'bg-indigo-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>GTEx V8 Organs</span>
          </button>

          <button
            onClick={() => setActiveTab('api')}
            className={`hidden lg:flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              activeTab === 'api'
                ? 'bg-indigo-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Terminal className="w-3.5 h-3.5" />
            <span>REST API</span>
          </button>

          <button
            onClick={() => setActiveTab('repo')}
            className={`hidden xl:flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              activeTab === 'repo'
                ? 'bg-indigo-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Model Card & Contract</span>
          </button>
        </nav>

        {/* AI Preclinical Advisor CTA */}
        <div className="flex items-center space-x-2">
          <button
            onClick={onOpenAiReport}
            className="flex items-center space-x-1.5 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:opacity-95 text-white text-xs font-extrabold px-3 py-2 rounded-xl shadow-lg shadow-indigo-500/20 transition-all active:scale-95"
          >
            <Sparkles className="w-3.5 h-3.5 animate-spin-slow" />
            <span className="hidden sm:inline">AI Safety Advisor</span>
            <span className="sm:hidden">AI Report</span>
          </button>
        </div>

      </div>
    </header>
  );
};
