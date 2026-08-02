import React from 'react';
import { Activity, Cpu, Database, Dna, FileText, Sparkles, Layers, ShieldCheck } from 'lucide-react';

interface NavbarProps {
  useTissueConditioning: boolean;
  onToggleTissueConditioning: () => void;
  onOpenAIAdvisor: () => void;
  activeTab: 'organ_prediction' | 'model_trainer' | 'explainable_xai' | 'benchmark_sider';
  onChangeTab: (tab: 'organ_prediction' | 'model_trainer' | 'explainable_xai' | 'benchmark_sider') => void;
  isTraining: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  useTissueConditioning,
  onToggleTissueConditioning,
  onOpenAIAdvisor,
  activeTab,
  onChangeTab,
  isTraining
}) => {
  return (
    <header className="bg-slate-900 border-b border-slate-800 text-slate-100 sticky top-0 z-30 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand & Badge */}
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-600 rounded-xl text-white shadow-md shadow-indigo-900/40">
            <Dna className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-indigo-300 bg-clip-text text-transparent">
                EpiADR-Net
              </h1>
              <span className="text-[10px] bg-indigo-950 text-indigo-300 border border-indigo-700/60 font-mono font-bold px-1.5 py-0.5 rounded uppercase tracking-wider">
                v2.1 GTEx V8
              </span>
            </div>
            <p className="text-xs text-slate-400 font-medium">Tissue-Conditioned Zero-Shot Toxicity AI</p>
          </div>
        </div>

        {/* Primary Navigation Tabs */}
        <nav className="hidden md:flex items-center space-x-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => onChangeTab('organ_prediction')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'organ_prediction'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Activity className="w-4 h-4" />
            <span>Organ Safety Profile</span>
          </button>

          <button
            onClick={() => onChangeTab('model_trainer')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'model_trainer'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Cpu className="w-4 h-4" />
            <span>Train & Finetune Model</span>
            {isTraining && (
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>
            )}
          </button>

          <button
            onClick={() => onChangeTab('explainable_xai')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'explainable_xai'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Layers className="w-4 h-4" />
            <span>XAI Hotspots</span>
          </button>

          <button
            onClick={() => onChangeTab('benchmark_sider')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'benchmark_sider'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Database className="w-4 h-4" />
            <span>SIDER 4.1 Benchmarks</span>
          </button>
        </nav>

        {/* Tissue Conditioning Toggle & Gemini AI Advisor */}
        <div className="flex items-center space-x-3">
          
          {/* Tissue Conditioning Mode Toggle */}
          <button
            onClick={onToggleTissueConditioning}
            className={`hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
              useTissueConditioning
                ? 'bg-emerald-950/80 border-emerald-700 text-emerald-300'
                : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
            title="Toggle between GTEx V8 Tissue-Conditioned 16-Head Attention Mode vs Molecule-Only Baseline"
          >
            <ShieldCheck className={`w-4 h-4 ${useTissueConditioning ? 'text-emerald-400' : 'text-slate-500'}`} />
            <span>{useTissueConditioning ? 'Tissue-Conditioned (GTEx V8)' : 'Molecule-Only Baseline'}</span>
          </button>

          {/* AI Preclinical Safety Advisor */}
          <button
            onClick={onOpenAIAdvisor}
            className="flex items-center space-x-2 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:opacity-90 text-white text-xs font-bold px-3 py-2 rounded-xl shadow-md transition-all active:scale-95"
          >
            <Sparkles className="w-4 h-4" />
            <span>AI Preclinical Report</span>
          </button>
        </div>
      </div>

      {/* Mobile Sub-Nav */}
      <div className="md:hidden flex overflow-x-auto p-2 bg-slate-950 border-t border-slate-800 space-x-2 no-scrollbar">
        <button
          onClick={() => onChangeTab('organ_prediction')}
          className={`whitespace-nowrap px-3 py-1 rounded text-xs font-medium ${
            activeTab === 'organ_prediction' ? 'bg-indigo-600 text-white' : 'bg-slate-900 text-slate-300'
          }`}
        >
          Organ Profile
        </button>
        <button
          onClick={() => onChangeTab('model_trainer')}
          className={`whitespace-nowrap px-3 py-1 rounded text-xs font-medium ${
            activeTab === 'model_trainer' ? 'bg-indigo-600 text-white' : 'bg-slate-900 text-slate-300'
          }`}
        >
          Train Model
        </button>
        <button
          onClick={() => onChangeTab('explainable_xai')}
          className={`whitespace-nowrap px-3 py-1 rounded text-xs font-medium ${
            activeTab === 'explainable_xai' ? 'bg-indigo-600 text-white' : 'bg-slate-900 text-slate-300'
          }`}
        >
          XAI Hotspots
        </button>
        <button
          onClick={() => onChangeTab('benchmark_sider')}
          className={`whitespace-nowrap px-3 py-1 rounded text-xs font-medium ${
            activeTab === 'benchmark_sider' ? 'bg-indigo-600 text-white' : 'bg-slate-900 text-slate-300'
          }`}
        >
          SIDER 4.1 Set
        </button>
      </div>
    </header>
  );
};
