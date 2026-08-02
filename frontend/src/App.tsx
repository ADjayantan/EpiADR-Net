import React, { useState } from 'react';
import { Header } from './components/Header';
import { PredictionPanel } from './components/PredictionPanel';
import { TrainingPanel } from './components/TrainingPanel';
import { OrganProfilesPanel } from './components/OrganProfilesPanel';
import { ApiDocsPanel } from './components/ApiDocsPanel';
import { RepositoryCardPanel } from './components/RepositoryCardPanel';
import { AiAdvisorModal } from './components/AiAdvisorModal';

export default function App() {
  const [activeTab, setActiveTab] = useState<'predict' | 'train' | 'organs' | 'api' | 'repo'>('predict');
  const [useTissueConditioning, setUseTissueConditioning] = useState<boolean>(true);
  const [isAiModalOpen, setIsAiModalOpen] = useState<boolean>(false);
  const [aiModalData, setAiModalData] = useState<any>(null);

  const handleOpenAiReportWithData = (data: any) => {
    setAiModalData(data);
    setIsAiModalOpen(true);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500 selection:text-white flex flex-col">
      
      {/* App Header & Navigation */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        useTissueConditioning={useTissueConditioning}
        setUseTissueConditioning={setUseTissueConditioning}
        onOpenAiReport={() => setIsAiModalOpen(true)}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        
        {activeTab === 'predict' && (
          <PredictionPanel
            useTissueConditioning={useTissueConditioning}
            onOpenAiReportWithData={handleOpenAiReportWithData}
          />
        )}

        {activeTab === 'train' && (
          <TrainingPanel
            useTissueConditioning={useTissueConditioning}
            setUseTissueConditioning={setUseTissueConditioning}
          />
        )}

        {activeTab === 'organs' && (
          <OrganProfilesPanel />
        )}

        {activeTab === 'api' && (
          <ApiDocsPanel />
        )}

        {activeTab === 'repo' && (
          <RepositoryCardPanel />
        )}

      </main>

      {/* Footer */}
      <footer className="bg-slate-900 border-t border-slate-800 py-4 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-wrap items-center justify-between gap-2">
          <span>EpiADR-Net © 2026 Tissue-Conditioned Zero-Shot AI Platform</span>
          <span className="font-mono text-[11px]">SIDER 4.1 & GTEx V8 Transcriptomics Integrated</span>
        </div>
      </footer>

      {/* AI Preclinical Safety Modal */}
      <AiAdvisorModal
        isOpen={isAiModalOpen}
        onClose={() => setIsAiModalOpen(false)}
        initialData={aiModalData}
      />

    </div>
  );
}
