import React, { useState } from 'react';
import { FileText, GitBranch, ShieldCheck, CheckCircle, ExternalLink, Database, Code } from 'lucide-react';

export const RepositoryCardPanel: React.FC = () => {
  const [activeDoc, setActiveDoc] = useState<'contract' | 'model' | 'provenance'>('contract');

  return (
    <div className="space-y-6">
      
      {/* Sync Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-emerald-500/10 text-emerald-400 rounded-xl border border-emerald-500/30">
              <CheckCircle className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white">GitHub & Hugging Face Sync Status</h2>
              <p className="text-xs text-slate-400">Commit Hash: <span className="font-mono text-indigo-300">643c0b7fe4f546b9910fd5b004fad8892cf1e127</span></p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-xs bg-emerald-950 text-emerald-300 px-3 py-1 rounded-full border border-emerald-800 font-extrabold uppercase tracking-wider">
              100% In Sync 🎉
            </span>
          </div>
        </div>

        {/* Sync Repositories Table */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex justify-between items-center">
            <div>
              <span className="text-slate-400 block text-[10px]">GitHub Repository</span>
              <span className="text-indigo-300 font-bold">github.com/ADjayantan/EpiADR-Net</span>
            </div>
            <span className="text-emerald-400 font-bold">main ✓</span>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex justify-between items-center">
            <div>
              <span className="text-slate-400 block text-[10px]">Hugging Face Space</span>
              <span className="text-amber-300 font-bold">huggingface.co/spaces/jayantan/EpiADR-Net</span>
            </div>
            <span className="text-emerald-400 font-bold">Live ✓</span>
          </div>
        </div>
      </div>

      {/* Docs Selector */}
      <div className="flex space-x-2 bg-slate-950 p-1.5 rounded-2xl border border-slate-800 max-w-fit">
        <button
          onClick={() => setActiveDoc('contract')}
          className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${
            activeDoc === 'contract'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          DATA_CONTRACT.md
        </button>
        <button
          onClick={() => setActiveDoc('model')}
          className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${
            activeDoc === 'model'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          MODEL_CARD.md
        </button>
        <button
          onClick={() => setActiveDoc('provenance')}
          className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${
            activeDoc === 'provenance'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          DATA_PROVENANCE.md
        </button>
      </div>

      {/* Markdown Document Content Viewer */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg text-slate-200 space-y-4">
        
        {activeDoc === 'contract' && (
          <div className="space-y-4 text-xs font-mono leading-relaxed">
            <h3 className="text-base font-bold text-white font-sans border-b border-slate-800 pb-2">DATA_CONTRACT.md</h3>
            
            <p className="text-slate-300 font-sans">
              This document specifies the authoritative schema, field types, validation constraints, and non-null guarantees for all dataset inputs ingested by EpiADR-Net.
            </p>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <span className="text-indigo-400 font-bold block">1. SIDER 4.1 Record Specification (`data/adr_records.csv`)</span>
              <ul className="list-disc list-inside text-slate-400 space-y-1">
                <li><strong className="text-slate-200">Total Validated Rows:</strong> 7,325 rows</li>
                <li><strong className="text-slate-200">Columns:</strong> drug_id (string), smiles (string, valid RDKit canonical), MedDRA_10_labels (binary 0/1 integers)</li>
                <li><strong className="text-slate-200">Missing Value Policy:</strong> No null SMILES or unmapped CID identifiers permitted.</li>
              </ul>
            </div>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <span className="text-emerald-400 font-bold block">2. GTEx V8 Transcriptomics Specification (`data/tissue_profiles.csv`)</span>
              <ul className="list-disc list-inside text-slate-400 space-y-1">
                <li><strong className="text-slate-200">Organs Covered:</strong> Liver, Heart (Left Ventricle), Kidney (Cortex), Brain (Cortex), Lung (Alveolar)</li>
                <li><strong className="text-slate-200">Dimension:</strong> 128-dimensional normalized Log2(TPM + 1) gene expression vectors</li>
                <li><strong className="text-slate-200">Validation:</strong> Standardized to zero mean and unit variance per organ.</li>
              </ul>
            </div>
          </div>
        )}

        {activeDoc === 'model' && (
          <div className="space-y-4 text-xs font-mono leading-relaxed">
            <h3 className="text-base font-bold text-white font-sans border-b border-slate-800 pb-2">MODEL_CARD.md</h3>
            
            <p className="text-slate-300 font-sans">
              EpiADR-Net is a Tissue-Conditioned Graph Transformer designed for Zero-Shot Adverse Drug Reaction and Organ Toxicity prediction.
            </p>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <span className="text-indigo-400 font-bold block">Architecture Overview</span>
              <p className="text-slate-400">
                Fuses 256-bit SMILES Morgan fingerprints with GTEx V8 transcriptomic organ expression profiles via 16-Head Bi-Directional Cross-Attention layers.
              </p>
            </div>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <span className="text-amber-400 font-bold block">Uncertainty & Domain Flagging</span>
              <p className="text-slate-400">
                Uses N=30 Monte Carlo Dropout stochastic forward passes to compute mean risk (μ) and uncertainty sigma (σ). Computes Tanimoto Morgan fingerprint similarity to flag out-of-domain novel chemical scaffolds.
              </p>
            </div>
          </div>
        )}

        {activeDoc === 'provenance' && (
          <div className="space-y-4 text-xs font-mono leading-relaxed">
            <h3 className="text-base font-bold text-white font-sans border-b border-slate-800 pb-2">DATA_PROVENANCE.md</h3>
            
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <span className="text-indigo-400 font-bold block">Data Origin & Licensing</span>
              <ul className="list-disc list-inside text-slate-400 space-y-1">
                <li><strong className="text-slate-200">SIDER 4.1:</strong> European Molecular Biology Laboratory (EMBL) side effect database. Creative Commons Attribution-NonCommercial 4.0 International.</li>
                <li><strong className="text-slate-200">GTEx V8:</strong> Genotype-Tissue Expression Project (dbGaP Accession phs000424.v8.p2).</li>
              </ul>
            </div>
          </div>
        )}

      </div>

    </div>
  );
};
