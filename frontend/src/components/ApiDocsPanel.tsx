import React, { useState } from 'react';
import { Code, Terminal, Play, CheckCircle, Copy, FileText, Server } from 'lucide-react';

export const ApiDocsPanel: React.FC = () => {
  const [activeEndpoint, setActiveEndpoint] = useState<'/predict' | '/explain' | '/health' | '/organs'>('/predict');
  const [testSmiles, setTestSmiles] = useState<string>('CC(=O)NC1=CC=C(O)C=C1');
  const [useTissueConditioning, setUseTissueConditioning] = useState<boolean>(true);
  const [apiResponse, setApiResponse] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  const handleRunRequest = async () => {
    setIsLoading(true);
    setApiResponse(null);

    await new Promise((r) => setTimeout(r, 400));

    if (activeEndpoint === '/predict') {
      setApiResponse(JSON.stringify({
        compound: "Acetaminophen",
        smiles: testSmiles,
        tissueConditioning: useTissueConditioning,
        organScores: {
          liver: { meanRisk: 0.92, uncertaintySigma: 0.03, riskLevel: "Severe" },
          heart: { meanRisk: 0.12, uncertaintySigma: 0.02, riskLevel: "Low" },
          kidney: { meanRisk: 0.54, uncertaintySigma: 0.04, riskLevel: "Moderate" },
          brain: { meanRisk: 0.15, uncertaintySigma: 0.02, riskLevel: "Low" },
          lung: { meanRisk: 0.10, uncertaintySigma: 0.02, riskLevel: "Low" }
        },
        tanimotoDomain: {
          score: 0.88,
          status: "High Confidence (In-Domain)"
        }
      }, null, 2));
    } else if (activeEndpoint === '/explain') {
      setApiResponse(JSON.stringify({
        smiles: testSmiles,
        atomicHotspots: [
          { atomIndex: 0, symbol: "C", attentionWeight: 0.15 },
          { atomIndex: 1, symbol: "O", attentionWeight: 0.72, toxicophore: "Reactive Carbonyl" },
          { atomIndex: 2, symbol: "N", attentionWeight: 0.81, toxicophore: "Amide Center" }
        ],
        primaryToxicophores: ["Reactive Carbonyl Center", "Substituted Amine Core"]
      }, null, 2));
    } else if (activeEndpoint === '/organs') {
      setApiResponse(JSON.stringify({
        organs: [
          { id: "liver", markerGenes: ["CYP3A4", "ALB"], expressionDim: 128 },
          { id: "heart", markerGenes: ["MYH6", "TNNT2"], expressionDim: 128 },
          { id: "kidney", markerGenes: ["SLC22A2", "UMOD"], expressionDim: 128 }
        ]
      }, null, 2));
    } else {
      setApiResponse(JSON.stringify({
        status: "ok",
        version: "2.1.0-EpiADR-Net",
        model: "EpiADR-Net Tissue-Conditioned Graph Transformer",
        dataset: "SIDER 4.1 & GTEx V8"
      }, null, 2));
    }

    setIsLoading(false);
  };

  const pythonSnippet = `import requests

url = "http://localhost:8000/predict"
payload = {
    "smiles": "${testSmiles}",
    "use_tissue_conditioning": ${useTissueConditioning ? "True" : "False"},
    "mc_dropout_passes": 30
}

response = requests.post(url, json=payload)
print(response.json())`;

  const curlSnippet = `curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"smiles": "${testSmiles}", "use_tissue_conditioning": ${useTissueConditioning}}'`;

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
        <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
          <Terminal className="w-5 h-5 text-indigo-400" />
          <div>
            <h2 className="text-base font-bold text-white">FastAPI REST Microservice & Swagger UI Simulator</h2>
            <p className="text-xs text-slate-400">Enterprise REST Endpoints for High-Throughput Preclinical Screening</p>
          </div>
        </div>

        {/* Endpoints Nav */}
        <div className="flex space-x-2 overflow-x-auto no-scrollbar">
          {(['/predict', '/explain', '/organs', '/health'] as const).map((ep) => (
            <button
              key={ep}
              onClick={() => {
                setActiveEndpoint(ep);
                setApiResponse(null);
              }}
              className={`px-3.5 py-1.5 rounded-xl text-xs font-mono font-bold transition-all border ${
                activeEndpoint === ep
                  ? 'bg-indigo-600 text-white border-indigo-400 shadow-md'
                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-800'
              }`}
            >
              {ep}
            </button>
          ))}
        </div>
      </div>

      {/* Main Interactive Testing Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Request Body & Test Form */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center space-x-2">
            <Server className="w-4 h-4 text-indigo-400" />
            <span>Request Simulator ({activeEndpoint})</span>
          </h3>

          {(activeEndpoint === '/predict' || activeEndpoint === '/explain') && (
            <div className="space-y-3 text-xs">
              <div className="space-y-1">
                <label className="text-slate-400 font-semibold block">Target SMILES String:</label>
                <input
                  type="text"
                  value={testSmiles}
                  onChange={(e) => setTestSmiles(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-indigo-300 font-mono"
                />
              </div>

              {activeEndpoint === '/predict' && (
                <div className="flex items-center justify-between bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                  <span className="text-slate-300 font-semibold">use_tissue_conditioning</span>
                  <input
                    type="checkbox"
                    checked={useTissueConditioning}
                    onChange={(e) => setUseTissueConditioning(e.target.checked)}
                    className="accent-indigo-500 w-4 h-4 cursor-pointer"
                  />
                </div>
              )}
            </div>
          )}

          <button
            onClick={handleRunRequest}
            disabled={isLoading}
            className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-bold text-xs shadow-lg flex items-center justify-center space-x-2 transition-all active:scale-95"
          >
            <Play className="w-4 h-4 fill-white" />
            <span>Execute API Request</span>
          </button>

          {/* Code Snippets */}
          <div className="pt-2 border-t border-slate-800 space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-slate-300">Python Integration Snippet</span>
              <button
                onClick={() => copyToClipboard(pythonSnippet)}
                className="text-indigo-400 hover:text-indigo-300 flex items-center space-x-1"
              >
                <Copy className="w-3.5 h-3.5" />
                <span>{copied ? 'Copied!' : 'Copy'}</span>
              </button>
            </div>
            <pre className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-[11px] font-mono text-emerald-300 overflow-x-auto leading-normal">
              {pythonSnippet}
            </pre>
          </div>

        </div>

        {/* Right Column: JSON Response Output */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <span className="text-xs font-bold text-slate-300 font-mono">Response Payload (200 OK)</span>
            <span className="text-[11px] text-emerald-400 font-mono">application/json</span>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 min-h-[280px]">
            {isLoading ? (
              <div className="flex items-center justify-center h-48 text-indigo-400 font-mono text-xs animate-pulse">
                <span>Executing forward pass...</span>
              </div>
            ) : apiResponse ? (
              <pre className="text-xs font-mono text-indigo-200 overflow-x-auto leading-relaxed">
                {apiResponse}
              </pre>
            ) : (
              <div className="flex flex-col items-center justify-center h-48 text-slate-500 text-xs">
                <span>Click "Execute API Request" above to view response payload</span>
              </div>
            )}
          </div>
        </div>

      </div>

    </div>
  );
};
