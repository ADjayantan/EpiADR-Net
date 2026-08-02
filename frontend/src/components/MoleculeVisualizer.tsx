import React, { useState } from 'react';
import { AtomicHotspot } from '../types';
import { Eye, Flame, Info, ShieldCheck, Zap } from 'lucide-react';

interface MoleculeVisualizerProps {
  smiles: string;
  compoundName: string;
  hotspots: AtomicHotspot[];
  toxicophores: string[];
}

export const MoleculeVisualizer: React.FC<MoleculeVisualizerProps> = ({
  smiles,
  compoundName,
  hotspots,
  toxicophores
}) => {
  const [hoveredAtom, setHoveredAtom] = useState<AtomicHotspot | null>(null);

  // Generate a clean 2D visual layout representation for the SMILES string
  const atoms = hotspots;
  const totalAtoms = atoms.length;

  // Simple ring/chain coordinate calculation for visual rendering
  const getAtomCoordinates = (index: number, total: number) => {
    const isBenzene = smiles.includes('C1=CC=CC=C1') || smiles.includes('c1ccccc1');
    if (isBenzene && index < 6) {
      // 6-member ring
      const angle = (index * 2 * Math.PI) / 6 - Math.PI / 2;
      const radius = 55;
      return {
        cx: 140 + radius * Math.cos(angle),
        cy: 110 + radius * Math.sin(angle)
      };
    } else {
      // Horizontal wave chain with branching
      const cols = 6;
      const row = Math.floor(index / cols);
      const col = index % cols;
      return {
        cx: 60 + col * 45 + (row % 2 === 1 ? 22 : 0),
        cy: 60 + row * 45
      };
    }
  };

  // Map attention weight to color gradient
  const getHotspotColor = (weight: number) => {
    if (weight >= 0.85) return { fill: '#ef4444', stroke: '#f87171', bg: 'bg-rose-500/20 text-rose-300 border-rose-500/40' }; // Severe toxicophore
    if (weight >= 0.65) return { fill: '#f97316', stroke: '#fb923c', bg: 'bg-amber-500/20 text-amber-300 border-amber-500/40' }; // Moderate toxicophore
    if (weight >= 0.40) return { fill: '#eab308', stroke: '#fde047', bg: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40' }; // Mild risk
    return { fill: '#3b82f6', stroke: '#60a5fa', bg: 'bg-sky-500/20 text-sky-300 border-sky-500/40' }; // Safe/neutral
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
      
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 bg-rose-500/10 text-rose-400 rounded-lg">
            <Flame className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white">XAI Atomic Toxicity Hotspot Map</h3>
            <p className="text-[11px] text-slate-400">16-Head Cross-Attention Toxicophore Subgraph Contributions</p>
          </div>
        </div>

        <span className="text-[11px] bg-slate-950 text-slate-300 px-2.5 py-1 rounded-full border border-slate-800 font-mono">
          {totalAtoms} Reactive Centers
        </span>
      </div>

      {/* SVG Chemical Graph Viewer */}
      <div className="relative bg-slate-950 rounded-xl p-4 border border-slate-800/80 flex flex-col items-center justify-center min-h-[220px]">
        
        <svg className="w-full h-[200px]" viewBox="0 0 320 200">
          <defs>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Connective Bonds */}
          {atoms.map((atom, i) => {
            if (i === 0) return null;
            const prevCoords = getAtomCoordinates(i - 1, totalAtoms);
            const currCoords = getAtomCoordinates(i, totalAtoms);
            return (
              <line
                key={`bond-${i}`}
                x1={prevCoords.cx}
                y1={prevCoords.cy}
                x2={currCoords.cx}
                y2={currCoords.cy}
                stroke="#475569"
                strokeWidth="2.5"
                strokeLinecap="round"
              />
            );
          })}

          {/* Atom Nodes with Attention Glow */}
          {atoms.map((atom, i) => {
            const { cx, cy } = getAtomCoordinates(i, totalAtoms);
            const colors = getHotspotColor(atom.attentionWeight);
            const isHovered = hoveredAtom?.atomIndex === atom.atomIndex;

            return (
              <g
                key={`atom-${i}`}
                onMouseEnter={() => setHoveredAtom(atom)}
                onMouseLeave={() => setHoveredAtom(null)}
                className="cursor-pointer transition-transform duration-200"
              >
                {/* Outer Attention Halo */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={16 + atom.attentionWeight * 12}
                  fill={colors.fill}
                  opacity={isHovered ? 0.45 : 0.22}
                  filter="url(#glow)"
                />

                {/* Inner Core Node */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 13 : 11}
                  fill="#0f172a"
                  stroke={colors.stroke}
                  strokeWidth="2"
                />

                {/* Symbol Label */}
                <text
                  x={cx}
                  y={cy + 4}
                  textAnchor="middle"
                  fill="#f8fafc"
                  fontSize="11"
                  fontWeight="bold"
                  fontFamily="monospace"
                >
                  {atom.symbol}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Hovered Atom Info Box */}
        {hoveredAtom ? (
          <div className="absolute bottom-2 left-2 right-2 bg-slate-900/95 border border-indigo-500/50 p-2 rounded-lg text-xs flex justify-between items-center shadow-xl backdrop-blur-sm animate-fade-in">
            <div>
              <span className="font-bold text-indigo-300 font-mono">Atom #{hoveredAtom.atomIndex + 1} ({hoveredAtom.symbol}): </span>
              <span className="text-slate-200 font-medium">{hoveredAtom.subgraphName}</span>
            </div>
            <span className="text-rose-400 font-bold font-mono">
              α = {hoveredAtom.attentionWeight.toFixed(2)}
            </span>
          </div>
        ) : (
          <div className="absolute bottom-2 text-[10px] text-slate-500 flex items-center space-x-1">
            <Eye className="w-3 h-3 text-slate-400" />
            <span>Hover over atomic nodes to inspect XAI attention weights & toxicophore subgraphs</span>
          </div>
        )}
      </div>

      {/* Identified Toxicophore Badges */}
      <div className="space-y-1.5">
        <span className="text-xs font-semibold text-slate-400 block">Detected Structural Toxicophores:</span>
        <div className="flex flex-wrap gap-1.5">
          {toxicophores.map((t, idx) => (
            <span
              key={idx}
              className="text-xs bg-rose-950/60 text-rose-300 border border-rose-800/80 px-2.5 py-1 rounded-lg font-medium flex items-center space-x-1"
            >
              <Zap className="w-3 h-3 text-rose-400" />
              <span>{t}</span>
            </span>
          ))}
        </div>
      </div>

      {/* Attention Heatmap Legend */}
      <div className="flex items-center justify-between text-[11px] text-slate-400 pt-2 border-t border-slate-800">
        <span className="font-medium">Attention Intensity Scale:</span>
        <div className="flex items-center space-x-2 font-mono">
          <span className="flex items-center space-x-1">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
            <span>Low (α &lt; 0.4)</span>
          </span>
          <span className="flex items-center space-x-1">
            <span className="w-2.5 h-2.5 rounded-full bg-yellow-500"></span>
            <span>Mid (0.4 - 0.7)</span>
          </span>
          <span className="flex items-center space-x-1">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
            <span>Toxicophore (α &gt; 0.8)</span>
          </span>
        </div>
      </div>

    </div>
  );
};
