import React, { useState, useEffect } from 'react';
import { 
  Briefcase, GraduationCap, Building2, Microscope, 
  Key, RefreshCw, ThumbsUp, ThumbsDown, Activity, Shield, AlertTriangle, CheckCircle, XCircle, Zap
} from 'lucide-react';
import { generateProfile, makeDecision, setApiKey as setLlmApiKey } from './services/llm';
import { getMemory, addFeedback, getStats, incrementDecisions } from './services/rl';
import { analyzeBias, applyCorrection, getRecommendation, storeDecision, rewardAttribute, penalizeAttribute, healthCheck } from './services/api';

function App() {
  const [apiKey, setApiKey] = useState(localStorage.getItem('bias_scope_apikey') || '');
  const [activeDomain, setActiveDomain] = useState('Job');
  const [loading, setLoading] = useState(false);
  const [profile, setProfile] = useState(null);
  const [decision, setDecision] = useState(null);
  const [stats, setStats] = useState(getStats());
  const [sessionHistory, setSessionHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('analysis');
  
  // AIF360 state
  const [biasMetrics, setBiasMetrics] = useState(null);
  const [correctionResult, setCorrectionResult] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [analyzingBias, setAnalyzingBias] = useState(false);
  const [correcting, setCorrecting] = useState(false);
  const [backendAlive, setBackendAlive] = useState(false);
  const [selectedProtected, setSelectedProtected] = useState('Gender');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('calibrated_eq_odds');

  useEffect(() => {
    if (apiKey) {
      localStorage.setItem('bias_scope_apikey', apiKey);
      setLlmApiKey(apiKey);
    }
  }, [apiKey]);

  useEffect(() => {
    healthCheck().then(setBackendAlive);
  }, []);

  const handleGenerate = async () => {
    if (!apiKey) return alert("Please enter a Gemini API Key.");
    setLoading(true);
    setProfile(null);
    setDecision(null);
    try {
      const newProfile = await generateProfile(activeDomain);
      setProfile(newProfile);
      const memory = getMemory();
      const newDecision = await makeDecision(activeDomain, newProfile, memory);
      setDecision(newDecision);
      incrementDecisions();
      setStats(getStats());
      
      const historyEntry = { domain: activeDomain, decision: newDecision, profile: newProfile, timestamp: Date.now() };
      setSessionHistory(prev => [historyEntry, ...prev]);
      
      // Store in backend for AIF360
      if (backendAlive) {
        storeDecision({
          domain: activeDomain.toLowerCase(),
          attributes: newProfile.attributes || {},
          decision: newDecision.decision,
          weighted_attributes: newDecision.weightedAttributes || [],
        });
      }
    } catch (err) {
      console.error(err);
      alert("Error: " + err.message);
    }
    setLoading(false);
  };

  const handleFeedback = async (attribute, type) => {
    if (!decision) return;
    addFeedback(activeDomain, attribute, type);
    setStats(getStats());
    const newDec = { ...decision };
    if (newDec.weightedAttributes) {
      const attrObj = newDec.weightedAttributes.find(a => a.attribute === attribute);
      if (attrObj) attrObj.feedbackGiven = true;
    }
    setDecision(newDec);
    
    if (backendAlive) {
      if (type === 'positive') rewardAttribute(activeDomain.toLowerCase(), attribute);
      else penalizeAttribute(activeDomain.toLowerCase(), attribute);
    }
  };

  const handleAnalyzeBias = async () => {
    if (sessionHistory.length < 2) return alert("Need at least 2 decisions for bias analysis.");
    setAnalyzingBias(true);
    const records = sessionHistory.map(sh => ({
      attributes: sh.profile?.attributes || {},
      decision: sh.decision?.decision || "Unknown",
    }));
    const metrics = await analyzeBias(activeDomain.toLowerCase(), records, selectedProtected);
    setBiasMetrics(metrics);
    const rec = await getRecommendation(activeDomain.toLowerCase(), records, selectedProtected);
    setRecommendation(rec);
    setAnalyzingBias(false);
  };

  const handleApplyCorrection = async () => {
    if (sessionHistory.length < 2) return;
    setCorrecting(true);
    const records = sessionHistory.map(sh => ({
      attributes: sh.profile?.attributes || {},
      decision: sh.decision?.decision || "Unknown",
    }));
    const result = await applyCorrection(activeDomain.toLowerCase(), records, selectedProtected, selectedAlgorithm);
    setCorrectionResult(result);
    setCorrecting(false);
  };

  const getBiasColor = (di) => {
    if (!di) return 'var(--text-muted)';
    if (di >= 0.8 && di <= 1.25) return 'var(--stat-pos)';
    if (di < 0.5 || di > 2.0) return 'var(--stat-neg)';
    return 'var(--stat-caution)';
  };

  return (
    <div className="app-container">
      <header className="top-bar">
        <div className="logo-section">
          <div className="logo-icon"><Activity color="white" size={24} /></div>
          <div className="logo-text">
            <h1>BiasScope</h1>
            <p>AIF360 + RL Bias Detection</p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className={`backend-badge ${backendAlive ? 'alive' : ''}`}>
            {backendAlive ? <CheckCircle size={12} /> : <XCircle size={12} />}
            AIF360 {backendAlive ? 'Connected' : 'Offline'}
          </div>
          <div className="api-status">
            <div className={`status-dot ${apiKey ? 'active' : ''}`}></div>
            <div style={{ position: 'relative' }}>
              <input type="password" className="api-key-input" placeholder="Gemini API Key..." value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
              <Key size={14} style={{ position: 'absolute', right: 10, top: 10, color: 'var(--text-muted)' }} />
            </div>
          </div>
        </div>
      </header>

      <main className="main-grid">
        {/* Left Column */}
        <div className="left-column" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="panel">
            <div className="panel-header">Domain</div>
            <div className="domains-grid">
              {[['Job', Briefcase], ['College', GraduationCap], ['Loan', Building2]].map(([d, Icon]) => (
                <button key={d} className={`btn-domain ${activeDomain === d ? 'active' : ''}`} onClick={() => setActiveDomain(d)}>
                  <Icon size={20} color={activeDomain === d ? "var(--accent-purple)" : "var(--text-muted)"} />
                  <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>{d.toUpperCase()}</span>
                </button>
              ))}
            </div>
            <p style={{ textAlign: 'center', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>Generate a profile to begin</p>
            <button className="btn-primary" onClick={handleGenerate} disabled={loading}>
              <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
              {loading ? "Generating..." : "Generate Profile"}
            </button>
          </div>

          <div className="panel" style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
            <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Session History</span>
              <span style={{ textTransform: 'none', color: 'var(--text-muted)' }}>{sessionHistory.length} decisions</span>
            </div>
            {sessionHistory.length === 0 ? (
              <div className="empty-state" style={{ fontSize: '0.85rem' }}>No decisions yet</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {sessionHistory.map((sh, idx) => (
                  <div key={idx} style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.03)', borderRadius: '6px', fontSize: '0.8rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <strong style={{ color: 'var(--accent-purple)' }}>{sh.domain}</strong>
                      <span style={{ color: 'var(--text-muted)' }}>{new Date(sh.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div>Decision: <strong>{sh.decision.decision}</strong></div>
                    <div>Top: <span style={{ color: 'var(--accent-blue)' }}>{sh.decision.weightedAttributes?.[0]?.attribute}</span></div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Center Column */}
        <div className="panel">
          <div className="center-tabs">
            <button className={`tab ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')}>Analysis</button>
            <button className={`tab ${activeTab === 'fairness' ? 'active' : ''}`} onClick={() => setActiveTab('fairness')}>
              <Shield size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />Fairness
            </button>
            <button className={`tab ${activeTab === 'insights' ? 'active' : ''}`} onClick={() => setActiveTab('insights')}>Insights</button>
          </div>

          {activeTab === 'analysis' && !profile && (
            <div className="empty-state">
              <Microscope size={48} color="var(--panel-border)" style={{ marginBottom: '1rem' }} />
              <p>Generate a profile and make a decision</p>
            </div>
          )}

          {activeTab === 'analysis' && profile && (
            <div className="animate-fade-in" style={{ flex: 1, overflowY: 'auto', paddingRight: '0.5rem' }}>
              <div className="profile-card">
                <h3>{profile.name} - Profile</h3>
                <div className="profile-grid">
                  {Object.entries(profile.attributes || {}).map(([key, val]) => (
                    <React.Fragment key={key}>
                      <div className="profile-label">{key}</div>
                      <div style={{ color: 'var(--text-main)' }}>{val.toString()}</div>
                    </React.Fragment>
                  ))}
                </div>
              </div>

              {decision && (
                <div className="decision-box animate-fade-in">
                  <h3 style={{ color: 'white', marginBottom: '1rem' }}>AI Decision: <span className="text-gradient">{decision.decision}</span></h3>
                  <div style={{ marginBottom: '1rem' }}>
                    <div className="profile-label" style={{ marginBottom: '1rem' }}>Top Weighted Attributes:</div>
                    {decision.weightedAttributes?.map((attr, idx) => (
                      <div key={idx} style={{ marginBottom: '1rem', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid var(--panel-border)' }}>
                        <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-pink)', marginBottom: '0.5rem' }}>
                          {attr.attribute} <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>({attr.weight})</span>
                        </div>
                        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>{attr.reasoning}</p>
                        {!attr.feedbackGiven ? (
                          <div className="feedback-actions" style={{ marginTop: 0 }}>
                            <button className="btn-feedback pos" style={{ padding: '0.5rem', fontSize: '0.8rem' }} onClick={() => handleFeedback(attr.attribute, 'positive')}><ThumbsUp size={14} /> Reward</button>
                            <button className="btn-feedback neg" style={{ padding: '0.5rem', fontSize: '0.8rem' }} onClick={() => handleFeedback(attr.attribute, 'negative')}><ThumbsDown size={14} /> Penalize</button>
                          </div>
                        ) : (
                          <div style={{ color: 'var(--stat-pos)', fontSize: '0.85rem', fontWeight: 500 }}>
                            <ThumbsUp size={14} style={{ verticalAlign: 'middle', marginRight: '0.5rem' }} /> Feedback recorded
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'fairness' && (
            <div className="animate-fade-in" style={{ flex: 1, overflowY: 'auto', paddingRight: '0.5rem' }}>
              <div className="profile-card">
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Shield size={20} color="var(--accent-blue)" /> AIF360 Post-Processing
                </h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                  IBM AIF360 analyzes your session decisions for statistical bias and can correct them using post-processing algorithms.
                </p>

                {!backendAlive && (
                  <div className="warning-box">
                    <AlertTriangle size={16} /> Backend offline. Start it with: <code>cd biasscope/backend && uvicorn main:app --reload</code>
                  </div>
                )}

                <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                  <select className="select-input" value={selectedProtected} onChange={e => setSelectedProtected(e.target.value)}>
                    <option value="Gender">Gender</option>
                    <option value="Age">Age</option>
                    <option value="Race">Race</option>
                  </select>
                  <select className="select-input" value={selectedAlgorithm} onChange={e => setSelectedAlgorithm(e.target.value)}>
                    <option value="calibrated_eq_odds">Calibrated Eq. Odds</option>
                    <option value="eq_odds">Equalized Odds</option>
                    <option value="reject_option">Reject Option</option>
                  </select>
                </div>

                <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem' }}>
                  <button className="btn-primary" onClick={handleAnalyzeBias} disabled={analyzingBias || sessionHistory.length < 2} style={{ flex: 1 }}>
                    <Microscope size={16} /> {analyzingBias ? 'Analyzing...' : `Analyze Bias (${sessionHistory.length} decisions)`}
                  </button>
                  <button className="btn-correction" onClick={handleApplyCorrection} disabled={correcting || sessionHistory.length < 2 || !biasMetrics?.bias_detected} style={{ flex: 1 }}>
                    <Zap size={16} /> {correcting ? 'Correcting...' : 'Apply Correction'}
                  </button>
                </div>

                {sessionHistory.length < 2 && (
                  <p style={{ fontSize: '0.8rem', color: 'var(--stat-caution)', textAlign: 'center' }}>
                    Need at least 2 decisions to run bias analysis
                  </p>
                )}
              </div>

              {biasMetrics && (
                <div className="profile-card animate-fade-in">
                  <h3 style={{ marginBottom: '1rem' }}>Bias Metrics</h3>
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <div className="metric-label">Disparate Impact</div>
                      <div className="metric-value" style={{ color: getBiasColor(biasMetrics.disparate_impact) }}>
                        {biasMetrics.disparate_impact ?? 'N/A'}
                      </div>
                      <div className="metric-range">Fair: 0.80 – 1.25</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Statistical Parity</div>
                      <div className="metric-value" style={{ color: getBiasColor(biasMetrics.disparate_impact) }}>
                        {biasMetrics.statistical_parity_diff ?? 'N/A'}
                      </div>
                      <div className="metric-range">Fair: close to 0</div>
                    </div>
                  </div>
                  <div className={`bias-badge ${biasMetrics.bias_detected ? 'biased' : 'fair'}`}>
                    {biasMetrics.bias_detected ? <><AlertTriangle size={14} /> Bias Detected ({biasMetrics.severity})</> : <><CheckCircle size={14} /> No Significant Bias</>}
                  </div>
                  {biasMetrics.threshold_info && (
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>{biasMetrics.threshold_info.interpretation}</p>
                  )}
                </div>
              )}

              {recommendation && recommendation.recommendation !== 'none' && (
                <div className="profile-card animate-fade-in" style={{ borderColor: 'var(--accent-purple)' }}>
                  <h3 style={{ color: 'var(--accent-purple)', marginBottom: '0.5rem' }}>Algorithm Recommendation</h3>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{recommendation.reason}</p>
                </div>
              )}

              {correctionResult && correctionResult.success && (
                <div className="profile-card animate-fade-in" style={{ borderColor: 'var(--stat-pos)' }}>
                  <h3 style={{ color: 'var(--stat-pos)', marginBottom: '1rem' }}>
                    <Zap size={18} style={{ verticalAlign: 'middle' }} /> Correction Applied
                  </h3>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>{correctionResult.algorithm_description}</p>
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <div className="metric-label">Before DI</div>
                      <div className="metric-value" style={{ color: 'var(--stat-neg)' }}>{correctionResult.original_metrics?.disparate_impact}</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">After DI</div>
                      <div className="metric-value" style={{ color: 'var(--stat-pos)' }}>{correctionResult.corrected_metrics?.disparate_impact}</div>
                    </div>
                  </div>
                  <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(6,214,160,0.1)', borderRadius: '6px', fontSize: '0.85rem' }}>
                    <strong>{correctionResult.improvement?.decisions_flipped}</strong> of {correctionResult.improvement?.total_decisions} decisions flipped ({correctionResult.improvement?.flip_percentage}%)
                  </div>
                </div>
              )}

              {correctionResult && !correctionResult.success && (
                <div className="profile-card animate-fade-in" style={{ borderColor: 'var(--stat-neg)' }}>
                  <h3 style={{ color: 'var(--stat-neg)' }}>Correction Failed</h3>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{correctionResult.error}</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'insights' && (
            <div className="animate-fade-in" style={{ flex: 1, overflowY: 'auto', paddingRight: '0.5rem' }}>
              <div className="profile-card">
                <h3 style={{ marginBottom: '0.5rem' }}>AI Learning Memory</h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                  The RL Memory Bank injected into the AI's prompt. It dictates what the model has learned from your feedback.
                </p>
                {['job', 'college', 'loan'].map(domain => {
                  const mem = getMemory()[domain];
                  if (!mem) return null;
                  return (
                    <div key={domain} style={{ marginBottom: '1.5rem', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid var(--panel-border)' }}>
                      <h4 style={{ color: 'var(--accent-purple)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '1rem' }}>{domain} Domain</h4>
                      <div style={{ marginBottom: '1rem' }}>
                        <div className="profile-label" style={{ color: 'var(--stat-pos)', marginBottom: '0.5rem' }}>Rewarded:</div>
                        {mem.positive.length > 0 ? mem.positive.map(attr => (
                          <span key={attr} className="attribute-tag pos">{attr}</span>
                        )) : <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>None yet</span>}
                      </div>
                      <div>
                        <div className="profile-label" style={{ color: 'var(--stat-neg)', marginBottom: '0.5rem' }}>Penalized:</div>
                        {mem.negative.length > 0 ? mem.negative.map(attr => (
                          <span key={attr} className="attribute-tag neg">{attr}</span>
                        )) : <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>None yet</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Bias Tracker */}
        <div className="left-column" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Bias Tracker</span>
              <span style={{ textTransform: 'none', color: 'var(--text-muted)' }}>{stats.allAttributes.length} patterns</span>
            </div>
            <div className="stats-grid">
              {[
                ['TOTAL', stats.totalDecisions, 'var(--stat-total)'],
                ['NEG BIAS', stats.negBiasCount, 'var(--stat-neg)'],
                ['POS BIAS', stats.posBiasCount, 'var(--stat-pos)'],
                ['CAUTION', stats.cautionCount, 'var(--stat-caution)'],
              ].map(([label, value, color]) => (
                <div key={label} className="stat-box">
                  <div style={{ width: '24px', height: '12px', borderRadius: '6px', border: `3px solid ${color}`, marginBottom: '0.5rem' }}></div>
                  <div className="stat-value">{value}</div>
                  <div className="stat-label">{label}</div>
                </div>
              ))}
            </div>

            {biasMetrics && (
              <div style={{ marginBottom: '1.5rem' }}>
                <div className="panel-header">AIF360 Score</div>
                <div className="aif360-score-bar">
                  <div className="score-fill" style={{ width: `${Math.min((biasMetrics.disparate_impact || 0) / 2 * 100, 100)}%`, background: getBiasColor(biasMetrics.disparate_impact) }}></div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  <span>0 (Biased)</span><span>1.0 (Fair)</span><span>2.0 (Over-favor)</span>
                </div>
              </div>
            )}

            <div className="panel-header" style={{ marginTop: '1.5rem' }}>Detected Attributes</div>
            <div style={{ flex: 1, overflowY: 'auto' }}>
              {stats.allAttributes.length === 0 ? (
                <div className="empty-state" style={{ fontSize: '0.85rem' }}>No patterns detected yet</div>
              ) : (
                <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                  {stats.allAttributes.map((attr, idx) => (
                    <span key={idx} className={`attribute-tag ${attr.type}`}>{attr.attr}</span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
