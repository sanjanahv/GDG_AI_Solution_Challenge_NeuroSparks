import React, { useState, useEffect } from 'react';
import { 
  Briefcase, GraduationCap, Building2, Microscope, 
  Key, RefreshCw, ThumbsUp, ThumbsDown, Activity, Save
} from 'lucide-react';
import { generateProfile, makeDecision, setApiKey as setLlmApiKey } from './services/llm';
import { getMemory, addFeedback, getStats, incrementDecisions } from './services/rl';

function App() {
  const [apiKey, setApiKey] = useState(localStorage.getItem('bias_scope_apikey') || '');
  const [activeDomain, setActiveDomain] = useState('Job');
  const [loading, setLoading] = useState(false);
  const [profile, setProfile] = useState(null);
  const [decision, setDecision] = useState(null);
  const [stats, setStats] = useState(getStats());
  const [sessionHistory, setSessionHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('analysis'); // 'analysis' or 'insights'

  // Sync API Key
  useEffect(() => {
    if (apiKey) {
      localStorage.setItem('bias_scope_apikey', apiKey);
      setLlmApiKey(apiKey);
    }
  }, [apiKey]);

  const handleGenerate = async () => {
    if (!apiKey) return alert("Please enter a Gemini API Key top right.");
    setLoading(true);
    setProfile(null);
    setDecision(null);
    try {
      const newProfile = await generateProfile(activeDomain);
      setProfile(newProfile);
      
      // Make decision based on current memory
      const memory = getMemory();
      const newDecision = await makeDecision(activeDomain, newProfile, memory);
      setDecision(newDecision);
      incrementDecisions();
      setStats(getStats());
      
      setSessionHistory(prev => [{ domain: activeDomain, decision: newDecision, timestamp: Date.now() }, ...prev]);
    } catch (err) {
      console.error(err);
      alert("Error: " + err.message);
    }
    setLoading(false);
  };

  const handleFeedback = (attribute, type) => {
    if (!decision) return;
    addFeedback(activeDomain, attribute, type);
    setStats(getStats());
    const newDec = { ...decision };
    if (newDec.weightedAttributes) {
      const attrObj = newDec.weightedAttributes.find(a => a.attribute === attribute);
      if (attrObj) attrObj.feedbackGiven = true;
    }
    setDecision(newDec);
  };

  return (
    <div className="app-container">
      {/* Top Bar */}
      <header className="top-bar">
        <div className="logo-section">
          <div className="logo-icon">
            <Activity color="white" size={24} />
          </div>
          <div className="logo-text">
            <h1>BiasScope</h1>
            <p>RL Bias Detection System</p>
          </div>
        </div>
        <div className="api-status">
          <div className={`status-dot ${apiKey ? 'active' : ''}`}></div>
          <div style={{ position: 'relative' }}>
            <input 
              type="password" 
              className="api-key-input" 
              placeholder="Gemini API Key..." 
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
            <Key size={14} style={{ position: 'absolute', right: 10, top: 10, color: 'var(--text-muted)' }} />
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="main-grid">
        
        {/* Left Column: Domain & History */}
        <div className="left-column" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="panel">
            <div className="panel-header">Domain</div>
            <div className="domains-grid">
              <button 
                className={`btn-domain ${activeDomain === 'Job' ? 'active' : ''}`}
                onClick={() => setActiveDomain('Job')}
              >
                <Briefcase size={20} color={activeDomain === 'Job' ? "var(--accent-purple)" : "var(--text-muted)"} />
                <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>JOB</span>
              </button>
              <button 
                className={`btn-domain ${activeDomain === 'College' ? 'active' : ''}`}
                onClick={() => setActiveDomain('College')}
              >
                <GraduationCap size={20} color={activeDomain === 'College' ? "var(--accent-purple)" : "var(--text-muted)"} />
                <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>COLLEGE</span>
              </button>
              <button 
                className={`btn-domain ${activeDomain === 'Loan' ? 'active' : ''}`}
                onClick={() => setActiveDomain('Loan')}
              >
                <Building2 size={20} color={activeDomain === 'Loan' ? "var(--accent-purple)" : "var(--text-muted)"} />
                <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>LOAN</span>
              </button>
            </div>
            
            <p style={{ textAlign: 'center', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
              Generate a profile to begin
            </p>
            
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
                    <div>Top Weight: <span style={{ color: 'var(--accent-blue)' }}>{sh.decision.weightedAttributes ? sh.decision.weightedAttributes[0]?.attribute : sh.decision.weightedAttribute}</span></div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Center Column: Analysis & Insights */}
        <div className="panel">
          <div className="center-tabs">
            <button className={`tab ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')}>Analysis</button>
            <button className={`tab ${activeTab === 'insights' ? 'active' : ''}`} onClick={() => setActiveTab('insights')}>Insights</button>
          </div>

          {!profile && activeTab === 'analysis' ? (
            <div className="empty-state">
              <Microscope size={48} color="var(--panel-border)" style={{ marginBottom: '1rem' }} />
              <p>Generate a profile and make a decision</p>
              <p style={{ fontSize: '0.75rem', opacity: 0.7 }}>The AI will analyze what drove your choice</p>
            </div>
          ) : activeTab === 'insights' ? (
            <div className="animate-fade-in" style={{ flex: 1, overflowY: 'auto', paddingRight: '0.5rem' }}>
              <div className="profile-card">
                <h3 style={{ marginBottom: '0.5rem' }}>AI Learning Memory</h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                  This is the actual "Memory Bank" that is injected into the AI's prompt. It dictates what the model has learned from your past rewards and penalties.
                </p>
                
                {['job', 'college', 'loan'].map(domain => {
                  const mem = getMemory()[domain];
                  if (!mem) return null;
                  return (
                    <div key={domain} style={{ marginBottom: '1.5rem', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid var(--panel-border)' }}>
                      <h4 style={{ color: 'var(--accent-purple)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '1rem' }}>{domain} Domain</h4>
                      
                      <div style={{ marginBottom: '1rem' }}>
                        <div className="profile-label" style={{ color: 'var(--stat-pos)', marginBottom: '0.5rem' }}>Rewarded (Positive Bias):</div>
                        {mem.positive.length > 0 ? mem.positive.map(attr => (
                          <span key={attr} className="attribute-tag pos">{attr}</span>
                        )) : <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>None yet</span>}
                      </div>

                      <div>
                        <div className="profile-label" style={{ color: 'var(--stat-neg)', marginBottom: '0.5rem' }}>Penalized (Negative Bias):</div>
                        {mem.negative.length > 0 ? mem.negative.map(attr => (
                          <span key={attr} className="attribute-tag neg">{attr}</span>
                        )) : <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>None yet</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
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
                    
                    {decision.weightedAttributes ? decision.weightedAttributes.map((attr, idx) => (
                      <div key={idx} style={{ marginBottom: '1rem', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid var(--panel-border)' }}>
                        <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-pink)', marginBottom: '0.5rem' }}>
                          {attr.attribute} <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>({attr.weight})</span>
                        </div>
                        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>{attr.reasoning}</p>
                        
                        {!attr.feedbackGiven ? (
                          <div className="feedback-actions" style={{ marginTop: 0 }}>
                            <button className="btn-feedback pos" style={{ padding: '0.5rem', fontSize: '0.8rem' }} onClick={() => handleFeedback(attr.attribute, 'positive')}>
                              <ThumbsUp size={14} /> Reward
                            </button>
                            <button className="btn-feedback neg" style={{ padding: '0.5rem', fontSize: '0.8rem' }} onClick={() => handleFeedback(attr.attribute, 'negative')}>
                              <ThumbsDown size={14} /> Penalize
                            </button>
                          </div>
                        ) : (
                          <div style={{ color: 'var(--stat-pos)', fontSize: '0.85rem', fontWeight: 500 }}>
                            <ThumbsUp size={14} style={{ verticalAlign: 'middle', marginRight: '0.5rem' }} /> Feedback recorded
                          </div>
                        )}
                      </div>
                    )) : (
                      <div>
                        <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-pink)', marginBottom: '0.5rem' }}>
                          {decision.weightedAttribute}
                        </div>
                        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>{decision.reasoning}</p>
                        {!decision.feedbackGiven && (
                          <div className="feedback-actions">
                            <button className="btn-feedback pos" onClick={() => handleFeedback(decision.weightedAttribute, 'positive')}><ThumbsUp size={16} /> Reward</button>
                            <button className="btn-feedback neg" onClick={() => handleFeedback(decision.weightedAttribute, 'negative')}><ThumbsDown size={16} /> Penalize</button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
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
              <div className="stat-box">
                <div style={{ width: '24px', height: '12px', borderRadius: '6px', border: '3px solid var(--stat-total)', marginBottom: '0.5rem' }}></div>
                <div className="stat-value">{stats.totalDecisions}</div>
                <div className="stat-label">TOTAL</div>
              </div>
              <div className="stat-box">
                <div style={{ width: '24px', height: '12px', borderRadius: '6px', border: '3px solid var(--stat-neg)', marginBottom: '0.5rem' }}></div>
                <div className="stat-value">{stats.negBiasCount}</div>
                <div className="stat-label">NEG BIAS</div>
              </div>
              <div className="stat-box">
                <div style={{ width: '24px', height: '12px', borderRadius: '6px', border: '3px solid var(--stat-pos)', marginBottom: '0.5rem' }}></div>
                <div className="stat-value">{stats.posBiasCount}</div>
                <div className="stat-label">POS BIAS</div>
              </div>
              <div className="stat-box">
                <div style={{ width: '24px', height: '12px', borderRadius: '6px', border: '3px solid var(--stat-caution)', marginBottom: '0.5rem' }}></div>
                <div className="stat-value">{stats.cautionCount}</div>
                <div className="stat-label">CAUTION</div>
              </div>
            </div>

            <div className="panel-header" style={{ marginTop: '1.5rem' }}>Detected Attributes</div>
            <div style={{ flex: 1, overflowY: 'auto' }}>
              {stats.allAttributes.length === 0 ? (
                <div className="empty-state" style={{ fontSize: '0.85rem' }}>No patterns detected yet</div>
              ) : (
                <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                  {stats.allAttributes.map((attr, idx) => (
                    <span key={idx} className={`attribute-tag ${attr.type}`}>
                      {attr.attr}
                    </span>
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
