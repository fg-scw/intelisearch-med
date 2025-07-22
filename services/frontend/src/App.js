// services/frontend/src/App.js
// Interface React pour démonstration du pipeline de recherche

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState('hybrid');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [specialty, setSpecialty] = useState('');
  const [patientContext, setPatientContext] = useState({ age: '', gender: '' });
  const [stats, setStats] = useState(null);

  const specialties = [
    'cardiologie', 'pneumologie', 'neurologie', 'oncologie',
    'gastroenterologie', 'endocrinologie', 'néphrologie'
  ];

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_BASE}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Erreur récupération stats:', error);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const searchData = {
        query: query.trim(),
        search_type: searchType,
        limit: 10,
        specialty: specialty || null,
        patient_context: patientContext.age || patientContext.gender ? patientContext : null
      };

      const response = await axios.post(`${API_BASE}/search`, searchData);
      setResults(response.data);
    } catch (error) {
      console.error('Erreur de recherche:', error);
      alert('Erreur lors de la recherche');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8001/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      if (response.data.status === 'success') {
        alert(`Document traité avec succès! ID: ${response.data.document_id}`);
        fetchStats(); // Rafraîchir les statistiques
      } else {
        alert(`Erreur: ${response.data.error}`);
      }
    } catch (error) {
      console.error('Erreur upload:', error);
      alert('Erreur lors de l\'upload');
    } finally {
      setUploading(false);
    }
  };

  const highlightText = (text, query) => {
    if (!query) return text;
    const regex = new RegExp(`(${query})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🏥 Recherche Sémantique Médicale</h1>
        <p>Pipeline de recherche hybride avec IA pour documents médicaux</p>
      </header>

      <div className="container">
        {/* Section Upload */}
        <section className="upload-section">
          <h2>📄 Upload de Document</h2>
          <div className="upload-area">
            <input 
              type="file" 
              onChange={handleFileUpload} 
              accept=".txt,.pdf,.docx"
              disabled={uploading}
            />
            {uploading && <p>Traitement en cours...</p>}
          </div>
        </section>

        {/* Section Recherche */}
        <section className="search-section">
          <h2>🔍 Recherche Intelligente</h2>
          
          <form onSubmit={handleSearch} className="search-form">
            <div className="search-input-container">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ex: douleur thoracique, diagnostic différentiel, traitement hypertension..."
                className="search-input"
              />
              <button type="submit" disabled={loading} className="search-button">
                {loading ? 'Recherche...' : 'Rechercher'}
              </button>
            </div>

            <div className="search-options">
              <div className="option-group">
                <label>Type de recherche:</label>
                <select value={searchType} onChange={(e) => setSearchType(e.target.value)}>
                  <option value="hybrid">Hybride (Textuel + Sémantique)</option>
                  <option value="semantic">Sémantique uniquement</option>
                  <option value="textual">Textuel uniquement</option>
                </select>
              </div>

              <div className="option-group">
                <label>Spécialité:</label>
                <select value={specialty} onChange={(e) => setSpecialty(e.target.value)}>
                  <option value="">Toutes spécialités</option>
                  {specialties.map(s => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>

              <div className="patient-context">
                <h4>Contexte Patient (optionnel):</h4>
                <input
                  type="number"
                  placeholder="Âge"
                  value={patientContext.age}
                  onChange={(e) => setPatientContext({...patientContext, age: e.target.value})}
                />
                <select 
                  value={patientContext.gender}
                  onChange={(e) => setPatientContext({...patientContext, gender: e.target.value})}
                >
                  <option value="">Genre</option>
                  <option value="M">Masculin</option>
                  <option value="F">Féminin</option>
                </select>
              </div>
            </div>
          </form>
        </section>

        {/* Statistiques */}
        {stats && (
          <section className="stats-section">
            <h3>📊 Statistiques du Système</h3>
            <div className="stats-grid">
              <div className="stat-card">
                <h4>Documents indexés</h4>
                <span className="stat-number">{stats.indexed_documents || 0}</span>
              </div>
              <div className="stat-card">
                <h4>Embeddings totaux</h4>
                <span className="stat-number">{stats.total_embeddings || 0}</span>
              </div>
              <div className="stat-card">
                <h4>Taille index</h4>
                <span className="stat-number">
                  {stats.index_size ? `${(stats.index_size / 1024 / 1024).toFixed(2)} MB` : '0 MB'}
                </span>
              </div>
            </div>
          </section>
        )}

        {/* Résultats */}
        <section className="results-section">
          <h2>📋 Résultats de Recherche</h2>
          
          {results.length > 0 && (
            <div className="results-info">
              <p>{results.length} résultat(s) trouvé(s)</p>
            </div>
          )}

          <div className="results-container">
            {results.map((result, index) => (
              <div key={result.document_id || index} className="result-card">
                <div className="result-header">
                  <h3 className="result-title">
                    {result.title || `Document ${result.document_id}`}
                  </h3>
                  <div className="result-scores">
                    {result.type === 'hybrid' && (
                      <>
                        <span className="score hybrid">
                          Score: {result.hybrid_score?.toFixed(3)}
                        </span>
                        <span className="score textual">
                          Textuel: {result.relevance_score?.toFixed(3)}
                        </span>
                        <span className="score semantic">
                          Sémantique: {result.semantic_score?.toFixed(3)}
                        </span>
                      </>
                    )}
                    {result.type !== 'hybrid' && (
                      <span className="score">
                        Score: {result.score?.toFixed(3)}
                      </span>
                    )}
                  </div>
                </div>

                <div className="result-content">
                  <div 
                    className="result-snippet"
                    dangerouslySetInnerHTML={{
                      __html: highlightText(result.snippet || result.content?.substring(0, 300) + '...', query)
                    }}
                  />
                </div>

                <div className="result-meta">
                  <span className="document-id">ID: {result.document_id}</span>
                  <span className="search-type">Type: {result.type}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;

