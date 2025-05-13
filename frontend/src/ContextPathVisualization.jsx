import React, { useState } from 'react';
import './style.css';

const ContextPathVisualization = ({ navigationHistory, facts, currentFactIndex, onSelectEntry }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Функция для переключения между свернутым и развернутым состоянием
  const toggleExpand = () => setIsExpanded(!isExpanded);

  // Получение текста факта по ID
  const getFactText = (factId) => {
    const fact = facts.find(f => f.id === factId);
    return fact ? fact.fact : 'Факт не найден';
  };

  // Получение текста с троеточием, если он слишком длинный
  const truncateText = (text, maxLength = 40) => {
    if (!text) return '';
    return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
  };

  // Функция для обработки выбора записи в истории
  const handleEntryClick = (entry) => {
    onSelectEntry(entry.factIndex);
    // Не сворачиваем панель после выбора, чтобы пользователь мог видеть свой путь
  };

  return (
    <div className={`context-path-container ${isExpanded ? 'context-path-expanded' : ''}`}>
      {!isExpanded ? (
        <button className="context-path-button" onClick={toggleExpand} title="Show journey history">
          ⚲
        </button>
      ) : (
        <>
          <div className="context-path-header">
            <button className="context-path-close-button" onClick={toggleExpand} title="Close">
              ✕
            </button>
          </div>
          <div className="path-container">
            <h3 className="path-title">History of your fact journey</h3>
            
            {navigationHistory.length > 1 ? (
              navigationHistory.map((entry, index) => {
                // Пропускаем первую запись, так как это начальная точка
                if (index === 0) return null;
                
                const isActive = currentFactIndex === entry.factIndex;
                const previousEntry = navigationHistory[index - 1];
                
                return (
                  <div 
                    key={`history-${index}`}
                    className={`path-entry ${isActive ? 'path-entry-active' : ''}`}
                    onClick={() => handleEntryClick(entry)}
                    title="Click to navigate to this fact"
                  >
                    <div className="path-entry-number">{index}</div>
                    <div className="path-entry-content">
                      {entry.source === 'navigation' ? (
                        // Навигация стрелками
                        <div className="fact-text">
                          Transition {entry.direction === 'prev' ? '⬅️' : '➡️'}: {truncateText(getFactText(entry.factId))}
                        </div>
                      ) : (
                        // Переход по подсказке
                        <>
                          <div className="fact-text">
                            {truncateText(getFactText(entry.factId))}
                          </div>
                          <div className="hint-text">
                            From: {truncateText(getFactText(previousEntry.factId), 30)}
                            <span className="arrow">→</span>
                            Hint: "{entry.hintText}"
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="empty-text">
                Start exploring facts by clicking on hints and navigation arrows.
              </div>
            )}
            
            <div className="legend">
            * Click on any entry to return to the corresponding fact
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ContextPathVisualization;