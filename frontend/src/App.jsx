import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './style.css';
import Header from './header';
import ContextPathVisualization from './ContextPathVisualization';
import MenuButton from './MenuButton';
import AboutProject from './AboutProject';
import AboutContent from './AboutContent';

const App = () => {
    const [facts, setFacts] = useState([]);
    const [currentFactIndex, setCurrentFactIndex] = useState(0);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [loadingHints, setLoadingHints] = useState(false);

    // Состояния для анимации
    const [phase, setPhase] = useState('idle');
    const [slideDirection, setSlideDirection] = useState(null);
    const [pendingIndex, setPendingIndex] = useState(null);

    // Добавляем историю переходов для отслеживания пути пользователя
    const [navigationHistory, setNavigationHistory] = useState([]);

    useEffect(() => {
        fetchInitialFacts();
    }, []);

    const fetchInitialFacts = async () => {
        setLoading(true);
        try {
            const response = await fetch('https://asf-backend.onrender.com/api/all_facts');
            if (!response.ok) throw new Error('Ошибка при загрузке фактов.');
            const data = await response.json();
            if (!data || !Array.isArray(data.facts)) throw new Error('Некорректные данные.');
            
            if (data.facts.length > 0) {
                setFacts(data.facts);
                setCurrentFactIndex(0);
                
                // Инициализируем историю переходов с первым фактом
                setNavigationHistory([{ 
                    factId: data.facts[0].id, 
                    factIndex: 0 
                }]);
            } else {
                // Если фактов нет, запросим новый
                await fetchNewFact();
            }
        } catch (error) {
            setError('Не удалось загрузить факты. Попробуйте позже.');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const fetchNewFact = async () => {
        setLoading(true);
        try {
            const response = await fetch('https://asf-backend.onrender.com/api/fact');
            if (!response.ok) throw new Error('Ошибка при загрузке факта.');
            const data = await response.json();
            
            // Добавляем новый факт в список фактов
            setFacts([data]);
            setCurrentFactIndex(0);
            
            // Добавляем в историю
            setNavigationHistory([{ factId: data.id, factIndex: 0 }]);
        } catch (error) {
            setError('Не удалось загрузить факт. Попробуйте позже.');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const handleFactNavigation = (direction) => {
        // Если фактов нет или анимация ещё не закончилась (phase != 'idle'), то не делаем ничего
        if (!facts.length || phase !== 'idle') return;

        let nextIndex;
        if (direction === 'prev') {
            nextIndex = (currentFactIndex - 1 + facts.length) % facts.length;
            setSlideDirection('left');
        } else {
            nextIndex = (currentFactIndex + 1) % facts.length;
            setSlideDirection('right');
        }

        // Запоминаем, какой индекс будет следующим
        setPendingIndex(nextIndex);

        // 1) Сначала запускаем фазу "out": старый текст уезжает
        setPhase('out');

        // Через 0.5с (сколько идёт "out"), обновим currentFactIndex и запустим "in"
        setTimeout(() => {
            setCurrentFactIndex(nextIndex);
            setPhase('in');
            
            // Добавляем переход в историю
            const newHistoryEntry = { 
                factId: facts[nextIndex].id, 
                factIndex: nextIndex,
                source: 'navigation',
                direction
            };
            setNavigationHistory(prev => [...prev, newHistoryEntry]);
        }, 500);

        // Ещё через 0.5с (всего 1с от начала) завершим анимацию и вернёмся в "idle"
        setTimeout(() => {
            setPhase('idle');
            setSlideDirection(null);
            setPendingIndex(null);
        }, 1000);
    };

    const handleHintClick = async (hint) => {
        if (!hint || hint.id === undefined) {
            console.warn("Ошибка: hint или hint.id не определен");
            return;
        }
        
        setLoadingHints(true);
    
        try {
            const currentFact = facts[currentFactIndex];
            
            const requestData = {
                factId: currentFact.id,
                hintId: hint.id,
                currentFact: currentFact.fact,
                hintText: hint.text
            };
    
            console.log('Отправляем запрос:', JSON.stringify(requestData)); // Логируем JSON-строку
    
            const response = await fetch('https://asf-backend.onrender.com/api/contextual_hint', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData),
            });
    
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Ошибка сервера:', response.status, errorText);
                throw new Error(`Ошибка при получении подсказки: ${response.status}`);
            }
            
            let data;
            try {
                data = await response.json();
                console.log('Ответ сервера:', data);
            } catch (jsonError) {
                console.error('Ошибка при разборе JSON ответа:', jsonError);
                throw new Error('Ошибка в формате ответа сервера.');
            }
    
            if (!data.fact || !data.hints) {
                console.warn('Ошибка: сервер вернул некорректные данные:', data);
                throw new Error('Сервер вернул некорректные данные.');
            }
    
            // Добавляем новый факт в список
            const updatedFacts = [...facts];
            
            // Проверяем, есть ли этот факт уже в списке
            const existingFactIndex = facts.findIndex(f => f.id === data.id);
            
            if (existingFactIndex >= 0) {
                // Если факт уже существует, просто переключаемся на него
                animateToFactIndex(existingFactIndex);
                
                // Добавляем в историю
                const newHistoryEntry = { 
                    factId: data.id, 
                    factIndex: existingFactIndex,
                    source: 'hint',
                    sourceFactId: currentFact.id,
                    sourceFactIndex: currentFactIndex,
                    hintId: hint.id,
                    hintText: hint.text
                };
                setNavigationHistory(prev => [...prev, newHistoryEntry]);
            } else {
                // Если факт новый, добавляем его в список
                updatedFacts.push(data);
                setFacts(updatedFacts);
                
                // Анимируем переход к новому факту
                const newFactIndex = updatedFacts.length - 1;
                animateToFactIndex(newFactIndex);
                
                // Добавляем в историю
                const newHistoryEntry = { 
                    factId: data.id, 
                    factIndex: newFactIndex,
                    source: 'hint',
                    sourceFactId: currentFact.id,
                    sourceFactIndex: currentFactIndex,
                    hintId: hint.id,
                    hintText: hint.text
                };
                setNavigationHistory(prev => [...prev, newHistoryEntry]);
            }
        } catch (error) {
            setError('Произошла ошибка при обработке подсказки.');
            console.error('Ошибка в handleHintClick:', error);
        } finally {
            setLoadingHints(false);
        }
    };
    
    // Вспомогательная функция для анимации перехода к заданному индексу факта
    const animateToFactIndex = (targetIndex) => {
        // Определяем направление в зависимости от индексов
        const direction = targetIndex > currentFactIndex ? 'right' : 'left';
        setSlideDirection(direction);
        
        // Запоминаем целевой индекс
        setPendingIndex(targetIndex);
        
        // Анимация выезда текущего факта
        setPhase('out');
        
        // Через 0.5с переходим к новому факту
        setTimeout(() => {
            setCurrentFactIndex(targetIndex);
            setPhase('in');
        }, 500);
        
        // Ещё через 0.5с завершаем анимацию
        setTimeout(() => {
            setPhase('idle');
            setSlideDirection(null);
            setPendingIndex(null);
        }, 1000);
    };

    if (loading) {
        return (
            <div className="loading-indicator">
                <div className="spinner" />
                <span>Загрузка...</span>
            </div>
        );
    }

    if (error) {
        return <div className="error-message">{error}</div>;
    }

    // Определяем классы анимации
    let factClass = 'fact';
    if (phase === 'out') {
        factClass += (slideDirection === 'left') ? ' slide-out-left' : ' slide-out-right';
    } else if (phase === 'in') {
        factClass += (slideDirection === 'left') ? ' slide-in-left' : ' slide-in-right';
    }

    return (
        <Router>
                <Header />
                <MenuButton />
                <Routes>
                    <Route 
                        path="/" 
                        element={
                             <div>
                                <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400&display=swap" rel="stylesheet"></link>
                                <div className="container">
                                    <div id="fact-text" className={factClass}>
                                        {facts.length > 0 ? facts[currentFactIndex]?.fact : 'No facts loaded.'}
                                    </div>

                                    <div id="hints-container" className="button-container">
                                        {facts.length > 0 && facts[currentFactIndex]?.hints?.length > 0 ? (
                                            facts[currentFactIndex].hints.map((hint, index) => (
                                                <button
                                                    key={`${hint.id || index}-${currentFactIndex}`}
                                                    className="hint-button"
                                                    onClick={() => handleHintClick(hint)}
                                                    disabled={loadingHints}
                                                >
                                                    {loadingHints ? <span style={{ color: 'gray' }}>Hint is loading...</span> : (hint.text || "No data loaded")}
                                                </button>
                                            ))
                                        ) : (
                                            <p>Подсказки отсутствуют</p>
                                        )}
                                    </div>

                                    <div className="gradient-button-left" onClick={() => handleFactNavigation('prev')}></div>
                                    <div className="gradient-button-right" onClick={() => handleFactNavigation('next')}></div>
                                    
                                    <ContextPathVisualization 
                                        navigationHistory={navigationHistory}
                                        facts={facts}
                                        currentFactIndex={currentFactIndex}
                                        onSelectEntry={(index) => animateToFactIndex(index)}
                                    />
                                </div>
                            </div>
                        }
                    />
                    <Route path="/about-project" element={
                        <div className="about-project-wrapper">
                            <AboutProject />
                        </div>
                    } />
                    <Route path="/about-content" element={
                        <div className="about-project-wrapper">
                            <AboutContent />
                        </div>
                    } />
                </Routes>
            </Router>
    );
};

export default App;