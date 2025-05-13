import React, { useState, useEffect, useRef } from 'react';

function GradientEffect({ width = '400px', height = '200px', children }) {
  const [style, setStyle] = useState({});
  const startTimeRef = useRef(performance.now());
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  // Отслеживаем движение мыши
  useEffect(() => {
    function handleMouseMove(e) {
      setMousePos({ x: e.clientX, y: e.clientY });
    }
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Анимация: пересчитываем градиент на каждом кадре
  useEffect(() => {
    let frameId;

    function animate(time) {
      const elapsed = time - startTimeRef.current;
      const { innerWidth, innerHeight } = window;
      
      // Нормализованные координаты
      const xPercent = mousePos.x / innerWidth;
      const yPercent = mousePos.y / innerHeight;
      
      // Угол градиента + лёгкая «турбулентность»
      const angle = 45 + xPercent * 180 + Math.sin(elapsed * 0.0005) * 20;

      // Изменяем параметры HSL:
      // - оставляем hue (оттенок) как есть
      // - уменьшаем saturation (насыщенность) до 60%
      // - увеличиваем lightness (яркость) до 70%
      const hue1 = Math.floor(xPercent * 360 + Math.sin(elapsed * 0.0003) * 30);
      const hue2 = Math.floor(yPercent * 360 + Math.cos(elapsed * 0.0003) * 30);
      const color1 = `hsl(${hue1}, 60%, 70%)`;
      const color2 = `hsl(${hue2}, 60%, 70%)`;

      // Записываем стили для фона
      setStyle({
        background: `linear-gradient(${angle}deg, ${color1}, ${color2})`
      });

      frameId = requestAnimationFrame(animate);
    }

    frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, [mousePos]);

  return (
    <div
      style={{
        ...style,
        width,
        height,
        position: 'relative',
        overflow: 'hidden',
        borderRadius: '20px',
        boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.1)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: height === 'auto' ? '80px' : height, // Минимальная высота для auto
      }}
    >
      {children}
    </div>
  );
}

export default GradientEffect;
