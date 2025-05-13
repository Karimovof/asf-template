import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './style.css';

const MenuButton = () => {
  const [isExpanded, setIsExpanded] = useState(false);
  const navigate = useNavigate();

  // Проверка, работает ли изменение класса
  useEffect(() => {
    console.log("Menu expanded state:", isExpanded);
  }, [isExpanded]);

  // Toggle function for expanded/collapsed state
  const toggleExpand = () => setIsExpanded(!isExpanded);

  
  // Handle menu item click
  const handleMenuItemClick = (item) => {
    navigate(`/${item}`);
    // Add your navigation or modal logic here
    
    // Optional: collapse menu after selection
    setIsExpanded(false);
  };

  return (
    <div className={`menu-container ${isExpanded ? 'menu-expanded' : ''}`}>
      {!isExpanded ? (
        <button className="menu-button" onClick={toggleExpand} title="Menu">
          ☰
        </button>
      ) : (
        <>
          <div className="menu-header">
            <button className="menu-close-button" onClick={toggleExpand} title="Close">
              ✕
            </button>
          </div>
          <div className="menu-content">
            <div
              className="menu-item"
              onClick={() => handleMenuItemClick('')}
            >
              Home
            </div>
            <div 
              className="menu-item"
              onClick={() => handleMenuItemClick('about-project')}
            >
              About the project
            </div>
            <div 
              className="menu-item"
              onClick={() => handleMenuItemClick('about-content')}
            >
              About the content
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default MenuButton;