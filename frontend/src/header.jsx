import React from 'react';
import Logo from './logo';
import GradientEffect from './GradientEffect';

function Header() {
  return (
    <GradientEffect width="90%" height="auto">
      <header className='header-style'>
        <Logo />
      </header>
    </GradientEffect>
  );
}

export default Header;
