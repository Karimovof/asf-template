import React from 'react';
import {Cigarette} from 'lucide-react';

function Logo() {
  return (
    <div className='logo-style'>
      <Cigarette style={{
        transform: 'rotate(45deg)',
        color: '#ff0000',
        width: '40px',
        height: '30px',
       }} />
      <h1 style={{ 
        margin: 0, 
        fontSize: '1.6rem', 
        color: '#8e44ad', 
        fontWeight: 'bold', 
        fontFamily: 'Montserrat',
        marginLeft: '10px',
      }}>
        antismokefacts.com
      </h1>
    </div>
  );
}

export default Logo;
