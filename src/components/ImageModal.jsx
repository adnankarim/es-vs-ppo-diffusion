import React, { useEffect, useState, useRef } from 'react';

const ImageModal = ({ src, alt, onClose }) => {
    const [scale, setScale] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [startPos, setStartPos] = useState({ x: 0, y: 0 });
    const imgRef = useRef(null);

    useEffect(() => {
        const handleEsc = (e) => {
            if (e.key === 'Escape') onClose();
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, [onClose]);

    // Reset when src changes
    useEffect(() => {
        setScale(1);
        setPosition({ x: 0, y: 0 });
    }, [src]);

    const handleWheel = (e) => {
        e.preventDefault();
        e.stopPropagation();

        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.min(Math.max(scale * delta, 0.5), 5); // Limit zoom 0.5x - 5x
        setScale(newScale);

        // Center on reset
        if (newScale <= 1) {
            setPosition({ x: 0, y: 0 });
        }
    };

    const handleMouseDown = (e) => {
        if (scale > 1) {
            e.preventDefault();
            setIsDragging(true);
            setStartPos({ x: e.clientX - position.x, y: e.clientY - position.y });
        }
    };

    const handleMouseMove = (e) => {
        if (!isDragging) return;
        e.preventDefault();
        const newX = e.clientX - startPos.x;
        const newY = e.clientY - startPos.y;
        setPosition({ x: newX, y: newY });
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    const resetZoom = (e) => {
        e.stopPropagation();
        setScale(1);
        setPosition({ x: 0, y: 0 });
    };

    const zoomIn = (e) => {
        e.stopPropagation();
        setScale(Math.min(scale * 1.2, 5));
    };

    const zoomOut = (e) => {
        e.stopPropagation();
        const newScale = Math.max(scale * 0.8, 0.5);
        setScale(newScale);
        if (newScale <= 1) setPosition({ x: 0, y: 0 });
    };

    if (!src) return null;

    return (
        <div
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: 'rgba(0, 0, 0, 0.9)',
                zIndex: 1000,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '40px',
                animation: 'fadeIn 0.2s ease-out',
                overflow: 'hidden'
            }}
            onClick={onClose}
            onWheel={handleWheel}
        >
            <div
                style={{
                    position: 'relative',
                    width: '100vw',
                    height: '100vh',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: scale > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default'
                }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onClick={(e) => e.stopPropagation()}
            >
                <img
                    ref={imgRef}
                    src={src}
                    alt={alt}
                    style={{
                        maxWidth: '90vw',
                        maxHeight: '90vh',
                        borderRadius: '4px',
                        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
                        objectFit: 'contain',
                        transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
                        transition: isDragging ? 'none' : 'transform 0.1s ease-out',
                        pointerEvents: 'none' // Let events bubble to container
                    }}
                />

                {/* Controls */}
                <div style={{
                    position: 'absolute',
                    bottom: '30px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    display: 'flex',
                    gap: '12px',
                    background: 'rgba(255, 255, 255, 0.1)',
                    backdropFilter: 'blur(5px)',
                    padding: '8px 16px',
                    borderRadius: '24px',
                    zIndex: 1001
                }}>
                    <button
                        onClick={zoomOut}
                        disabled={scale <= 0.5}
                        style={{ background: 'none', border: 'none', color: 'white', fontSize: '20px', cursor: 'pointer', opacity: scale <= 0.5 ? 0.3 : 1 }}
                    >
                        −
                    </button>
                    <span style={{ color: 'white', minWidth: '40px', textAlign: 'center', lineHeight: '24px', fontSize: '14px' }}>
                        {Math.round(scale * 100)}%
                    </span>
                    <button
                        onClick={zoomIn}
                        disabled={scale >= 5}
                        style={{ background: 'none', border: 'none', color: 'white', fontSize: '20px', cursor: 'pointer', opacity: scale >= 5 ? 0.3 : 1 }}
                    >
                        +
                    </button>
                    <button
                        onClick={resetZoom}
                        style={{ background: 'none', border: 'none', color: '#94a3b8', fontSize: '14px', cursor: 'pointer', marginLeft: '8px', borderLeft: '1px solid rgba(255,255,255,0.2)', paddingLeft: '12px' }}
                    >
                        Reset
                    </button>
                </div>

                <div style={{
                    position: 'absolute',
                    top: '20px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    color: 'rgba(255, 255, 255, 0.5)',
                    fontSize: '12px',
                    pointerEvents: 'none'
                }}>
                    Scroll to zoom • Drag to pan
                </div>

                <button
                    onClick={onClose}
                    style={{
                        position: 'absolute',
                        top: '20px',
                        right: '40px',
                        background: 'transparent',
                        border: 'none',
                        color: 'white',
                        fontSize: '32px',
                        cursor: 'pointer',
                        padding: '8px',
                        zIndex: 1001
                    }}
                >
                    ×
                </button>
            </div>
            <style>
                {`
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
        `}
            </style>
        </div>
    );
};

export default ImageModal;
