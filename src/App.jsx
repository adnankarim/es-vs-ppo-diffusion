import { useState } from 'react'
import ResearchPaper from './components/ResearchPaper'
import PresentationMode from './components/PresentationMode'
import './App.css'

function App() {
  const [isPresentationMode, setIsPresentationMode] = useState(false);

  return (
    <div className="app">
      {isPresentationMode ? (
        <PresentationMode onExit={() => setIsPresentationMode(false)} />
      ) : (
        <ResearchPaper onEnterPresentation={() => setIsPresentationMode(true)} />
      )}
    </div>
  )
}

export default App

