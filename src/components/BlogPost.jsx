import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import remarkGfm from 'remark-gfm'
import rehypeKatex from 'rehype-katex'
import ImageModal from './ImageModal'
import { useState } from 'react'
import './BlogPost.css'

function BlogPost({ content }) {
  const [modalImage, setModalImage] = useState(null)
  // Ensure content is a string and handle any encoding issues
  let processedContent = typeof content === 'string' ? content : ''

  // Convert LaTeX delimiters to dollar signs for remark-math
  // Convert \[ ... \] to $$ ... $$ (display math - must be done first)
  processedContent = processedContent.replace(/\\\[([\s\S]*?)\\\]/g, (match, content) => `$$\n${content}\n$$`)
  // Convert \( ... \) to $ ... $ (inline math - single $)
  // Use non-greedy match to handle nested parentheses
  processedContent = processedContent.replace(/\\\(([\s\S]*?)\\\)/g, (match, content) => `$${content}$`)

  return (
    <article className="blog-post">
      <div className="blog-container">
        <ReactMarkdown
          remarkPlugins={[remarkMath, remarkGfm]}
          rehypePlugins={[rehypeKatex]}
          components={{
            // Custom rendering for images with error handling
            img: ({ node, ...props }) => (
              <img
                {...props}
                alt={props.alt || 'Research figure'}
                style={{ cursor: 'pointer' }}
                onClick={() => setModalImage({ src: props.src, alt: props.alt || 'Research figure' })}
                onError={(e) => {
                  e.target.style.display = 'none'
                  const caption = document.createElement('p')
                  caption.className = 'image-placeholder'
                  caption.textContent = `📊 Figure: ${props.alt || props.src}`
                  e.target.parentNode.insertBefore(caption, e.target)
                }}
              />
            ),
            // Custom table styling
            table: ({ node, ...props }) => (
              <div className="table-wrapper">
                <table {...props} />
              </div>
            ),
            // Custom code block styling
            code: ({ node, inline, className, children, ...props }) => {
              const match = /language-(\w+)/.exec(className || '')
              return inline ? (
                <code className="inline-code" {...props}>
                  {children}
                </code>
              ) : (
                <code className={className} {...props}>
                  {children}
                </code>
              )
            },
          }}
        >
          {processedContent}
        </ReactMarkdown>
      </div>
      <ImageModal
        src={modalImage?.src}
        alt={modalImage?.alt}
        onClose={() => setModalImage(null)}
      />
    </article>
  )
}

export default BlogPost

