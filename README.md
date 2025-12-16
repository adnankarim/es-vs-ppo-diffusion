# Research Blog - Evolution Strategies vs PPO

A beautiful, modern blog application to display your research report with full LaTeX mathematical equation support.

## Features

✨ **Beautiful Design**: Modern gradient background with a clean, readable content area  
📐 **LaTeX Support**: Full mathematical equation rendering using KaTeX  
📱 **Responsive**: Works perfectly on desktop, tablet, and mobile devices  
🌓 **Dark Mode**: Automatic dark mode support based on system preferences  
📊 **Tables & Figures**: Styled tables and image support with graceful fallbacks  
🎨 **Syntax Highlighting**: Code blocks with syntax highlighting support  

## Tech Stack

- **React 18** - Modern React with hooks
- **Vite** - Lightning-fast build tool and dev server
- **react-markdown** - Markdown parsing and rendering
- **remark-math** - Math equation support in markdown
- **rehype-katex** - LaTeX rendering with KaTeX
- **remark-gfm** - GitHub Flavored Markdown support

## Installation

### Prerequisites

- Node.js 18+ and npm (or yarn/pnpm)

### Setup

1. **Install dependencies:**

```bash
npm install
```

or with yarn:

```bash
yarn install
```

2. **Start the development server:**

```bash
npm run dev
```

or with yarn:

```bash
yarn dev
```

3. **Open your browser:**

Navigate to `http://localhost:5173` (or the URL shown in your terminal)

## Build for Production

To create a production build:

```bash
npm run build
```

To preview the production build:

```bash
npm run preview
```

The production files will be in the `dist/` directory.

## Project Structure

```
research-blog/
├── public/
│   └── content/
│       └── index.md         # Your research report
├── src/
│   ├── components/
│   │   ├── BlogPost.jsx     # Main blog post component
│   │   └── BlogPost.css     # Blog post styles
│   ├── App.jsx              # Main app component
│   ├── App.css              # App styles
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── index.html               # HTML template
├── package.json             # Dependencies
└── vite.config.js           # Vite configuration
```

## Customization

### Changing Colors

Edit `src/App.css` to change the gradient background:

```css
.app {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

Edit `src/components/BlogPost.css` to change heading colors and other styles.

### Adding Your Own Content

Replace the file `public/content/index.md` with your own markdown content. The blog supports:

- Standard markdown formatting
- LaTeX equations (inline with `$...$` or `\\(...\\)`, display with `$$...$$` or `\\[...\\]`)
- Tables (GitHub Flavored Markdown)
- Code blocks with syntax highlighting
- Images

### Math Equation Examples

**Inline math:** `$E = mc^2$` renders as $E = mc^2$

**Display math:**

```
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$
```

## Features in Detail

### LaTeX Rendering

The blog uses KaTeX for fast, high-quality LaTeX rendering. All standard LaTeX commands are supported. Math is rendered using:

- **remark-math**: Parses math notation in markdown
- **rehype-katex**: Renders LaTeX using KaTeX
- **katex.min.css**: KaTeX stylesheet for proper formatting

### Responsive Tables

Tables automatically become scrollable on mobile devices and feature:
- Gradient headers
- Hover effects
- Clean borders
- Mobile-optimized padding

### Image Handling

Images that fail to load show a placeholder with the alt text, preventing broken layouts.

### Dark Mode

The blog automatically detects your system's color scheme preference and adjusts:
- Background colors
- Text colors
- Code block themes
- Table styles

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Opera 76+

## Performance

- ⚡ Fast initial load with Vite
- 📦 Optimized production builds with code splitting
- 🎯 Lazy loading for images
- 💨 Smooth rendering with React 18

## License

MIT

## Credits

Built with ❤️ using React, Vite, and KaTeX

---

**Note:** This blog is optimized for displaying research papers and technical content with mathematical equations. Feel free to customize the styling to match your preferences!

