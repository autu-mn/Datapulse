/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // 深色主题色板 - 赛博朋克风格
        'cyber': {
          'bg': '#0a0e17',
          'surface': '#111827',
          'card': '#1a2332',
          'border': '#2d3a4f',
          'primary': '#00f5d4',
          'secondary': '#7b61ff',
          'accent': '#ff6b9d',
          'warning': '#ffd93d',
          'success': '#00ff88',
          'text': '#e8ecf4',
          'muted': '#8b97a8'
        }
      },
      fontFamily: {
        'display': ['Orbitron', 'monospace'],
        'body': ['JetBrains Mono', 'Fira Code', 'monospace'],
        'chinese': ['Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', 'sans-serif']
      },
      backgroundImage: {
        'grid-pattern': `linear-gradient(rgba(0, 245, 212, 0.03) 1px, transparent 1px),
                         linear-gradient(90deg, rgba(0, 245, 212, 0.03) 1px, transparent 1px)`,
        'glow-radial': 'radial-gradient(ellipse at center, rgba(0, 245, 212, 0.15) 0%, transparent 70%)',
      },
      backgroundSize: {
        'grid': '50px 50px',
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'slide-up': 'slide-up 0.5s ease-out',
        'fade-in': 'fade-in 0.6s ease-out',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 20px rgba(0, 245, 212, 0.3)' },
          '50%': { boxShadow: '0 0 40px rgba(0, 245, 212, 0.6)' },
        },
        'slide-up': {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        }
      }
    },
  },
  plugins: [],
}







