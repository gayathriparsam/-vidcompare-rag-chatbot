/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        ink: { 950: "#0a0b10", 900: "#0f1117", 800: "#161922", 700: "#1f2330" },
        neon: { 500: "#7c3aed", 400: "#a78bfa" },
      },
    },
  },
  plugins: [],
};
