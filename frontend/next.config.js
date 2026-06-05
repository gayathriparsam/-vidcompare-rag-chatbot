/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    const base = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
    return [{ source: "/api/:path*", destination: `${base}/api/:path*` }];
  },
};
module.exports = nextConfig;
