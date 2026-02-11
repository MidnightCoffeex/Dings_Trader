/** @type {import('next').NextConfig} */
const nextConfig = {
  // output: 'export',  // Disabled - API routes need server mode
  // distDir: 'dist',
  images: {
    unoptimized: true,
  },
}

export default nextConfig
