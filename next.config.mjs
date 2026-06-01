/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static HTML export — produces an `out/` folder Netlify serves directly.
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
