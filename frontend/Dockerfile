FROM node:22-alpine AS base

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .

# Development target
FROM base AS development
EXPOSE 5173
CMD ["npm", "run", "dev"]

# Production target - builds static files
FROM base AS production
RUN npm run build
# dist/ contains the built files
