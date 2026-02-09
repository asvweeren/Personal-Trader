FROM node:22-alpine AS base

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .

# Development target
FROM base AS development
EXPOSE 5173
CMD ["npm", "run", "dev"]

# Production target
FROM base AS production
RUN npm run build

FROM nginx:alpine AS production-serve
COPY --from=production /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
