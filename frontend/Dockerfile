# 1. Use Node base image
FROM node:20-alpine

WORKDIR /app

# 2. Install dependencies first (for better caching)
COPY package*.json ./
RUN npm install

# 3. Copy the rest of your frontend code
COPY . .

# 4. Expose the port
EXPOSE 8081

# 5. Use CI=1 to signal a non-interactive environment (Docker)
ENV CI=1

CMD ["npx", "expo", "start", "--web"]