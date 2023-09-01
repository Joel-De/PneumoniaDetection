
FROM node:20-alpine3.17

WORKDIR /app
ENV PATH /app/node_modules/.bin:$PATH

# Install deps
COPY package.json ./
COPY package-lock.json ./
RUN npm install --silent
RUN npm install react-scripts@3.4.1 -g --silent

# Copy application files
COPY . ./

CMD ["npm", "start"]