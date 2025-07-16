## DevBot Chatbot — Developer Assistant

This full-stack AI chatbot helps developers learn, debug, and compile code across Python, C, C++, Java, and R.

### 🚀 Live Setup
- **Backend:** FastAPI hosted on Render
- **Frontend:** React (Vite) hosted on Vercel
- **AI:** GPT-4 (OpenAI API)
- **Compiler:** Judge0 (RapidAPI)

### 🛠️ To Deploy
1. Add your `.env` in backend:
   ```
   OPENAI_API_KEY=your_openai_key
   RAPIDAPI_KEY=your_rapidapi_key
   ```
2. Deploy backend to Render
3. Update frontend `App.jsx` with Render URL
4. Deploy frontend to Vercel
5. Done 🎉
