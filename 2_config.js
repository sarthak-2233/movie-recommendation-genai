import dotenv from 'dotenv';
import neo4j from 'neo4j-driver';
import { Pinecone } from "@pinecone-database/pinecone";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenAI } from "@google/genai";


// CONFIG ENV
dotenv.config();

// Neo4j
// Neo4j-driver == creates connection pool(not single connection)
// neo4j+s:// = Bolt protocol with TLS (required for Aura cloud)

const driver=neo4j.driver(
    process.env.NEO4J_URI,
    neo4j.auth.basic(process.env.NEO4J_USERNAME, process.env.NEO4J_PASSWORD)
)


// Pinecone
const pinecone= new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const pineconeIndex = pinecone.index(process.env.PINECONE_INDEX_NAME);

// GEMINI LLM
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
  apiKey: process.env.GEMINI_API_KEY,
  temperature: 0,
});
// GOOGLE GENAI SDK (needed for embeddings + PDF file uploads)
const genai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });


// Embed one text
async function embedText(text)
{
    const response = await genai.models.embedContent({
    model: "gemini-embedding-001",
    contents: text,
  });
  // response.embeddings is an ARRAY (even for single text)
  // Each element has .values (the actual vector)
  return response.embeddings[0].values;
}

// Embed multiple texts
async function embedTexts(texts) {
  const response = await genai.models.embedContent({
    model: "gemini-embedding-001",
    contents: texts,
  });
  return response.embeddings.map((e) => e.values);
}


// CLOSE ALL CONNECTION AFTER FINISH
async function closeConnections() {
  await driver.close();
  console.log("✅ All connections closed.");
}

export { driver, pinecone, pineconeIndex, llm, genai, embedText, embedTexts, closeConnections };