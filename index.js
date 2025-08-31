import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import multer from "multer";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAI } from "@google/generative-ai";
import fs from "fs";
import "dotenv/config";

const app = express();

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = "./uploads";
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname);
  },
});

const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === "application/pdf") {
      cb(null, true);
    } else {
      cb(new Error("Only PDF files are allowed!"), false);
    }
  },
});

app.use(cors());
app.use(bodyParser.json());

// Store vector stores in memory (in production, use a persistent database)
const vectorStores = new Map();

// Initialize Google Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GOOGLE);

// Upload PDF and create embeddings
app.post("/upload-pdf", upload.single("pdf"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No PDF file uploaded" });
    }

    const filePath = req.file.path;
    const documentId = req.file.filename.split(".")[0]; // Use filename without extension as ID

    // Load PDF document
    const loader = new PDFLoader(filePath);
    const docs = await loader.load();

    console.log(`Loaded ${docs.length} pages from PDF`);

    // Split documents into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000, // Size of each chunk
      chunkOverlap: 200, // Overlap between chunks
      separators: ["\n\n", "\n", " ", ""], // Separators to split on
    });

    const splitDocs = await textSplitter.splitDocuments(docs);
    console.log(`Split into ${splitDocs.length} chunks`);

    // Create embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: "text-embedding-004",
      taskType: TaskType.RETRIEVAL_DOCUMENT,
      title: req.file.originalname,
      apiKey: process.env.GOOGLE,
    });

    // Create vector store
    const vectorStore = await MemoryVectorStore.fromDocuments(
      splitDocs,
      embeddings
    );

    // Store the vector store for later queries
    vectorStores.set(documentId, {
      vectorStore,
      filename: req.file.originalname,
      chunks: splitDocs.length,
      createdAt: new Date(),
    });

    // Clean up uploaded file (optional)
    fs.unlinkSync(filePath);

    res.json({
      success: true,
      documentId,
      filename: req.file.originalname,
      chunks: splitDocs.length,
      message: "PDF processed and embeddings created successfully",
    });
  } catch (error) {
    console.error("Error processing PDF:", error);
    res.status(500).json({ error: error.message });
  }
});

// Query the PDF document
app.post("/query/:documentId", async (req, res) => {
  try {
    const { documentId } = req.params;
    const { question, topK = 3 } = req.body;

    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }

    const docData = vectorStores.get(documentId);
    if (!docData) {
      return res.status(404).json({ error: "Document not found" });
    }

    // Retrieve relevant chunks
    const retriever = docData.vectorStore.asRetriever(topK);
    const retrievedDocuments = await retriever.invoke(question);

    // Generate answer using Gemini
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    const context = retrievedDocuments
      .map((doc) => doc.pageContent)
      .join("\n\n");

    const prompt = `Based on the following context from the document, answer the question. If the answer cannot be found in the context, say "I cannot find the answer in the provided document."
try to read the context and tell a better answer in a good format also try to provide the examples according to document context you have and emit the response in proper markdown.
Context:
${context}

Question: ${question}

Answer:`;

    const result = await model.generateContent(prompt);
    const answer = result.response.text();

    res.json({
      success: true,
      question,
      answer,
      sources: retrievedDocuments.map((doc) => ({
        content: doc.pageContent,
        metadata: doc.metadata,
      })),
      documentInfo: {
        filename: docData.filename,
        totalChunks: docData.chunks,
      },
    });
  } catch (error) {
    console.error("Error querying document:", error);
    res.status(500).json({ error: error.message });
  }
});

// Get list of uploaded documents
app.get("/documents", (req, res) => {
  const documents = Array.from(vectorStores.entries()).map(([id, data]) => ({
    id,
    filename: data.filename,
    chunks: data.chunks,
    createdAt: data.createdAt,
  }));

  res.json({
    success: true,
    documents,
  });
});

// Delete a document
app.delete("/documents/:documentId", (req, res) => {
  const { documentId } = req.params;

  if (vectorStores.has(documentId)) {
    vectorStores.delete(documentId);
    res.json({ success: true, message: "Document deleted successfully" });
  } else {
    res.status(404).json({ error: "Document not found" });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === "LIMIT_FILE_SIZE") {
      return res.status(400).json({ error: "File too large" });
    }
  }
  res.status(500).json({ error: error.message });
});

app.listen(8000, () => {
  console.log("Server is running on port 8000");
});
