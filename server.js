const express = require("express");
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const pdfParse = require('pdf-parse');
const { OpenAI } = require('openai');
const cors = require("cors");
const { Pinecone } = require("@pinecone-database/pinecone");

require('dotenv').config();

const app = express();
app.use(cors());

const port = 5000;
const uploadsDir = path.join(__dirname, "uploads");

if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir);
}

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    },
});

const upload = multer({ storage });
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,  // Use the API key from your environment variable
});


const indexName = 'rag-agent';  // The name of your index
const dimension = 1536;  // The dimension of your model
const metric = 'cosine';  // Cosine similarity metric

async function initPinecone() {
    try {
        // List existing indexes
        const indexes = await pc.listIndexes();

        // Check if the index exists

        await pc.createIndex(indexName, {
            dimension: 1536,  // Adjust dimension size based on your embeddings model
            metric: 'cosine',  // Use cosine similarity metric
            shardSize: 1,
        });
        console.log('Index created successfully');

    } catch (error) {
        console.error('Error initializing Pinecone:', error);
    }
}

async function upsertEmbeddings(indexName, embeddings, metadata, fileName) {
    const index = pc.index(indexName);
    try {
        await index.namespace('ns1').upsert(
            embeddings.map((embedding, i) => ({
                id: `${fileName}_vec${i + 1}`,  // Use the filename and index to create a unique ID
                values: embedding,  // Embedding vector
                metadata: metadata[i],  // Metadata for the vector
            }))
        );
        console.log("Embeddings successfully upserted");
    } catch (error) {
        console.error("Error upserting embeddings:", error);
    }
}


async function generateEmbeddings(text) {
    const response = await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text,
    });
    return response.data;
}

// await initPinecone();
app.use(express.json());



// Temporarily store file content by filename
let fileContents = {};

app.post("/api/upload", upload.single("file"), async (req, res) => {
    try {
        const file = req.file;
        if (!file) return res.status(400).send("No file uploaded.");

        if (file.mimetype === "application/pdf") {
            const pdfBuffer = fs.readFileSync(file.path);
            try {
                const pdfData = await pdfParse(pdfBuffer);

                // Store the extracted text by filename (use a unique key for each file)
                fileContents[file.filename] = pdfData.text;

                const chunkSize = 500;
                const textChunks = [];

                for (let i = 0; i < pdfData.text.length; i += chunkSize) {
                    textChunks.push(pdfData.text.slice(i, i + chunkSize));
                }


                const embeddingsPromises = textChunks.map(async (chunk) => {
                    const embedding = await generateEmbeddings(chunk);
                    return embedding[0].embedding; // Extract the embedding vector
                });

                const embeddings = await Promise.all(embeddingsPromises);

                // Prepare metadata
                const metadata = textChunks.map((chunk, index) => ({
                    genre: index % 2 === 0 ? 'drama' : 'action',  // Example metadata: alternating genres
                    content: chunk,
                }));

                await initPinecone();
                await upsertEmbeddings(indexName, embeddings, metadata, file.filename);

                // Optionally delete the file after processing
                fs.unlinkSync(file.path);

                return res.json({
                    message: "File uploaded, processed, and embeddings stored successfully",
                    fileName: file.filename,
                    text: pdfData.text,
                });
            } catch (error) {
                console.error("Error parsing PDF:", error);
                return res.status(500).json({ error: "Error parsing PDF", details: error.message });
            }
        } else {
            fs.unlinkSync(file.path); // Delete unsupported files
            return res.status(400).send("Unsupported file type.");
        }
    } catch (error) {
        console.error("Error handling file upload:", error);
        return res.status(500).json({ error: "Internal server error" });
    }
});

app.post('/api/search', async (req, res) => {
    const { fileName, query } = req.body;

    console.log("query", query)

    if (!query) {
        return res.status(400).send(' query not provided.');
    }

    // Retrieve the extracted text using the file name
    if (fileName) {
        // Retrieve the extracted text if fileName is provided
        const fileText = fileContents[fileName];
        if (!fileText) {
            return res.status(404).send('File not found or text not extracted.');
        }
    }

    try {
        console.log("Input Query for Embedding:", query);
        const queryEmbedding = await generateEmbeddings(query);
        const queryVector = queryEmbedding[0].embedding;


        const index = pc.Index(indexName);
        console.log("index", index)
        const queryRequest = {
            vector: queryVector, // Query vector (embedding of the search query)
            topK: 10, // Number of nearest neighbors to retrieve
            includeMetadata: true,
        };

        console.log('Query Request:', queryRequest);


        const result = await index.namespace('ns1').query(queryRequest);

        console.log('Pinecone Query Result:', result);


        // if (!result.matches || result.matches.length === 0) {
        //     return res.status(404).send('No relevant content found.');
        // }

        const context = result.matches.map(match => match.metadata.content).join("\n\n");

        console.log("context", context)


        const response = await openai.chat.completions.create({
            model: 'gpt-4', // Use OpenAI GPT model
            messages: [
                { role: 'system', content: 'You are an assistant that answers questions based on the file content.' },
                { role: 'user', content: `Here are some relevant sections of the file content:\n\n${context}` },
                { role: 'user', content: `The user query is: ${query}` },
            ],
        });

        const resultContent = response.choices[0].message.content;
        return res.json([{ title: 'Generated Answer', content: resultContent }]);
    } catch (error) {
        console.error("Error generating response:", error);
        return res.status(500).json({ error: 'Error generating response' });
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
