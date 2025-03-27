RAG -> Retrieval Augumented Generation

Documentes are indexed so that there is some heuristic information on the bases of which LLM's retrieve's relevant data from these documents to answer the questions asked to it . 

Steps to build RAG:-
1. Query Translation (Translate the ques into a form that is better suited for query retrieval)
2. Routing (Logical / Semantic)
3. Query Construction
(main steps :)
4. Indexing
5. Retrieval
6. Generation (Active Retrieval) 




Phase 1 -> Indexing 
Raw PDf files(Medical books) -> Kwonledge Source -> Chunking -> Embeddings -> Knowledge Base (Vector Store FAISS) 

Phase 2 
Connecting LLM (Mistral) with Knowledge Base 




