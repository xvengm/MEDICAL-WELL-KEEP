from __future__ import annotations

import io
import os
import uuid
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

#  Optional OCR dependencies for images 
try:
    import pytesseract  
    from PIL import Image 
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
    try:
        from PIL import Image  
    except Exception:
        Image = None  

#  Document parsers 
import fitz  
import docx  
try:
    import textract  
    TEXTRACT_AVAILABLE = True
except Exception:
    TEXTRACT_AVAILABLE = False

#  LLM + embeddings + vector store 
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma as LangChainChroma
from chromadb import PersistentClient

#  Validation model for LLM output 
from pydantic import BaseModel, Field, ValidationError

import requests  



# Pydantic response schema

class MedicalResponse(BaseModel):
    summary: str
    confidence: str
    main_issue: str = Field(alias="main issue")
    key_findings: str = Field(alias="key findings")
    treatment_plan: str = Field(alias="treatment / plan")
    important_notes: str = Field(alias="important notes")
    codes: Optional[List[str]] = None
    limitations: Optional[List[str]] = None
    followUpQuestions: Optional[List[str]] = None
    disclaimer: str



# Helpers: text chunking

def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text or "")


def _guess_image_mime(filename: Optional[str]) -> str:
    name = (filename or "").lower()
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        return "image/jpeg"
    elif name.endswith(".png"):
        return "image/png"
    
    return "image/png"


def _gemini_extract_text_from_image_bytes(img_bytes: bytes, mime_type: str, model: str = "gemini-1.5-flash") -> str:
    """
    Use Gemini multimodal to extract legible text from a medical image (scan/report/photo).
    Assumes genai.configure(api_key=...) has already been called.
    """
    
    gm = genai.GenerativeModel(model)
    parts = [
        {"mime_type": mime_type, "data": img_bytes},
        {
            "text": (
                "Extract all legible text from this medical image or scan. "
                "Return plain UTF-8 text only (no markdown, no commentary). "
                "Preserve medically relevant structure and units where possible."
            )
        },
    ]
    resp = gm.generate_content(parts)
    extracted = getattr(resp, "text", None)
    if not extracted:
        try:
            extracted = resp.candidates[0].content.parts[0].text  [attr-defined]
        except Exception:
            extracted = ""
    return extracted or ""



# Document parsing for multiple types

class DocumentParser:
    @staticmethod
    def from_pdf(file_bytes: bytes) -> str:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()

    @staticmethod
    def from_docx(file_bytes: bytes) -> str:
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs)

    @staticmethod
    def from_doc(file_bytes: bytes) -> str:
        if TEXTRACT_AVAILABLE:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".doc") as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                raw = textract.process(tmp.name)  
            return raw.decode("utf-8", errors="ignore")
        raise RuntimeError(
            "Legacy .doc parsing requires 'textract' and system dependencies. "
            "Please convert the file to DOCX/PDF."
        )

    @staticmethod
    def from_txt(file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8", errors="ignore")

    @staticmethod
    def from_image(file_bytes: bytes, filename: Optional[str] = None) -> str:
        """
        Preferred path:
          1) Try Tesseract OCR (if available) by opening the image from memory (BytesIO).
          2) If OCR not available OR returns too little text, fallback to Gemini multimodal
             to extract legible text from the image.
        """
        
        if OCR_AVAILABLE and Image is not None:
            try:
                img = Image.open(io.BytesIO(file_bytes))
                text = pytesseract.image_to_string(img)  
                
                if (text or "").strip():
                    return text
            except Exception:
                pass

        # 2) Gemini multimodal fallback 
        try:
            mime = _guess_image_mime(filename)
            extracted = _gemini_extract_text_from_image_bytes(file_bytes, mime)
            if extracted.strip():
                return extracted
        except Exception as e:
            
            raise RuntimeError(
                f"Image text extraction failed. Ensure either Tesseract OCR is installed "
                f"or Gemini Vision API access is available. Details: {e}"
            ) from e
        return ""



# RAG Agent

class RAGAgent:
    def __init__(
        self,
        chroma_dir: str = "chroma_db",
        collection_name: str = "medical_kb",
        model_name: str = "gemini-1.5-flash",
        embedding_model: str = "models/embedding-001",
        preload_sources: bool = True,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not found in environment (create a .env or export it).")

       
        genai.configure(api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, api_key=api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, api_key=api_key)

        
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(exist_ok=True)

       
        self._client = PersistentClient(path=str(self.chroma_dir))
        self.collection_name = collection_name

        
        self.vectorstore = LangChainChroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_dir),
        )

        
        if preload_sources:
            self._preload_trusted_sources_if_needed()

        # System prompt
        self.system_msg = (
            "You are a medical summarization assistant. Your purpose is to transform complex medical "
            "records into clear, structured, plain-language summaries for non-technical users. "
            "Follow these principles:\n\n"
            "1. Do not include the patient’s name or identifiers. Always begin with: "
            "'Based on your uploaded record…'.\n\n"
            "2. Use short, simple sentences. Avoid medical jargon whenever possible. "
            "If an abbreviation appears, expand it the first time (e.g., 'BP (blood pressure)').\n\n"
            "3. Organize your summary into the following sections:\n"
            "   • Main Issue – the primary diagnosis or health concern.\n"
            "   • Key Findings – important observations, test results, or symptoms.\n"
            "   • Treatment / Plan – recommended care, medication, or procedures.\n"
            "   • Important Notes – any additional details the patient should know.\n\n"
            "4. After the summary, always include:\n"
            "   • A Confidence Score (High / Medium / Low) based on how clear and complete "
            "the medical record is.\n"
            "   • A Disclaimer: 'This summary is for informational purposes only. Always seek advice "
            "from your doctor or qualified health practitioner for professional medical guidance.'\n\n"
            "5. Balance accuracy with readability: remain faithful to the medical content while ensuring "
            "the explanation is approachable for someone with no medical background.\n\n"
            "Your tone should be supportive, professional, and easy to follow."
        )

        # User prompt 
        self.user_msg = (
            "Report:\n{report}\n\n"
            "Context from trusted sources:\n{context}\n\n"
            "Answer ONLY with the following JSON structure:\n"
            "{{\n"
            '  "summary": "...",\n'
            '  "confidence": "...",\n'
            '  "main issue": "...",\n'
            '  "key findings": "...",\n'
            '  "treatment / plan": "...",\n'
            '  "important notes": "...",\n'
            '  "codes": ["ICD-10 codes if available"],\n'
            '  "limitations": ["..."],\n'
            '  "followUpQuestions": ["..."],\n'
            '  "disclaimer": "..." \n'
            "}}\n"
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_msg),
                HumanMessagePromptTemplate.from_template(self.user_msg),
            ]
        )
        self.chain = self.prompt | self.llm

    # - Public API -
    def process_document(self, file_bytes: bytes, filename: str) -> Dict:
        """
        Returns:
          dict containing the validated fields + 'session_id' + 'context' (for follow-ups)
        """
        report_text = self._extract_text(file_bytes, filename)
        if not report_text.strip():
            raise RuntimeError("No readable text found in the uploaded file.")

        context = self._retrieve_context(report_text)
        raw = self._invoke_llm(report_text[:2000], context)  
        data = self._parse_llm_json(raw)

        session_id = uuid.uuid4().hex
        shaped = self._shape_response(data, include_session_id=session_id)
        shaped["context"] = context  
        return shaped

    def process_question(self, question: str, context: str) -> Dict:
        raw = self._invoke_llm(question, context)
        data = self._parse_llm_json(raw)
        return self._shape_response(data)

    # - Internals -
    def _extract_text(self, file_bytes: bytes, filename: str) -> str:
        name = (filename or "").lower()
        if name.endswith(".pdf"):
            return DocumentParser.from_pdf(file_bytes)
        if name.endswith(".docx"):
            return DocumentParser.from_docx(file_bytes)
        if name.endswith(".doc"):
            return DocumentParser.from_doc(file_bytes)
        if name.endswith(".txt"):
            return DocumentParser.from_txt(file_bytes)
        if any(name.endswith(ext) for ext in (".png", ".jpg", ".jpeg")):
            
            return DocumentParser.from_image(file_bytes, filename)
        
        return DocumentParser.from_txt(file_bytes)

    def _retrieve_context(self, query: str, k: int = 5) -> str:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        try:
            docs = retriever.get_relevant_documents(query)  
        except Exception:
            docs = retriever.invoke(query)  
        return "\n\n".join(getattr(d, "page_content", "") for d in docs if getattr(d, "page_content", ""))

    def _invoke_llm(self, report: str, context: str) -> str:
        resp = self.chain.invoke({"report": report, "context": context})
        return getattr(resp, "content", "") or str(resp)

    def _parse_llm_json(self, text: str) -> MedicalResponse:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("The model did not return valid JSON.")
        try:
            parsed = json.loads(text[start : end + 1])
            return MedicalResponse.model_validate(parsed)
        except ValidationError as ve:
            raise RuntimeError(f"Response failed schema validation: {ve}") from ve
        except json.JSONDecodeError as je:
            raise RuntimeError(f"Failed to parse JSON from model output: {je}") from je

    def _shape_response(self, valid: MedicalResponse, include_session_id: Optional[str] = None) -> Dict:
        shaped = {
            "summary": valid.summary,
            "confidence": valid.confidence,
            "main_issue": valid.main_issue,
            "key_findings": valid.key_findings,
            "treatment_plan": valid.treatment_plan,
            "important_notes": valid.important_notes,
            "codes": valid.codes or [],
            "follow_up_questions": valid.followUpQuestions or [],
            "disclaimer": valid.disclaimer,
            "limitations": valid.limitations or [],
        }
        if include_session_id:
            shaped["session_id"] = include_session_id
        return shaped

    def _preload_trusted_sources_if_needed(self) -> None:
        try:
            collection = self._client.get_collection(self.collection_name)
            count = collection.count()
        except Exception:
            count = 0

        if count and count > 0:
            return

        sources = {
            "WHO": "https://en.wikipedia.org/wiki/World_Health_Organization",
            "NIH": "https://en.wikipedia.org/wiki/United_States_National_Institutes_of_Health",
            "Mayo_Clinic": "https://en.wikipedia.org/wiki/Mayo_Clinic",
            "NICE": "https://en.wikipedia.org/wiki/National_Institute_for_Health_and_Care_Excellence",
            "FDA": "https://en.wikipedia.org/wiki/United_States_Food_and_Drug_Administration",
            "MedlinePlus": "https://en.wikipedia.org/wiki/MedlinePlus",
        }

        for name, url in sources.items():
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                text = r.text
                if text:
                    chunks = chunk_text(text)
                    self.vectorstore.add_texts(texts=chunks, metadatas=[{"source": name}] * len(chunks))
            except Exception:
                continue
