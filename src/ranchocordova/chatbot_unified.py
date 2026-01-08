
import os
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_loader import get_data_loader

# Globals
_llm = None
_embedder = None
_chunks = None
_chunk_embeddings = None
_energy_df = None
_cs_df = None
_dept_df = None


def initialize_models():
    """Load LLM, embedder, KB and dataframes once."""
    print("##### CALLING initialize_models()\n")
    global _llm, _embedder, _chunks, _chunk_embeddings
    global _energy_df, _cs_df, _dept_df

    if _llm is not None:
        return

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    print("Loading Rancho Cordova models with PDF support...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    )
    model.eval()

    _llm = (model, tokenizer)
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    base_path = os.path.join(os.path.dirname(__file__), "data")
    _energy_df = pd.read_csv(os.path.join(base_path, "Energy.txt"))
    _cs_df = pd.read_csv(os.path.join(base_path, "CustomerService.txt"))

    # Load department data if available
    dept_file = os.path.join(base_path, "Department-city of Rancho Cordova.txt")
    if os.path.exists(dept_file):
        _dept_df = pd.read_csv(dept_file)
    else:
        _dept_df = pd.DataFrame()

    print("Loading enhanced energy datasets with PDF support...")
    loader = get_data_loader()
    print("✅ Enhanced datasets loaded")

    # Build chunk KB
    _chunks = []

    # ENERGY TABLE
    for _, row in _energy_df.iterrows():
        _chunks.append(
            f"ENERGY_RECORD | "
            f"CustomerID={row['CustomerID']} | "
            f"AccountType={row['AccountType']} | "
            f"Month={row['Month']} | "
            f"EnergyConsumption_kWh={row['EnergyConsumption_kWh']}"
        )

    # CUSTOMER SERVICE
    for _, row in _cs_df.iterrows():
        text_row = " | ".join([f"{col}={row[col]}" for col in _cs_df.columns])
        _chunks.append(f"CS_RECORD | {text_row}")

    # DEPARTMENTS
    if not _dept_df.empty:
        for _, row in _dept_df.iterrows():
            text_row = " | ".join([f"{col}={row[col]}" for col in _dept_df.columns])
            _chunks.append(f"DEPT_RECORD | {text_row}")

    _chunks.extend(_extract_benchmark_insights(base_path))
    _chunks.extend(_extract_tou_rate_insights(base_path))
    _chunks.extend(_extract_rebate_insights(base_path))
    _chunks.extend(_extract_pdf_knowledge())

    print(f"✅ Total RAG chunks: {len(_chunks)}")
    _chunk_embeddings = _embedder.encode(_chunks, convert_to_numpy=True)
    print("✅ Rancho models initialized with PDF support.")


def _extract_pdf_knowledge() -> list:
    """Extract knowledge chunks from PDF documents."""
    chunks = []
    loader = get_data_loader()
    pdf_contents = loader.get_all_pdf_contents()

    if not pdf_contents:
        print("  ⚠️  No PDF documents found")
        return chunks

    print(f"\n📄 Extracting knowledge from {len(pdf_contents)} PDF documents...")

    for filename, pdf_data in pdf_contents.items():
        doc_type = _identify_document_type(filename)
        text = pdf_data["text"]
        sections = re.split(r"\n\n+", text)

        chunk_count = 0
        for i, section in enumerate(sections):
            section = section.strip()
            if len(section) > 50:
                chunk = (
                    f"PDF_DOCUMENT | "
                    f"Source={filename} | "
                    f"Type={doc_type} | "
                    f"Section={i + 1} | "
                    f"Content={section[:1000]}"
                )
                chunks.append(chunk)
                chunk_count += 1

        print(f"  ✓ Extracted {chunk_count} chunks from {filename}")

    print(f"  ✓ Total PDF chunks: {len(chunks)}")
    return chunks


def _identify_document_type(filename: str) -> str:
    """Identify document type based on filename"""
    filename_lower = filename.lower()
    if "annual" in filename_lower or "report" in filename_lower:
        return "Annual_Report"
    elif "cec" in filename_lower or "california" in filename_lower:
        return "Technical_Standard"
    elif "manual" in filename_lower:
        return "Manual"
    elif "policy" in filename_lower:
        return "Policy_Document"
    else:
        return "General_Document"


def _extract_benchmark_insights(base_path: str) -> list:
    """Dynamically extract utility comparison insights from CSV."""
    chunks = []
    csv_path = os.path.join(base_path, "CA_Benchmarks.csv")

    if not os.path.exists(csv_path):
        return chunks

    try:
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            chunks.append(
                f"UTILITY_COMPARISON | "
                f"Utility={row['Utility_or_CCA']} | "
                f"Type={row.get('Utility_Type', 'N/A')} | "
                f"Home_Type={row['Home_Type']} | "
                f"Avg_Monthly_kWh={row['Avg_Monthly_Usage_kWh']} | "
                f"Avg_Annual_kWh={row['Avg_Annual_Usage_kWh']} | "
                f"Rate_per_kWh=${row['Avg_Rate_usd_per_kWh']} | "
                f"Avg_Monthly_Bill=${row['Est_Avg_Monthly_Bill_usd']}"
            )

        smud_data = df[df["Utility_or_CCA"] == "SMUD"]
        if not smud_data.empty:
            smud_avg_rate = smud_data["Avg_Rate_usd_per_kWh"].mean()
            pge_data = df[df["Utility_or_CCA"] == "PG&E"]
            if not pge_data.empty:
                pge_avg_rate = pge_data["Avg_Rate_usd_per_kWh"].mean()
                savings_pct = (pge_avg_rate - smud_avg_rate) / pge_avg_rate * 100
                chunks.append(
                    f"UTILITY_SAVINGS | "
                    f"Comparison=SMUD_vs_PGE | "
                    f"SMUD_Rate=${smud_avg_rate:.3f}/kWh | "
                    f"PGE_Rate=${pge_avg_rate:.3f}/kWh | "
                    f"Savings={savings_pct:.0f}% | "
                    f"Description=SMUD residential customers save approximately {savings_pct:.0f}% on electricity "
                    f"rates compared to PG&E."
                )

    except Exception as e:
        print(f"  ⚠️  Error processing benchmarks: {e}")

    return chunks


def _extract_tou_rate_insights(base_path: str) -> list:
    """Dynamically extract TOU rate information from CSV."""
    chunks = []
    csv_path = os.path.join(base_path, "SMUD_TOU_Rates.csv")

    if not os.path.exists(csv_path):
        return chunks

    try:
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            chunks.append(
                f"TOU_RATE | "
                f"Plan={row['plan']} | "
                f"Period={row['period']} | "
                f"Season={row['season']} | "
                f"DayType={row['day_type']} | "
                f"Hours={row['start_time']}-{row['end_time']} | "
                f"Rate=${row['rate_per_kwh_usd']}/kWh"
            )

        if "period" in df.columns:
            peak_rates = df[df["period"].str.contains("Peak", case=False, na=False)]
            offpeak_rates = df[df["period"].str.contains("Off", case=False, na=False)]

            if not peak_rates.empty and not offpeak_rates.empty:
                avg_peak = peak_rates["rate_per_kwh_usd"].mean()
                avg_offpeak = offpeak_rates["rate_per_kwh_usd"].mean()
                savings_pct = (avg_peak - avg_offpeak) / avg_peak * 100
                chunks.append(
                    f"TOU_SAVINGS | "
                    f"Savings={savings_pct:.0f}% | "
                    f"Description=You can save up to {savings_pct:.0f}% by shifting energy usage from peak hours to off-peak."
                )

    except Exception as e:
        print(f"  ⚠️  Error processing TOU rates: {e}")

    return chunks


def _extract_rebate_insights(base_path: str) -> list:
    """Dynamically extract rebate program information from CSV."""
    chunks = []
    csv_path = os.path.join(base_path, "SMUD_Rebates.csv")

    if not os.path.exists(csv_path):
        return chunks

    try:
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            chunks.append(
                f"REBATE_PROGRAM | "
                f"CustomerType={row['customer_type']} | "
                f"Category={row['program_category']} | "
                f"Program={row['rebate_name']} | "
                f"Technologies={row['eligible_technologies']} | "
                f"Amount={row['typical_rebate_range_usd']}"
            )

    except Exception as e:
        print(f"  ⚠️  Error processing rebates: {e}")

    return chunks


def retrieve_top_k(query: str, k: int = 5) -> list:
    """Retrieve top-k most relevant chunks for a query."""
    if _chunks is None or _chunk_embeddings is None:
        initialize_models()

    query_emb = _embedder.encode([query], convert_to_numpy=True)
    similarities = np.dot(_chunk_embeddings, query_emb.T).flatten()
    top_indices = similarities.argsort()[-k:][::-1]

    return [_chunks[i] for i in top_indices]


def detect_agent_type(prompt: str) -> str:
    """Determine which agent should handle this request."""
    prompt_lower = prompt.lower()

    # Visualization keywords
    viz_keywords = [
        "chart", "graph", "plot", "visual", "show me", "display",
        "compare", "trend", "pattern", "distribution", "over time",
        "forecast", "history", "timeline"
    ]
    if any(kw in prompt_lower for kw in viz_keywords):
        return "visualization"

    # Customer service keywords
    cs_keywords = [
        "pothole", "water bill", "trash", "garbage", "recycling",
        "permit", "complaint", "report", "request", "service",
        "problem", "issue", "fix", "repair"
    ]
    if any(kw in prompt_lower for kw in cs_keywords):
        return "customer_service"

    # Energy keywords
    energy_keywords = [
        "energy", "electricity", "power", "kwh", "bill",
        "consumption", "usage", "rate", "peak", "rebate", "save"
    ]
    if any(kw in prompt_lower for kw in energy_keywords):
        return "energy"

    return "general"


def generate_response(prompt: str, use_rag: bool = True, agent_type: str = None) -> str:
    """
    Generate CUSTOMER-FRIENDLY response (NO CODE EVER).
    """
    if _llm is None:
        initialize_models()

    model, tokenizer = _llm

    if agent_type is None:
        agent_type = detect_agent_type(prompt)

    # Build context
    if use_rag:
        context_chunks = retrieve_top_k(prompt, k=5)
        context = "\n".join(context_chunks)
    else:
        context = ""

    # ========================================================================
    # CRITICAL: System prompts that PREVENT code generation
    # ========================================================================
    
    if agent_type == "visualization":
        system_msg = (
            "You are a friendly energy data assistant for Rancho Cordova residents. "
            "When asked about trends, patterns, or visualizations, explain the data insights "
            "in simple, clear language. DO NOT write any code. DO NOT show Python, SQL, or any "
            "programming language. Just explain what the data shows in plain English. "
            "A chart will be automatically generated and shown to the user."
        )
    elif agent_type == "energy":
        system_msg = (
            "You are an energy efficiency expert for Rancho Cordova and SMUD. "
            "Provide helpful, accurate information in plain English. "
            "NEVER write code or technical commands. Speak naturally like a helpful neighbor."
        )
    elif agent_type == "customer_service":
        system_msg = (
            "You are a friendly customer service representative for the City of Rancho Cordova. "
            "Help residents with city services. Be conversational and helpful. "
            "Never show code or technical details."
        )
    else:
        system_msg = (
            "You are a helpful assistant for Rancho Cordova residents. "
            "Answer in simple, friendly language. Never write code."
        )

    # Build full prompt
    if context:
        full_prompt = (
            f"{system_msg}\n\n"
            f"Context Data:\n{context}\n\n"
            f"Resident Question: {prompt}\n\n"
            f"Your Response (in plain English, NO CODE):"
        )
    else:
        full_prompt = (
            f"{system_msg}\n\n"
            f"Resident Question: {prompt}\n\n"
            f"Your Response:"
        )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": full_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=300,  # Shorter to prevent code generation
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # ========================================================================
    # SAFETY CHECK: Remove any code that slipped through
    # ========================================================================
    response = _remove_code_from_response(response)

    return response.strip()


def _remove_code_from_response(text: str) -> str:
    """
    Safety filter: Remove any code blocks that accidentally got generated.
    """
    # Remove Python code blocks
    text = re.sub(r'```python.*?```', '[Chart generated automatically]', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '[Chart generated automatically]', text, flags=re.DOTALL)
    
    # Remove import statements
    text = re.sub(r'import\s+\w+.*', '', text)
    text = re.sub(r'from\s+\w+.*', '', text)
    
    # Remove common code patterns
    code_patterns = [
        r'def\s+\w+\(.*?\):',
        r'class\s+\w+:',
        r'plt\.\w+\(',
        r'pd\.\w+\(',
        r'df\[.*?\]',
    ]
    
    for pattern in code_patterns:
        text = re.sub(pattern, '', text)
    
    return text.strip()


def generate_answer(prompt: str, agent_type: str = None) -> dict:
    """
    Main entry point for Flask app.
    Returns natural language + optional visualization (NO CODE TO CUSTOMER).
    """
    if _llm is None:
        initialize_models()

    if agent_type is None or agent_type == "":
        agent_type = detect_agent_type(prompt)

    print(f"🤖 Agent Type: {agent_type}")
    print(f"📝 Query: {prompt}")

    # Get natural language response (NO CODE)
    response_text = generate_response(prompt, use_rag=True, agent_type=agent_type)
    
    # Make sure no code leaked through
    response_text = _remove_code_from_response(response_text)
    
    print(f"✅ Response generated: {len(response_text)} chars")

    # Generate visualization if needed
    visualization = None
    
    if agent_type == "visualization":
        print(f"📊 Generating visualization...")
        try:
            from .viz import generate_simple_visualization
            visualization = generate_simple_visualization(prompt, _energy_df, _cs_df)
            
            if visualization:
                print(f"✅ Visualization generated")
                # Add friendly message about the chart
                response_text += "\n\n📊 I've created a chart below to show you this visually."
            else:
                print(f"⚠️ Visualization generation returned None")
                response_text += "\n\nI'd love to show you a chart, but I'm having trouble generating it right now."
                
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            import traceback
            traceback.print_exc()

    result = {
        "answer": response_text,
        "visualization": visualization
    }
    
    print(f"📤 Returning to customer: {len(result['answer'])} chars")
    
    return result


def chat(user_message: str, conversation_history: list = None) -> dict:
    """
    Main chat interface with enhanced PDF support.
    """
    if _llm is None:
        initialize_models()

    agent_type = detect_agent_type(user_message)

    # Check for document references
    if any(keyword in user_message.lower() for keyword in ["pdf", "document", "report", "annual"]):
        loader = get_data_loader()
        pdfs = loader.get_all_pdf_contents()
        if pdfs:
            pdf_names = ", ".join(pdfs.keys())
            user_message += f" (Available documents: {pdf_names})"

    response = generate_response(user_message, use_rag=True, agent_type=agent_type)
    
    # Safety check
    response = _remove_code_from_response(response)

    return {
        "response": response,
        "agent_type": agent_type,
        "context_used": True
    }
